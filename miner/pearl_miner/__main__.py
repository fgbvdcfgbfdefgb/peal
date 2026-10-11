"""
Pearl solo miner - offline, source-only, GitHub-relayed.

Usage (on the offline Linux mining box):

    python -m pearl_miner --jobs jobs.txt --shares share.txt
    python -m pearl_miner --backend cuda --devices 0,1,2,3

The miner:

  1. watches ``jobs.txt`` for jobs published by the Windows relay
     (``git pull`` on the mining machine - the miner never pulls by itself),
  2. mines with the GPU engine (CuPy/NVRTC, no compiler needed) or the
     pure-numpy engine as a fallback,
  3. verifies every share locally before writing it,
  4. appends shares to ``share.txt`` (``git push`` on the mining machine).

Mining configuration defaults mirror a consensus-valid dense setup:

    m = 128, n = 128, k = 4096, rank = 128
    rows pattern [0,1,2,3] (tile 4), cols pattern [0..7] (tile 8)

One worker thread runs per GPU when ``--devices`` lists more than one id;
each worker watches jobs.txt independently and appends verified shares.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time

from .core import IncompleteBlockHeader, MiningConfiguration
from .patterns import PeriodicPattern
from .engine_np import NumpyEngine
from .jobfile import JobWatcher, ShareWriter, utcnow_iso
from .proof import plain_proof_from_hit, verify_plain_proof
from .core import derivation_for_cert_version


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pearl_miner",
        description="Offline Pearl (PRL) solo miner - reads jobs.txt, "
                    "appends shares to share.txt.")
    p.add_argument("--jobs", default="jobs.txt",
                   help="path to jobs.txt (default: jobs.txt)")
    p.add_argument("--shares", default="share.txt",
                   help="path to share.txt (default: share.txt)")
    p.add_argument("--m", type=int, default=128, help="rows of A (tokens)")
    p.add_argument("--n", type=int, default=128, help="columns of B")
    p.add_argument("--k", type=int, default=4096, help="common dimension")
    p.add_argument("--rank", type=int, default=128, help="noise rank")
    p.add_argument("--rows", default="0,1,2,3",
                   help="rows pattern (comma separated indices)")
    p.add_argument("--cols", default="0,1,2,3,4,5,6,7",
                   help="cols pattern (comma separated indices)")
    p.add_argument("--backend", choices=["auto", "cuda", "numpy"],
                   default="auto",
                   help="auto = use the GPU when CuPy and a supported "
                        "configuration are available (default: auto)")
    p.add_argument("--devices", default="0", metavar="IDS",
                   help="comma separated GPU ids to use, e.g. 0,1,2,3 "
                        "(default: 0).  Ignored by the numpy backend.")
    p.add_argument("--batch", type=int, default=256,
                   help="trials per batch for the numpy engine "
                        "(lower it if RAM is tight)")
    p.add_argument("--gpu-batch", type=int, default=512,
                   help="trials per batch per GPU (default: 512)")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed (default: random)")
    p.add_argument("--poll", type=float, default=2.0,
                   help="jobs.txt poll interval in seconds")
    p.add_argument("--stats", type=float, default=30.0,
                   help="stats print interval in seconds")
    p.add_argument("--once", type=int, default=0, metavar="BATCHES",
                   help="mine BATCHES batches on the current job then exit "
                        "(0 = run forever)")
    p.add_argument("--max-shares", type=int, default=0, metavar="N",
                   help="stop after writing N shares (0 = unlimited)")
    return p


def parse_pattern(text: str) -> PeriodicPattern:
    idx = [int(x) for x in text.split(",") if x.strip() != ""]
    return PeriodicPattern.from_list(idx)


def parse_devices(text: str):
    ids = []
    for part in text.split(","):
        part = part.strip()
        if part == "":
            continue
        try:
            ids.append(int(part))
        except ValueError:
            raise SystemExit(f"[miner] bad --devices value: {part!r}")
    if not ids:
        raise SystemExit("[miner] --devices is empty")
    return ids


def gpu_status_line() -> str:
    """One-line GPU status for the fallback warning."""
    try:
        from .cuda_kernels import gpu_status
        return gpu_status()
    except Exception:  # pragma: no cover
        return "GPU status unavailable"


def usable_gpu_ids(ids):
    """Filter GPU ids to those CuPy can actually run on."""
    try:
        from .cuda_kernels import gpu_devices
    except Exception as exc:  # CuPy missing or unusable
        print(f"[miner] GPU backend unavailable: {exc}", file=sys.stderr)
        return []
    available = gpu_devices()
    usable = [i for i in ids if i in available]
    if len(usable) != len(ids):
        missing = [i for i in ids if i not in available]
        print(f"[miner] ignoring unusable GPU ids {missing} "
              f"(available: {available})", file=sys.stderr)
    return usable


def make_engine(args, rows, cols, device_id=None):
    """Create the mining engine for one worker; returns (engine, backend)."""
    if args.backend in ("auto", "cuda"):
        from .engine_cuda import GpuEngine
        ok, why = GpuEngine.supports(args.m, args.n, args.k, args.rank,
                                     rows, cols)
        if ok:
            try:
                engine = GpuEngine(args.m, args.n, args.k, args.rank, rows,
                                   cols, batch_size=args.gpu_batch,
                                   device_id=device_id or 0, seed=args.seed)
                return engine, "cuda"
            except Exception as exc:  # pragma: no cover - driver problems
                if args.backend == "cuda":
                    print(f"[miner] GPU engine failed on device "
                          f"{device_id}: {exc}", file=sys.stderr)
                    print(f"[miner] {gpu_status_line()}", file=sys.stderr)
                    print("[miner] run ./run_miner.sh --gpu-info to diagnose, "
                          "or --backend numpy to mine on the CPU",
                          file=sys.stderr)
                    raise SystemExit(2)
                print(f"[miner] WARNING: GPU engine unavailable ({exc}); "
                      f"falling back to the numpy engine", file=sys.stderr)
        elif args.backend == "cuda":
            print(f"[miner] configuration not supported by the GPU engine: "
                  f"{why}", file=sys.stderr)
            raise SystemExit(2)
        else:
            print(f"[miner] WARNING: GPU engine cannot mine this "
                  f"configuration ({why}); using the numpy engine",
                  file=sys.stderr)
    return NumpyEngine(args.m, args.n, args.k, args.rank, rows, cols,
                       batch_size=args.batch, seed=args.seed), "numpy"


class Worker(threading.Thread):
    """Mines one job with one engine and appends verified shares."""

    def __init__(self, name, engine, backend, args, writer, lock, stop_event,
                 total_shares):
        super().__init__(name=name, daemon=True)
        self.engine = engine
        self.backend = backend
        self.args = args
        self.writer = writer
        self.lock = lock
        self.stop_event = stop_event
        self.total_shares = total_shares      # shared counter (with lock)
        self.job = None
        self.header = None
        self.job_trials = 0
        self.job_shares = 0
        self.total_trials = 0
        self.batches_done = 0
        self.t_job = time.time()
        self.t_stats = time.time()

    def _log(self, msg):
        print(f"[{self.name}] {msg}", flush=True)

    def run(self):
        args = self.args
        host = socket.gethostname()
        watcher = JobWatcher(args.jobs, poll_interval=args.poll)
        config_desc = (f"m={args.m} n={args.n} k={args.k} rank={args.rank} "
                       f"rows={args.rows} cols={args.cols}")
        batch = args.gpu_batch if self.backend == "cuda" else args.batch
        self._log(f"backend={self.backend} {config_desc} "
                  f"batch={batch} waiting for a job in {args.jobs} ...")

        try:
            while not self.stop_event.is_set():
                new_job = watcher.poll()
                if new_job is not None:
                    if self.job is None or new_job.job_id != self.job.job_id:
                        self._adopt(new_job)
                if self.job is None:
                    time.sleep(args.poll)
                    continue

                hits = self.engine.run_batch()
                self.total_trials += batch
                self.job_trials += batch

                for hit in hits:
                    proof = plain_proof_from_hit(hit, self.engine.config)
                    ok, msg = verify_plain_proof(
                        self.header, proof, self.header.nbits,
                        derivation_for_cert_version(self.job.cert_version))
                    if not ok:
                        self._log(f"WARNING: local verification failed for a "
                                  f"hit ({msg}) - share discarded")
                        continue
                    share = {
                        "job_id": self.job.job_id,
                        "mining_job": self.job.gateway_job_dict(),
                        "plain_proof": proof.to_base64(),
                        "found_at": utcnow_iso(),
                        "miner": host,
                        "worker": self.name,
                        "attempts": self.job_trials,
                        "elapsed_s": round(time.time() - self.t_job, 3),
                        "backend": self.backend,
                        "config": config_desc,
                    }
                    with self.lock:
                        self.writer.append(share)
                        self.total_shares[0] += 1
                    self.job_shares += 1
                    self._log(f"*** SHARE FOUND and verified locally *** "
                              f"job={self.job.job_id} rows={hit['a_rows'][0]}.."
                              f" cols={hit['b_cols'][0]}.. -> appended to "
                              f"{args.shares} (push it!)")
                    if args.max_shares and self.total_shares[0] >= args.max_shares:
                        self.stop_event.set()
                        return

                if args.once:
                    self.batches_done += 1
                    if self.batches_done >= args.once:
                        self._log(f"--once finished ({args.once} batches)")
                        self.stop_event.set()
                        return

                now = time.time()
                if now - self.t_stats >= args.stats:
                    rate = self.job_trials / max(1e-9, now - self.t_job)
                    macs = rate * args.m * args.n * \
                        self.engine.config.dot_product_length
                    self._log(f"job={self.job.job_id} trials={self.job_trials} "
                              f"({rate:.1f} trials/s, {macs / 1e9:.2f} GMAC/s) "
                              f"shares={self.job_shares}")
                    self.t_stats = now
        except KeyboardInterrupt:
            self._log("interrupted")
        self._log(f"stopped. trials={self.total_trials} "
                  f"shares={self.job_shares}")

    def _adopt(self, new_job):
        try:
            header = IncompleteBlockHeader.from_bytes(
                new_job.incomplete_header_bytes)
        except ValueError as exc:
            self._log(f"bad job header: {exc}")
            return
        try:
            self.engine.set_job(header, new_job.cert_version)
        except ValueError as exc:
            self._log(f"job rejected ({exc}); waiting for a usable job")
            return
        self.job = new_job
        self.header = header
        self.job_trials = 0
        self.job_shares = 0
        self.batches_done = 0
        self.t_job = time.time()
        self.t_stats = time.time()
        self._log(f"new job {new_job.job_id}: height={new_job.height} "
                  f"prev={new_job.prev_hash} nbits=0x{header.nbits:08x} "
                  f"cert_v={new_job.cert_version} target={new_job.target}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    rows = parse_pattern(args.rows)
    cols = parse_pattern(args.cols)
    try:
        MiningConfiguration(args.k, args.rank, rows, cols).sanity_check(
            args.m, args.n)
    except ValueError as exc:
        print(f"[miner] invalid mining configuration: {exc}", file=sys.stderr)
        return 2

    writer = ShareWriter(args.shares)
    lock = threading.Lock()
    stop_event = threading.Event()
    total_shares = [0]

    device_ids = parse_devices(args.devices)
    if args.backend == "numpy":
        device_ids = device_ids[:1]
        engines = [make_engine(args, rows, cols)[0]]
        backends = ["numpy"]
    else:
        usable = usable_gpu_ids(device_ids)
        if not usable:
            if args.backend == "cuda":
                print("[miner] no usable GPU found", file=sys.stderr)
                print(f"[miner] {gpu_status_line()}", file=sys.stderr)
                print("[miner] install CuPy with ./run_miner.sh --install-gpu, "
                      "diagnose with ./run_miner.sh --gpu-info, or mine on "
                      "the CPU with --backend numpy", file=sys.stderr)
                return 2
            print("[miner] WARNING: no usable GPU - using the numpy engine "
                  "(run ./run_miner.sh --gpu-info to see why)",
                  file=sys.stderr)
            engines, backends = [make_engine(args, rows, cols)[0]], ["numpy"]
        else:
            engines, backends = [], []
            for dev in usable:
                try:
                    engine, backend = make_engine(args, rows, cols,
                                                  device_id=dev)
                except SystemExit:
                    raise
                except Exception as exc:
                    print(f"[miner] device {dev} failed to start: {exc}",
                          file=sys.stderr)
                    continue
                engines.append(engine)
                backends.append(backend)

    if not engines:
        print("[miner] no engines started", file=sys.stderr)
        return 2

    print("[miner] Pearl solo miner (source build, offline)")
    if backends and all(b == "numpy" for b in backends):
        print("[miner] *** CPU (numpy) engine - the GPUs are NOT being "
              "used.  Run ./run_miner.sh --gpu-info to find out why. ***")
    print(f"[miner] jobs file : {args.jobs}")
    print(f"[miner] share file: {args.shares}")
    labels = []
    for i, backend in enumerate(backends):
        labels.append(f"{backend}@gpu{device_ids[i]}"
                      if backend == "cuda" else backend)
    print(f"[miner] engines  : {', '.join(labels)}")

    workers = []
    for i, (engine, backend) in enumerate(zip(engines, backends)):
        name = f"{backend}-{device_ids[i]}" if backend == "cuda" else backend
        w = Worker(name, engine, backend, args, writer, lock, stop_event,
                   total_shares)
        w.start()
        workers.append(w)

    try:
        while any(w.is_alive() for w in workers):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[miner] interrupted - stopping workers")
        stop_event.set()
        for w in workers:
            w.join(timeout=10)

    print(f"[miner] done. total shares written={total_shares[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
