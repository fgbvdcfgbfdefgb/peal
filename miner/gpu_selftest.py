"""
GPU engine validation - run this ONCE on the mining machine.

The CUDA kernels cannot be exercised without a GPU, so this script compares
the GPU pipeline against the pure-Python/numpy reference on identical
inputs, stage by stage:

  1. commitment roots      (keyed BLAKE3 Merkle roots)
  2. noise-seed derivation (Legacy and Salted)
  3. noise matrices        (uniform rows + permutation pairs)
  4. noised GEMM + the jackpot fold
  5. jackpot hashes and the hit set (both a saturating bound, where every
     partition must hit, and a realistic bound, where the hit sets must
     match exactly)
  6. a mined share is verified locally and by the official Rust module
     when it is installed

Usage:

    pip install cupy-cuda12x numpy blake3
    python gpu_selftest.py                 # default configuration
    python gpu_selftest.py --m 64 --n 64 --k 2048 --rank 64
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from pearl_miner.core import (
    IncompleteBlockHeader, MiningConfiguration, compute_job_key,
    compute_jackpot_hash, compute_noise_for_indices, derive_noise_seeds,
    extract_difficulty_bound, hash_meets_bound, matrix_roots,
    derivation_for_cert_version, SIGNAL_MAX, SIGNAL_MIN,
)
from pearl_miner.patterns import PeriodicPattern
from pearl_miner.proof import plain_proof_from_hit, verify_plain_proof

try:
    from pearl_miner.engine_cuda import GpuEngine
    from pearl_miner.cuda_kernels import gpu_devices
except Exception as exc:  # pragma: no cover
    print(f"GPU engine unavailable: {exc}")
    print("Install CuPy first:  pip install cupy-cuda12x")
    raise SystemExit(2)

try:
    import pearl_mining as _rust
except Exception:
    _rust = None

PASS = 0
FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [ok] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}" + (f" - {detail}" if detail else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=128)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--rows", default="0,1,2,3")
    ap.add_argument("--cols", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--cert-version", type=int, default=3)
    args = ap.parse_args()

    devices = gpu_devices()
    if not devices:
        print("No usable GPU found (need compute capability >= 7.5).")
        return 2
    if args.device not in devices:
        print(f"Device {args.device} is not usable; available: {devices}")
        return 2

    rows = PeriodicPattern.from_list(
        [int(x) for x in args.rows.split(",") if x.strip()])
    cols = PeriodicPattern.from_list(
        [int(x) for x in args.cols.split(",") if x.strip()])
    cfg = MiningConfiguration(args.k, args.rank, rows, cols)

    ok, why = GpuEngine.supports(args.m, args.n, args.k, args.rank, rows, cols)
    if not ok:
        print(f"Configuration not supported by the GPU engine: {why}")
        return 2

    print(f"GPU selftest: m={args.m} n={args.n} k={args.k} rank={args.rank} "
          f"rows={args.rows} cols={args.cols} batch={args.batch} "
          f"device={args.device}")

    header = IncompleteBlockHeader(
        version=0x20000000, prev_block=bytes(range(31, -1, -1)),
        merkle_root=bytes([9]) * 32, timestamp=0x66666666, nbits=0x1d00ffff)
    derivation = derivation_for_cert_version(args.cert_version)

    engine = GpuEngine(args.m, args.n, args.k, args.rank, rows, cols,
                       batch_size=args.batch, device_id=args.device, seed=7)
    engine.set_job(header, args.cert_version)
    print(f"  engine ready on device {args.device}")

    # ---- fixed inputs -----------------------------------------------------
    rng = np.random.default_rng(1234)
    A = rng.integers(SIGNAL_MIN, SIGNAL_MAX + 1,
                     size=(args.batch, args.m, args.k), dtype=np.int8)
    Bt = rng.integers(SIGNAL_MIN, SIGNAL_MAX + 1,
                      size=(args.batch, args.n, args.k), dtype=np.int8)

    # ---- 1. commitment roots ---------------------------------------------
    print("\n== commitment roots ==")
    t0 = time.time()
    engine.run_batch(matrices=(A, Bt))
    dt = time.time() - t0
    gpu_roots_a = engine.d_roots_a.get().astype("<u4").tobytes()
    gpu_roots_b = engine.d_roots_b.get().astype("<u4").tobytes()
    ok_a = ok_b = True
    for t in range(args.batch):
        ra, rb = matrix_roots(A[t], Bt[t], engine.job_key)
        if ra != gpu_roots_a[t * 32:(t + 1) * 32]:
            ok_a = False
        if rb != gpu_roots_b[t * 32:(t + 1) * 32]:
            ok_b = False
    check("A roots match the CPU reference", ok_a)
    check("B^T roots match the CPU reference", ok_b)

    # ---- 2/3/4/5. saturating bound: every partition must hit -------------
    print("\n== saturating bound (every partition must hit) ==")
    sat_header = IncompleteBlockHeader(
        version=0x20000000, prev_block=bytes(range(31, -1, -1)),
        merkle_root=bytes([9]) * 32, timestamp=0x66666666, nbits=0x207fffff)
    engine.set_job(sat_header, args.cert_version)
    hits = engine.run_batch(matrices=(A, Bt))
    expected_hits = args.batch * engine.P
    check(f"hit count == {expected_hits} (got {len(hits)})",
          len(hits) == expected_hits)

    from pearl_miner.trial import compute_jackpot_words
    gpu_map = {(h["trial"], tuple(h["a_rows"]), tuple(h["b_cols"])):
               (tuple(h["jackpot"]), h["jackpot_hash"]) for h in hits}
    jackpot_ok = hash_ok = 0
    jackpot_total = 0
    for t in range(args.batch):
        job_key = engine.job_key
        ra, rb = matrix_roots(A[t], Bt[t], job_key)
        b_seed, a_seed = derive_noise_seeds(job_key, ra, rb, args.m, args.n,
                                            derivation)
        noise_a, noise_b = compute_noise_for_indices(
            args.k, args.rank, (b_seed, a_seed), range(args.m), range(args.n))
        a_noised = A[t].astype(np.int32) + noise_a.astype(np.int32)
        bt_noised = Bt[t].astype(np.int32) + noise_b.astype(np.int32)
        for a_rows in engine.partitions_a:
            for b_cols in engine.partitions_b:
                words = compute_jackpot_words(a_noised, bt_noised, a_rows,
                                              b_cols, args.k, args.rank)
                digest = compute_jackpot_hash(words, a_seed)
                jackpot_total += 1
                got = gpu_map.get((t, tuple(a_rows), tuple(b_cols)))
                if got is None:
                    continue
                if got[0] == tuple(int(w) for w in words):
                    jackpot_ok += 1
                if got[1] == digest:
                    hash_ok += 1
    check(f"all {jackpot_total} (trial, partition) pairs present on the GPU",
          len(gpu_map) == jackpot_total, f"{len(gpu_map)} found")
    check(f"jackpot words match the CPU reference ({jackpot_ok}/"
          f"{jackpot_total})", jackpot_ok == jackpot_total)
    check(f"jackpot hashes match the CPU reference ({hash_ok}/"
          f"{jackpot_total})", hash_ok == jackpot_total)

    # ---- realistic bound: hit sets must agree ----------------------------
    print("\n== realistic bound (hit sets must match) ==")
    engine.set_job(header, args.cert_version)
    hits = engine.run_batch(matrices=(A, Bt))
    bound = extract_difficulty_bound(header.nbits, cfg)
    from pearl_miner.trial import compute_jackpot_words
    expected = set()
    for t in range(args.batch):
        job_key = engine.job_key
        ra, rb = matrix_roots(A[t], Bt[t], job_key)
        b_seed, a_seed = derive_noise_seeds(job_key, ra, rb, args.m, args.n,
                                            derivation)
        noise_a, noise_b = compute_noise_for_indices(
            args.k, args.rank, (b_seed, a_seed), range(args.m), range(args.n))
        a_noised = A[t].astype(np.int32) + noise_a.astype(np.int32)
        bt_noised = Bt[t].astype(np.int32) + noise_b.astype(np.int32)
        for a_rows in engine.partitions_a:
            for b_cols in engine.partitions_b:
                words = compute_jackpot_words(a_noised, bt_noised, a_rows,
                                              b_cols, args.k, args.rank)
                digest = compute_jackpot_hash(words, a_seed)
                if hash_meets_bound(digest, bound):
                    expected.add((t, tuple(a_rows), tuple(b_cols),
                                  tuple(int(w) for w in words), digest))
    got = {(h["trial"], tuple(h["a_rows"]), tuple(h["b_cols"]),
            tuple(h["jackpot"]), h["jackpot_hash"]) for h in hits}
    check(f"hit count matches the CPU reference ({len(hits)} vs "
          f"{len(expected)})", len(hits) == len(expected))
    check("hit contents match exactly", got == expected,
          f"{len(got & expected)} of {len(expected)} matched")

    # ---- share verification ----------------------------------------------
    print("\n== share verification ==")
    if hits:
        proof = plain_proof_from_hit(hits[0], cfg)
        ok, msg = verify_plain_proof(header, proof, header.nbits, derivation)
        check("GPU share verifies locally", ok, msg)
        if _rust is not None:
            rp = _rust.PlainProof.from_base64(proof.to_base64())
            rh = _rust.IncompleteBlockHeader(
                0x20000000, bytes(range(31, -1, -1)), bytes([9]) * 32,
                0x66666666, 0x1d00ffff)
            fn = (_rust.verify_plain_proof_v3 if args.cert_version == 3
                  else _rust.verify_plain_proof_v2)
            ok2, msg2 = fn(rh, rp)
            check("GPU share verifies with the Rust module", ok2, msg2)
        else:
            print("  (pearl_mining not installed - skipped)")
    else:
        print("  (no hits to verify with this bound)")

    # ---- throughput -------------------------------------------------------
    print("\n== throughput (random matrices, realistic bound) ==")
    engine.set_job(header, args.cert_version)
    t0 = time.time()
    n_trials = 0
    for _ in range(3):
        engine.run_batch()
        n_trials += args.batch
    dt = time.time() - t0
    macs = n_trials * args.m * args.n * engine.config.dot_product_length
    print(f"  {n_trials} trials in {dt:.2f}s -> "
          f"{n_trials / dt:,.0f} trials/s, {macs / dt / 1e9:,.1f} GMAC/s "
          f"on device {args.device}")

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
