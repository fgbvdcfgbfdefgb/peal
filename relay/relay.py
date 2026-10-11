"""
Pearl relay (Windows): bridges the local ``pearld`` node and the offline
Linux mining machine through GitHub files.

    pearld + pearl-gateway  <--JSON-RPC-->  relay  <--GitHub-->  miner
        (Windows)                                        (Linux, offline)

What it does, continuously:

  1. asks the gateway for the current mining job (``getMiningInfo``),
  2. publishes new jobs to ``jobs.txt`` in the GitHub repo (the miner
     ``git pull``s them on its side),
  3. reads ``share.txt`` from the repo (the miner ``git push``es it),
  4. submits every new share to the gateway (``submitPlainProof``), skipping
     shares whose header no longer matches the current job (the gateway
     would skip them anyway, and building a ZK proof for them is expensive).

Configuration (environment variables):

    GITHUB_TOKEN   - token with contents:write on the repo   (required)
    GITHUB_REPO    - "owner/name", default fgbvdcfgbfdefgb/peal
    GITHUB_BRANCH  - default main
    JOBS_FILE      - default jobs.txt
    SHARES_FILE    - default share.txt
    RELAY_POLL_SECONDS   - gateway poll interval, default 1.0
    RELAY_SYNC_SECONDS   - GitHub sync interval, default 5.0
    RELAY_STATE_FILE     - default .relay-state.json (consumed-offset memory)
    RELAY_DRY_RUN        - "1" = read-only, do not touch GitHub or the gateway

Run it:

    pip install requests
    set GITHUB_TOKEN=ghp_xxx
    python -m relay
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import time

from .github_sync import GitHubSync, GitHubError
from .gateway_client import GatewayClient, GatewayError

DEFAULT_REPO = "fgbvdcfgbfdefgb/peal"


def log(msg: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def header_job_id(header_bytes: bytes) -> str:
    """Stable id for a job: short hash of the 76-byte header."""
    return hashlib.sha256(header_bytes).hexdigest()[:16]


def header_prev_hex(header_bytes: bytes) -> str:
    """Display order of the previous-block hash (bytes 4..36, reversed)."""
    if len(header_bytes) >= 36:
        return header_bytes[4:36][::-1].hex()
    return "?"


class RelayState:
    """Remembers which job was published and how far share.txt was consumed."""

    def __init__(self, path: str):
        self.path = path
        self.last_job_id = None
        self.shares_offset = 0
        self.shares_sha = None
        self.submitted = 0
        self._load()

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.last_job_id = data.get("last_job_id")
            self.shares_offset = int(data.get("shares_offset", 0))
            self.shares_sha = data.get("shares_sha")
            self.submitted = int(data.get("submitted", 0))
        except Exception:
            pass

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({
                "last_job_id": self.last_job_id,
                "shares_offset": self.shares_offset,
                "shares_sha": self.shares_sha,
                "submitted": self.submitted,
            }, fh)
        os.replace(tmp, self.path)


class Relay:
    def __init__(self):
        self.token = os.environ.get("GITHUB_TOKEN", "")
        self.repo = os.environ.get("GITHUB_REPO", DEFAULT_REPO)
        self.branch = os.environ.get("GITHUB_BRANCH", "main")
        self.jobs_file = os.environ.get("JOBS_FILE", "jobs.txt")
        self.shares_file = os.environ.get("SHARES_FILE", "share.txt")
        self.poll = float(os.environ.get("RELAY_POLL_SECONDS", "1.0"))
        self.sync = float(os.environ.get("RELAY_SYNC_SECONDS", "5.0"))
        self.dry_run = os.environ.get("RELAY_DRY_RUN", "") not in ("", "0", "false")
        self.state = RelayState(os.environ.get("RELAY_STATE_FILE",
                                               ".relay-state.json"))
        self.github = None
        self.gateway = GatewayClient()
        self.current_job = None       # dict from getMiningInfo
        self.current_header = None    # bytes

    # -- jobs ---------------------------------------------------------------

    def publish_job(self, job: dict) -> None:
        header_b64 = job["incomplete_header_bytes"]
        header_bytes = base64.b64decode(header_b64)
        job_id = header_job_id(header_bytes)
        if job_id == self.state.last_job_id:
            return
        line = {
            "job_id": job_id,
            "incomplete_header_bytes": header_b64,
            "target": int(job["target"]),
            "cert_version": int(job["cert_version"]),
            "prev_hash": header_prev_hex(header_bytes),
            "published_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if self.dry_run:
            log(f"[dry-run] would publish job {job_id} to {self.jobs_file}")
        else:
            self.github.append_line(
                self.jobs_file,
                json.dumps(line, separators=(",", ":")),
                f"relay: publish job {job_id}")
            log(f"published job {job_id} (prev={line['prev_hash'][:16]}... "
                f"cert_v={line['cert_version']} target={line['target']})")
        self.state.last_job_id = job_id
        self.state.save()

    # -- shares -------------------------------------------------------------

    def process_shares(self) -> None:
        content, sha = self.github.get_file(self.shares_file)
        if content is None:
            return
        self.state.shares_sha = sha
        text = content.decode("utf-8", errors="replace")
        if len(text.encode("utf-8")) <= self.state.shares_offset:
            return  # nothing new
        new_text = text.encode("utf-8")[self.state.shares_offset:].decode(
            "utf-8", errors="replace")
        self.state.shares_offset = len(text.encode("utf-8"))
        self.state.save()

        for line in new_text.splitlines():
            line = line.strip()
            if not line:
                continue
            self.submit_share(line)

    def submit_share(self, line: str) -> None:
        try:
            share = json.loads(line)
            mining_job = share["mining_job"]
            plain_proof = share["plain_proof"]
        except Exception as exc:
            log(f"skipping malformed share line: {exc}")
            return

        share_header = base64.b64decode(mining_job["incomplete_header_bytes"])
        if self.current_header is None or share_header != self.current_header:
            log(f"skipping stale share (job {share.get('job_id', '?')}): "
                f"header no longer current")
            return

        if self.dry_run:
            log(f"[dry-run] would submit share for job {share.get('job_id')}")
            return
        try:
            result = self.gateway.submit_plain_proof(plain_proof, mining_job)
            self.state.submitted += 1
            self.state.save()
            log(f"submitted share for job {share.get('job_id')} -> {result}")
        except GatewayError as exc:
            log(f"gateway rejected share for job {share.get('job_id')}: {exc}")

    # -- main loop ----------------------------------------------------------

    def run(self) -> int:
        if not self.token and not self.dry_run:
            log("GITHUB_TOKEN is not set - export it first")
            return 2
        if not self.dry_run:
            api_base = os.environ.get(
                "GITHUB_API_BASE", "https://api.github.com")
            self.github = GitHubSync(self.token, self.repo, self.branch,
                                     api_base=api_base)
            log(f"relay started: repo={self.repo} branch={self.branch} "
                f"jobs={self.jobs_file} shares={self.shares_file}")
        else:
            log("relay started in DRY-RUN mode")

        last_sync = 0.0
        while True:
            try:
                job = self.gateway.get_mining_info()
                if job != self.current_job:
                    self.current_job = job
                    self.current_header = base64.b64decode(
                        job["incomplete_header_bytes"])
                    log(f"gateway job: nbits header "
                        f"prev={header_prev_hex(self.current_header)[:16]}... "
                        f"target={job['target']} cert_v={job['cert_version']}")
                    self.publish_job(job)
            except GatewayError as exc:
                log(f"gateway not reachable ({exc}); retrying")
                time.sleep(max(self.poll, 2.0))
                continue

            now = time.time()
            if not self.dry_run and now - last_sync >= self.sync:
                last_sync = now
                try:
                    self.process_shares()
                except GitHubError as exc:
                    log(f"github sync error: {exc}")

            time.sleep(self.poll)


def main() -> int:
    try:
        return Relay().run()
    except KeyboardInterrupt:
        log("relay stopped")
        return 0
    except GitHubError as exc:
        log(f"fatal: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
