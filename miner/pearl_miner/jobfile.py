"""
jobs.txt / share.txt file handling.

The offline miner never touches the network: it reads mining jobs that the
Windows relay published to ``jobs.txt`` (one JSON object per line) and
appends the shares it finds to ``share.txt`` (one JSON object per line).
The user syncs both files with ``git pull`` / ``git push`` on the mining
machine.

Job line (published by the relay from the gateway's getMiningInfo):

    {"job_id": "...", "incomplete_header_bytes": "<base64, 76 bytes>",
     "target": 123456789, "cert_version": 3, "height": 114000,
     "prev_hash": "...", "published_at": "2026-10-11T12:00:00Z"}

Share line (appended by the miner):

    {"job_id": "...", "mining_job": {...the job dict...},
     "plain_proof": "<base64 bincode PlainProof>", "found_at": "...",
     "miner": "hostname", "attempts": 123456, "elapsed_s": 12.3,
     "backend": "numpy", "config": "m=128,n=128,k=4096,rank=128"}
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass


@dataclass
class MiningJobInfo:
    job_id: str
    incomplete_header_bytes: bytes
    target: int
    cert_version: int
    height: int | None = None
    prev_hash: str | None = None
    published_at: str | None = None
    raw: dict | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "MiningJobInfo":
        return cls(
            job_id=str(data.get("job_id", "")),
            incomplete_header_bytes=base64.b64decode(data["incomplete_header_bytes"]),
            target=int(data["target"]),
            cert_version=int(data["cert_version"]),
            height=data.get("height"),
            prev_hash=data.get("prev_hash"),
            published_at=data.get("published_at"),
            raw=data,
        )

    def gateway_job_dict(self) -> dict:
        """The dict the gateway expects for submitPlainProof."""
        return {
            "incomplete_header_bytes": self.raw["incomplete_header_bytes"],
            "target": self.target,
            "cert_version": self.cert_version,
        }


def read_jobs(path: str) -> list:
    """Parse every valid JSON line of jobs.txt (oldest first)."""
    jobs = []
    if not os.path.exists(path):
        return jobs
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                jobs.append(MiningJobInfo.from_dict(data))
            except Exception:
                continue  # tolerate partially-written lines
    return jobs


def latest_job(path: str):
    """The newest parseable job in jobs.txt (what the miner works on)."""
    jobs = read_jobs(path)
    return jobs[-1] if jobs else None


class JobWatcher:
    """Polls jobs.txt for new content (the user pulls it with git)."""

    def __init__(self, path: str, poll_interval: float = 2.0):
        self.path = path
        self.poll_interval = poll_interval
        self._sig = None
        self._job = None

    def _signature(self):
        try:
            st = os.stat(self.path)
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return None

    def poll(self):
        """Return the newest job when the file changed, else None."""
        sig = self._signature()
        if sig == self._sig:
            return None
        self._sig = sig
        job = latest_job(self.path)
        if job is None:
            return None
        if self._job is not None and job.job_id == self._job.job_id:
            return None
        self._job = job
        return job


class ShareWriter:
    """Appends share lines to share.txt (the user pushes the file)."""

    def __init__(self, path: str):
        self.path = path
        self.count = 0

    def append(self, share: dict) -> None:
        line = json.dumps(share, separators=(",", ":"))
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self.count += 1


def utcnow_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
