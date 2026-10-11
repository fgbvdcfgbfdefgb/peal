"""
GitHub file sync for the Pearl relay (Windows side).

The offline Linux mining machine has no internet except pip and git
pull/push, so the Windows relay talks to GitHub through the REST Contents
API (no git binary required on Windows).

Two files are synchronised in the repository ``fgbvdcfgbfdefgb/peal``:

  * ``jobs.txt``   - written by the relay (one JSON job per line, append-only
                     from the relay's point of view), pulled by the miner.
  * ``share.txt``  - appended by the miner (pushed manually), read by the
                     relay which submits the shares to the local gateway.

Only the Python standard library plus ``requests`` are used.
"""

from __future__ import annotations

import base64
import time

try:
    import requests
except Exception:  # pragma: no cover
    requests = None


class GitHubError(RuntimeError):
    pass


class GitHubSync:
    """Minimal GitHub Contents API client."""

    def __init__(self, token: str, repo: str, branch: str = "main",
                 api_base: str = "https://api.github.com",
                 timeout: float = 30.0, max_retries: int = 4):
        if requests is None:
            raise GitHubError("the 'requests' package is required "
                              "(pip install requests)")
        if "/" not in repo:
            raise GitHubError(f"repo must be 'owner/name', got {repo!r}")
        self.owner, self.name = repo.split("/", 1)
        self.branch = branch
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "pearl-relay",
        })

    # -- low level ----------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs):
        url = f"{self.api_base}{path}"
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                resp = self.session.request(method, url, timeout=self.timeout,
                                            **kwargs)
            except Exception as exc:  # network hiccup
                last_exc = exc
                time.sleep(1.5 * (attempt + 1))
                continue
            if resp.status_code in (408, 429, 500, 502, 503, 504):
                # retryable
                time.sleep(1.5 * (attempt + 1))
                continue
            return resp
        raise GitHubError(f"{method} {url} failed: {last_exc or 'retries exhausted'}")

    # -- contents API -------------------------------------------------------

    def get_file(self, path: str):
        """
        Fetch a file.  Returns ``(content_bytes, sha)`` or ``(None, None)``
        when the file does not exist yet.
        """
        resp = self._request(
            "GET", f"/repos/{self.owner}/{self.name}/contents/{path}",
            params={"ref": self.branch})
        if resp.status_code == 404:
            return None, None
        if resp.status_code >= 400:
            raise GitHubError(f"GET {path}: {resp.status_code} {resp.text[:200]}")
        data = resp.json()
        if isinstance(data, list):  # directory listing
            raise GitHubError(f"{path} is a directory")
        content = base64.b64decode(data["content"])
        return content, data["sha"]

    def put_file(self, path: str, content: bytes, message: str,
                 sha: str | None = None, retries: int = 3):
        """
        Create or update a file.  On a stale-sha conflict the current file is
        re-fetched and the caller-supplied ``content`` is written as-is, so
        callers that append should re-read the file first (see
        :meth:`append_line`).
        """
        payload = {
            "message": message,
            "content": base64.b64encode(content).decode("ascii"),
            "branch": self.branch,
        }
        if sha:
            payload["sha"] = sha
        last = None
        for _ in range(retries):
            resp = self._request(
                "PUT", f"/repos/{self.owner}/{self.name}/contents/{path}",
                json=payload)
            if resp.status_code < 300:
                return resp.json()
            if resp.status_code == 409:  # conflict: stale sha
                content_now, sha_now = self.get_file(path)
                if sha_now == payload.get("sha"):
                    raise GitHubError(f"PUT {path}: conflict persists")
                payload["sha"] = sha_now
                last = f"{resp.status_code} {resp.text[:200]}"
                continue
            if resp.status_code == 422 and "empty" in resp.text.lower():
                # nothing to change
                return {"content": None}
            raise GitHubError(f"PUT {path}: {resp.status_code} {resp.text[:200]}")
        raise GitHubError(f"PUT {path}: conflict after {retries} tries ({last})")

    def append_line(self, path: str, line: str, message: str):
        """Append one line to a remote file (re-reading it first)."""
        if not line.endswith("\n"):
            line += "\n"
        content, sha = self.get_file(path)
        new = (content or b"") + line.encode("utf-8")
        if content is not None and content.endswith(line.encode("utf-8")) \
                and content.count(b"\n") == 1:
            pass  # fine, first write
        return self.put_file(path, new, message, sha=sha)
