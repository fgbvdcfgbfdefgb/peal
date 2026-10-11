"""
Line-delimited JSON-RPC client for the local Pearl gateway.

The gateway (``pearl-gateway`` from the Pearl monorepo) exposes
``getMiningInfo`` and ``submitPlainProof`` over a UNIX socket or TCP
(127.0.0.1:8337 by default).  Configure it with the same environment
variables the gateway itself uses:

    MINER_RPC_TRANSPORT=tcp|uds
    MINER_RPC_HOST=localhost
    MINER_RPC_PORT=8337
    MINER_RPC_SOCKET_PATH=/tmp/pearlgw.sock
"""

from __future__ import annotations

import json
import os
import socket
import time


class GatewayError(RuntimeError):
    pass


class GatewayClient:
    """Synchronous JSON-RPC client (one connection, kept open)."""

    def __init__(self, transport: str | None = None, host: str | None = None,
                 port: int | None = None, socket_path: str | None = None,
                 timeout: float = 30.0):
        self.transport = (transport
                          or os.environ.get("MINER_RPC_TRANSPORT", "tcp")
                          ).lower()
        self.host = host or os.environ.get("MINER_RPC_HOST", "127.0.0.1")
        self.port = int(port or os.environ.get("MINER_RPC_PORT", "8337"))
        self.socket_path = (socket_path
                            or os.environ.get("MINER_RPC_SOCKET_PATH",
                                              "/tmp/pearlgw.sock"))
        self.timeout = timeout
        self._sock = None
        self._reader = None
        self._id = 0

    # -- connection ---------------------------------------------------------

    def connect(self):
        self.close()
        if self.transport == "uds":
            if not os.path.exists(self.socket_path):
                raise GatewayError(f"gateway socket not found: {self.socket_path}")
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect(self.socket_path)
        elif self.transport == "tcp":
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((self.host, self.port))
        else:
            raise GatewayError(f"unknown transport: {self.transport}")
        self._sock = sock
        self._reader = sock.makefile("rb")
        return self

    def close(self):
        try:
            if self._reader is not None:
                self._reader.close()
        except Exception:
            pass
        try:
            if self._sock is not None:
                self._sock.close()
        except Exception:
            pass
        self._sock = None
        self._reader = None

    def _ensure(self):
        if self._sock is None:
            self.connect()

    # -- RPC ----------------------------------------------------------------

    def call(self, method: str, params: dict | None = None, retries: int = 2):
        self._id += 1
        request = {"jsonrpc": "2.0", "method": method, "id": self._id}
        if params is not None:
            request["params"] = params
        line = json.dumps(request).encode("utf-8") + b"\n"

        last_exc = None
        for attempt in range(retries + 1):
            try:
                self._ensure()
                self._sock.sendall(line)
                while True:
                    raw = self._reader.readline()
                    if not raw:
                        raise GatewayError("gateway closed the connection")
                    raw = raw.strip()
                    if not raw:
                        continue
                    response = json.loads(raw.decode("utf-8"))
                    break
                if "error" in response and response["error"] is not None:
                    err = response["error"]
                    raise GatewayError(f"{err.get('code')}: {err.get('message')}"
                                       + (f" ({err.get('data')})" if err.get("data") else ""))
                return response.get("result")
            except (GatewayError, OSError, json.JSONDecodeError) as exc:
                last_exc = exc
                self.close()
                if attempt >= retries:
                    break
                time.sleep(1.0 * (attempt + 1))
        raise GatewayError(f"{method} failed: {last_exc}")

    # -- gateway methods ----------------------------------------------------

    def get_mining_info(self) -> dict:
        """``{incomplete_header_bytes: b64, target: int, cert_version: int}``."""
        result = self.call("getMiningInfo")
        if not isinstance(result, dict):
            raise GatewayError(f"unexpected getMiningInfo result: {result!r}")
        return result

    def submit_plain_proof(self, plain_proof_b64: str, mining_job: dict):
        """Submit a share; the gateway returns 'submitted' (async)."""
        return self.call("submitPlainProof", {
            "plain_proof": plain_proof_b64,
            "mining_job": mining_job,
        })
