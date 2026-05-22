from __future__ import annotations

import argparse
import json
import os
import select
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socketserver import ThreadingUnixStreamServer
from typing import Any
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .process_env import build_subprocess_env
from .repository import ControlPlaneRepository


def proxy_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def unix_proxy_url(socket_path: Path) -> str:
    return f"unix://{socket_path.resolve()}"


def start_network_proxy_process(
    *,
    host: str | None = None,
    port: int | None = None,
    socket_path: Path | None = None,
    database_path: Path,
    policy: dict[str, Any],
    metadata: dict[str, Any],
    env: dict[str, str] | None = None,
) -> subprocess.Popen[str]:
    args = [
        sys.executable,
        "-m",
        "agentic_opt.control_plane.network_proxy",
        "--db",
        str(database_path),
        "--policy-json",
        json.dumps(policy, sort_keys=True),
        "--metadata-json",
        json.dumps(metadata, sort_keys=True),
    ]
    if socket_path is not None:
        args.extend(["--socket", str(socket_path)])
    else:
        if host is None or port is None:
            raise ValueError("network proxy requires either socket_path or host+port")
        args.extend(["--host", host, "--port", str(port)])
    return subprocess.Popen(
        args,
        cwd=str(Path(__file__).resolve().parents[3]),
        env=env or build_subprocess_env(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


class NetworkAuditProxy(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        repository: ControlPlaneRepository,
        policy: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        super().__init__(server_address, NetworkAuditProxyHandler)
        self.repository = repository
        self.policy = policy
        self.metadata = metadata


class NetworkAuditUnixProxy(ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        socket_path: Path,
        *,
        repository: ControlPlaneRepository,
        policy: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        self.socket_path = socket_path.resolve()
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            self.socket_path.unlink()
        super().__init__(str(self.socket_path), NetworkAuditProxyHandler)
        os.chmod(self.socket_path, 0o600)
        self.repository = repository
        self.policy = policy
        self.metadata = metadata

    def server_close(self) -> None:
        super().server_close()
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass


class NetworkAuditProxyHandler(BaseHTTPRequestHandler):
    server: NetworkAuditProxy | NetworkAuditUnixProxy
    timeout = 30

    def do_CONNECT(self) -> None:
        host, port = _split_connect_destination(self.path)
        decision = self._record(destination=f"{host}:{port}", access_type="http-connect")
        if decision == "denied":
            self._reject(403, "network destination denied by experiment policy")
            return
        try:
            upstream = socket.create_connection((host, port), timeout=self.timeout)
        except OSError as exc:
            self._reject(502, f"network proxy connect failed: {exc}")
            return
        self.send_response(200, "Connection Established")
        self.end_headers()
        self._tunnel(upstream)

    def do_GET(self) -> None:
        self._proxy_http()

    def do_POST(self) -> None:
        self._proxy_http()

    def do_PUT(self) -> None:
        self._proxy_http()

    def do_PATCH(self) -> None:
        self._proxy_http()

    def do_DELETE(self) -> None:
        self._proxy_http()

    def do_HEAD(self) -> None:
        self._proxy_http()

    def log_message(self, format: str, *args: object) -> None:
        return

    def _proxy_http(self) -> None:
        target = self.path
        if not target.startswith(("http://", "https://")):
            host = self.headers.get("Host")
            if not host:
                self._reject(400, "proxy request missing Host header")
                return
            target = f"http://{host}{self.path}"
        parsed = urlsplit(target)
        destination = parsed.netloc
        decision = self._record(destination=destination, access_type=f"http-{self.command.lower()}")
        if decision == "denied":
            self._reject(403, "network destination denied by experiment policy")
            return
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        headers = {key: value for key, value in self.headers.items() if key.lower() not in {"proxy-connection", "connection", "host"}}
        request = Request(target, data=body, method=self.command, headers=headers)
        try:
            with urlopen(request, timeout=self.timeout) as response:  # noqa: S310 - this is the audited proxy target.
                payload = response.read()
                self.send_response(response.status)
                for key, value in response.headers.items():
                    if key.lower() in {"connection", "transfer-encoding"}:
                        continue
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(payload)
        except Exception as exc:
            self._reject(502, f"network proxy request failed: {type(exc).__name__}: {exc}")

    def _record(self, *, destination: str, access_type: str) -> str:
        policy = self.server.policy
        decision = "denied" if _host_denied(destination, policy) else "audit"
        client = self.client_address[0] if isinstance(self.client_address, tuple) and self.client_address else "unix-socket"
        metadata = {**self.server.metadata, "policy": policy, "proxy": {"client": client}}
        self.server.repository.record_network_access_event(
            {
                "experiment_id": metadata.get("experiment_id"),
                "assignment_id": metadata.get("assignment_id"),
                "session_id": metadata.get("session_id"),
                "task_id": metadata.get("task_id"),
                "agent_id": metadata.get("agent_id"),
                "destination": destination,
                "access_type": access_type,
                "decision": decision,
                "reason": "outbound audit proxy" if decision == "audit" else "destination denied by network policy",
                "metadata": metadata,
            }
        )
        return decision

    def _tunnel(self, upstream: socket.socket) -> None:
        sockets = [self.connection, upstream]
        try:
            while True:
                readable, _, _ = select.select(sockets, [], [], self.timeout)
                if not readable:
                    return
                for sock in readable:
                    data = sock.recv(8192)
                    if not data:
                        return
                    target = upstream if sock is self.connection else self.connection
                    target.sendall(data)
        finally:
            upstream.close()

    def _reject(self, status: int, message: str) -> None:
        payload = json.dumps({"error": message}, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def serve(
    *,
    host: str | None = None,
    port: int | None = None,
    socket_path: Path | None = None,
    database_path: Path,
    policy: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    repository = ControlPlaneRepository(database_path)
    if socket_path is not None:
        server: NetworkAuditProxy | NetworkAuditUnixProxy = NetworkAuditUnixProxy(
            socket_path,
            repository=repository,
            policy=policy,
            metadata=metadata,
        )
    else:
        if host is None or port is None:
            raise ValueError("host and port are required for TCP network proxy")
        server = NetworkAuditProxy((host, port), repository=repository, policy=policy, metadata=metadata)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auditing HTTP proxy for Docker-backed workers and jobs.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument("--socket", type=Path)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--policy-json", default="{}")
    parser.add_argument("--metadata-json", default="{}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    serve(
        host=args.host,
        port=args.port,
        socket_path=args.socket,
        database_path=args.db,
        policy=json.loads(args.policy_json),
        metadata=json.loads(args.metadata_json),
    )
    return 0


def _split_connect_destination(raw: str) -> tuple[str, int]:
    if ":" not in raw:
        return raw, 443
    host, port_raw = raw.rsplit(":", 1)
    return host, int(port_raw)


def _host_denied(destination: str, policy: dict[str, Any]) -> bool:
    host = destination.rsplit(":", 1)[0].strip("[]").lower()
    denied = {str(item).lower() for item in (policy.get("denied_hosts") or [])}
    allowed = {str(item).lower() for item in (policy.get("allowed_hosts") or [])}
    if host in denied:
        return True
    if str(policy.get("external_internet")) == "deny":
        return bool(allowed) and host not in allowed
    return False


def wait_for_tcp(host: str, port: int, *, timeout_s: float = 5.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.1):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"network audit proxy did not start on {host}:{port}")


def wait_for_unix_socket(socket_path: Path, *, timeout_s: float = 5.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.1)
                sock.connect(str(socket_path))
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"network audit proxy did not start on {socket_path}")


if __name__ == "__main__":
    raise SystemExit(main())
