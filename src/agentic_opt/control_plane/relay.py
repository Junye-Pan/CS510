from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingUnixStreamServer
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


ALLOWED_METHODS = {"GET", "POST", "PATCH"}
ALLOWED_PATH_PREFIXES = ("/api/v1/", "/healthz")
DEFAULT_MAX_BODY_BYTES = 25 * 1024 * 1024


def relay_url(socket_path: Path) -> str:
    return f"unix://{socket_path.resolve()}"


def start_relay_process(
    *,
    socket_path: Path,
    target_url: str,
    python_path: str | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.Popen[str]:
    socket_path = socket_path.resolve()
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        python_path or sys.executable,
        "-m",
        "agentic_opt.control_plane.relay",
        "--socket",
        str(socket_path),
        "--target-url",
        target_url,
    ]
    return subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env,
    )


class ControlPlaneRelayServer(ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, socket_path: Path, target_url: str, *, max_body_bytes: int = DEFAULT_MAX_BODY_BYTES) -> None:
        self.socket_path = socket_path.resolve()
        self.target_url = target_url.rstrip("/")
        self.max_body_bytes = max_body_bytes
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            self.socket_path.unlink()
        super().__init__(str(self.socket_path), ControlPlaneRelayHandler)
        os.chmod(self.socket_path, 0o600)

    def server_close(self) -> None:
        super().server_close()
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass


class ControlPlaneRelayHandler(BaseHTTPRequestHandler):
    server: ControlPlaneRelayServer
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        self._handle()

    def do_POST(self) -> None:
        self._handle()

    def do_PATCH(self) -> None:
        self._handle()

    def do_PUT(self) -> None:
        self._reject(405, "method not allowed")

    def do_DELETE(self) -> None:
        self._reject(405, "method not allowed")

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _handle(self) -> None:
        if self.command not in ALLOWED_METHODS:
            self._reject(405, "method not allowed")
            return
        error = _validate_relay_path(self.path)
        if error:
            self._reject(403, error)
            return
        try:
            body = self._read_body()
        except ValueError as exc:
            self._reject(413, str(exc))
            return
        target = f"{self.server.target_url}{self.path}"
        headers = {
            "Accept": self.headers.get("Accept") or "application/json",
            "Content-Type": self.headers.get("Content-Type") or "application/json",
            "X-Agentic-Opt-Relay": "control-plane",
        }
        request = Request(target, data=body, method=self.command, headers=headers)
        try:
            with urlopen(request, timeout=120.0) as response:
                raw = response.read()
                status = response.status
                content_type = response.headers.get("Content-Type") or "application/json"
        except HTTPError as exc:
            raw = exc.read()
            status = exc.code
            content_type = exc.headers.get("Content-Type") or "application/json"
        except Exception as exc:
            self._reject(502, f"control-plane relay target failed: {type(exc).__name__}: {exc}")
            return
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _read_body(self) -> bytes | None:
        raw_length = self.headers.get("Content-Length")
        if not raw_length:
            return None
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length > self.server.max_body_bytes:
            raise ValueError(f"request body too large: {length} > {self.server.max_body_bytes}")
        return self.rfile.read(length)

    def _reject(self, status: int, message: str) -> None:
        raw = json.dumps({"error": message, "error_type": "ControlPlaneRelayError"}, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def _validate_relay_path(path: str) -> str | None:
    parsed = urlsplit(path)
    if parsed.scheme or parsed.netloc:
        return "relay only accepts origin-form control-plane paths"
    if not any(parsed.path == prefix.rstrip("/") or parsed.path.startswith(prefix) for prefix in ALLOWED_PATH_PREFIXES):
        return "relay only forwards control-plane API paths"
    return None


def serve(*, socket_path: Path, target_url: str, max_body_bytes: int = DEFAULT_MAX_BODY_BYTES) -> None:
    with ControlPlaneRelayServer(socket_path, target_url, max_body_bytes=max_body_bytes) as server:
        server.serve_forever()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m agentic_opt.control_plane.relay")
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--target-url", required=True)
    parser.add_argument("--max-body-bytes", type=int, default=DEFAULT_MAX_BODY_BYTES)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    serve(socket_path=args.socket, target_url=args.target_url, max_body_bytes=args.max_body_bytes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
