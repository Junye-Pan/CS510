from __future__ import annotations

import http.client
import json
import os
import socket
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class ControlPlaneClientError(RuntimeError):
    pass


DEFAULT_CONTROL_PLANE_CLIENT_TIMEOUT_S = 300.0


class ControlPlaneClient:
    def __init__(self, base_url: str, *, timeout_s: float | None = None) -> None:
        parsed = urlparse(base_url)
        self.socket_path: Path | None = None
        if parsed.scheme == "unix":
            if not parsed.path:
                raise ValueError("unix control-plane URL must include a socket path")
            self.socket_path = Path(parsed.path)
            self.base_url = "http://agentic-opt-control-plane"
        else:
            self.base_url = base_url.rstrip("/")
        self.timeout_s = _control_plane_client_timeout_s(timeout_s)

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        suffix = path if path.startswith("/") else f"/{path}"
        if params:
            query = urlencode({key: value for key, value in params.items() if value is not None})
            if query:
                suffix = f"{suffix}?{query}"
        if self.socket_path is not None:
            return self._request_unix("GET", suffix)
        url = f"{self.base_url}{suffix}"
        return self._request("GET", url)

    def post(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        suffix = path if path.startswith("/") else f"/{path}"
        if self.socket_path is not None:
            return self._request_unix("POST", suffix, payload or {})
        return self._request("POST", f"{self.base_url}{suffix}", payload or {})

    def patch(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        suffix = path if path.startswith("/") else f"/{path}"
        if self.socket_path is not None:
            return self._request_unix("PATCH", suffix, payload or {})
        return self._request("PATCH", f"{self.base_url}{suffix}", payload or {})

    def _request(self, method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            url,
            data=data,
            method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            raw_error = exc.read().decode("utf-8", errors="replace")
            raise ControlPlaneClientError(f"{method} {url} failed: HTTP {exc.code}: {raw_error}") from exc
        if not raw:
            return {}
        decoded = json.loads(raw)
        if isinstance(decoded, dict) and decoded.get("error"):
            raise ControlPlaneClientError(f"{method} {url} failed: {decoded['error']}")
        if not isinstance(decoded, dict):
            return {"result": decoded}
        return decoded

    def _request_unix(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        assert self.socket_path is not None
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        conn = _UnixHTTPConnection(str(self.socket_path), timeout=self.timeout_s)
        try:
            conn.request(method, path, body=data, headers=headers)
            response = conn.getresponse()
            raw = response.read().decode("utf-8")
        except OSError as exc:
            raise ControlPlaneClientError(f"{method} unix://{self.socket_path}{path} failed: {exc}") from exc
        finally:
            conn.close()
        if response.status >= 400:
            raise ControlPlaneClientError(f"{method} unix://{self.socket_path}{path} failed: HTTP {response.status}: {raw}")
        if not raw:
            return {}
        decoded = json.loads(raw)
        if isinstance(decoded, dict) and decoded.get("error"):
            raise ControlPlaneClientError(f"{method} unix://{self.socket_path}{path} failed: {decoded['error']}")
        if not isinstance(decoded, dict):
            return {"result": decoded}
        return decoded


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: str, *, timeout: float) -> None:
        super().__init__("agentic-opt-control-plane", timeout=timeout)
        self.socket_path = socket_path

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self.socket_path)
        self.sock = sock


def _control_plane_client_timeout_s(explicit: float | None) -> float:
    if explicit is not None:
        return float(explicit)
    raw = os.environ.get("AO_CONTROL_CLIENT_TIMEOUT_S")
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return DEFAULT_CONTROL_PLANE_CLIENT_TIMEOUT_S
