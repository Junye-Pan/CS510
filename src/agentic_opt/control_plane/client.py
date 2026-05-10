from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class ControlPlaneClientError(RuntimeError):
    pass


class ControlPlaneClient:
    def __init__(self, base_url: str, *, timeout_s: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        suffix = path if path.startswith("/") else f"/{path}"
        url = f"{self.base_url}{suffix}"
        if params:
            query = urlencode({key: value for key, value in params.items() if value is not None})
            if query:
                url = f"{url}?{query}"
        return self._request("GET", url)

    def post(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        suffix = path if path.startswith("/") else f"/{path}"
        return self._request("POST", f"{self.base_url}{suffix}", payload or {})

    def patch(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        suffix = path if path.startswith("/") else f"/{path}"
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
