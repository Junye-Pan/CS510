from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class AppServerClientError(RuntimeError):
    pass


class AppServerProtocolError(AppServerClientError):
    pass


_PROJECT_SECTION_RE = re.compile(r'^\[projects\."(?P<path>.+)"\]\s*$')


@dataclass
class _PendingRequest:
    event: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: dict[str, Any] | None = None


class AppServerClient:
    def __init__(
        self,
        *,
        codex_binary: str = "codex",
        root_cwd: str | None = None,
        codex_home: str | None = None,
        config_overrides: list[str] | None = None,
        extra_env: dict[str, str] | None = None,
        startup_timeout_s: float = 20.0,
    ) -> None:
        self.codex_binary = codex_binary
        self.root_cwd = root_cwd
        self.codex_home = codex_home
        self.config_overrides = list(config_overrides or ())
        self.extra_env = extra_env or {}
        self.startup_timeout_s = startup_timeout_s

        self._process: subprocess.Popen[str] | None = None
        self._next_request_id = 1
        self._lock = threading.Lock()
        self._pending: dict[int, _PendingRequest] = {}
        self._notifications: list[dict[str, Any]] = []
        self._stderr_lines: list[str] = []
        self._condition = threading.Condition()
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None

    def ensure_started(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        if shutil.which(self.codex_binary) is None:
            raise AppServerClientError(f"Unable to find Codex binary: {self.codex_binary}")
        command = [self.codex_binary, "app-server", "--listen", "stdio://"]
        for override in self.config_overrides:
            command.extend(["-c", override])
        self._process = subprocess.Popen(
            command,
            cwd=self.root_cwd,
            env=self._build_env(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if self._process.stdin is None or self._process.stdout is None or self._process.stderr is None:
            raise AppServerClientError("Failed to open stdio pipes to codex app-server")

        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True, name="codex-app-server-stdout")
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True, name="codex-app-server-stderr")
        self._stdout_thread.start()
        self._stderr_thread.start()

        self.request(
            "initialize",
            {
                "clientInfo": {
                    "name": "agentic-opt",
                    "title": "Agentic Optimization",
                    "version": "0.1.0",
                },
                "capabilities": {
                    "experimentalApi": True,
                },
            },
            timeout_s=self.startup_timeout_s,
        )

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        with self._condition:
            for pending in self._pending.values():
                pending.error = {"message": "codex app-server client closed"}
                pending.event.set()
            self._pending.clear()
            self._condition.notify_all()
        for stream_name in ("stdin", "stdout", "stderr"):
            stream = getattr(process, stream_name, None)
            if stream is None:
                continue
            try:
                stream.close()
            except OSError:
                pass

    def request(self, method: str, params: dict[str, Any] | None, *, timeout_s: float | None = None) -> Any:
        self.ensure_started()
        request_id = self._claim_request_id()
        pending = _PendingRequest()
        with self._condition:
            self._pending[request_id] = pending
        self._send({"id": request_id, "method": method, "params": params or {}})
        if not pending.event.wait(timeout=timeout_s):
            with self._condition:
                self._pending.pop(request_id, None)
            raise TimeoutError(f"Timed out waiting for App Server response to {method}")
        if pending.error is not None:
            message = pending.error.get("message", "unknown error")
            raise AppServerClientError(f"{method} failed: {message}")
        return pending.result

    def notification_cursor(self) -> int:
        with self._condition:
            return len(self._notifications)

    def stderr_cursor(self) -> int:
        with self._condition:
            return len(self._stderr_lines)

    def notifications_slice(self, start: int, end: int | None = None) -> list[dict[str, Any]]:
        with self._condition:
            return list(self._notifications[start:end])

    def stderr_slice(self, start: int) -> list[str]:
        with self._condition:
            return list(self._stderr_lines[start:])

    def wait_for_turn_completion(
        self,
        *,
        notification_cursor: int,
        thread_id: str,
        turn_id: str,
        timeout_s: float | None,
    ) -> tuple[dict[str, Any], int]:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        cursor = notification_cursor
        with self._condition:
            while True:
                for index in range(cursor, len(self._notifications)):
                    notification = self._notifications[index]
                    if notification.get("method") != "turn/completed":
                        continue
                    params = notification.get("params") or {}
                    if params.get("threadId") != thread_id:
                        continue
                    turn = params.get("turn") or {}
                    if turn.get("id") == turn_id:
                        return notification, index + 1
                cursor = len(self._notifications)
                if self._process is not None and self._process.poll() is not None:
                    raise AppServerClientError("codex app-server exited before the turn completed")
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(f"Timed out waiting for turn completion: {turn_id}")
                self._condition.wait(timeout=remaining)

    def interrupt_turn(self, *, thread_id: str, turn_id: str, timeout_s: float | None = None) -> None:
        self.request(
            "turn/interrupt",
            {
                "threadId": thread_id,
                "turnId": turn_id,
            },
            timeout_s=timeout_s,
        )

    def _claim_request_id(self) -> int:
        with self._lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            return request_id

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(self.extra_env)
        if self.codex_home:
            target_home = Path(self.codex_home).resolve()
            if self.root_cwd is not None and _is_relative_to(target_home, Path(self.root_cwd).resolve()):
                raise AppServerClientError("codex_home must be outside the agent workspace/root cwd")
            target_home.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(target_home, 0o700)
            except OSError:
                pass
            self._prepare_codex_home(target_home, env)
            env["CODEX_HOME"] = str(target_home)
        return env

    def _prepare_codex_home(self, target_home: Path, env: dict[str, str]) -> None:
        source_home = self._source_codex_home(env)
        if source_home is None or not source_home.exists():
            return
        if source_home.resolve() == target_home.resolve():
            return
        for filename in (
            "auth.json",
            "installation_id",
            ".codex-global-state.json",
            "version.json",
            ".personality_migration",
        ):
            self._sync_optional_file(source_home / filename, target_home / filename)
        self._merge_config_file(source_home / "config.toml", target_home / "config.toml")

    def _source_codex_home(self, env: dict[str, str]) -> Path | None:
        source_value = env.get("CODEX_HOME")
        if source_value:
            return Path(source_value).expanduser()
        return Path.home() / ".codex"

    def _sync_optional_file(self, source: Path, target: Path) -> None:
        if not source.exists() or not source.is_file():
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            try:
                if target.read_bytes() == source.read_bytes():
                    return
            except OSError:
                pass
        shutil.copy2(source, target)
        if target.name == "auth.json":
            try:
                os.chmod(target, 0o600)
            except OSError:
                pass

    def _merge_config_file(self, source: Path, target: Path) -> None:
        if not source.exists() or not source.is_file():
            return
        source_text = source.read_text(encoding="utf-8").rstrip()
        target_text = target.read_text(encoding="utf-8").rstrip() if target.exists() else ""
        source_blocks = self._extract_project_blocks(source_text)
        target_blocks = self._extract_project_blocks(target_text)
        merged_parts = [source_text] if source_text else []
        merged_parts.extend(block for project_path, block in target_blocks.items() if project_path not in source_blocks)
        merged_text = "\n\n".join(part for part in merged_parts if part).rstrip() + "\n"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_text(encoding="utf-8") == merged_text:
            return
        target.write_text(merged_text, encoding="utf-8")
        try:
            os.chmod(target, source.stat().st_mode & 0o777)
        except OSError:
            pass

    def _extract_project_blocks(self, text: str) -> dict[str, str]:
        blocks: dict[str, str] = {}
        if not text.strip():
            return blocks
        lines = text.splitlines()
        index = 0
        while index < len(lines):
            match = _PROJECT_SECTION_RE.match(lines[index].strip())
            if not match:
                index += 1
                continue
            start = index
            index += 1
            while index < len(lines):
                candidate = lines[index].strip()
                if candidate.startswith("[") and candidate.endswith("]"):
                    break
                index += 1
            blocks[match.group("path")] = "\n".join(lines[start:index]).rstrip()
        return blocks

    def _send(self, payload: dict[str, Any]) -> None:
        if self._process is None or self._process.stdin is None:
            raise AppServerClientError("codex app-server is not running")
        try:
            self._process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
            self._process.stdin.flush()
        except BrokenPipeError as exc:
            raise AppServerClientError("Lost connection to codex app-server") from exc

    def _read_stdout(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        try:
            for raw_line in self._process.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AppServerProtocolError(f"Invalid JSON from codex app-server: {line}") from exc
                self._handle_message(message)
        except Exception as exc:  # pragma: no cover
            with self._condition:
                for pending in self._pending.values():
                    pending.error = {"message": str(exc)}
                    pending.event.set()
                self._condition.notify_all()

    def _read_stderr(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        for raw_line in self._process.stderr:
            with self._condition:
                self._stderr_lines.append(raw_line.rstrip("\n"))
                self._condition.notify_all()

    def _handle_message(self, message: dict[str, Any]) -> None:
        if "method" in message and "id" in message:
            self._record_notification(
                {
                    "method": "__server_request__",
                    "params": {
                        "request_id": message["id"],
                        "original_method": message["method"],
                        "params": message.get("params"),
                    },
                }
            )
            self._send(
                {
                    "id": message["id"],
                    "error": {
                        "code": -32601,
                        "message": f"Server request not supported by agentic-opt: {message['method']}",
                    },
                }
            )
            return

        if "method" in message:
            self._record_notification({"method": message["method"], "params": message.get("params")})
            return

        if "id" in message:
            with self._condition:
                pending = self._pending.pop(message["id"], None)
                if pending is None:
                    return
                if "error" in message:
                    pending.error = message["error"]
                else:
                    pending.result = message.get("result")
                pending.event.set()
                self._condition.notify_all()
            return

        raise AppServerProtocolError(f"Unexpected App Server message: {message}")

    def _record_notification(self, notification: dict[str, Any]) -> None:
        payload = {
            "received_at": time.time(),
            "method": notification["method"],
            "params": notification.get("params"),
        }
        with self._condition:
            self._notifications.append(payload)
            self._condition.notify_all()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
