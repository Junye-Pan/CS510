from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentic_opt.common.atomic import atomic_write_text
from agentic_opt.common.ids import isoformat_z, make_session_id

from .app_server_client import AppServerClient
from .base import BudgetPolicy, CodexAdapter, InstructionBundle, SessionHandle, TraceBundle, TurnHandle, TurnResult, WorkspacePolicy


@dataclass(frozen=True)
class AppServerAdapterConfig:
    codex_binary: str = "codex"
    startup_timeout_s: float = 20.0
    request_timeout_s: float = 30.0
    approval_policy: str = "never"
    model: str | None = "gpt-5.4"
    reasoning_effort: str | None = "high"
    personality: str | None = "pragmatic"
    ephemeral_thread: bool = False


@dataclass
class _TurnRuntime:
    session_id: str
    thread_id: str
    run_id: str
    kind: str
    workspace_root: Path
    notification_cursor: int
    stderr_cursor: int
    started_at: str
    trace_dir: Path


class AppServerCodexAdapter(CodexAdapter):
    def __init__(
        self,
        *,
        client: AppServerClient,
        config: AppServerAdapterConfig | None = None,
    ) -> None:
        self.client = client
        self.config = config or AppServerAdapterConfig()
        self.sessions: dict[str, SessionHandle] = {}
        self.session_runtime: dict[str, dict[str, object]] = {}
        self.turns: dict[str, _TurnRuntime] = {}

    def start_session(
        self,
        *,
        session_id: str | None = None,
        task_id: str,
        agent_id: str,
        run_id: str,
        workspace: WorkspacePolicy,
        instructions: InstructionBundle,
        budget: BudgetPolicy,
    ) -> SessionHandle:
        response = self.client.request(
            "thread/start",
            {
                "cwd": workspace.workspace_root,
                "approvalPolicy": self.config.approval_policy,
                "sandbox": workspace.sandbox_mode,
                "baseInstructions": None,
                "developerInstructions": self._developer_instructions(instructions, workspace=workspace),
                "ephemeral": self.config.ephemeral_thread,
                **({"model": self.config.model} if self.config.model else {}),
                **({"personality": self.config.personality} if self.config.personality else {}),
            },
            timeout_s=self.config.request_timeout_s,
        )
        thread = response["thread"]
        session_id = session_id or make_session_id()
        handle = SessionHandle(
            session_id=session_id,
            task_id=task_id,
            agent_id=agent_id,
            run_id=run_id,
            provider="codex",
            thread_id=thread["id"],
            root_thread_id=thread.get("forkedFromId"),
            status="active",
        )
        self.sessions[session_id] = handle
        self.session_runtime[session_id] = {
            "workspace": workspace,
            "instructions": instructions,
            "budget": budget,
        }
        return handle

    def resume_session(
        self,
        *,
        session_id: str,
        run_id: str,
        workspace: WorkspacePolicy,
        instructions: InstructionBundle,
        budget: BudgetPolicy,
    ) -> SessionHandle:
        handle = self.sessions[session_id]
        response = self.client.request(
            "thread/resume",
            {
                "threadId": handle.thread_id,
                "cwd": workspace.workspace_root,
                "approvalPolicy": self.config.approval_policy,
                "sandbox": workspace.sandbox_mode,
                "baseInstructions": None,
                "developerInstructions": self._developer_instructions(instructions, workspace=workspace),
                **({"model": self.config.model} if self.config.model else {}),
                **({"personality": self.config.personality} if self.config.personality else {}),
            },
            timeout_s=self.config.request_timeout_s,
        )
        handle.thread_id = response["thread"]["id"]
        handle.run_id = run_id
        handle.status = "active"
        self.session_runtime[session_id] = {
            "workspace": workspace,
            "instructions": instructions,
            "budget": budget,
        }
        return handle

    def start_turn(
        self,
        *,
        session_id: str,
        kind: str,
        prompt: str,
        budget: BudgetPolicy | None = None,
    ) -> TurnHandle:
        handle = self.sessions[session_id]
        runtime = self.session_runtime[session_id]
        workspace = runtime["workspace"]
        budget = budget or runtime["budget"]
        notification_cursor = self.client.notification_cursor()
        stderr_cursor = self.client.stderr_cursor()
        response = self.client.request(
            "turn/start",
            {
                "threadId": handle.thread_id,
                "input": [{"type": "text", "text": prompt}],
                "cwd": workspace.workspace_root,
                "approvalPolicy": self.config.approval_policy,
                "sandboxPolicy": self._sandbox_policy(workspace),
                **({"model": self.config.model} if self.config.model else {}),
                **({"effort": self.config.reasoning_effort} if self.config.reasoning_effort else {}),
                "summary": "concise",
                "personality": self.config.personality,
            },
            timeout_s=min(self.config.request_timeout_s, max(5, budget.max_turn_wall_time_s)),
        )
        turn = response["turn"]
        started_at = self._unix_to_iso(turn.get("startedAt")) or isoformat_z()
        turn_handle = TurnHandle(
            session_id=session_id,
            turn_id=turn["id"],
            kind=kind,
            started_at=started_at,
        )
        trace_dir = Path(workspace.workspace_root) / ".run" / "traces" / handle.run_id / turn_handle.turn_id
        trace_dir.mkdir(parents=True, exist_ok=True)
        self.turns[turn_handle.turn_id] = _TurnRuntime(
            session_id=session_id,
            thread_id=handle.thread_id,
            run_id=handle.run_id,
            kind=kind,
            workspace_root=Path(workspace.workspace_root),
            notification_cursor=notification_cursor,
            stderr_cursor=stderr_cursor,
            started_at=started_at,
            trace_dir=trace_dir,
        )
        return turn_handle

    def wait_turn(
        self,
        *,
        session_id: str,
        turn_id: str,
        timeout_s: int | None = None,
    ) -> TurnResult:
        del session_id
        runtime = self.turns[turn_id]
        try:
            completion, end_index = self.client.wait_for_turn_completion(
                notification_cursor=runtime.notification_cursor,
                thread_id=runtime.thread_id,
                turn_id=turn_id,
                timeout_s=timeout_s,
            )
            notifications = self.client.notifications_slice(runtime.notification_cursor, end_index)
            stderr_lines = self.client.stderr_slice(runtime.stderr_cursor)
        except Exception as exc:
            notifications = self.client.notifications_slice(runtime.notification_cursor)
            stderr_lines = self.client.stderr_slice(runtime.stderr_cursor)
            trace_bundle = self._write_trace_bundle(
                runtime=runtime,
                turn_id=turn_id,
                notifications=notifications,
                stderr_lines=stderr_lines,
                outcome="incomplete",
            )
            message = f"{exc}; partial trace at {trace_bundle.events_path}"
            if isinstance(exc, TimeoutError):
                raise TimeoutError(message) from exc
            raise RuntimeError(message) from exc
        turn = (completion.get("params") or {}).get("turn") or {}
        outcome = turn.get("status") or "completed"
        finished_at = self._unix_to_iso(turn.get("completedAt")) or isoformat_z()
        error_message = None
        if isinstance(turn.get("error"), dict):
            error_message = turn["error"].get("message")
        final_message = self._extract_final_message(notifications)
        trace_bundle = self._write_trace_bundle(
            runtime=runtime,
            turn_id=turn_id,
            notifications=notifications,
            stderr_lines=stderr_lines,
            outcome=outcome,
        )
        return TurnResult(
            session_id=runtime.session_id,
            turn_id=turn_id,
            kind=runtime.kind,
            outcome=outcome,
            finished_at=finished_at,
            final_message=final_message,
            trace_bundle=trace_bundle,
            error_message=error_message,
        )

    def get_session_state(self, *, session_id: str) -> SessionHandle:
        return self.sessions[session_id]

    def close_session(self, *, session_id: str, final_status: str) -> None:
        handle = self.sessions[session_id]
        try:
            self.client.request(
                "thread/unsubscribe",
                {"threadId": handle.thread_id},
                timeout_s=self.config.request_timeout_s,
            )
        except Exception:
            pass
        handle.status = final_status

    def interrupt_turn(self, *, session_id: str, turn_id: str) -> None:
        del session_id
        runtime = self.turns[turn_id]
        self.client.interrupt_turn(
            thread_id=runtime.thread_id,
            turn_id=turn_id,
            timeout_s=self.config.request_timeout_s,
        )

    def _developer_instructions(self, instructions: InstructionBundle, *, workspace: WorkspacePolicy) -> str:
        hints = []
        if instructions.agents_md_path:
            hints.append(f"AGENTS.md: {instructions.agents_md_path}")
        if instructions.skills_root_path:
            hints.append(f"skills: {instructions.skills_root_path}")
        if workspace.allow_network:
            access_guidance = (
                "Use semantic server tools (`ctx`, `artifact`, `eval`, `finding`, `notebook`, `job`) for "
                "experiment history, artifacts, feedback, and durable state. Public network search is allowed when it "
                "has expected value; do not browse or depend on hidden evaluator logic, private assets, or non-public archives."
            )
        else:
            access_guidance = (
                "Network access is disabled. Use semantic server tools (`ctx`, `artifact`, `eval`, `finding`, "
                "`notebook`, `job`) for experiment history, artifacts, feedback, and durable state; do not browse or "
                "depend on hidden evaluator logic, private assets, or non-public archives."
            )
        guidance = [
            "Project-local instructions are available on disk.",
            *hints,
            "",
            access_guidance,
        ]
        return "\n".join(guidance).strip()

    def _sandbox_policy(self, workspace: WorkspacePolicy) -> dict[str, Any]:
        readable_roots = list(workspace.readable_roots)
        for root in self._tool_support_roots():
            if root not in readable_roots:
                readable_roots.append(root)
        return {
            "type": "workspaceWrite",
            "writableRoots": workspace.writable_roots,
            "readOnlyAccess": {
                "type": "restricted",
                "includePlatformDefaults": True,
                "readableRoots": readable_roots,
            },
            "networkAccess": workspace.allow_network,
            "excludeTmpdirEnvVar": False,
            "excludeSlashTmp": False,
        }

    def _write_trace_bundle(
        self,
        *,
        runtime: _TurnRuntime,
        turn_id: str,
        notifications: list[dict[str, Any]],
        stderr_lines: list[str],
        outcome: str,
    ) -> TraceBundle:
        events_path = runtime.trace_dir / "events.jsonl"
        stdout_log_path = runtime.trace_dir / "stdout.log"
        with events_path.open("w", encoding="utf-8") as handle:
            for notification in notifications:
                handle.write(json.dumps(notification, sort_keys=True) + "\n")
        message_chunks = []
        command_output = []
        for notification in notifications:
            method = notification.get("method")
            params = notification.get("params") or {}
            if method in {"item/agentMessage/delta", "agent_message.delta", "agent_message.chunk"}:
                delta = params.get("delta") or params.get("text")
                if isinstance(delta, str):
                    message_chunks.append(delta)
            if method in {"item/commandExecution/outputDelta", "command/exec/outputDelta"}:
                delta = params.get("delta")
                if isinstance(delta, str):
                    command_output.append(delta)
        sections = []
        if stderr_lines:
            sections.append("# codex app-server stderr\n" + "\n".join(stderr_lines))
        if message_chunks:
            sections.append("# agent messages\n" + "".join(message_chunks))
        if command_output:
            sections.append("# command output\n" + "".join(command_output))
        if not sections:
            sections.append("# trace\nNo stderr or command output captured.")
        atomic_write_text(stdout_log_path, "\n\n".join(sections) + "\n")
        return TraceBundle(
            session_id=runtime.session_id,
            run_id=runtime.run_id,
            turn_id=turn_id,
            events_path=str(events_path),
            stdout_log_path=str(stdout_log_path),
            summary={
                "event_count": len(notifications),
                "stderr_line_count": len(stderr_lines),
                "outcome": outcome,
            },
        )

    def _extract_final_message(self, notifications: list[dict[str, Any]]) -> str | None:
        completed_messages: list[str] = []
        streaming_chunks: list[str] = []
        for notification in notifications:
            method = notification.get("method")
            params = notification.get("params") or {}
            item = params.get("item") or {}
            if method == "item/completed":
                text = item.get("text")
                if isinstance(text, str) and text:
                    completed_messages.append(text)
            elif method in {"item/agentMessage/delta", "agent_message.delta", "agent_message.chunk"}:
                delta = params.get("delta") or params.get("text")
                if isinstance(delta, str):
                    streaming_chunks.append(delta)
        if completed_messages:
            return "\n\n".join(completed_messages).strip()
        text = "".join(streaming_chunks).strip()
        return text or None

    def _tool_support_roots(self) -> list[str]:
        candidates = {
            str(Path.home() / ".codex"),
            "/bin",
            "/usr/bin",
            "/usr/lib",
            "/System",
            "/Library",
            "/opt/homebrew",
        }
        for executable in ("node", "python3", "zsh", "codex"):
            path = shutil.which(executable)
            if path:
                resolved = Path(path).resolve()
                candidates.add(str(resolved))
                candidates.add(str(resolved.parent))
        current_python = Path(sys.executable).resolve()
        candidates.add(str(current_python))
        candidates.add(str(current_python.parent))
        resolved_python = current_python.resolve()
        candidates.add(str(resolved_python))
        candidates.add(str(resolved_python.parent))
        return sorted(candidates)

    def _unix_to_iso(self, timestamp: int | float | None) -> str | None:
        if timestamp is None:
            return None
        return datetime.fromtimestamp(timestamp, tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
