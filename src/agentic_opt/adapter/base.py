from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class WorkspacePolicy:
    workspace_root: str
    writable_roots: list[str]
    readable_roots: list[str]
    sandbox_mode: str = "workspace-write"
    allow_network: bool = False


@dataclass(frozen=True)
class InstructionBundle:
    task_id: str
    startup_prompt: str
    agents_md_path: str | None = None
    skills_root_path: str | None = None


@dataclass(frozen=True)
class BudgetPolicy:
    max_turn_wall_time_s: int | None
    max_turns: int | None = None
    max_model_turns: int | None = None
    max_evaluator_runs_per_session: int | None = None
    evaluator_run_budget: int | None = None
    max_consecutive_invalid_submits: int | None = None
    max_consecutive_infra_errors: int | None = None
    idle_timeout_s: int | None = None
    max_tool_calls: int | None = None


@dataclass
class SessionHandle:
    session_id: str
    task_id: str
    agent_id: str
    run_id: str
    provider: str
    thread_id: str
    root_thread_id: str | None
    status: str


@dataclass
class TurnHandle:
    session_id: str
    turn_id: str
    kind: str
    started_at: str


@dataclass
class TraceBundle:
    session_id: str
    run_id: str
    turn_id: str | None
    events_path: str | None
    stdout_log_path: str | None
    summary: dict


@dataclass
class TurnResult:
    session_id: str
    turn_id: str
    kind: str
    outcome: str
    finished_at: str
    final_message: str | None
    trace_bundle: TraceBundle
    error_message: str | None = None


class CodexAdapter(Protocol):
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
        ...

    def resume_session(
        self,
        *,
        session_id: str,
        run_id: str,
        workspace: WorkspacePolicy,
        instructions: InstructionBundle,
        budget: BudgetPolicy,
    ) -> SessionHandle:
        ...

    def start_turn(
        self,
        *,
        session_id: str,
        kind: str,
        prompt: str,
        budget: BudgetPolicy | None = None,
    ) -> TurnHandle:
        ...

    def wait_turn(
        self,
        *,
        session_id: str,
        turn_id: str,
        timeout_s: int | None = None,
    ) -> TurnResult:
        ...

    def get_session_state(self, *, session_id: str) -> SessionHandle:
        ...

    def close_session(self, *, session_id: str, final_status: str) -> None:
        ...
