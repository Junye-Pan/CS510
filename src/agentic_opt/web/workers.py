from __future__ import annotations

import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_opt.control_plane.process_env import build_subprocess_env
from agentic_opt.control_plane.relay import relay_url, start_relay_process
from agentic_opt.control_plane.repository import ControlPlaneRepository


DOCKER_WORKER_BACKENDS = {"docker", "docker_image", "local-docker", "local-docker-strict"}
TERMINAL_SESSION_STATUSES = {"completed", "failed", "cancelled", "interrupted", "stopped", "blocked"}
CONTINUABLE_ASSIGNMENT_STATUSES = {"completed", "stopped"}
CONTINUABLE_STOP_REASONS = {"turn_completed", "turn_timeout"}


@dataclass
class WorkerProcess:
    assignment_id: str
    session_id: str
    experiment_id: str
    process: subprocess.Popen[str]
    relay_process: subprocess.Popen[str] | None = None
    api_url: str = ""
    dry_run: bool = False
    max_turn_wall_time_s: int | None = None


class WorkerManager:
    def __init__(
        self,
        *,
        repo_root: Path,
        state_root: Path,
        control: ControlPlaneRepository,
        reaper_interval_s: float | None = 1.0,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.state_root = state_root.resolve()
        self.control = control
        self._assignment_processes: dict[str, WorkerProcess] = {}
        self._lock = threading.RLock()
        self._reaper_stop = threading.Event()
        self._reaper_thread: threading.Thread | None = None
        if reaper_interval_s is not None and reaper_interval_s > 0:
            self._reaper_thread = threading.Thread(
                target=self._reaper_loop,
                args=(float(reaper_interval_s),),
                name="agentic-opt-worker-reaper",
                daemon=True,
            )
            self._reaper_thread.start()

    def start_control_assignment(
        self,
        *,
        assignment_id: str,
        api_url: str,
        dry_run: bool = False,
        max_turn_wall_time_s: int | None = None,
    ) -> dict[str, Any]:
        self.reap_finished_processes()
        with self._lock:
            existing = self._assignment_processes.get(assignment_id)
            if existing is not None and self._reap_worker_locked(assignment_id, existing) is None:
                raise RuntimeError(f"assignment already running: {assignment_id}")
        assignment = self.control.get_assignment(assignment_id)
        if assignment is None:
            raise KeyError(assignment_id)
        session = self.control.create_session(
            {
                "assignment_id": assignment_id,
                "status": "starting",
                "worker_backend": assignment["worker_backend"],
                "details": {"api_url": api_url, "dry_run": dry_run},
            }
        )
        workspace_root = self.state_root / "workspaces" / assignment_id / session["session_id"]
        log_dir = self.state_root / "worker_logs" / assignment_id / session["session_id"]
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / "stdout.log"
        stderr_path = log_dir / "stderr.log"
        worker_api_url = api_url
        relay_process: subprocess.Popen[str] | None = None
        relay_details: dict[str, Any] = {}
        if _needs_docker_control_plane_relay(assignment=assignment, experiment=self.control.get_experiment(assignment["experiment_id"])):
            relay_socket = self.state_root / "relays" / assignment_id / session["session_id"] / "control.sock"
            relay_process = start_relay_process(
                socket_path=relay_socket,
                target_url=api_url,
                env=_worker_process_env(self.repo_root),
            )
            _wait_for_socket(relay_socket)
            worker_api_url = relay_url(relay_socket)
            relay_details = {
                "control_plane_relay": {
                    "relay_url": worker_api_url,
                    "relay_socket_path": str(relay_socket),
                    "relay_pid": relay_process.pid,
                    "target_url": api_url,
                    "transport": "unix-socket",
                }
            }
        cmd = [
            sys.executable,
            "-m",
            "agentic_opt.adapter.semantic_worker",
            "--assignment-id",
            assignment_id,
            "--session-id",
            session["session_id"],
            "--api-url",
            worker_api_url,
            "--workspace-root",
            str(workspace_root),
        ]
        if dry_run:
            cmd.append("--dry-run")
        if max_turn_wall_time_s is not None:
            cmd.extend(["--max-turn-wall-time-s", str(max_turn_wall_time_s)])
        env = _worker_process_env(self.repo_root)
        env["AO_CONTROL_API_URL"] = worker_api_url
        env["AO_ASSIGNMENT_ID"] = assignment_id
        env["AO_SESSION_ID"] = session["session_id"]
        env["AO_TASK_ID"] = assignment["task_id"]
        env["AO_EXPERIMENT_ID"] = assignment["experiment_id"]
        env["AO_AGENT_ID"] = assignment["agent_id"]
        try:
            with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
                process = subprocess.Popen(
                    cmd,
                    cwd=str(self.repo_root),
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                )
        except Exception:
            _terminate_process(relay_process)
            raise
        with self._lock:
            self._assignment_processes[assignment_id] = WorkerProcess(
                assignment_id=assignment_id,
                session_id=session["session_id"],
                experiment_id=assignment["experiment_id"],
                process=process,
                relay_process=relay_process,
                api_url=api_url,
                dry_run=dry_run,
                max_turn_wall_time_s=max_turn_wall_time_s,
            )
        return self.control.update_session(
            session["session_id"],
            {
                "status": "running",
                "pid": process.pid,
                "workspace_path": str(workspace_root),
                "details": {
                    "api_url": api_url,
                    "worker_api_url": worker_api_url,
                    "dry_run": dry_run,
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                    **relay_details,
                },
            },
        )

    def worker_status(self, assignment_id: str) -> dict[str, Any] | None:
        summary: dict[str, Any] | None = None
        with self._lock:
            worker = self._assignment_processes.get(assignment_id)
            if worker is None:
                return None
            reaped = self._reap_worker_locked(assignment_id, worker)
            if reaped is not None:
                summary = reaped
            else:
                return {
                    "assignment_id": worker.assignment_id,
                    "session_id": worker.session_id,
                    "experiment_id": worker.experiment_id,
                    "pid": worker.process.pid,
                    "status": "running",
                }
        continuation = self._maybe_continue_assignment(summary)
        if continuation is not None:
            summary["continuation"] = continuation
        return summary

    def reap_finished_processes(self) -> list[dict[str, Any]]:
        with self._lock:
            reaped: list[dict[str, Any]] = []
            for assignment_id, worker in list(self._assignment_processes.items()):
                summary = self._reap_worker_locked(assignment_id, worker)
                if summary is not None:
                    reaped.append(summary)
        for summary in reaped:
            continuation = self._maybe_continue_assignment(summary)
            if continuation is not None:
                summary["continuation"] = continuation
        return reaped

    def close(self) -> None:
        self._reaper_stop.set()
        if self._reaper_thread is not None:
            self._reaper_thread.join(timeout=2.0)

    def _reaper_loop(self, interval_s: float) -> None:
        while not self._reaper_stop.wait(interval_s):
            try:
                self.reap_finished_processes()
            except Exception:
                continue

    def _reap_worker_locked(self, assignment_id: str, worker: WorkerProcess) -> dict[str, Any] | None:
        returncode = worker.process.poll()
        if returncode is None:
            return None
        try:
            returncode = worker.process.wait(timeout=0)
        except subprocess.TimeoutExpired:
            return None
        _terminate_process(worker.relay_process)
        if self._assignment_processes.get(assignment_id) is worker:
            self._assignment_processes.pop(assignment_id, None)
        self._finalize_session_if_needed(worker, returncode)
        return {
            "assignment_id": worker.assignment_id,
            "session_id": worker.session_id,
            "experiment_id": worker.experiment_id,
            "pid": worker.process.pid,
            "status": "finished",
            "returncode": returncode,
            "api_url": worker.api_url,
            "dry_run": worker.dry_run,
            "max_turn_wall_time_s": worker.max_turn_wall_time_s,
        }

    def _maybe_continue_assignment(self, summary: dict[str, Any] | None) -> dict[str, Any] | None:
        if not summary or summary.get("returncode") != 0:
            return None
        assignment_id = str(summary["assignment_id"])
        assignment = self.control.get_assignment(assignment_id)
        session = self.control.get_session(str(summary["session_id"]))
        if assignment is None or session is None:
            return None
        if assignment.get("status") not in CONTINUABLE_ASSIGNMENT_STATUSES:
            return None
        session_details = session.get("details") or {}
        if session_details.get("stop_reason") not in CONTINUABLE_STOP_REASONS:
            return None
        metadata = assignment.get("metadata") or {}
        if metadata.get("global_stop_condition") or metadata.get("stop_condition") or metadata.get("auto_continue_disabled"):
            return None

        budget_state = self._evaluator_budget_state(assignment)
        if not budget_state["has_budget"]:
            return None
        if int(budget_state["remaining"]) <= 0:
            self.control.update_assignment_status(
                assignment_id,
                "completed",
                metadata={
                    "budget_exhausted": {
                        "source": "worker_reaper",
                        **budget_state,
                    }
                },
            )
            self.control.record_event(
                {
                    "experiment_id": assignment["experiment_id"],
                    "assignment_id": assignment_id,
                    "session_id": session["session_id"],
                    "task_id": assignment["task_id"],
                    "agent_id": assignment["agent_id"],
                    "event_type": "assignment.budget_exhausted",
                    "summary": "assignment evaluator budget exhausted",
                    "payload": budget_state,
                }
            )
            return None

        self.control.update_experiment_status(
            assignment["experiment_id"],
            "running",
            metadata={
                "auto_continue": {
                    "source": "worker_reaper",
                    "assignment_id": assignment_id,
                    "previous_session_id": session["session_id"],
                    "budget": budget_state,
                }
            },
        )
        self.control.record_event(
            {
                "experiment_id": assignment["experiment_id"],
                "assignment_id": assignment_id,
                "session_id": session["session_id"],
                "task_id": assignment["task_id"],
                "agent_id": assignment["agent_id"],
                "event_type": "assignment.auto_continue",
                "summary": "worker completed before evaluator budget was exhausted; starting another session",
                "payload": budget_state,
            }
        )
        return self.start_control_assignment(
            assignment_id=assignment_id,
            api_url=str(summary.get("api_url") or (session_details.get("api_url") or "")),
            dry_run=bool(summary.get("dry_run")),
            max_turn_wall_time_s=summary.get("max_turn_wall_time_s"),
        )

    def _evaluator_budget_state(self, assignment: dict[str, Any]) -> dict[str, Any]:
        experiment = self.control.get_experiment(assignment["experiment_id"])
        assignment_limit = _positive_int((assignment.get("budget") or {}).get("evaluator_runs"))
        experiment_budget = (experiment or {}).get("budget") or {}
        experiment_limit = _positive_int(experiment_budget.get("total_evaluator_runs") or experiment_budget.get("evaluator_runs"))
        assignment_used = _count_evaluator_runs(self.control.list_evaluations(assignment_id=assignment["assignment_id"]))
        experiment_used = _count_evaluator_runs(self.control.list_evaluations(experiment_id=assignment["experiment_id"]))
        remaining_values: list[int] = []
        if assignment_limit is not None:
            remaining_values.append(assignment_limit - assignment_used)
        if experiment_limit is not None:
            remaining_values.append(experiment_limit - experiment_used)
        remaining = min(remaining_values) if remaining_values else 0
        return {
            "has_budget": bool(remaining_values),
            "remaining": max(0, remaining),
            "assignment_limit": assignment_limit,
            "assignment_used": assignment_used,
            "experiment_limit": experiment_limit,
            "experiment_used": experiment_used,
        }

    def _finalize_session_if_needed(self, worker: WorkerProcess, returncode: int) -> None:
        try:
            session = self.control.get_session(worker.session_id)
        except Exception:
            return
        if session is None or session.get("status") in TERMINAL_SESSION_STATUSES:
            return
        details = {
            **(session.get("details") or {}),
            "worker_exit": {
                "pid": worker.process.pid,
                "returncode": returncode,
                "source": "worker_reaper",
            },
        }
        self.control.update_session(
            worker.session_id,
            {
                "status": "completed" if returncode == 0 else "failed",
                "details": details,
            },
        )


def _count_evaluator_runs(evaluations: list[dict[str, Any]]) -> int:
    return sum(1 for item in evaluations if item.get("status") != "cancelled")


def _positive_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _needs_docker_control_plane_relay(*, assignment: dict[str, Any], experiment: dict[str, Any] | None) -> bool:
    if assignment.get("worker_backend") not in DOCKER_WORKER_BACKENDS:
        return False
    raw_policy = ((experiment or {}).get("policy") or {}).get("network") or ((experiment or {}).get("config") or {}).get("network") or {}
    external = raw_policy.get("external_internet")
    if external is None:
        external = "deny" if raw_policy.get("allow_external_internet") is False else "allow"
    control_plane = raw_policy.get("control_plane") or raw_policy.get("control_plane_network") or "allow"
    return str(external) == "deny" and str(control_plane) == "allow"


def _wait_for_socket(path: Path, *, timeout_s: float = 5.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise TimeoutError(f"control-plane relay socket did not appear: {path}")


def _terminate_process(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return
    if process.poll() is not None:
        try:
            process.wait(timeout=0)
        except subprocess.TimeoutExpired:
            pass
        return
    process.terminate()
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass


def _worker_process_env(repo_root: Path) -> dict[str, str]:
    env = build_subprocess_env()
    repo_src = str(repo_root / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = repo_src if not existing_pythonpath else repo_src + ":" + existing_pythonpath
    return env
