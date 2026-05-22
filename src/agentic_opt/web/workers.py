from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentic_opt.control_plane.docker_runtime import (
    DockerMount,
    DockerNetworkPolicyError,
    build_docker_run_command,
)
from agentic_opt.control_plane.process_env import build_subprocess_env
from agentic_opt.control_plane.relay import relay_url, start_relay_process, tcp_relay_url
from agentic_opt.control_plane.network_proxy import proxy_url, start_network_proxy_process, wait_for_unix_socket
from agentic_opt.control_plane.repository import ControlPlaneRepository
from agentic_opt.control_plane.task_context import (
    append_docker_task_context_mount,
    docker_task_context_enforcement,
    ensure_task_context_snapshot,
)
from agentic_opt.control_plane.traces import AgentTraceService


DOCKER_WORKER_BACKENDS = {"docker", "docker_image", "local-docker", "local-docker-strict"}
TERMINAL_SESSION_STATUSES = {"completed", "failed", "cancelled", "interrupted", "stopped", "blocked"}
CONTINUABLE_ASSIGNMENT_STATUSES = {"completed", "stopped"}
CONTINUABLE_STOP_REASONS = {"turn_completed", "turn_timeout"}
CONTAINER_CONTROL_PLANE_SOCKET_PATH = "/ao-control/control.sock"
CONTAINER_OUTBOUND_PROXY_SOCKET_PATH = "/ao-network/proxy.sock"
CONTAINER_OUTBOUND_PROXY_BRIDGE_PORT = 8765
DEFAULT_STALE_SESSION_GRACE_S = 300.0
_DOCKER_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass
class WorkerProcess:
    assignment_id: str
    session_id: str
    experiment_id: str
    process: subprocess.Popen[str]
    relay_process: subprocess.Popen[str] | None = None
    proxy_process: subprocess.Popen[str] | None = None
    api_url: str = ""
    dry_run: bool = False
    max_turn_wall_time_s: int | None = None
    docker_container_name: str | None = None


class WorkerManager:
    def __init__(
        self,
        *,
        repo_root: Path,
        state_root: Path,
        control: ControlPlaneRepository,
        reaper_interval_s: float | None = 1.0,
        stale_session_grace_s: float = DEFAULT_STALE_SESSION_GRACE_S,
        default_api_url: str | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.state_root = state_root.resolve()
        self.control = control
        self._assignment_processes: dict[str, WorkerProcess] = {}
        self._lock = threading.RLock()
        self._recovering_stale_sessions = False
        self.stale_session_grace_s = float(stale_session_grace_s)
        self.default_api_url = default_api_url or os.environ.get("AO_CONTROL_API_URL")
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
        self.default_api_url = api_url
        self.reap_finished_processes()
        with self._lock:
            existing = self._assignment_processes.get(assignment_id)
            if existing is not None and self._reap_worker_locked(assignment_id, existing) is None:
                raise RuntimeError(f"assignment already running: {assignment_id}")
        assignment = self.control.get_assignment(assignment_id)
        if assignment is None:
            raise KeyError(assignment_id)
        experiment = self.control.get_experiment(assignment["experiment_id"])
        if experiment is None:
            raise KeyError(assignment["experiment_id"])
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
        workspace_root.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / "stdout.log"
        stderr_path = log_dir / "stderr.log"
        relay_audit_path = log_dir / "relay_audit.jsonl"
        worker_api_url = api_url
        relay_process: subprocess.Popen[str] | None = None
        proxy_process: subprocess.Popen[str] | None = None
        relay_socket: Path | None = None
        relay_details: dict[str, Any] = {}
        proxy_details: dict[str, Any] = {}
        docker_worker = _is_docker_worker_backend(str(assignment.get("worker_backend") or ""))
        docker_container_name: str | None = None
        docker_details: dict[str, Any] = {}
        network_policy = _network_policy_for_experiment(experiment)
        if docker_worker:
            _prepare_docker_worker_host_paths(state_root=self.state_root)
        if docker_worker and network_policy.get("control_plane") != "allow":
            blocked = self._block_starting_session(
                session=session,
                assignment=assignment,
                reason="docker_control_plane_denied_by_policy",
                message="Docker semantic worker requires control-plane access, but network policy denies control_plane",
                details={"network_policy": network_policy},
            )
            raise RuntimeError(blocked["details"]["startup_error"]["message"])
        if _needs_docker_control_plane_relay(assignment=assignment, experiment=experiment):
            relay_transport = _docker_control_plane_relay_transport(assignment=assignment, experiment=experiment)
            if relay_transport == "tcp":
                relay_host = "127.0.0.1"
                relay_port = _free_tcp_port()
                relay_process = start_relay_process(
                    socket_path=None,
                    target_url=api_url,
                    env=_worker_process_env(self.repo_root),
                    audit_log_path=relay_audit_path,
                    transport="tcp",
                    tcp_host=relay_host,
                    tcp_port=relay_port,
                )
                _wait_for_tcp(relay_host, relay_port)
                host_relay_url = tcp_relay_url(relay_host, relay_port)
                container_relay_host = _docker_container_relay_host(assignment=assignment, experiment=experiment)
                worker_api_url = tcp_relay_url(container_relay_host, relay_port) if docker_worker else host_relay_url
                relay_details = {
                    "control_plane_relay": {
                        "relay_url": host_relay_url,
                        "worker_relay_url": worker_api_url,
                        "relay_tcp_host": relay_host,
                        "relay_tcp_port": relay_port,
                        "relay_pid": relay_process.pid,
                        "target_url": api_url,
                        "transport": "tcp",
                        "audit_log_path": str(relay_audit_path),
                        "policy_weakened": True,
                        "policy_weakened_reason": "tcp_control_plane_relay_requires_docker_network",
                    }
                }
            else:
                relay_socket = _relay_socket_path(
                    state_root=self.state_root,
                    assignment_id=assignment_id,
                    session_id=session["session_id"],
                )
                relay_process = start_relay_process(
                    socket_path=relay_socket,
                    target_url=api_url,
                    env=_worker_process_env(self.repo_root),
                    audit_log_path=relay_audit_path,
                )
                _wait_for_socket(relay_socket)
                host_relay_url = relay_url(relay_socket)
                worker_api_url = f"unix://{CONTAINER_CONTROL_PLANE_SOCKET_PATH}" if docker_worker else host_relay_url
                relay_details = {
                    "control_plane_relay": {
                        "relay_url": host_relay_url,
                        "worker_relay_url": worker_api_url,
                        "relay_socket_path": str(relay_socket),
                        "container_socket_path": CONTAINER_CONTROL_PLANE_SOCKET_PATH if docker_worker else None,
                        "relay_pid": relay_process.pid,
                        "target_url": api_url,
                        "transport": "unix-socket",
                        "audit_log_path": str(relay_audit_path),
                    }
                }
        if docker_worker and _needs_outbound_audit_proxy(network_policy):
            proxy_metadata = {
                "experiment_id": assignment["experiment_id"],
                "assignment_id": assignment_id,
                "session_id": session["session_id"],
                "task_id": assignment["task_id"],
                "agent_id": assignment["agent_id"],
                "worker_backend": assignment.get("worker_backend"),
            }
            proxy_transport = _docker_outbound_proxy_transport(assignment=assignment, experiment=experiment)
            if proxy_transport == "unix-socket":
                proxy_socket = _outbound_proxy_socket_path(
                    state_root=self.state_root,
                    assignment_id=assignment_id,
                    session_id=session["session_id"],
                )
                proxy_process = start_network_proxy_process(
                    socket_path=proxy_socket,
                    database_path=self.control.db_path,
                    policy=network_policy,
                    metadata=proxy_metadata,
                    env=_worker_process_env(self.repo_root),
                )
                try:
                    wait_for_unix_socket(proxy_socket)
                except Exception:
                    _terminate_process(proxy_process)
                    _terminate_process(relay_process)
                    raise
                container_proxy_url = f"http://127.0.0.1:{CONTAINER_OUTBOUND_PROXY_BRIDGE_PORT}"
                network_policy = {
                    **network_policy,
                    "outbound_proxy_url": container_proxy_url,
                    "outbound_proxy_socket": str(proxy_socket),
                    "outbound_proxy_container_socket": CONTAINER_OUTBOUND_PROXY_SOCKET_PATH,
                    "outbound_proxy_bridge_port": CONTAINER_OUTBOUND_PROXY_BRIDGE_PORT,
                    "outbound_proxy_no_proxy": "127.0.0.1,localhost",
                }
                proxy_details = {
                    "outbound_audit_proxy": {
                        "proxy_url": f"unix://{proxy_socket.resolve()}",
                        "worker_proxy_url": container_proxy_url,
                        "proxy_socket_path": str(proxy_socket),
                        "container_socket_path": CONTAINER_OUTBOUND_PROXY_SOCKET_PATH,
                        "bridge_port": CONTAINER_OUTBOUND_PROXY_BRIDGE_PORT,
                        "proxy_pid": proxy_process.pid,
                        "transport": "unix-socket-http-proxy",
                    }
                }
            else:
                proxy_host = "127.0.0.1"
                proxy_port = _free_tcp_port()
                proxy_process = start_network_proxy_process(
                    host=proxy_host,
                    port=proxy_port,
                    database_path=self.control.db_path,
                    policy=network_policy,
                    metadata=proxy_metadata,
                    env=_worker_process_env(self.repo_root),
                )
                try:
                    _wait_for_tcp(proxy_host, proxy_port)
                except Exception:
                    _terminate_process(proxy_process)
                    _terminate_process(relay_process)
                    raise
                container_proxy_url = proxy_url(_docker_container_relay_host(assignment=assignment, experiment=experiment), proxy_port)
                network_policy = {
                    **network_policy,
                    "outbound_proxy_url": container_proxy_url,
                    "outbound_proxy_no_proxy": "127.0.0.1,localhost",
                }
                proxy_details = {
                    "outbound_audit_proxy": {
                        "proxy_url": proxy_url(proxy_host, proxy_port),
                        "worker_proxy_url": container_proxy_url,
                        "proxy_tcp_host": proxy_host,
                        "proxy_tcp_port": proxy_port,
                        "proxy_pid": proxy_process.pid,
                        "transport": "http-proxy",
                    }
                }
        if docker_worker:
            try:
                worker_image = _resolve_docker_worker_image(assignment=assignment, experiment=experiment)
                docker_container_name = _docker_container_name(assignment_id=assignment_id, session_id=session["session_id"])
                cmd, network_enforcement = build_docker_worker_command(
                    image=worker_image,
                    assignment=assignment,
                    session_id=session["session_id"],
                    api_url=worker_api_url,
                    workspace_root=workspace_root,
                    state_root=self.state_root,
                    network_policy=network_policy,
                    requested_network_mode=_requested_docker_network_mode(assignment=assignment, experiment=experiment),
                    control_plane_relay_socket=relay_socket if relay_socket is not None else None,
                    control_plane_relay_url=worker_api_url if (relay_details.get("control_plane_relay") or {}).get("transport") == "tcp" else None,
                    dry_run=dry_run,
                    max_turn_wall_time_s=max_turn_wall_time_s,
                    container_name=docker_container_name,
                    codex_source_home=_host_codex_source_home(),
                )
            except DockerNetworkPolicyError as exc:
                _terminate_process(relay_process)
                _terminate_process(proxy_process)
                blocked = self._block_starting_session(
                    session=session,
                    assignment=assignment,
                    reason=exc.reason,
                    message=str(exc),
                    details={
                        "network_policy": network_policy,
                        "network_enforcement": exc.enforcement,
                        **relay_details,
                        **proxy_details,
                    },
                )
                raise RuntimeError(blocked["details"]["startup_error"]["message"]) from exc
            except Exception as exc:
                _terminate_process(relay_process)
                _terminate_process(proxy_process)
                blocked = self._block_starting_session(
                    session=session,
                    assignment=assignment,
                    reason=type(exc).__name__,
                    message=str(exc),
                    details={"network_policy": network_policy, **relay_details, **proxy_details},
                )
                raise RuntimeError(blocked["details"]["startup_error"]["message"]) from exc
            docker_details = {
                "docker_worker": {
                    "image": worker_image,
                    "container_name": docker_container_name,
                    "workspace_mount": str(workspace_root),
                    "environment_mount": str(self.state_root / "envs"),
                    "provider_state_mount": str(self.state_root / "provider_state"),
                },
                "network_enforcement": network_enforcement,
            }
        else:
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
            _terminate_process(proxy_process)
            raise
        with self._lock:
            self._assignment_processes[assignment_id] = WorkerProcess(
                assignment_id=assignment_id,
                session_id=session["session_id"],
                experiment_id=assignment["experiment_id"],
                process=process,
                relay_process=relay_process,
                proxy_process=proxy_process,
                api_url=api_url,
                dry_run=dry_run,
                max_turn_wall_time_s=max_turn_wall_time_s,
                docker_container_name=docker_container_name,
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
                    **proxy_details,
                    **docker_details,
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
        if not self._recovering_stale_sessions:
            reaped.extend(self.recover_stale_sessions())
        return reaped

    def recover_stale_sessions(self) -> list[dict[str, Any]]:
        if self._recovering_stale_sessions:
            return []
        self._recovering_stale_sessions = True
        try:
            with self._lock:
                managed_assignments = set(self._assignment_processes)
            recovered: list[dict[str, Any]] = []
            for assignment in self.control.list_assignments():
                assignment_id = str(assignment["assignment_id"])
                if assignment_id in managed_assignments:
                    continue
                recovered_item = self._recover_assignment_without_managed_process(assignment)
                if recovered_item is not None:
                    recovered.append(recovered_item)
            return recovered
        finally:
            self._recovering_stale_sessions = False

    def _recover_assignment_without_managed_process(self, assignment: dict[str, Any]) -> dict[str, Any] | None:
        assignment_id = str(assignment["assignment_id"])
        sessions = self.control.list_sessions(assignment_id=assignment_id)
        if not sessions:
            return None

        active_sessions = [session for session in sessions if session.get("status") in {"starting", "running"}]
        if active_sessions:
            session = active_sessions[0]
            pid = _positive_int(session.get("pid"))
            if pid is not None and _pid_is_alive(pid):
                return None
            age_s = _session_age_s(session)
            if age_s < self.stale_session_grace_s:
                return None
            return self._recover_stale_active_session(assignment, session, age_s=age_s)

        session = sessions[0]
        if session.get("status") not in {"completed", "stopped"}:
            return None
        if assignment.get("status") not in CONTINUABLE_ASSIGNMENT_STATUSES:
            return None
        session_details = session.get("details") or {}
        if session_details.get("stop_reason") not in CONTINUABLE_STOP_REASONS:
            return None
        return self._restart_assignment_from_session(
            assignment,
            session,
            reason="unmanaged_terminal_session",
            event_type="assignment.auto_continue",
            summary="worker session ended before evaluator budget was exhausted; starting another session",
        )

    def _recover_stale_active_session(
        self,
        assignment: dict[str, Any],
        session: dict[str, Any],
        *,
        age_s: float,
    ) -> dict[str, Any] | None:
        details = {
            **(session.get("details") or {}),
            "stale_session_recovery": {
                "source": "worker_reaper",
                "reason": "no_managed_worker_process",
                "previous_status": session.get("status"),
                "pid": session.get("pid"),
                "age_s": age_s,
            },
        }
        updated = self.control.update_session(
            session["session_id"],
            {
                "status": "failed",
                "details": details,
            },
        )
        self.control.record_event(
            {
                "experiment_id": assignment["experiment_id"],
                "assignment_id": assignment["assignment_id"],
                "session_id": session["session_id"],
                "task_id": assignment["task_id"],
                "agent_id": assignment["agent_id"],
                "event_type": "worker.session.stale",
                "summary": "worker session had no managed process; marked stale and failed",
                "payload": details["stale_session_recovery"],
            }
        )
        restarted = self._restart_assignment_from_session(
            assignment,
            updated,
            reason="stale_active_session",
            event_type="assignment.stale_session_restarted",
            summary="stale worker session recovered by starting another session",
        )
        if restarted is None:
            return {"session": updated, "recovered": False, "reason": "stale_active_session"}
        return {"session": updated, "recovered": True, "continuation": restarted}

    def _restart_assignment_from_session(
        self,
        assignment: dict[str, Any],
        session: dict[str, Any],
        *,
        reason: str,
        event_type: str,
        summary: str,
    ) -> dict[str, Any] | None:
        metadata = assignment.get("metadata") or {}
        if metadata.get("global_stop_condition") or metadata.get("stop_condition") or metadata.get("auto_continue_disabled"):
            return None
        budget_state = self._evaluator_budget_state(assignment)
        if not budget_state["has_budget"]:
            return None
        if int(budget_state["remaining"]) <= 0:
            self.control.update_assignment_status(
                assignment["assignment_id"],
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
                    "assignment_id": assignment["assignment_id"],
                    "session_id": session["session_id"],
                    "task_id": assignment["task_id"],
                    "agent_id": assignment["agent_id"],
                    "event_type": "assignment.budget_exhausted",
                    "summary": "assignment evaluator budget exhausted",
                    "payload": budget_state,
                }
            )
            return None
        session_details = session.get("details") or {}
        api_url = str(session_details.get("api_url") or self.default_api_url or "")
        if not api_url:
            self.control.record_event(
                {
                    "experiment_id": assignment["experiment_id"],
                    "assignment_id": assignment["assignment_id"],
                    "session_id": session["session_id"],
                    "task_id": assignment["task_id"],
                    "agent_id": assignment["agent_id"],
                    "event_type": "assignment.auto_continue.blocked",
                    "summary": "cannot restart assignment without control-plane api_url",
                    "payload": {"reason": "missing_api_url", "budget": budget_state, "source": "worker_reaper"},
                }
            )
            return None
        self.control.update_experiment_status(
            assignment["experiment_id"],
            "running",
            metadata={
                "auto_continue": {
                    "source": "worker_reaper",
                    "reason": reason,
                    "assignment_id": assignment["assignment_id"],
                    "previous_session_id": session["session_id"],
                    "budget": budget_state,
                }
            },
        )
        self.control.record_event(
            {
                "experiment_id": assignment["experiment_id"],
                "assignment_id": assignment["assignment_id"],
                "session_id": session["session_id"],
                "task_id": assignment["task_id"],
                "agent_id": assignment["agent_id"],
                "event_type": event_type,
                "summary": summary,
                "payload": {"reason": reason, **budget_state},
            }
        )
        return self.start_control_assignment(
            assignment_id=assignment["assignment_id"],
            api_url=api_url,
            dry_run=bool(session_details.get("dry_run")),
            max_turn_wall_time_s=session_details.get("max_turn_wall_time_s"),
        )

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
        _terminate_process(worker.proxy_process)
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
        assignment_used = _leaderboard_budget_used(
            self.control,
            experiment_id=assignment["experiment_id"],
            assignment_id=assignment["assignment_id"],
        )
        experiment_used = _leaderboard_budget_used(
            self.control,
            experiment_id=assignment["experiment_id"],
        )
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
        updated = self.control.update_session(
            worker.session_id,
            {
                "status": "completed" if returncode == 0 else "failed",
                "details": details,
            },
        )
        self._register_unregistered_session_traces(updated, outcome="worker_exit", returncode=returncode)

    def _block_starting_session(
        self,
        *,
        session: dict[str, Any],
        assignment: dict[str, Any],
        reason: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "reason": reason,
            "message": message,
            "source": "worker_manager",
        }
        updated = self.control.update_session(
            session["session_id"],
            {
                "status": "blocked",
                "details": {
                    **(details or {}),
                    "startup_error": payload,
                },
            },
        )
        self.control.record_event(
            {
                "experiment_id": assignment["experiment_id"],
                "assignment_id": assignment["assignment_id"],
                "session_id": session["session_id"],
                "task_id": assignment["task_id"],
                "agent_id": assignment["agent_id"],
                "event_type": "worker.start.blocked",
                "summary": f"worker start blocked: {reason}",
                "payload": payload,
            }
        )
        return updated

    def _register_unregistered_session_traces(self, session: dict[str, Any], *, outcome: str, returncode: int | None = None) -> None:
        workspace_path = session.get("workspace_path")
        if not workspace_path:
            return
        traces_root = Path(workspace_path) / ".run" / "traces"
        if not traces_root.exists():
            return
        trace_service = AgentTraceService(repository=self.control, artifact_root=self.state_root / "artifacts")
        for events_path in sorted(traces_root.glob("*/*/events.jsonl")):
            trace_dir = events_path.parent
            run_id = trace_dir.parent.name
            turn_id = trace_dir.name
            try:
                trace_service.register_trace_directory(
                    {
                        "experiment_id": session["experiment_id"],
                        "assignment_id": session["assignment_id"],
                        "session_id": session["session_id"],
                        "task_id": session["task_id"],
                        "agent_id": session["agent_id"],
                        "run_id": run_id,
                        "turn_id": turn_id,
                        "worker_backend": session.get("worker_backend"),
                        "trace_dir": str(trace_dir),
                        "outcome": outcome,
                        "status": "completed" if returncode == 0 else "partial",
                        "metadata": {"registered_by": "worker_reaper", "worker_returncode": returncode},
                    }
                )
            except Exception:
                continue


def _leaderboard_budget_used(
    control: ControlPlaneRepository,
    *,
    experiment_id: str,
    assignment_id: str | None = None,
) -> int:
    leaderboard_used = sum(
        1
        for entry in control.list_leaderboard_entries(experiment_id=experiment_id, limit=1_000_000)
        if assignment_id is None or entry.get("assignment_id") == assignment_id
    )
    pending_used = sum(
        1
        for evaluation in control.list_evaluations(experiment_id=experiment_id, assignment_id=assignment_id)
        if _pending_evaluation_counts_toward_leaderboard_budget(evaluation)
    )
    return leaderboard_used + pending_used


def _pending_evaluation_counts_toward_leaderboard_budget(evaluation: dict[str, Any]) -> bool:
    if evaluation.get("status") not in {"queued", "running"}:
        return False
    request = evaluation.get("request") or {}
    kind = evaluation.get("kind") or request.get("kind")
    if kind not in {"submit", "official"}:
        return False
    if request.get("count_budget") is False:
        return False
    if request.get("publish_leaderboard") is False:
        return False
    if request.get("replay") and request.get("publish_leaderboard") is not True:
        return False
    return True


def _session_age_s(session: dict[str, Any]) -> float:
    updated_at = session.get("updated_at") or session.get("started_at")
    timestamp = _parse_iso_z(str(updated_at)) if updated_at else None
    if timestamp is None:
        return float("inf")
    return max(0.0, (datetime.now(UTC) - timestamp).total_seconds())


def _parse_iso_z(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _positive_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def build_docker_worker_command(
    *,
    image: str,
    assignment: dict[str, Any],
    session_id: str,
    api_url: str,
    workspace_root: Path,
    state_root: Path,
    network_policy: dict[str, Any],
    requested_network_mode: str | None = None,
    control_plane_relay_socket: Path | None = None,
    control_plane_relay_url: str | None = None,
    dry_run: bool = False,
    max_turn_wall_time_s: int | None = None,
    container_name: str | None = None,
    codex_source_home: Path | None = None,
) -> tuple[list[str], dict[str, Any]]:
    envs_root = state_root / "envs"
    provider_state_root = state_root / "provider_state"
    task_context = ensure_task_context_snapshot(task_id=assignment["task_id"], state_root=state_root)
    (workspace_root / "task").mkdir(parents=True, exist_ok=True)
    mounts = [
        DockerMount(source=workspace_root, target=str(workspace_root)),
        DockerMount(source=envs_root, target=str(envs_root), read_only=True),
        DockerMount(source=provider_state_root, target=str(provider_state_root)),
    ]
    mounts = append_docker_task_context_mount(
        mounts=mounts,
        snapshot=task_context,
        workspace_root=workspace_root,
    )
    task_context_enforcement = docker_task_context_enforcement(snapshot=task_context, workspace_root=workspace_root)
    command = [
        "python",
        "-m",
        "agentic_opt.adapter.semantic_worker",
        "--assignment-id",
        assignment["assignment_id"],
        "--session-id",
        session_id,
        "--api-url",
        api_url,
        "--workspace-root",
        str(workspace_root),
    ]
    if dry_run:
        command.append("--dry-run")
    if max_turn_wall_time_s is not None:
        command.extend(["--max-turn-wall-time-s", str(max_turn_wall_time_s)])

    container_env = {
        "AO_CONTROL_API_URL": api_url,
        "AO_ASSIGNMENT_ID": assignment["assignment_id"],
        "AO_SESSION_ID": session_id,
        "AO_TASK_ID": assignment["task_id"],
        "AO_EXPERIMENT_ID": assignment["experiment_id"],
        "AO_AGENT_ID": assignment["agent_id"],
        "AO_TASK_RUNTIME_ENVS_ROOT": str(envs_root),
        "AO_WORKER_RUNTIME_PYTHON": "/usr/local/bin/python",
        "AO_WORKER_RUNTIME_ROOT": "/opt/agentic-opt",
        "AO_WORKER_RUNTIME_VENV": "/usr/local",
        "AO_WORKER_RUNTIME_MANIFEST": "/opt/agentic-opt/docker_worker_runtime.json",
        "PYTHONUNBUFFERED": "1",
    }
    if codex_source_home is not None and codex_source_home.exists():
        mounts.append(DockerMount(source=codex_source_home, target="/ao-codex-source", read_only=True))
        container_env["AO_CODEX_SOURCE_HOME"] = "/ao-codex-source"
    command_result, network_enforcement = build_docker_run_command(
        image=image,
        command=command,
        mounts=mounts,
        workdir=str(workspace_root),
        network_policy=network_policy,
        requested_network_mode=requested_network_mode,
        requires_control_plane=True,
        control_plane_relay_socket=control_plane_relay_socket,
        control_plane_relay_url=control_plane_relay_url,
        container_control_plane_socket_path=CONTAINER_CONTROL_PLANE_SOCKET_PATH,
        env=container_env,
        container_name=container_name,
        pids_limit=512,
        add_hosts=_docker_relay_add_hosts(control_plane_relay_url, network_policy.get("outbound_proxy_url")),
    )
    return command_result, {
        **network_enforcement,
        "task_context": task_context,
        "task_context_enforcement": task_context_enforcement,
    }


def _is_docker_worker_backend(worker_backend: str) -> bool:
    return worker_backend in DOCKER_WORKER_BACKENDS


def _network_policy_for_experiment(experiment: dict[str, Any] | None) -> dict[str, Any]:
    raw_policy = ((experiment or {}).get("policy") or {}).get("network") or ((experiment or {}).get("config") or {}).get("network") or {}
    external = raw_policy.get("external_internet")
    if external is None:
        external = "deny" if raw_policy.get("allow_external_internet") is False else "allow"
    control_plane = raw_policy.get("control_plane") or raw_policy.get("control_plane_network") or "allow"
    return {
        "control_plane": str(control_plane),
        "external_internet": str(external),
        "package_indexes": raw_policy.get("package_indexes") or "policy",
        "allowed_hosts": raw_policy.get("allowed_hosts") or ["127.0.0.1", "localhost"],
        "denied_hosts": raw_policy.get("denied_hosts") or [],
        "audit_external_attempts": bool(raw_policy.get("audit_external_attempts", True)),
        "outbound_proxy": raw_policy.get("outbound_proxy") or raw_policy.get("audit_proxy"),
    }


def _needs_docker_control_plane_relay(*, assignment: dict[str, Any], experiment: dict[str, Any] | None) -> bool:
    if not _is_docker_worker_backend(str(assignment.get("worker_backend") or "")):
        return False
    network_policy = _network_policy_for_experiment(experiment)
    return network_policy.get("external_internet") in {"deny", "audit"} and network_policy.get("control_plane") == "allow"


def _docker_control_plane_relay_transport(*, assignment: dict[str, Any], experiment: dict[str, Any] | None) -> str:
    assignment_metadata = assignment.get("metadata") or {}
    experiment_config = (experiment or {}).get("config") or {}
    experiment_policy = (experiment or {}).get("policy") or {}
    docker_metadata = assignment_metadata.get("docker") if isinstance(assignment_metadata.get("docker"), dict) else {}
    docker_config = experiment_config.get("docker") if isinstance(experiment_config.get("docker"), dict) else {}
    network_policy = experiment_policy.get("network") if isinstance(experiment_policy.get("network"), dict) else {}
    raw = (
        assignment_metadata.get("control_plane_relay_transport")
        or docker_metadata.get("control_plane_relay_transport")
        or experiment_config.get("control_plane_relay_transport")
        or docker_config.get("control_plane_relay_transport")
        or network_policy.get("control_plane_relay_transport")
        or "unix-socket"
    )
    value = str(raw).lower().replace("_", "-")
    if value == "auto":
        return "tcp" if sys.platform == "darwin" else "unix-socket"
    if value in {"unix", "unix-socket", "socket"}:
        return "unix-socket"
    if value in {"tcp", "http"}:
        return "tcp"
    raise ValueError(f"unsupported Docker control-plane relay transport: {raw}")


def _docker_container_relay_host(*, assignment: dict[str, Any], experiment: dict[str, Any] | None) -> str:
    assignment_metadata = assignment.get("metadata") or {}
    experiment_config = (experiment or {}).get("config") or {}
    docker_metadata = assignment_metadata.get("docker") if isinstance(assignment_metadata.get("docker"), dict) else {}
    docker_config = experiment_config.get("docker") if isinstance(experiment_config.get("docker"), dict) else {}
    return str(
        docker_metadata.get("container_relay_host")
        or assignment_metadata.get("container_relay_host")
        or docker_config.get("container_relay_host")
        or experiment_config.get("container_relay_host")
        or "host.docker.internal"
    )


def _docker_outbound_proxy_transport(*, assignment: dict[str, Any], experiment: dict[str, Any] | None) -> str:
    assignment_metadata = assignment.get("metadata") or {}
    experiment_config = (experiment or {}).get("config") or {}
    docker_metadata = assignment_metadata.get("docker") if isinstance(assignment_metadata.get("docker"), dict) else {}
    docker_config = experiment_config.get("docker") if isinstance(experiment_config.get("docker"), dict) else {}
    raw = (
        assignment_metadata.get("outbound_proxy_transport")
        or docker_metadata.get("outbound_proxy_transport")
        or experiment_config.get("outbound_proxy_transport")
        or docker_config.get("outbound_proxy_transport")
        or "unix-socket"
    )
    value = str(raw).lower().replace("_", "-")
    if value in {"unix", "unix-socket", "socket"}:
        return "unix-socket"
    if value in {"tcp", "http"}:
        return "tcp"
    raise ValueError(f"unsupported Docker outbound proxy transport: {raw}")


def _needs_outbound_audit_proxy(network_policy: dict[str, Any]) -> bool:
    if network_policy.get("outbound_proxy") is False:
        return False
    return str(network_policy.get("external_internet")) == "audit" and bool(network_policy.get("audit_external_attempts", True))


def _docker_relay_add_hosts(control_plane_relay_url: str | None, outbound_proxy_url: Any = None) -> list[str]:
    if (control_plane_relay_url and "host.docker.internal" in control_plane_relay_url) or (
        outbound_proxy_url and "host.docker.internal" in str(outbound_proxy_url)
    ):
        return ["host.docker.internal:host-gateway"]
    return []


def _resolve_docker_worker_image(*, assignment: dict[str, Any], experiment: dict[str, Any] | None) -> str:
    assignment_metadata = assignment.get("metadata") or {}
    experiment_config = (experiment or {}).get("config") or {}
    experiment_metadata = (experiment or {}).get("metadata") or {}
    experiment_policy = (experiment or {}).get("policy") or {}
    candidates = [
        assignment_metadata.get("worker_image"),
        assignment_metadata.get("docker_worker_image"),
        assignment_metadata.get("docker_image"),
        (assignment_metadata.get("docker") or {}).get("worker_image") if isinstance(assignment_metadata.get("docker"), dict) else None,
        (assignment_metadata.get("docker") or {}).get("image") if isinstance(assignment_metadata.get("docker"), dict) else None,
        experiment_config.get("worker_image"),
        experiment_config.get("docker_worker_image"),
        (experiment_config.get("worker") or {}).get("image") if isinstance(experiment_config.get("worker"), dict) else None,
        (experiment_config.get("docker") or {}).get("worker_image") if isinstance(experiment_config.get("docker"), dict) else None,
        (experiment_config.get("docker") or {}).get("image") if isinstance(experiment_config.get("docker"), dict) else None,
        experiment_metadata.get("worker_image"),
        experiment_metadata.get("docker_worker_image"),
        (experiment_policy.get("worker") or {}).get("image") if isinstance(experiment_policy.get("worker"), dict) else None,
    ]
    for candidate in candidates:
        if candidate:
            return str(candidate)
    raise ValueError(
        "Docker worker backend requires a worker image in assignment.metadata.worker_image "
        "or experiment.config.worker_image; the image must have codex and agentic_opt preinstalled"
    )


def _requested_docker_network_mode(*, assignment: dict[str, Any], experiment: dict[str, Any] | None) -> str | None:
    assignment_metadata = assignment.get("metadata") or {}
    experiment_config = (experiment or {}).get("config") or {}
    docker_metadata = assignment_metadata.get("docker") if isinstance(assignment_metadata.get("docker"), dict) else {}
    docker_config = experiment_config.get("docker") if isinstance(experiment_config.get("docker"), dict) else {}
    return (
        assignment_metadata.get("network_mode")
        or docker_metadata.get("network_mode")
        or experiment_config.get("network_mode")
        or docker_config.get("network_mode")
    )


def _docker_container_name(*, assignment_id: str, session_id: str) -> str:
    raw = f"agentic-opt-{assignment_id}-{session_id}"
    safe = _DOCKER_NAME_RE.sub("-", raw).strip("-._")
    return safe[:120] or "agentic-opt-worker"


def _relay_socket_path(*, state_root: Path, assignment_id: str, session_id: str) -> Path:
    nested = state_root / "relays" / assignment_id / session_id / "control.sock"
    if len(str(nested)) < 100:
        return nested
    safe_session = _DOCKER_NAME_RE.sub("-", session_id).strip("-._") or "session"
    return Path("/tmp") / "agentic-opt-relays" / f"{safe_session}.sock"


def _outbound_proxy_socket_path(*, state_root: Path, assignment_id: str, session_id: str) -> Path:
    nested = state_root / "proxies" / assignment_id / session_id / "outbound.sock"
    if len(str(nested)) < 100:
        return nested
    safe_session = _DOCKER_NAME_RE.sub("-", session_id).strip("-._") or "session"
    return Path("/tmp") / "agentic-opt-proxies" / f"{safe_session}.sock"


def _prepare_docker_worker_host_paths(*, state_root: Path) -> None:
    for path in (state_root / "envs", state_root / "provider_state"):
        path.mkdir(parents=True, exist_ok=True)


def _host_codex_source_home() -> Path | None:
    raw = os.environ.get("CODEX_HOME")
    candidate = Path(raw).expanduser() if raw else Path.home() / ".codex"
    return candidate if candidate.exists() else None


def _wait_for_socket(path: Path, *, timeout_s: float = 5.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise TimeoutError(f"control-plane relay socket did not appear: {path}")


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_tcp(host: str, port: int, *, timeout_s: float = 5.0) -> None:
    deadline = time.time() + timeout_s
    last_error: OSError | None = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.05)
    raise TimeoutError(f"control-plane TCP relay did not accept connections at {host}:{port}: {last_error}")


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
