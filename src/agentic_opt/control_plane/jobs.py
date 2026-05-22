from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from agentic_opt.common.atomic import atomic_write_text
from agentic_opt.common.ids import isoformat_z, make_run_id

from .docker_runtime import DockerMount, DockerNetworkPolicyError, build_local_docker_command
from .environment_providers import (
    DockerImageProvider,
    EnvironmentProviderError,
    EnvironmentRunSpec,
    docker_workdir,
)
from .policy import PolicyService, estimated_cost
from .process_env import build_subprocess_env, sanitize_env
from .network_proxy import proxy_url, start_network_proxy_process, wait_for_tcp
from .repository import ControlPlaneRepository
from .runpod_provider import RunPodCapacityError, RunPodProvider
from .task_context import (
    TaskContextMountConflictError,
    append_docker_task_context_mount,
    ensure_task_context_snapshot,
    local_task_context_enforcement,
    verify_task_context_path,
)


TERMINAL_JOB_STATUSES = {"completed", "failed", "cancelled", "blocked"}
DOCKER_JOB_PROVIDERS = {"local-docker", "local-docker-strict", "docker_image"}


class JobService:
    """Server-owned durable job launcher.

    This is the first provider-independent layer for compute jobs. The initial
    provider is local subprocess execution; cloud providers should implement the
    same resource contract instead of adding worker-specific launch code.
    """

    def __init__(
        self,
        *,
        repository: ControlPlaneRepository,
        job_root: Path,
        database_path: Path,
        environments: Any | None = None,
        reaper_interval_s: float | None = 1.0,
    ) -> None:
        self.repository = repository
        self.job_root = job_root.resolve()
        self.database_path = database_path.resolve()
        self.environments = environments
        self.job_root.mkdir(parents=True, exist_ok=True)
        self._local_processes: dict[str, subprocess.Popen[str]] = {}
        self._job_relay_processes: dict[str, subprocess.Popen[str]] = {}
        self._job_proxy_processes: dict[str, subprocess.Popen[str]] = {}
        self._lock = threading.RLock()
        self._reaper_stop = threading.Event()
        self._reaper_thread: threading.Thread | None = None
        self.policy = PolicyService(repository)
        if reaper_interval_s is not None and reaper_interval_s > 0:
            self._reaper_thread = threading.Thread(
                target=self._reaper_loop,
                args=(float(reaper_interval_s),),
                name="agentic-opt-job-reaper",
                daemon=True,
            )
            self._reaper_thread.start()

    def launch(self, payload: dict[str, Any]) -> dict[str, Any]:
        decision = self.policy.decide_job(payload)
        if not decision.allowed:
            return self.repository.create_job(
                {
                    **payload,
                    "provider": payload.get("provider") or "local",
                    "status": "blocked",
                    "cost": decision.estimated_cost or estimated_cost(payload),
                    "details": {**(payload.get("details") or {}), "policy_block": {"reason": decision.reason, **(decision.details or {})}, "policy_decision": decision.to_dict()},
                }
            )
        payload = {
            **payload,
            "cost": decision.estimated_cost or estimated_cost(payload),
            "details": {**(payload.get("details") or {}), "policy_decision": decision.to_dict()},
        }
        provider = payload.get("provider") or "local"
        if provider == "runpod":
            return self.launch_runpod(payload)
        if provider in DOCKER_JOB_PROVIDERS:
            return self.launch_local_docker(payload)
        if provider != "local":
            return self.repository.create_job({**payload, "provider": provider, "status": payload.get("status") or "queued"})
        return self.launch_local(payload)

    def launch_runpod(self, payload: dict[str, Any]) -> dict[str, Any]:
        preview: dict[str, Any] | None = None
        try:
            preview = self.repository.create_job({**payload, "provider": "runpod", "status": "queued"})
            result = RunPodProvider().launch({**payload, "job_id": preview["job_id"]})
            return self.repository.update_job(
                preview["job_id"],
                {
                    "status": "running" if not result.dry_run else "queued",
                    "outputs": {"pod_id": result.pod_id},
                    "details": {
                        **(preview.get("details") or {}),
                        "runpod": {
                            "pod_id": result.pod_id,
                            "payload": result.payload,
                            "response": result.response,
                            "dry_run": result.dry_run,
                        },
                    },
                },
            )
        except RunPodCapacityError as exc:
            if preview is not None:
                return self.repository.update_job(
                    preview["job_id"],
                    {"status": "blocked", "details": {"provider_error": {"type": "capacity", "message": str(exc), "retryable": True}}},
                )
            return self.repository.create_job({**payload, "provider": "runpod", "status": "blocked", "details": {"provider_error": {"type": "capacity", "message": str(exc), "retryable": True}}})
        except Exception as exc:
            if preview is not None:
                return self.repository.update_job(preview["job_id"], {"status": "failed", "details": {"provider_error": {"type": type(exc).__name__, "message": str(exc), "retryable": False}}})
            raise

    def launch_local_docker(self, payload: dict[str, Any]) -> dict[str, Any]:
        inputs = dict(payload.get("inputs") or {})
        provider = payload.get("provider") or "local-docker"
        image = payload.get("image") or inputs.get("image")
        environment_id = payload.get("environment_id") or inputs.get("environment_id")
        environment_overlay_id = payload.get("environment_overlay_id") or inputs.get("environment_overlay_id")
        environment = self.repository.get_environment(str(environment_id)) if environment_id else None
        overlay = self.repository.get_environment_overlay(str(environment_overlay_id)) if environment_overlay_id else None
        if environment is None and overlay is not None:
            environment = self.repository.get_environment(str(overlay["base_environment_id"]))
        docker_execution = _docker_execution_from_records(environment=environment, overlay=overlay) if provider == "docker_image" and environment is not None else None
        provider_adapter = "docker_image" if docker_execution is not None else "local-docker"
        if not image and environment is not None and docker_execution is None:
            image = ((environment.get("metadata") or {}).get("image_ref") or (environment.get("lock") or {}).get("image_ref"))
        command = payload.get("command", inputs.get("command"))
        if not image and docker_execution is None:
            raise ValueError(f"{provider} job requires image or inputs.image")
        if not command:
            raise ValueError(f"{provider} job requires command or inputs.command")
        cwd = Path(payload.get("cwd") or inputs.get("cwd") or os.getcwd()).resolve()
        docker_env = sanitize_env(payload.get("env") or inputs.get("env") or {})
        launcher_env = sanitize_env(payload.get("launcher_env") or inputs.get("launcher_env") or {})
        mounts = _docker_mounts_from_payload(payload=payload, inputs=inputs, default_cwd=cwd)
        workdir = str(payload.get("workdir") or inputs.get("workdir") or "/workspace")
        task_context = _ensure_task_context_for_payload(
            repository=self.repository,
            payload=payload,
            inputs=inputs,
            state_root=self.database_path.parent,
        )
        task_context_enforcement: dict[str, Any] = {}
        if task_context is not None:
            task_context_target_root = Path(workdir if workdir.startswith("/") else "/workspace")
            try:
                mounts = append_docker_task_context_mount(
                    mounts=mounts,
                    snapshot=task_context,
                    workspace_root=task_context_target_root,
                )
            except TaskContextMountConflictError as exc:
                return self.repository.create_job(
                    {
                        **payload,
                        "provider": provider,
                        "status": "blocked",
                        "details": {
                            **(payload.get("details") or {}),
                            "provider_adapter": provider_adapter,
                            "task_context": task_context,
                            "policy_block": {"reason": type(exc).__name__, "message": str(exc)},
                        },
                    }
                )
            task_context_enforcement = {
                "provider": provider_adapter,
                "workspace_path": "task",
                "digest": task_context.get("digest"),
                "snapshot_task_path": task_context.get("task_path"),
                "container_task_path": str(task_context_target_root / "task"),
                "mechanism": "docker_readonly_bind_mount",
                "provider_enforced_readonly": True,
                "policy_weakened": False,
            }
        relay_process: subprocess.Popen[str] | None = None
        relay_socket_path: Path | None = None
        relay_metadata: dict[str, Any] = {}
        proxy_process: subprocess.Popen[str] | None = None
        proxy_metadata: dict[str, Any] = {}
        requires_control_plane = bool(payload.get("requires_control_plane") or inputs.get("requires_control_plane"))
        control_plane_url = self._resolve_control_plane_target_url(payload=payload, inputs=inputs)
        network_policy = self._network_policy_for_payload(payload)
        if requires_control_plane and network_policy.get("external_internet") == "deny" and control_plane_url:
            from .relay import relay_url, start_relay_process

            relay_socket_path = self.job_root / "relays" / f"{make_run_id('relay')}.sock"
            relay_process = start_relay_process(
                socket_path=relay_socket_path,
                target_url=str(control_plane_url),
                env=_worker_process_env(),
            )
            try:
                _wait_for_socket(relay_socket_path)
            except Exception:
                _terminate_process(relay_process)
                raise
            relay_metadata = {
                "relay_socket_path": str(relay_socket_path),
                "relay_url": relay_url(relay_socket_path),
                "relay_pid": relay_process.pid,
                "target_url": str(control_plane_url),
            }
        if _needs_outbound_audit_proxy(network_policy):
            proxy_host = "127.0.0.1"
            proxy_port = _free_tcp_port()
            proxy_process = start_network_proxy_process(
                host=proxy_host,
                port=proxy_port,
                database_path=self.database_path,
                policy=network_policy,
                metadata={
                    "experiment_id": payload.get("experiment_id"),
                    "assignment_id": payload.get("assignment_id"),
                    "session_id": payload.get("session_id"),
                    "task_id": payload.get("task_id"),
                    "agent_id": payload.get("agent_id"),
                    "job_provider": provider,
                },
                env=_worker_process_env(),
            )
            try:
                wait_for_tcp(proxy_host, proxy_port)
            except Exception:
                _terminate_process(proxy_process)
                raise
            host_proxy_url = proxy_url(proxy_host, proxy_port)
            container_proxy_url = proxy_url(_docker_container_proxy_host(payload=payload, inputs=inputs), proxy_port)
            network_policy = {
                **network_policy,
                "outbound_proxy_url": container_proxy_url,
                "outbound_proxy_no_proxy": "127.0.0.1,localhost",
            }
            proxy_metadata = {
                "outbound_audit_proxy": {
                    "proxy_url": host_proxy_url,
                    "container_proxy_url": container_proxy_url,
                    "proxy_pid": proxy_process.pid,
                }
            }
        try:
            if docker_execution is not None:
                plan = DockerImageProvider().build_run_plan(
                    EnvironmentRunSpec(
                        execution=docker_execution,
                        command=command,
                        cwd=cwd,
                        env=docker_env,
                        mounts=mounts,
                        workdir=workdir if workdir != "/workspace" else docker_workdir(docker_execution),
                        network_policy=network_policy,
                        requested_network_mode=payload.get("network_mode") or inputs.get("network_mode"),
                        requires_control_plane=requires_control_plane,
                        control_plane_relay_socket=relay_socket_path,
                        container_name=payload.get("container_name") or inputs.get("container_name"),
                        pids_limit=payload.get("pids_limit") or inputs.get("pids_limit"),
                        memory=payload.get("memory") or inputs.get("memory"),
                        cpus=payload.get("cpus") or inputs.get("cpus"),
                        add_hosts=_docker_proxy_add_hosts(network_policy.get("outbound_proxy_url")),
                        require_immutable_image=True,
                    )
                )
                docker_command = plan.command
                network_enforcement = plan.network_enforcement
                runner_metadata = plan.metadata
                image = ((runner_metadata.get("image") or {}).get("reference") or image)
                workdir = str(runner_metadata.get("workdir") or workdir)
            else:
                docker_command, network_enforcement = build_local_docker_command(
                    image=str(image),
                    command=command,
                    cwd=cwd,
                    network_policy=network_policy,
                    requested_network_mode=payload.get("network_mode") or inputs.get("network_mode"),
                    requires_control_plane=requires_control_plane,
                    control_plane_relay_socket=relay_socket_path,
                    mounts=mounts,
                    workdir=workdir,
                    env=docker_env,
                    container_name=payload.get("container_name") or inputs.get("container_name"),
                    pids_limit=payload.get("pids_limit") or inputs.get("pids_limit"),
                    memory=payload.get("memory") or inputs.get("memory"),
                    cpus=payload.get("cpus") or inputs.get("cpus"),
                    add_hosts=_docker_proxy_add_hosts(network_policy.get("outbound_proxy_url")),
                )
                runner_metadata = {
                    "provider": provider,
                    "image": {"reference": str(image), "kind": "mutable_ref", "immutable": False},
                    "workdir": workdir,
                    "mounts": [_docker_mount_to_json(mount) for mount in mounts],
                    "network_enforcement": network_enforcement,
                }
        except (DockerNetworkPolicyError, EnvironmentProviderError, TaskContextMountConflictError) as exc:
            if relay_process is not None:
                _terminate_process(relay_process)
            if proxy_process is not None:
                _terminate_process(proxy_process)
            reason = exc.reason if isinstance(exc, DockerNetworkPolicyError) else type(exc).__name__
            enforcement = exc.enforcement if isinstance(exc, DockerNetworkPolicyError) else {}
            return self.repository.create_job(
                {
                    **payload,
                    "provider": provider,
                    "status": "blocked",
                    "details": {
                        **(payload.get("details") or {}),
                        "provider_adapter": provider_adapter,
                        "environment": _docker_environment_job_metadata(environment),
                        "network_enforcement": enforcement,
                        "task_context": task_context,
                        "task_context_enforcement": task_context_enforcement,
                        "control_plane_relay": relay_metadata,
                        **proxy_metadata,
                        "policy_block": {"reason": reason, "message": str(exc)},
                    },
                }
            )
        try:
            record = self.launch_local(
                {
                    **payload,
                    "provider": provider,
                    "command": docker_command,
                    "cwd": str(cwd),
                    "env": launcher_env,
                    "inputs": {
                        **inputs,
                        "image": image,
                        "environment_id": environment_id,
                        "environment_overlay_id": environment_overlay_id,
                        "original_command": command,
                        "command": docker_command,
                        "cwd": str(cwd),
                        "env": launcher_env,
                        "docker_env": docker_env,
                        "mounts": [_docker_mount_to_json(mount) for mount in mounts],
                        "workdir": workdir,
                        "task_context": task_context,
                        "task_context_enforcement": task_context_enforcement,
                    },
                    "details": {
                        **(payload.get("details") or {}),
                        "provider_adapter": provider_adapter,
                        "environment": _docker_environment_job_metadata(environment),
                        "runner": runner_metadata,
                        "network_enforcement": network_enforcement,
                        "task_context": task_context,
                        "task_context_enforcement": task_context_enforcement,
                        "control_plane_relay": relay_metadata,
                        **proxy_metadata,
                    },
                }
            )
        except Exception:
            if relay_process is not None:
                _terminate_process(relay_process)
            if proxy_process is not None:
                _terminate_process(proxy_process)
            raise
        if relay_process is not None:
            with self._lock:
                self._job_relay_processes[record["job_id"]] = relay_process
        if proxy_process is not None:
            with self._lock:
                self._job_proxy_processes[record["job_id"]] = proxy_process
        if network_enforcement.get("external_internet") == "deny" and network_enforcement.get("external_internet_enforced"):
            experiment = self.repository.get_experiment(payload.get("experiment_id")) if payload.get("experiment_id") else None
            self.repository.record_network_access_event(
                {
                    "experiment_id": payload.get("experiment_id"),
                    "assignment_id": payload.get("assignment_id"),
                    "session_id": payload.get("session_id"),
                    "task_id": payload.get("task_id") or (experiment or {}).get("task_id"),
                    "agent_id": payload.get("agent_id"),
                    "destination": "external_internet",
                    "access_type": "docker-egress",
                    "decision": "denied",
                    "reason": "Docker job launched with --network none",
                    "metadata": {"job_id": record["job_id"], "provider": provider, "network_enforcement": network_enforcement},
                }
            )
        return record

    def launch_local(self, payload: dict[str, Any]) -> dict[str, Any]:
        inputs = dict(payload.get("inputs") or {})
        provider = payload.get("provider") or "local"
        command = payload.get("command", inputs.get("command"))
        if not command:
            raise ValueError("local job requires command or inputs.command")
        env_inputs = sanitize_env(payload.get("env") or inputs.get("env") or {})
        environment_metadata: dict[str, Any] = {}
        task_context = _ensure_task_context_for_payload(
            repository=self.repository,
            payload=payload,
            inputs=inputs,
            state_root=self.database_path.parent,
        )
        workspace_root = _workspace_root_for_payload(repository=self.repository, payload=payload, inputs=inputs)
        task_context_precheck: dict[str, Any] | None = None
        task_context_enforcement: dict[str, Any] = {}
        if provider == "local" and task_context is not None:
            if workspace_root is not None and (workspace_root / "task").exists():
                task_context_precheck = verify_task_context_path(
                    task_path=workspace_root / "task",
                    expected_digest=str(task_context.get("digest") or ""),
                )
            else:
                task_context_precheck = {"ok": True, "status": "not_materialized_in_job_workspace"}
            task_context_enforcement = local_task_context_enforcement(
                snapshot=task_context,
                workspace_root=workspace_root,
                verification=task_context_precheck,
            )
        if provider == "local" and self._should_enforce_local_environment(payload=payload, inputs=inputs):
            execution = self._local_job_execution_environment(payload=payload, inputs=inputs)
            if execution is not None:
                env_inputs = _merge_execution_env(execution=execution, env_inputs=env_inputs)
                inputs["task_id"] = execution["task_id"]
                inputs["environment_id"] = execution.get("environment_id")
                if execution.get("environment_overlay_id"):
                    inputs["environment_overlay_id"] = execution["environment_overlay_id"]
                inputs["environment_enforced"] = True
                environment_metadata = {
                    "enforced": True,
                    "kind": execution.get("kind"),
                    "provider": execution.get("provider"),
                    "task_id": execution.get("task_id"),
                    "environment_id": execution.get("environment_id"),
                    "environment_overlay_id": execution.get("environment_overlay_id"),
                    "python_path": execution.get("python_path"),
                    "root_path": execution.get("root_path"),
                }
        record = self.repository.create_job(
            {
                **payload,
                "provider": provider,
                "status": "queued",
                "inputs": {
                    **inputs,
                    "command": command,
                    "cwd": str(Path(payload.get("cwd") or inputs.get("cwd") or os.getcwd()).resolve()),
                    "env": env_inputs,
                    **({"task_context": task_context, "task_context_enforcement": task_context_enforcement} if task_context is not None else {}),
                },
                "outputs": {},
                "details": {
                    **(payload.get("details") or {}),
                    "launcher": "agentic_opt.control_plane.job_worker",
                    **({"environment": environment_metadata} if environment_metadata else {}),
                    **({"task_context": task_context, "task_context_enforcement": task_context_enforcement} if task_context is not None else {}),
                },
            }
        )
        job_dir = self.job_root / record["job_id"]
        job_dir.mkdir(parents=True, exist_ok=True)
        command_path = job_dir / "command.json"
        stdout_path = job_dir / "stdout.log"
        stderr_path = job_dir / "stderr.log"
        atomic_write_text(
            command_path,
            json.dumps({"job": record, "inputs": record["inputs"]}, indent=2, sort_keys=True) + "\n",
        )
        record = self.repository.update_job(
            record["job_id"],
            {
                "outputs": {
                    "job_dir": str(job_dir),
                    "command_path": str(command_path),
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                },
            },
        )
        src_root = Path(__file__).resolve().parents[2]
        env = build_subprocess_env({"PYTHONPATH": str(src_root)})
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "agentic_opt.control_plane.job_worker",
                "--db",
                str(self.database_path),
                "--job-id",
                record["job_id"],
            ],
            cwd=str(src_root.parent),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        with self._lock:
            self._local_processes[record["job_id"]] = process
        return self.repository.update_job(
            record["job_id"],
            {"status": "running", "pid": process.pid, "details": {**(record.get("details") or {}), "launcher_pid": process.pid}},
        )

    def _should_enforce_local_environment(self, *, payload: dict[str, Any], inputs: dict[str, Any]) -> bool:
        if self.environments is None:
            return False
        details = payload.get("details") if isinstance(payload.get("details"), dict) else {}
        if details.get("kind") == "evaluation":
            return False
        if payload.get("environment_enforced") is False or inputs.get("environment_enforced") is False:
            return False
        return self._local_job_task_id(payload=payload, inputs=inputs) is not None

    def _local_job_execution_environment(self, *, payload: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any] | None:
        task_id = self._local_job_task_id(payload=payload, inputs=inputs)
        if not task_id:
            return None
        execution = self.environments.get_execution_environment(
            task_id=task_id,
            experiment_id=payload.get("experiment_id") or inputs.get("experiment_id"),
            environment_id=payload.get("environment_id") or inputs.get("environment_id"),
            overlay_id=payload.get("environment_overlay_id") or inputs.get("environment_overlay_id") or payload.get("overlay_id") or inputs.get("overlay_id"),
            allow_overlay=True,
        )
        return {**execution, "task_id": task_id}

    def _local_job_task_id(self, *, payload: dict[str, Any], inputs: dict[str, Any]) -> str | None:
        task_id = payload.get("task_id") or inputs.get("task_id")
        if task_id:
            return str(task_id)
        assignment_id = payload.get("assignment_id") or inputs.get("assignment_id")
        assignment = self.repository.get_assignment(str(assignment_id)) if assignment_id else None
        if assignment and assignment.get("task_id"):
            return str(assignment["task_id"])
        experiment_id = payload.get("experiment_id") or inputs.get("experiment_id")
        experiment = self.repository.get_experiment(str(experiment_id)) if experiment_id else None
        if experiment and experiment.get("task_id"):
            return str(experiment["task_id"])
        return None

    def get(self, job_id: str) -> dict[str, Any]:
        self.reap_finished_processes()
        record = self.repository.get_job(job_id)
        if record is None:
            raise KeyError(job_id)
        if record["provider"] == "runpod":
            record = self._refresh_runpod_status(record)
        if record["status"] in TERMINAL_JOB_STATUSES:
            self._cleanup_job_relay(job_id)
        return record

    def attach(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Record that the current worker session/attempt is using a durable job.

        Attach is intentionally an observation marker. It does not restart the
        job, take exclusive ownership, or rewrite the job's original
        session_id/attempt_id.
        """

        record = self.get(job_id)
        experiment_id = str(payload.get("experiment_id") or record["experiment_id"])
        if experiment_id != record["experiment_id"]:
            raise ValueError("job belongs to a different experiment")

        assignment_id = payload.get("assignment_id")
        session_id = payload.get("session_id")
        attempt_id = payload.get("attempt_id")
        agent_id = payload.get("agent_id")
        mode = str(payload.get("mode") or "observe")
        if mode not in {"observe", "continue"}:
            raise ValueError("job attach mode must be 'observe' or 'continue'")

        assignment = self.repository.get_assignment(str(assignment_id)) if assignment_id else None
        if assignment_id and assignment is None:
            raise KeyError(str(assignment_id))
        if assignment is not None and assignment["experiment_id"] != experiment_id:
            raise ValueError("assignment belongs to a different experiment")
        if assignment is not None and record.get("assignment_id") and assignment["assignment_id"] != record["assignment_id"]:
            raise ValueError("job belongs to a different assignment")

        session = self.repository.get_session(str(session_id)) if session_id else None
        if session_id and session is None:
            raise KeyError(str(session_id))
        if session is not None:
            if session["experiment_id"] != experiment_id:
                raise ValueError("session belongs to a different experiment")
            if record.get("assignment_id") and session["assignment_id"] != record["assignment_id"]:
                raise ValueError("session belongs to a different assignment")
            if assignment is not None and session["assignment_id"] != assignment["assignment_id"]:
                raise ValueError("session belongs to a different assignment")
            assignment_id = assignment_id or session["assignment_id"]
            agent_id = agent_id or session.get("agent_id")

        attempt = self.repository.get_attempt(str(attempt_id)) if attempt_id else None
        if attempt_id and attempt is None:
            raise KeyError(str(attempt_id))
        if attempt is not None:
            if attempt["experiment_id"] != experiment_id:
                raise ValueError("attempt belongs to a different experiment")
            if record.get("assignment_id") and attempt.get("assignment_id") and attempt["assignment_id"] != record["assignment_id"]:
                raise ValueError("attempt belongs to a different assignment")
            if assignment is not None and attempt.get("assignment_id") and attempt["assignment_id"] != assignment["assignment_id"]:
                raise ValueError("attempt belongs to a different assignment")
            assignment_id = assignment_id or attempt.get("assignment_id")
            session_id = session_id or attempt.get("session_id")
            agent_id = agent_id or attempt.get("agent_id")

        now = isoformat_z()
        attachment = {
            "attachment_id": make_run_id("job_attach"),
            "job_id": job_id,
            "mode": mode,
            "attached_at": now,
            "experiment_id": experiment_id,
            "assignment_id": assignment_id,
            "session_id": session_id,
            "attempt_id": attempt_id,
            "agent_id": agent_id,
            "note": payload.get("note"),
        }
        details = dict(record.get("details") or {})
        attachments = list(details.get("attachments") or [])
        attachments.append({key: value for key, value in attachment.items() if value is not None})
        details["attachments"] = attachments
        details["last_attachment"] = attachments[-1]
        updated = self.repository.update_job(job_id, {"details": details})
        self.repository.record_event(
            {
                "experiment_id": experiment_id,
                "assignment_id": assignment_id,
                "session_id": session_id,
                "task_id": (assignment or session or attempt or {}).get("task_id"),
                "agent_id": agent_id,
                "event_type": "job.attached",
                "summary": f"job attached: {job_id}",
                "payload": {"job_id": job_id, "attachment": attachments[-1]},
            }
        )
        return {"job": updated, "attachment": attachments[-1]}

    def cancel(self, job_id: str) -> dict[str, Any]:
        record = self.get(job_id)
        if record["status"] in TERMINAL_JOB_STATUSES:
            return record
        pid = record.get("pid")
        if pid:
            try:
                os.kill(int(pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        if record["provider"] == "runpod":
            pod_id = ((record.get("details") or {}).get("runpod") or {}).get("pod_id") or (record.get("outputs") or {}).get("pod_id")
            if pod_id:
                try:
                    RunPodProvider().stop(str(pod_id))
                except Exception as exc:
                    return self.repository.update_job(job_id, {"status": "cancelled", "details": {"cancel_error": str(exc)}})
        self._cleanup_job_relay(job_id)
        return self.repository.update_job(job_id, {"status": "cancelled"})

    def read_logs(self, job_id: str, *, max_bytes: int = 200_000) -> dict[str, Any]:
        record = self.get(job_id)
        outputs = record.get("outputs") or {}
        return {
            "job_id": job_id,
            "status": record["status"],
            "stdout": _tail_text(outputs.get("stdout_path"), max_bytes=max_bytes),
            "stderr": _tail_text(outputs.get("stderr_path"), max_bytes=max_bytes),
            "outputs": outputs,
        }

    def reap_finished_processes(self) -> list[dict[str, Any]]:
        with self._lock:
            reaped: list[dict[str, Any]] = []
            for job_id, process in list(self._local_processes.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                try:
                    returncode = process.wait(timeout=0)
                except subprocess.TimeoutExpired:
                    continue
                self._local_processes.pop(job_id, None)
                record = self.repository.get_job(job_id)
                if record is not None:
                    task_context_check = _verify_job_task_context(repository=self.repository, record=record)
                    if record["status"] not in TERMINAL_JOB_STATUSES:
                        final_status = "completed" if returncode == 0 else "failed"
                        if task_context_check is not None and not task_context_check.get("ok", False):
                            final_status = "failed"
                        record = self.repository.update_job(
                            job_id,
                            {
                                "status": final_status,
                                "outputs": {
                                    **(record.get("outputs") or {}),
                                    "launcher_returncode": returncode,
                                },
                                "details": {
                                    **(record.get("details") or {}),
                                    **({"task_context_postcheck": task_context_check} if task_context_check is not None else {}),
                                },
                            },
                        )
                    elif task_context_check is not None:
                        final_status = record["status"]
                        if record["status"] == "completed" and not task_context_check.get("ok", False):
                            final_status = "failed"
                        record = self.repository.update_job(
                            job_id,
                            {
                                "status": final_status,
                                "details": {
                                    **(record.get("details") or {}),
                                    "task_context_postcheck": task_context_check,
                                },
                            },
                        )
                    if record["status"] in TERMINAL_JOB_STATUSES:
                        self._cleanup_job_relay(job_id)
                reaped.append({"job_id": job_id, "pid": process.pid, "returncode": returncode})
            return reaped

    def close(self) -> None:
        self._reaper_stop.set()
        if self._reaper_thread is not None:
            self._reaper_thread.join(timeout=2.0)
        deadline = time.time() + 2.0
        while time.time() < deadline:
            self.reap_finished_processes()
            with self._lock:
                if not self._local_processes:
                    break
            time.sleep(0.05)
        with self._lock:
            remaining_processes = list(self._local_processes.items())
            self._local_processes.clear()
            relay_job_ids = set(self._job_relay_processes) | set(self._job_proxy_processes)
        for job_id, process in remaining_processes:
            _terminate_process(process)
            record = self.repository.get_job(job_id)
            if record is not None and record.get("status") not in TERMINAL_JOB_STATUSES:
                self.repository.update_job(job_id, {"status": "cancelled"})
            self._cleanup_job_relay(job_id)
        for job_id in relay_job_ids:
            self._cleanup_job_relay(job_id)

    def _reaper_loop(self, interval_s: float) -> None:
        while not self._reaper_stop.wait(interval_s):
            try:
                self.reap_finished_processes()
            except Exception:
                continue

    def _refresh_runpod_status(self, record: dict[str, Any]) -> dict[str, Any]:
        runpod = (record.get("details") or {}).get("runpod") or {}
        pod_id = runpod.get("pod_id") or (record.get("outputs") or {}).get("pod_id")
        if not pod_id or runpod.get("dry_run"):
            return record
        try:
            status = RunPodProvider().status(str(pod_id))
        except Exception as exc:
            return self.repository.update_job(record["job_id"], {"details": {"runpod_status_error": str(exc)}})
        normalized = str(status.get("status") or status.get("desiredStatus") or status.get("runtimeStatus") or "").lower()
        terminal = {
            "exited": "completed",
            "terminated": "cancelled",
            "stopped": "cancelled",
            "failed": "failed",
        }
        updates: dict[str, Any] = {"details": {"runpod_status": status}}
        if normalized in terminal:
            updates["status"] = terminal[normalized]
        return self.repository.update_job(record["job_id"], updates)

    def _cleanup_job_relay(self, job_id: str) -> None:
        with self._lock:
            process = self._job_relay_processes.pop(job_id, None)
            proxy_process = self._job_proxy_processes.pop(job_id, None)
        if process is None:
            _terminate_process(proxy_process)
            return
        _terminate_process(process)
        _terminate_process(proxy_process)

    def _network_policy_for_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        experiment = self.repository.get_experiment(payload.get("experiment_id")) if payload.get("experiment_id") else None
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

    def _resolve_control_plane_target_url(self, *, payload: dict[str, Any], inputs: dict[str, Any]) -> str | None:
        raw_url = payload.get("control_plane_url") or inputs.get("control_plane_url")
        if raw_url and not str(raw_url).startswith("unix://"):
            return str(raw_url)
        session_id = payload.get("session_id") or inputs.get("session_id")
        session = self.repository.get_session(session_id) if session_id else None
        relay = ((session or {}).get("details") or {}).get("control_plane_relay") or {}
        target_url = relay.get("target_url")
        if target_url:
            return str(target_url)
        return str(raw_url) if raw_url else None


def _wait_for_socket(path: Path, *, timeout_s: float = 5.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise TimeoutError(f"control-plane relay socket did not appear: {path}")


def _ensure_task_context_for_payload(
    *,
    repository: ControlPlaneRepository,
    payload: dict[str, Any],
    inputs: dict[str, Any],
    state_root: Path,
) -> dict[str, Any] | None:
    task_id = _task_id_for_payload(repository=repository, payload=payload, inputs=inputs)
    if not task_id:
        return None
    return ensure_task_context_snapshot(task_id=task_id, state_root=state_root)


def _task_id_for_payload(
    *,
    repository: ControlPlaneRepository,
    payload: dict[str, Any],
    inputs: dict[str, Any],
) -> str | None:
    if payload.get("task_id") or inputs.get("task_id"):
        return str(payload.get("task_id") or inputs.get("task_id"))
    assignment_id = payload.get("assignment_id") or inputs.get("assignment_id")
    if assignment_id:
        assignment = repository.get_assignment(str(assignment_id))
        if assignment is not None:
            return str(assignment.get("task_id") or "")
    session_id = payload.get("session_id") or inputs.get("session_id")
    if session_id:
        session = repository.get_session(str(session_id))
        if session is not None:
            return str(session.get("task_id") or "")
    experiment_id = payload.get("experiment_id") or inputs.get("experiment_id")
    if experiment_id:
        experiment = repository.get_experiment(str(experiment_id))
        if experiment is not None:
            return str(experiment.get("task_id") or "")
    return None


def _workspace_root_for_payload(
    *,
    repository: ControlPlaneRepository,
    payload: dict[str, Any],
    inputs: dict[str, Any],
) -> Path | None:
    raw = payload.get("workspace_root") or inputs.get("workspace_root")
    if raw:
        return Path(str(raw)).resolve()
    session_id = payload.get("session_id") or inputs.get("session_id")
    if session_id:
        session = repository.get_session(str(session_id))
        workspace_path = (session or {}).get("workspace_path")
        if workspace_path:
            return Path(str(workspace_path)).resolve()
    return None


def _verify_job_task_context(*, repository: ControlPlaneRepository, record: dict[str, Any]) -> dict[str, Any] | None:
    details = record.get("details") or {}
    inputs = record.get("inputs") or {}
    task_context = details.get("task_context") or inputs.get("task_context")
    if not isinstance(task_context, dict) or not task_context.get("digest"):
        return None
    expected = str(task_context.get("digest") or "")
    snapshot_path = Path(str(task_context.get("task_path")))
    snapshot_check = verify_task_context_path(task_path=snapshot_path, expected_digest=expected)
    workspace_root = _workspace_root_for_payload(repository=repository, payload=record, inputs=inputs)
    workspace_check = None
    if workspace_root is not None and (workspace_root / "task").exists():
        workspace_check = verify_task_context_path(task_path=workspace_root / "task", expected_digest=expected)
    ok = bool(snapshot_check.get("ok")) and (workspace_check is None or bool(workspace_check.get("ok")))
    return {"ok": ok, "snapshot": snapshot_check, "workspace": workspace_check}


def _merge_execution_env(*, execution: dict[str, Any], env_inputs: dict[str, str]) -> dict[str, str]:
    exports = {str(key): str(value) for key, value in (execution.get("exports") or {}).items() if key and value is not None}
    env = {**exports, **env_inputs}
    python_path = execution.get("python_path")
    if python_path:
        python_path_obj = Path(str(python_path))
        runtime_bin = str(python_path_obj.parent)
        env["PATH"] = _prepend_path(runtime_bin, env.get("PATH") or os.environ.get("PATH") or "")
        python_parent = python_path_obj.parent
        if python_parent.name == "bin":
            env.setdefault("VIRTUAL_ENV", str(python_parent.parent))
        env["AO_TASK_RUNTIME_PYTHON"] = str(python_path_obj)
    root_path = execution.get("root_path")
    if root_path:
        env["AO_TASK_RUNTIME_ROOT"] = str(root_path)
    if execution.get("environment_id"):
        env["AO_ENVIRONMENT_ID"] = str(execution["environment_id"])
    if execution.get("environment_overlay_id"):
        env["AO_ENVIRONMENT_OVERLAY_ID"] = str(execution["environment_overlay_id"])
    return sanitize_env(env)


def _prepend_path(prefix: str, existing: str) -> str:
    parts = [part for part in existing.split(os.pathsep) if part]
    if prefix in parts:
        parts.remove(prefix)
    return os.pathsep.join([prefix, *parts])


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _needs_outbound_audit_proxy(network_policy: dict[str, Any]) -> bool:
    if network_policy.get("outbound_proxy") is False:
        return False
    return str(network_policy.get("external_internet")) == "audit" and bool(network_policy.get("audit_external_attempts", True))


def _docker_container_proxy_host(*, payload: dict[str, Any], inputs: dict[str, Any]) -> str:
    docker = payload.get("docker") if isinstance(payload.get("docker"), dict) else {}
    input_docker = inputs.get("docker") if isinstance(inputs.get("docker"), dict) else {}
    return str(
        payload.get("container_proxy_host")
        or docker.get("container_proxy_host")
        or inputs.get("container_proxy_host")
        or input_docker.get("container_proxy_host")
        or "host.docker.internal"
    )


def _docker_proxy_add_hosts(proxy_url_value: Any) -> list[str]:
    if proxy_url_value and "host.docker.internal" in str(proxy_url_value):
        return ["host.docker.internal:host-gateway"]
    return []


def _worker_process_env() -> dict[str, str]:
    env = build_subprocess_env()
    src_root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = str(src_root) if not env.get("PYTHONPATH") else f"{src_root}:{env['PYTHONPATH']}"
    return env


def _docker_mounts_from_payload(*, payload: dict[str, Any], inputs: dict[str, Any], default_cwd: Path) -> list[DockerMount]:
    raw_mounts = payload.get("mounts") or inputs.get("mounts")
    if not raw_mounts:
        return [DockerMount(source=default_cwd.resolve(), target="/workspace")]
    mounts: list[DockerMount] = []
    for raw in raw_mounts:
        if isinstance(raw, DockerMount):
            mounts.append(raw)
            continue
        if not isinstance(raw, dict):
            raise ValueError("docker mounts must be objects with source and target")
        source = raw.get("source")
        target = raw.get("target")
        if not source or not target:
            raise ValueError("docker mount requires source and target")
        mounts.append(DockerMount(source=Path(str(source)).expanduser().resolve(), target=str(target), read_only=bool(raw.get("read_only"))))
    return mounts


def _docker_mount_to_json(mount: DockerMount) -> dict[str, Any]:
    return {"source": str(mount.source), "target": mount.target, "read_only": mount.read_only}


def _docker_environment_job_metadata(environment: dict[str, Any] | None) -> dict[str, Any]:
    if environment is None:
        return {}
    metadata = environment.get("metadata") or {}
    lock = environment.get("lock") or {}
    return {
        "environment_id": environment.get("environment_id"),
        "provider": metadata.get("provider") or (environment.get("spec") or {}).get("provider"),
        "image_ref": metadata.get("image_ref") or lock.get("image_ref"),
        "image_digest": metadata.get("image_digest") or lock.get("image_digest"),
        "fingerprint": environment.get("fingerprint"),
    }


def _docker_execution_from_records(
    *,
    environment: dict[str, Any] | None,
    overlay: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if environment is None:
        return None
    if overlay is not None:
        metadata = {**(environment.get("metadata") or {}), **(overlay.get("metadata") or {})}
        return {
            "kind": "overlay",
            "environment_id": environment["environment_id"],
            "environment_overlay_id": overlay["overlay_id"],
            "provider": "docker_image",
            "python_path": str(metadata.get("container_python") or environment.get("python_path") or "/usr/local/bin/python"),
            "root_path": str(metadata.get("container_root") or environment.get("root_path") or "/opt/agentic-opt"),
            "exports": _docker_environment_exports(environment=environment, metadata=metadata),
            "record": overlay,
            "base_record": environment,
        }
    metadata = environment.get("metadata") or {}
    return {
        "kind": "environment",
        "environment_id": environment["environment_id"],
        "environment_overlay_id": None,
        "provider": "docker_image",
        "python_path": str(metadata.get("container_python") or environment.get("python_path") or "/usr/local/bin/python"),
        "root_path": str(metadata.get("container_root") or environment.get("root_path") or "/opt/agentic-opt"),
        "exports": _docker_environment_exports(environment=environment, metadata=metadata),
        "record": environment,
    }


def _docker_environment_exports(*, environment: dict[str, Any], metadata: dict[str, Any]) -> dict[str, str]:
    container_root = str(metadata.get("container_root") or "/opt/agentic-opt")
    container_python = str(metadata.get("container_python") or environment.get("python_path") or "/usr/local/bin/python")
    container_src = str(metadata.get("container_src_path") or f"{container_root}/src")
    container_tasks = str(metadata.get("container_tasks_root") or f"{container_root}/tasks")
    return {
        **_string_env(metadata.get("default_env")),
        "AO_ENVIRONMENT_ID": str(environment.get("environment_id") or ""),
        "AO_ENVIRONMENT_PROVIDER": "docker_image",
        "AO_ENVIRONMENT_ROOT": container_root,
        "AO_ENVIRONMENT_PYTHON": container_python,
        "AO_TASK_RUNTIME_ROOT": container_root,
        "AO_TASK_RUNTIME_PYTHON": container_python,
        "PYTHONPATH": container_src,
        "AO_TASKS_ROOTS": container_tasks,
    }


def _string_env(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items() if key and value is not None}


def _terminate_process(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return
    if process.poll() is not None:
        try:
            process.wait(timeout=0)
        except subprocess.TimeoutExpired:
            pass
        return
    terminate = getattr(process, "terminate", None)
    if terminate is None:
        return
    terminate()
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        kill = getattr(process, "kill", None)
        if kill is None:
            return
        kill()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass


def _tail_text(path_value: str | None, *, max_bytes: int) -> str:
    if not path_value:
        return ""
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        return ""
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > max_bytes:
            handle.seek(size - max_bytes)
        return handle.read().decode("utf-8", errors="replace")
