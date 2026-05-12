from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from agentic_opt.common.atomic import atomic_write_text
from agentic_opt.common.ids import make_run_id

from .policy import PolicyService, estimated_cost
from .process_env import build_subprocess_env, sanitize_env
from .repository import ControlPlaneRepository
from .runpod_provider import RunPodCapacityError, RunPodProvider


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
        reaper_interval_s: float | None = 1.0,
    ) -> None:
        self.repository = repository
        self.job_root = job_root.resolve()
        self.database_path = database_path.resolve()
        self.job_root.mkdir(parents=True, exist_ok=True)
        self._local_processes: dict[str, subprocess.Popen[str]] = {}
        self._job_relay_processes: dict[str, subprocess.Popen[str]] = {}
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
        command = payload.get("command", inputs.get("command"))
        if not image:
            raise ValueError(f"{provider} job requires image or inputs.image")
        if not command:
            raise ValueError(f"{provider} job requires command or inputs.command")
        cwd = Path(payload.get("cwd") or inputs.get("cwd") or os.getcwd()).resolve()
        relay_process: subprocess.Popen[str] | None = None
        relay_socket_path: Path | None = None
        relay_metadata: dict[str, Any] = {}
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
        try:
            docker_command, network_enforcement = build_local_docker_command(
                image=str(image),
                command=command,
                cwd=cwd,
                network_policy=network_policy,
                requested_network_mode=payload.get("network_mode") or inputs.get("network_mode"),
                requires_control_plane=requires_control_plane,
                control_plane_relay_socket=relay_socket_path,
            )
        except DockerNetworkPolicyError as exc:
            if relay_process is not None:
                _terminate_process(relay_process)
            return self.repository.create_job(
                {
                    **payload,
                    "provider": provider,
                    "status": "blocked",
                    "details": {
                        **(payload.get("details") or {}),
                        "provider_adapter": "local-docker",
                        "network_enforcement": exc.enforcement,
                        "control_plane_relay": relay_metadata,
                        "policy_block": {"reason": exc.reason, "message": str(exc)},
                    },
                }
            )
        try:
            record = self.launch_local(
                {
                    **payload,
                    "provider": provider,
                    "inputs": {
                        **inputs,
                        "image": image,
                        "original_command": command,
                        "command": docker_command,
                        "cwd": str(cwd),
                    },
                    "details": {
                        **(payload.get("details") or {}),
                        "provider_adapter": "local-docker",
                        "network_enforcement": network_enforcement,
                        "control_plane_relay": relay_metadata,
                    },
                }
            )
        except Exception:
            if relay_process is not None:
                _terminate_process(relay_process)
            raise
        if relay_process is not None:
            with self._lock:
                self._job_relay_processes[record["job_id"]] = relay_process
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
        command = payload.get("command", inputs.get("command"))
        if not command:
            raise ValueError("local job requires command or inputs.command")
        env_inputs = sanitize_env(payload.get("env") or inputs.get("env") or {})
        record = self.repository.create_job(
            {
                **payload,
                "provider": payload.get("provider") or "local",
                "status": "queued",
                "inputs": {
                    **inputs,
                    "command": command,
                    "cwd": str(Path(payload.get("cwd") or inputs.get("cwd") or os.getcwd()).resolve()),
                    "env": env_inputs,
                },
                "outputs": {},
                "details": {
                    **(payload.get("details") or {}),
                    "launcher": "agentic_opt.control_plane.job_worker",
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
                    if record["status"] not in TERMINAL_JOB_STATUSES:
                        record = self.repository.update_job(
                            job_id,
                            {
                                "status": "completed" if returncode == 0 else "failed",
                                "outputs": {
                                    **(record.get("outputs") or {}),
                                    "launcher_returncode": returncode,
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
        if process is None:
            return
        _terminate_process(process)

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


class DockerNetworkPolicyError(ValueError):
    def __init__(self, reason: str, message: str, *, enforcement: dict[str, Any]) -> None:
        super().__init__(message)
        self.reason = reason
        self.enforcement = enforcement


def build_local_docker_command(
    *,
    image: str,
    command: str | list[Any],
    cwd: Path,
    network_policy: dict[str, Any],
    requested_network_mode: str | None = None,
    requires_control_plane: bool = False,
    control_plane_relay_socket: Path | None = None,
    container_control_plane_socket_path: str = "/ao-control/control.sock",
) -> tuple[list[str], dict[str, Any]]:
    external = str(network_policy.get("external_internet") or "allow")
    control_plane = str(network_policy.get("control_plane") or "allow")
    network_mode = requested_network_mode
    relay_configured = control_plane_relay_socket is not None
    if external == "deny":
        if requested_network_mode and requested_network_mode != "none":
            enforcement = {
                "external_internet": external,
                "control_plane": control_plane,
                "requested_network_mode": requested_network_mode,
                "docker_network_mode": None,
                "external_internet_enforced": False,
                "control_plane_available": False,
            }
            raise DockerNetworkPolicyError(
                "docker_network_mode_violates_external_deny",
                f"external_internet=deny requires Docker network_mode=none, got {requested_network_mode!r}",
                enforcement=enforcement,
            )
        if requires_control_plane and control_plane == "allow" and not relay_configured:
            enforcement = {
                "external_internet": external,
                "control_plane": control_plane,
                "requested_network_mode": requested_network_mode,
                "docker_network_mode": "none",
                "external_internet_enforced": True,
                "control_plane_available": False,
                "control_plane_requires_relay": True,
                "control_plane_relay_configured": False,
            }
            raise DockerNetworkPolicyError(
                "docker_control_plane_relay_required",
                "Docker --network none enforces external_internet=deny but cannot reach the control plane; use a control-plane relay provider",
                enforcement=enforcement,
            )
        network_mode = "none"
    enforcement = {
        "external_internet": external,
        "control_plane": control_plane,
        "requested_network_mode": requested_network_mode,
        "docker_network_mode": network_mode or "default",
        "external_internet_enforced": external == "deny" and network_mode == "none",
        "control_plane_available": control_plane == "allow" and (network_mode != "none" or relay_configured),
        "control_plane_requires_relay": external == "deny" and requires_control_plane and control_plane == "allow",
        "control_plane_relay_configured": relay_configured,
    }
    docker_command: list[str] = [
        "docker",
        "run",
        "--rm",
    ]
    if network_mode:
        docker_command.extend(["--network", network_mode])
    if control_plane_relay_socket is not None:
        docker_command.extend(
            [
                "-v",
                f"{control_plane_relay_socket.resolve()}:{container_control_plane_socket_path}",
                "-e",
                f"AO_CONTROL_API_URL=unix://{container_control_plane_socket_path}",
            ]
        )
    docker_command.extend(
        [
            "-v",
            f"{cwd}:/workspace",
            "-w",
            "/workspace",
            str(image),
        ]
    )
    if isinstance(command, str):
        docker_command.extend(["sh", "-lc", command])
    else:
        docker_command.extend(str(item) for item in command)
    return docker_command, enforcement


def _wait_for_socket(path: Path, *, timeout_s: float = 5.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise TimeoutError(f"control-plane relay socket did not appear: {path}")


def _worker_process_env() -> dict[str, str]:
    env = build_subprocess_env()
    src_root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = str(src_root) if not env.get("PYTHONPATH") else f"{src_root}:{env['PYTHONPATH']}"
    return env


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
