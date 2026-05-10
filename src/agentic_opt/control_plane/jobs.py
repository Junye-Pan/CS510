from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from agentic_opt.common.atomic import atomic_write_text

from .policy import PolicyService, estimated_cost
from .repository import ControlPlaneRepository
from .runpod_provider import RunPodCapacityError, RunPodProvider


TERMINAL_JOB_STATUSES = {"completed", "failed", "cancelled", "blocked"}


class JobService:
    """Server-owned durable job launcher.

    This is the first provider-independent layer for compute jobs. The initial
    provider is local subprocess execution; cloud providers should implement the
    same resource contract instead of adding worker-specific launch code.
    """

    def __init__(self, *, repository: ControlPlaneRepository, job_root: Path, database_path: Path) -> None:
        self.repository = repository
        self.job_root = job_root.resolve()
        self.database_path = database_path.resolve()
        self.job_root.mkdir(parents=True, exist_ok=True)
        self._local_processes: dict[str, subprocess.Popen[str]] = {}
        self.policy = PolicyService(repository)

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
        if provider == "local-docker":
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
        image = payload.get("image") or inputs.get("image")
        command = payload.get("command", inputs.get("command"))
        if not image:
            raise ValueError("local-docker job requires image or inputs.image")
        if not command:
            raise ValueError("local-docker job requires command or inputs.command")
        cwd = Path(payload.get("cwd") or inputs.get("cwd") or os.getcwd()).resolve()
        docker_command: list[str] = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{cwd}:/workspace",
            "-w",
            "/workspace",
            str(image),
        ]
        if isinstance(command, str):
            docker_command.extend(["sh", "-lc", command])
        else:
            docker_command.extend(str(item) for item in command)
        return self.launch_local(
            {
                **payload,
                "provider": "local-docker",
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
                },
            }
        )

    def launch_local(self, payload: dict[str, Any]) -> dict[str, Any]:
        inputs = dict(payload.get("inputs") or {})
        command = payload.get("command", inputs.get("command"))
        if not command:
            raise ValueError("local job requires command or inputs.command")
        record = self.repository.create_job(
            {
                **payload,
                "provider": payload.get("provider") or "local",
                "status": "queued",
                "inputs": {
                    **inputs,
                    "command": command,
                    "cwd": str(Path(payload.get("cwd") or inputs.get("cwd") or os.getcwd()).resolve()),
                    "env": payload.get("env") or inputs.get("env") or {},
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
        env = dict(os.environ)
        src_root = Path(__file__).resolve().parents[2]
        env["PYTHONPATH"] = str(src_root) if not env.get("PYTHONPATH") else f"{src_root}:{env['PYTHONPATH']}"
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
        self._local_processes[record["job_id"]] = process
        return self.repository.update_job(
            record["job_id"],
            {"status": "running", "pid": process.pid, "details": {"launcher_pid": process.pid}},
        )

    def get(self, job_id: str) -> dict[str, Any]:
        process = self._local_processes.get(job_id)
        record = self.repository.get_job(job_id)
        if record is None:
            raise KeyError(job_id)
        if process is not None:
            if process.poll() is not None or record["status"] in TERMINAL_JOB_STATUSES:
                try:
                    process.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    pass
            if process.poll() is not None:
                self._local_processes.pop(job_id, None)
        record = self.repository.get_job(job_id)
        assert record is not None
        if record["provider"] == "runpod":
            record = self._refresh_runpod_status(record)
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
