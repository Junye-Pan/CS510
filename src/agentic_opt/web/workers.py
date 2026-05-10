from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_opt.control_plane.repository import ControlPlaneRepository


@dataclass
class WorkerProcess:
    assignment_id: str
    session_id: str
    experiment_id: str
    process: subprocess.Popen[str]


class WorkerManager:
    def __init__(
        self,
        *,
        repo_root: Path,
        state_root: Path,
        control: ControlPlaneRepository,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.state_root = state_root.resolve()
        self.control = control
        self._assignment_processes: dict[str, WorkerProcess] = {}

    def start_control_assignment(
        self,
        *,
        assignment_id: str,
        api_url: str,
        dry_run: bool = False,
        max_turn_wall_time_s: int | None = None,
    ) -> dict[str, Any]:
        existing = self._assignment_processes.get(assignment_id)
        if existing is not None and existing.process.poll() is None:
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
        cmd = [
            sys.executable,
            "-m",
            "agentic_opt.adapter.semantic_worker",
            "--assignment-id",
            assignment_id,
            "--session-id",
            session["session_id"],
            "--api-url",
            api_url,
            "--workspace-root",
            str(workspace_root),
        ]
        if dry_run:
            cmd.append("--dry-run")
        if max_turn_wall_time_s is not None:
            cmd.extend(["--max-turn-wall-time-s", str(max_turn_wall_time_s)])
        env = dict(os.environ)
        repo_src = str(self.repo_root / "src")
        env["PYTHONPATH"] = repo_src if not env.get("PYTHONPATH") else f"{repo_src}:{env['PYTHONPATH']}"
        env["AO_CONTROL_API_URL"] = api_url
        env["AO_ASSIGNMENT_ID"] = assignment_id
        env["AO_SESSION_ID"] = session["session_id"]
        env["AO_TASK_ID"] = assignment["task_id"]
        env["AO_EXPERIMENT_ID"] = assignment["experiment_id"]
        env["AO_AGENT_ID"] = assignment["agent_id"]
        process = subprocess.Popen(
            cmd,
            cwd=str(self.repo_root),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._assignment_processes[assignment_id] = WorkerProcess(
            assignment_id=assignment_id,
            session_id=session["session_id"],
            experiment_id=assignment["experiment_id"],
            process=process,
        )
        return self.control.update_session(
            session["session_id"],
            {
                "status": "running",
                "pid": process.pid,
                "workspace_path": str(workspace_root),
                "details": {"api_url": api_url, "dry_run": dry_run},
            },
        )

    def worker_status(self, assignment_id: str) -> dict[str, Any] | None:
        worker = self._assignment_processes.get(assignment_id)
        if worker is None:
            return None
        status = "running" if worker.process.poll() is None else "finished"
        return {
            "assignment_id": worker.assignment_id,
            "session_id": worker.session_id,
            "experiment_id": worker.experiment_id,
            "pid": worker.process.pid,
            "status": status,
        }
