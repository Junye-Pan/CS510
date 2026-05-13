from __future__ import annotations

import hashlib
import json
import sys
import os
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Any

from agentic_opt.common.atomic import atomic_write_text
from agentic_opt.common.config import get_repo_root
from agentic_opt.common.numpy_loader import import_numpy
from agentic_opt.common.snapshot import copy_snapshot
from agentic_opt.common.ids import make_run_id
from agentic_opt.task_api import candidate_snapshot_paths, candidate_spec_for
from agentic_opt.task_registry import get_task

from .environments import EnvironmentService
from .jobs import JobService
from .process_env import build_subprocess_env, sanitize_env
from .repository import ControlPlaneRepository


class EvaluationService:
    """Server-owned evaluation service.

    Task evaluator functions are called here as task-level implementation
    details. The worker talks to Evaluation resources, not directly to hidden
    evaluator internals.
    """

    def __init__(
        self,
        *,
        repository: ControlPlaneRepository,
        jobs: JobService,
        environments: EnvironmentService,
        database_path: Path,
        artifact_root: Path,
    ) -> None:
        self.repository = repository
        self.jobs = jobs
        self.environments = environments
        self.database_path = database_path.resolve()
        self.artifact_root = artifact_root.resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = self._normalize_request(payload)
        self._enforce_evaluator_budget(request)
        if _leaderboard_eligible_kind(request["kind"]) and request.get("artifact_id") is None and payload.get("snapshot_candidate", True):
            artifact = self._snapshot_candidate_for_evaluation(request)
            request = {
                **request,
                "artifact_id": artifact["artifact_id"],
                "input_kind": "artifact_snapshot",
                "entry_path": _entry_path_for_artifact(artifact),
            }
        run_async = _evaluation_async_requested(payload, request["kind"])
        execution = self._resolve_execution_environment(request)
        request = {
            **request,
            "environment_id": execution["environment_id"],
            "environment_overlay_id": execution["environment_overlay_id"],
            "environment_kind": execution["kind"],
        }
        record = self.repository.create_evaluation(
            {
                "experiment_id": request["experiment_id"],
                "assignment_id": request.get("assignment_id"),
                "attempt_id": payload.get("attempt_id"),
                "artifact_id": request.get("artifact_id"),
                "kind": request["kind"],
                "status": "queued" if run_async else "running",
                "request": request,
            }
        )
        if not run_async:
            return self._run_in_environment(record["evaluation_id"], request, execution)

        job = self.jobs.launch_local(
            {
                "experiment_id": request["experiment_id"],
                "assignment_id": request.get("assignment_id"),
                "session_id": payload.get("session_id"),
                "inputs": {
                    "command": [
                        str(execution["python_path"]),
                        "-m",
                        "agentic_opt.control_plane.evaluation_worker",
                        "--db",
                        str(self.database_path),
                        "--evaluation-id",
                        record["evaluation_id"],
                    ],
                    "cwd": str(get_repo_root()),
                    "env": _execution_subprocess_env(execution),
                },
                "details": {
                    "kind": "evaluation",
                    "evaluation_id": record["evaluation_id"],
                    "environment_id": execution["environment_id"],
                    "environment_overlay_id": execution["environment_overlay_id"],
                },
            }
        )
        record = self.repository.update_evaluation(record["evaluation_id"], {"status": "queued", "job_id": job["job_id"]})
        self.repository.record_event(
            {
                "experiment_id": request["experiment_id"],
                "assignment_id": request.get("assignment_id"),
                "task_id": request["task_id"],
                "agent_id": request.get("agent_id"),
                "event_type": f"evaluation.{request['kind']}.queued",
                "summary": f"{request['kind']} evaluation queued",
                "payload": {"evaluation_id": record["evaluation_id"], "job_id": job["job_id"]},
            }
        )
        return record

    def run(self, evaluation_id: str) -> dict[str, Any]:
        record = self.repository.get_evaluation(evaluation_id)
        if record is None:
            raise KeyError(evaluation_id)
        request = record.get("request") or {}
        self.repository.update_evaluation(evaluation_id, {"status": "running"})
        try:
            result, valid, score, public_feedback = self._run_request(request)
            updated = self.repository.update_evaluation(
                evaluation_id,
                {
                    "status": "completed",
                    "valid": valid,
                    "score": score,
                    "result": result,
                    "public_feedback": public_feedback,
                },
            )
            leaderboard_entry = self._record_leaderboard_entry(updated)
            if leaderboard_entry is not None:
                updated = self.repository.get_evaluation(evaluation_id) or updated
            event_type = f"evaluation.{request.get('kind', record['kind'])}.completed"
            summary = f"{request.get('kind', record['kind'])} evaluation completed"
        except Exception as exc:
            failure = {"error": str(exc), "traceback": traceback.format_exc()}
            updated = self.repository.update_evaluation(
                evaluation_id,
                {
                    "status": "failed",
                    "valid": False,
                    "score": None,
                    "result": failure,
                    "public_feedback": {"error": str(exc)},
                },
            )
            event_type = f"evaluation.{request.get('kind', record['kind'])}.failed"
            summary = f"{request.get('kind', record['kind'])} evaluation failed"
        self.repository.record_event(
            {
                "experiment_id": updated["experiment_id"],
                "assignment_id": updated.get("assignment_id"),
                "task_id": request.get("task_id"),
                "agent_id": request.get("agent_id"),
                "event_type": event_type,
                "summary": summary,
                "payload": {
                    "evaluation_id": updated["evaluation_id"],
                    "valid": updated.get("valid"),
                    "score": updated.get("score"),
                    "status": updated["status"],
                },
            }
        )
        return updated

    def _normalize_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        assignment_id = payload.get("assignment_id")
        assignment = self.repository.get_assignment(assignment_id) if assignment_id else None
        experiment_id = payload.get("experiment_id") or (assignment or {}).get("experiment_id")
        if not experiment_id:
            raise ValueError("experiment_id or assignment_id is required")
        task_id = payload.get("task_id") or (assignment or {}).get("task_id")
        if not task_id:
            raise ValueError("task_id or assignment_id is required")
        artifact_id = payload.get("artifact_id")
        input_kind = "entry_path"
        if "entry_path" in payload:
            entry_path = Path(payload["entry_path"]).resolve()
        elif artifact_id:
            artifact = self.repository.get_artifact(str(artifact_id))
            if artifact is None:
                raise KeyError(str(artifact_id))
            if not artifact.get("local_path"):
                raise ValueError(f"artifact {artifact_id} has no local_path for evaluation")
            entry_path = Path(_entry_path_for_artifact(artifact)).resolve()
            input_kind = "artifact"
        else:
            raise ValueError("entry_path or artifact_id is required")
        return {
            "experiment_id": experiment_id,
            "assignment_id": assignment_id,
            "artifact_id": artifact_id,
            "task_id": task_id,
            "agent_id": (assignment or {}).get("agent_id"),
            "kind": payload.get("kind") or "submit",
            "input_kind": input_kind,
            "entry_path": str(entry_path),
            "workspace_root": payload.get("workspace_root"),
            "probe_kind": payload.get("probe_kind"),
            "requested_environment_id": payload.get("environment_id"),
            "requested_environment_overlay_id": payload.get("environment_overlay_id") or payload.get("overlay_id"),
        }

    def _enforce_evaluator_budget(self, request: dict[str, Any]) -> None:
        experiment_id = request.get("experiment_id")
        experiment = self.repository.get_experiment(experiment_id) if experiment_id else None
        experiment_budget = (experiment or {}).get("budget") or {}
        experiment_limit = _positive_int(experiment_budget.get("total_evaluator_runs") or experiment_budget.get("evaluator_runs"))
        if experiment_limit is not None and experiment_id:
            experiment_used = _count_evaluator_runs(self.repository.list_evaluations(experiment_id=experiment_id))
            if experiment_used >= experiment_limit:
                raise ValueError("evaluator_budget_exhausted: experiment evaluator budget exhausted")

        assignment_id = request.get("assignment_id")
        assignment = self.repository.get_assignment(assignment_id) if assignment_id else None
        assignment_limit = _positive_int(((assignment or {}).get("budget") or {}).get("evaluator_runs"))
        if assignment_limit is not None and assignment_id:
            assignment_used = _count_evaluator_runs(self.repository.list_evaluations(assignment_id=assignment_id))
            if assignment_used >= assignment_limit:
                raise ValueError("evaluator_budget_exhausted: assignment evaluator budget exhausted")

    def _run_request(self, request: dict[str, Any]) -> tuple[dict[str, Any], bool, float | None, dict[str, Any]]:
        task = get_task(request["task_id"])
        entry_path = Path(request["entry_path"]).resolve()
        kind = request.get("kind") or "submit"
        if kind == "verify":
            result = _json_compatible(task.verify_entry(entry_path))
            valid = bool(result.get("valid"))
            score = None
            public_feedback = result.get("feedback") or result
        elif kind == "probe":
            probe_kind = str(request.get("probe_kind") or "diagnostics")
            result = _json_compatible(task.probe_entry(entry_path, kind=probe_kind))
            valid = bool(result.get("valid", True))
            score = result.get("score")
            public_feedback = result
        elif kind in {"submit", "official"}:
            verifier = _json_compatible(task.verify_entry(entry_path))
            if not verifier.get("valid"):
                result = {"verifier": verifier, "evaluated": False}
                valid = False
                score = 0.0
                public_feedback = verifier.get("feedback") or verifier
            else:
                result = _json_compatible(task.evaluate_entry(entry_path))
                valid = bool(result.get("correct", {}).get("correct", True))
                score = float(result["score"])
                public_feedback = result.get("evaluator", {}).get("public_details") or {}
        else:
            raise ValueError(f"unknown evaluation kind: {kind}")
        return result, valid, score, public_feedback

    def _resolve_execution_environment(self, request: dict[str, Any]) -> dict[str, Any]:
        kind = request.get("kind") or "submit"
        allow_overlay = kind in {"verify", "probe"}
        return self.environments.get_execution_environment(
            task_id=request["task_id"],
            experiment_id=request.get("experiment_id"),
            environment_id=request.get("requested_environment_id"),
            overlay_id=request.get("requested_environment_overlay_id"),
            allow_overlay=allow_overlay,
        )

    def _snapshot_candidate_for_evaluation(self, request: dict[str, Any]) -> dict[str, Any]:
        task = get_task(request["task_id"])
        spec = candidate_spec_for(task)
        entry_path = Path(request["entry_path"]).resolve()
        workspace_root = Path(request["workspace_root"]).resolve() if request.get("workspace_root") else None
        source_root, entry_relative_path, candidate_root = candidate_snapshot_paths(
            entry_path=entry_path,
            workspace_root=workspace_root,
            spec=spec,
        )
        artifact_id = make_run_id("artifact")
        artifact_dir = self.artifact_root / artifact_id
        artifact_dir.mkdir(parents=True, exist_ok=False)
        manifest_path = artifact_dir / "manifest.json"
        content_path = artifact_dir / "content"
        metadata = {
            "storage_provider": "local",
            "source_path": str(entry_path),
            "snapshot_reason": "official_evaluation",
            "task_id": request["task_id"],
            "entry_relative_path": entry_relative_path,
            "candidate_root": candidate_root,
            "workspace_root": str(workspace_root) if workspace_root else None,
        }
        if candidate_root is None:
            content_path.mkdir(parents=True, exist_ok=True)
            destination = content_path / entry_path.name
            shutil.copy2(entry_path, destination)
            local_path = destination
            digest = _digest_file(destination)
            file_count = 1
            size_bytes = destination.stat().st_size
        else:
            copied = copy_snapshot(source_root, content_path)
            local_path = content_path
            digest = _digest_directory(content_path)
            file_count = len(copied)
            size_bytes = _size_bytes(content_path)
            metadata["copied_files"] = copied
        metadata["size_bytes"] = size_bytes
        metadata["file_count"] = file_count
        manifest = {
            "artifact_id": artifact_id,
            "kind": "candidate",
            "content_path": str(local_path),
            "uri": local_path.as_uri(),
            "digest": digest,
            "metadata": metadata,
        }
        atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        metadata["manifest_path"] = str(manifest_path)
        return self.repository.create_artifact(
            {
                "artifact_id": artifact_id,
                "experiment_id": request.get("experiment_id"),
                "assignment_id": request.get("assignment_id"),
                "kind": "candidate",
                "uri": local_path.as_uri(),
                "local_path": str(local_path),
                "digest": digest,
                "metadata": metadata,
            }
        )

    def _record_leaderboard_entry(self, evaluation: dict[str, Any]) -> dict[str, Any] | None:
        request = evaluation.get("request") or {}
        if not _leaderboard_eligible_kind(evaluation.get("kind") or request.get("kind")):
            return None
        if evaluation.get("status") != "completed" or not evaluation.get("valid") or evaluation.get("score") is None:
            return None
        incumbent_before = self.repository.get_incumbent(experiment_id=evaluation["experiment_id"])
        score = float(evaluation["score"])
        assignment = self.repository.get_assignment(evaluation.get("assignment_id")) if evaluation.get("assignment_id") else None
        entry = self.repository.create_leaderboard_entry(
            {
                "experiment_id": evaluation["experiment_id"],
                "task_id": request["task_id"],
                "assignment_id": evaluation.get("assignment_id"),
                "direction_id": (assignment or {}).get("direction_id"),
                "evaluation_id": evaluation["evaluation_id"],
                "artifact_id": evaluation.get("artifact_id"),
                "score": score,
                "environment_id": request.get("environment_id"),
                "environment_overlay_id": request.get("environment_overlay_id"),
                "metadata": {
                    "input_kind": request.get("input_kind"),
                    "entry_path": request.get("entry_path"),
                    "agent_id": request.get("agent_id"),
                    "environment_kind": request.get("environment_kind"),
                    "public_feedback": evaluation.get("public_feedback") or {},
                },
            }
        )
        if incumbent_before is None or score > float(incumbent_before["score"]):
            self.repository.record_event(
                {
                    "experiment_id": evaluation["experiment_id"],
                    "assignment_id": evaluation.get("assignment_id"),
                    "task_id": request.get("task_id"),
                    "agent_id": request.get("agent_id"),
                    "event_type": "leaderboard.incumbent_updated",
                    "summary": f"new incumbent score={score}",
                    "payload": {
                        "leaderboard_entry_id": entry["leaderboard_entry_id"],
                        "evaluation_id": evaluation["evaluation_id"],
                        "artifact_id": evaluation.get("artifact_id"),
                        "score": score,
                        "previous_score": incumbent_before.get("score") if incumbent_before else None,
                    },
                }
            )
        return entry

    def _run_in_environment(self, evaluation_id: str, request: dict[str, Any], execution: dict[str, Any]) -> dict[str, Any]:
        proc = subprocess.run(
            [
                str(execution["python_path"]),
                "-m",
                "agentic_opt.control_plane.evaluation_worker",
                "--db",
                str(self.database_path),
                "--evaluation-id",
                evaluation_id,
            ],
            cwd=str(get_repo_root()),
            env=_execution_subprocess_env(execution),
            capture_output=True,
            text=True,
            check=False,
            timeout=float(request.get("timeout_s") or 3600),
        )
        record = self.repository.get_evaluation(evaluation_id)
        if record is None:
            raise KeyError(evaluation_id)
        if record["status"] == "running":
            record = self.repository.update_evaluation(
                evaluation_id,
                {
                    "status": "failed",
                    "valid": False,
                    "result": {
                        "error": "evaluation worker exited before updating status",
                        "returncode": proc.returncode,
                        "stdout": proc.stdout,
                        "stderr": proc.stderr,
                    },
                    "public_feedback": {"error": proc.stderr.strip() or proc.stdout.strip()},
                },
            )
        return record


def _json_compatible(value: Any) -> Any:
    np = import_numpy()
    ndarray_type = getattr(np, "ndarray", None)
    generic_type = getattr(np, "generic", None)
    if ndarray_type is not None and isinstance(value, ndarray_type):
        return value.tolist()
    if generic_type is not None and isinstance(value, generic_type):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _evaluation_async_requested(payload: dict[str, Any], kind: str) -> bool:
    if "async" in payload:
        return bool(payload["async"])
    if "run_async" in payload:
        return bool(payload["run_async"])
    return kind in {"submit", "official"}


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


def _leaderboard_eligible_kind(kind: str | None) -> bool:
    return (kind or "submit") in {"submit", "official"}


def _entry_path_for_artifact(artifact: dict[str, Any]) -> str:
    local_path = Path(artifact["local_path"]).resolve()
    metadata = artifact.get("metadata") or {}
    entry_relative_path = metadata.get("entry_relative_path")
    if local_path.is_dir() and entry_relative_path:
        return str((local_path / str(entry_relative_path)).resolve())
    return str(local_path)


def _execution_subprocess_env(execution: dict[str, Any]) -> dict[str, str]:
    repo_root = get_repo_root()
    env = build_subprocess_env(execution.get("exports") or {})
    env["PYTHONPATH"] = str(repo_root / "src")
    if "AO_TASKS_ROOTS" in os.environ:
        env.update(sanitize_env({"AO_TASKS_ROOTS": os.environ["AO_TASKS_ROOTS"]}))
    elif "AO_TASKS_ROOT" in os.environ:
        env.update(sanitize_env({"AO_TASKS_ROOT": os.environ["AO_TASKS_ROOT"]}))
    return env


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _digest_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
