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
from agentic_opt.common.runtime_env import RuntimeEnvironmentError
from agentic_opt.common.snapshot import copy_snapshot
from agentic_opt.common.ids import make_run_id
from agentic_opt.task_api import candidate_snapshot_paths, candidate_spec_for
from agentic_opt.task_registry import get_task

from .docker_runtime import DockerMount
from .environments import EnvironmentService
from .environment_providers import (
    EnvironmentRunSpec,
    docker_image_reference,
    docker_runner_metadata,
    docker_workdir,
    provider_for_execution,
)
from .jobs import JobService
from .process_env import build_subprocess_env, sanitize_env
from .repository import ControlPlaneRepository
from .task_context import (
    append_docker_task_context_mount,
    docker_task_context_enforcement,
    ensure_task_context_snapshot,
    local_task_context_enforcement,
    verify_request_task_context,
)


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
        self.state_root = self.artifact_root.parent
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
        framework_environment = self.environments.ensure_framework_environment()
        task_context = ensure_task_context_snapshot(task_id=request["task_id"], state_root=self.state_root)
        expected_task_context_digest = payload.get("expected_task_context_digest") or request.get("expected_task_context_digest")
        if expected_task_context_digest and expected_task_context_digest != task_context.get("digest"):
            raise ValueError(
                f"task_context_digest_mismatch: expected {expected_task_context_digest}, got {task_context.get('digest')}"
            )
        task_context_enforcement = (
            docker_task_context_enforcement(snapshot=task_context, workspace_root=Path(request["workspace_root"]).resolve())
            if _execution_uses_docker(execution) and request.get("workspace_root")
            else local_task_context_enforcement(
                snapshot=task_context,
                workspace_root=Path(request["workspace_root"]).resolve() if request.get("workspace_root") else None,
            )
        )
        request = {
            **request,
            "environment_id": execution["environment_id"],
            "environment_overlay_id": execution["environment_overlay_id"],
            "environment_kind": execution["kind"],
            "environment_provider": execution.get("provider"),
            "environment_lock": _execution_lock_summary(execution),
            "runner": _execution_runner_metadata(execution),
            "framework_environment_id": framework_environment["environment_id"],
            "framework_environment_lock": _framework_environment_lock_summary(framework_environment),
            "framework_runner": _framework_runner_metadata(framework_environment),
            "task_context": task_context,
            "task_context_enforcement": task_context_enforcement,
        }
        record = self.repository.create_evaluation(
                {
                    "experiment_id": request["experiment_id"],
                    "assignment_id": request.get("assignment_id"),
                    "attempt_id": request.get("attempt_id"),
                    "artifact_id": request.get("artifact_id"),
                    "kind": request["kind"],
                    "status": "queued" if run_async else "running",
                    "request": request,
                }
        )
        if not run_async:
            return self._run_in_environment(record["evaluation_id"], request, execution)

        if _execution_uses_docker(execution):
            job = self.jobs.launch(
                {
                    "experiment_id": request["experiment_id"],
                    "assignment_id": request.get("assignment_id"),
                    "session_id": payload.get("session_id"),
                    "attempt_id": request.get("attempt_id"),
                    "task_id": request["task_id"],
                    "provider": "docker_image",
                    "environment_id": execution["environment_id"],
                    "environment_overlay_id": execution["environment_overlay_id"],
                    "command": _evaluation_worker_command(record["evaluation_id"], execution, self.database_path),
                    "cwd": str(self.database_path.parent),
                    "inputs": {
                        "environment_id": execution["environment_id"],
                        "environment_overlay_id": execution["environment_overlay_id"],
                        "mounts": _evaluation_docker_mount_specs(request=request, database_path=self.database_path),
                        "workdir": docker_workdir(execution),
                    },
                    "env": _execution_subprocess_env(execution),
                    "details": {
                        "kind": "evaluation",
                        "evaluation_id": record["evaluation_id"],
                        "environment_id": execution["environment_id"],
                        "environment_overlay_id": execution["environment_overlay_id"],
                        "environment_provider": execution.get("provider"),
                        "runner": _execution_runner_metadata(execution),
                    },
                }
            )
        else:
            job = self.jobs.launch_local(
                {
                    "experiment_id": request["experiment_id"],
                    "assignment_id": request.get("assignment_id"),
                    "session_id": payload.get("session_id"),
                    "attempt_id": request.get("attempt_id"),
                    "inputs": {
                        "command": _evaluation_worker_command(record["evaluation_id"], execution, self.database_path),
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
            pre_task_context = verify_request_task_context(request)
            if not pre_task_context["ok"]:
                raise RuntimeError(f"task_context_verification_failed_before_run: {pre_task_context}")
            result, valid, score, public_feedback = self._run_request(request)
            post_task_context = verify_request_task_context(request)
            if not post_task_context["ok"]:
                raise RuntimeError(f"task_context_verification_failed_after_run: {post_task_context}")
            result = {
                **result,
                "task_context_verification": {
                    "before": pre_task_context,
                    "after": post_task_context,
                },
            }
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
            "attempt_id": payload.get("attempt_id"),
            "task_id": task_id,
            "agent_id": (assignment or {}).get("agent_id"),
            "kind": payload.get("kind") or "submit",
            "input_kind": input_kind,
            "entry_path": str(entry_path),
            "workspace_root": payload.get("workspace_root"),
            "probe_kind": payload.get("probe_kind"),
            "requested_environment_id": payload.get("environment_id"),
            "requested_environment_overlay_id": payload.get("environment_overlay_id") or payload.get("overlay_id"),
            "requested_environment_provider": payload.get("environment_provider") or payload.get("provider"),
            "replay": payload.get("replay"),
            "publish_leaderboard": payload.get("publish_leaderboard"),
            "count_budget": payload.get("count_budget"),
            "expected_task_context_digest": payload.get("expected_task_context_digest"),
        }

    def _enforce_evaluator_budget(self, request: dict[str, Any]) -> None:
        if not _request_counts_toward_leaderboard_budget(request):
            return
        experiment_id = request.get("experiment_id")
        experiment = self.repository.get_experiment(experiment_id) if experiment_id else None
        experiment_budget = (experiment or {}).get("budget") or {}
        experiment_limit = _positive_int(experiment_budget.get("total_evaluator_runs") or experiment_budget.get("evaluator_runs"))
        if experiment_limit is not None and experiment_id:
            experiment_used = _leaderboard_budget_used(self.repository, experiment_id=experiment_id)
            if experiment_used >= experiment_limit:
                raise ValueError("evaluator_budget_exhausted: experiment evaluator budget exhausted")

        assignment_id = request.get("assignment_id")
        assignment = self.repository.get_assignment(assignment_id) if assignment_id else None
        assignment_limit = _positive_int(((assignment or {}).get("budget") or {}).get("evaluator_runs"))
        if assignment_limit is not None and assignment_id:
            assignment_used = _leaderboard_budget_used(
                self.repository,
                experiment_id=experiment_id,
                assignment_id=assignment_id,
            )
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
        experiment = self.repository.get_experiment(request.get("experiment_id")) if request.get("experiment_id") else None
        requested_overlay_id = request.get("requested_environment_overlay_id")
        allow_overlay = kind in {"verify", "probe"}
        if kind in {"submit", "official"} and requested_overlay_id:
            if not _official_overlay_submit_allowed(experiment):
                raise ValueError("official_overlay_submit_disabled: official submit may not use an environment overlay unless policy.environments.allow_official_overlay_submit is true")
            overlay = self.repository.get_environment_overlay(str(requested_overlay_id))
            if overlay is None:
                raise KeyError(str(requested_overlay_id))
            if overlay.get("status") != "ready":
                raise RuntimeEnvironmentError(f"environment overlay is not ready: {requested_overlay_id} status={overlay.get('status')}")
            if not overlay.get("approved"):
                raise ValueError("official_overlay_submit_unapproved: official submit requires an approved environment overlay")
            allow_overlay = True
        provider, provider_config = _evaluation_environment_provider(request, experiment=experiment)
        return self.environments.get_execution_environment(
            task_id=request["task_id"],
            experiment_id=request.get("experiment_id"),
            environment_id=request.get("requested_environment_id"),
            overlay_id=request.get("requested_environment_overlay_id"),
            allow_overlay=allow_overlay,
            provider=provider,
            provider_config=provider_config,
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
                "attempt_id": request.get("attempt_id"),
                "kind": "candidate",
                "uri": local_path.as_uri(),
                "local_path": str(local_path),
                "digest": digest,
                "metadata": metadata,
            }
        )

    def _record_leaderboard_entry(self, evaluation: dict[str, Any]) -> dict[str, Any] | None:
        request = evaluation.get("request") or {}
        if request.get("publish_leaderboard") is False:
            return None
        if request.get("replay") and request.get("publish_leaderboard") is not True:
            return None
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
                    "environment_provider": request.get("environment_provider"),
                    "environment_lock": request.get("environment_lock"),
                    "runner": request.get("runner"),
                    "task_context": {
                        "digest": (request.get("task_context") or {}).get("digest"),
                        "enforcement": request.get("task_context_enforcement"),
                    },
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
        provider = provider_for_execution(execution)
        plan = provider.build_run_plan(
            EnvironmentRunSpec(
                execution=execution,
                command=_evaluation_worker_command(evaluation_id, execution, self.database_path),
                cwd=self.database_path.parent if _execution_uses_docker(execution) else get_repo_root(),
                env=_execution_subprocess_env(execution),
                mounts=_evaluation_docker_mounts(request=request, database_path=self.database_path) if _execution_uses_docker(execution) else [],
                workdir=docker_workdir(execution) if _execution_uses_docker(execution) else None,
                network_policy=_network_policy_for_experiment(self.repository.get_experiment(request.get("experiment_id"))),
                pids_limit=512 if _execution_uses_docker(execution) else None,
                require_immutable_image=True,
            )
        )
        proc = subprocess.run(
            plan.command,
            cwd=str(plan.cwd),
            env=plan.env,
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
                        "error": (
                            "docker evaluation worker exited before updating status"
                            if _execution_uses_docker(execution)
                            else "evaluation worker exited before updating status"
                        ),
                        "returncode": proc.returncode,
                        "stdout": proc.stdout,
                        "stderr": proc.stderr,
                        "runner": plan.metadata,
                        "network_enforcement": plan.network_enforcement,
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


def _leaderboard_budget_used(
    repository: ControlPlaneRepository,
    *,
    experiment_id: str | None,
    assignment_id: str | None = None,
) -> int:
    if not experiment_id:
        return 0
    leaderboard_used = sum(
        1
        for entry in repository.list_leaderboard_entries(experiment_id=experiment_id, limit=1_000_000)
        if assignment_id is None or entry.get("assignment_id") == assignment_id
    )
    pending_used = sum(
        1
        for evaluation in repository.list_evaluations(experiment_id=experiment_id, assignment_id=assignment_id)
        if _pending_evaluation_counts_toward_leaderboard_budget(evaluation)
    )
    return leaderboard_used + pending_used


def _pending_evaluation_counts_toward_leaderboard_budget(evaluation: dict[str, Any]) -> bool:
    if evaluation.get("status") not in {"queued", "running"}:
        return False
    request = evaluation.get("request") or {}
    kind = evaluation.get("kind") or request.get("kind")
    return _request_counts_toward_leaderboard_budget({**request, "kind": kind})


def _request_counts_toward_leaderboard_budget(request: dict[str, Any]) -> bool:
    if not _leaderboard_eligible_kind(request.get("kind")):
        return False
    if request.get("count_budget") is False:
        return False
    if request.get("publish_leaderboard") is False:
        return False
    if request.get("replay") and request.get("publish_leaderboard") is not True:
        return False
    return True


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
    if _execution_uses_docker(execution):
        env = sanitize_env(execution.get("exports") or {})
        exports = execution.get("exports") or {}
        env["PYTHONPATH"] = str(exports.get("PYTHONPATH") or "/opt/agentic-opt/src")
        env["AO_TASKS_ROOTS"] = str(exports.get("AO_TASKS_ROOTS") or "/opt/agentic-opt/tasks")
        env["PYTHONUNBUFFERED"] = "1"
        return env
    env = build_subprocess_env(execution.get("exports") or {})
    env["PYTHONPATH"] = str(repo_root / "src")
    if "AO_TASKS_ROOTS" in os.environ:
        env.update(sanitize_env({"AO_TASKS_ROOTS": os.environ["AO_TASKS_ROOTS"]}))
    elif "AO_TASKS_ROOT" in os.environ:
        env.update(sanitize_env({"AO_TASKS_ROOT": os.environ["AO_TASKS_ROOT"]}))
    return env


def _execution_uses_docker(execution: dict[str, Any]) -> bool:
    return str(execution.get("provider") or "") == "docker_image"


def _execution_lock_summary(execution: dict[str, Any]) -> dict[str, Any]:
    record = execution.get("record") or {}
    if execution.get("kind") == "overlay":
        base = execution.get("base_record") or {}
        return {
            "base_environment_id": execution.get("environment_id"),
            "base_lock": base.get("lock") or {},
            "overlay_lock": record.get("lock") or {},
        }
    lock = record.get("lock") or {}
    return {
        "provider": execution.get("provider"),
        "image_ref": lock.get("image_ref"),
        "image_digest": lock.get("image_digest"),
        "image_id": lock.get("image_id"),
        "requirements": lock.get("requirements"),
        "format": lock.get("format"),
    } if execution.get("provider") == "docker_image" else lock


def _execution_runner_metadata(execution: dict[str, Any]) -> dict[str, Any]:
    if _execution_uses_docker(execution):
        image = docker_image_reference(execution, require_immutable=True, allow_mutable=False)
        return docker_runner_metadata(execution, image)
    record = execution.get("record") or {}
    metadata = record.get("metadata") or {}
    if execution.get("kind") == "overlay":
        metadata = {**((execution.get("base_record") or {}).get("metadata") or {}), **metadata}
    lock = record.get("lock") or {}
    return {
        "provider": execution.get("provider"),
        "image_ref": metadata.get("image_ref") or lock.get("image_ref"),
        "image_digest": metadata.get("image_digest") or lock.get("image_digest"),
        "workdir": (metadata.get("runner") or {}).get("workdir") or execution.get("root_path"),
        "python_path": execution.get("python_path"),
    }


def _framework_environment_lock_summary(environment: dict[str, Any]) -> dict[str, Any]:
    lock = environment.get("lock") or {}
    return {
        "environment_id": environment.get("environment_id"),
        "fingerprint": environment.get("fingerprint"),
        "python_path": environment.get("python_path"),
        "lock": lock,
    }


def _framework_runner_metadata(environment: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": "framework",
        "environment_id": environment.get("environment_id"),
        "python_path": environment.get("python_path"),
        "root_path": environment.get("root_path"),
        "fingerprint": environment.get("fingerprint"),
    }


def _evaluation_worker_command(evaluation_id: str, execution: dict[str, Any], database_path: Path) -> list[str]:
    return [
        str(execution["python_path"]),
        "-m",
        "agentic_opt.control_plane.evaluation_worker",
        "--db",
        str(database_path),
        "--evaluation-id",
        evaluation_id,
    ]


def _evaluation_docker_mount_specs(*, request: dict[str, Any], database_path: Path) -> list[dict[str, Any]]:
    return [
        {"source": str(mount.source), "target": mount.target, "read_only": mount.read_only}
        for mount in _evaluation_docker_mounts(request=request, database_path=database_path)
    ]


def _evaluation_docker_mounts(*, request: dict[str, Any], database_path: Path) -> list[DockerMount]:
    state_root = database_path.resolve().parent
    repo_root = get_repo_root().resolve()
    mounts = [
        DockerMount(source=state_root, target=str(state_root)),
        DockerMount(source=repo_root, target=str(repo_root), read_only=True),
    ]
    entry_path = Path(request["entry_path"]).resolve()
    if not _path_is_relative_to(entry_path, state_root) and not _path_is_relative_to(entry_path, repo_root):
        mounts.append(DockerMount(source=entry_path.parent, target=str(entry_path.parent), read_only=True))
    task_context = request.get("task_context") or {}
    if task_context.get("task_path") and request.get("workspace_root"):
        mounts = append_docker_task_context_mount(
            mounts=mounts,
            snapshot=task_context,
            workspace_root=Path(str(request["workspace_root"])),
        )
    return _dedupe_mounts(mounts)


def _dedupe_mounts(mounts: list[DockerMount]) -> list[DockerMount]:
    result: list[DockerMount] = []
    seen: set[tuple[str, str]] = set()
    for mount in mounts:
        key = (str(mount.source.resolve()), mount.target)
        if key in seen:
            continue
        seen.add(key)
        result.append(mount)
    return result


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _network_policy_for_experiment(experiment: dict[str, Any] | None) -> dict[str, Any]:
    raw_policy = ((experiment or {}).get("policy") or {}).get("network") or ((experiment or {}).get("config") or {}).get("network") or {}
    external = raw_policy.get("external_internet")
    if external is None:
        external = "deny" if raw_policy.get("allow_external_internet") is False else "allow"
    return {
        "control_plane": str(raw_policy.get("control_plane") or raw_policy.get("control_plane_network") or "allow"),
        "external_internet": str(external),
        "package_indexes": raw_policy.get("package_indexes") or "policy",
        "allowed_hosts": raw_policy.get("allowed_hosts") or ["127.0.0.1", "localhost"],
        "denied_hosts": raw_policy.get("denied_hosts") or [],
    }


def _evaluation_environment_provider(request: dict[str, Any], *, experiment: dict[str, Any] | None) -> tuple[str | None, dict[str, Any]]:
    config = (experiment or {}).get("config") or {}
    env_config = config.get("environment") if isinstance(config.get("environment"), dict) else {}
    docker_config = config.get("docker") if isinstance(config.get("docker"), dict) else {}
    provider = request.get("requested_environment_provider") or config.get("environment_provider") or env_config.get("provider") or env_config.get("kind")
    provider_config: dict[str, Any] = {}
    provider_config.update(env_config)
    provider_config.update(docker_config)
    if provider:
        provider_config["provider"] = provider
    return (str(provider) if provider else None), provider_config


def _official_overlay_submit_allowed(experiment: dict[str, Any] | None) -> bool:
    policy = (experiment or {}).get("policy") or {}
    config = (experiment or {}).get("config") or {}
    candidates = [
        policy.get("environments") if isinstance(policy.get("environments"), dict) else {},
        policy.get("environment") if isinstance(policy.get("environment"), dict) else {},
        config.get("environments") if isinstance(config.get("environments"), dict) else {},
        config.get("environment") if isinstance(config.get("environment"), dict) else {},
    ]
    for item in candidates:
        if item.get("allow_official_overlay_submit") is True or item.get("allow_submit_overlay") is True:
            return True
    return False


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
