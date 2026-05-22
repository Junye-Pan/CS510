from __future__ import annotations

import hashlib
import json
import re
import shutil
import tarfile
from pathlib import Path
from typing import Any

from agentic_opt.common.atomic import atomic_write_text
from agentic_opt.common.ids import make_run_id
from agentic_opt.task_api import candidate_spec_for
from agentic_opt.task_registry import get_task

from .environments import EnvironmentService
from .evaluation import EvaluationService
from .jobs import JobService
from .object_store import S3CompatibleObjectStore
from .repository import ControlPlaneRepository
from .task_context import ensure_task_context_snapshot
from .telemetry import TelemetryService
from .trace_exports import TraceExportService
from .traces import AgentTraceService


def task_contract(task_id: str) -> dict[str, Any]:
    task = get_task(task_id)
    metadata = task.metadata
    spec = candidate_spec_for(task)
    task_md = (task.public_dir / "TASK.md").read_text(encoding="utf-8")
    public_contract = (task.public_dir / "public_contract.md").read_text(encoding="utf-8")
    public_files = _read_public_context_files(task.public_dir)
    task_knowledge = _task_knowledge_inventory(task_id=metadata.task_id, public_dir=task.public_dir)
    return {
        "task_id": metadata.task_id,
        "title": metadata.title,
        "public_context": {
            "task_markdown": task_md,
            "public_contract_markdown": public_contract,
            "public_dir": str(task.public_dir),
            "public_files": public_files,
            "research_directions": _load_research_directions(task.public_dir, public_files=public_files),
        },
        "task_knowledge": task_knowledge,
        "candidate_contract": {
            "entrypoint_name": spec.entrypoint_name,
            "workspace_entrypoint": spec.workspace_entrypoint.as_posix(),
            "candidate_root": spec.candidate_root,
            "public_seed_root": spec.public_seed_root,
            "description": spec.description,
        },
        "validation_contract": {"operation": "verify", "entry_path_field": "entry_path"},
        "probe_contract": {"operation": "probe", "entry_path_field": "entry_path", "kind_field": "kind"},
        "evaluation_contract": {
            "operation": "submit",
            "entry_path_field": "entry_path",
            "server_owned": True,
            "async_default": True,
        },
        "runtime_policy": task.runtime_spec.to_jsonable(),
        "artifact_policy": {
            "candidate_snapshots": True,
            "large_artifacts": "artifact_registry",
            "official_results": "evaluation_service",
        },
    }


def _read_public_context_files(public_dir: Path) -> list[dict[str, Any]]:
    allowed_suffixes = {".md", ".txt", ".json", ".yaml", ".yml"}
    max_bytes = 200_000
    files: list[dict[str, Any]] = []
    for path in sorted(public_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in allowed_suffixes:
            continue
        size_bytes = path.stat().st_size
        relative_path = path.relative_to(public_dir).as_posix()
        if relative_path == "knowledge" or relative_path.startswith("knowledge/"):
            continue
        if size_bytes > max_bytes:
            files.append(
                {
                    "path": relative_path,
                    "media_type": _media_type_for(path),
                    "size_bytes": size_bytes,
                    "truncated": True,
                    "content": path.read_text(encoding="utf-8", errors="replace")[:max_bytes],
                }
            )
            continue
        files.append(
            {
                "path": relative_path,
                "media_type": _media_type_for(path),
                "size_bytes": size_bytes,
                "truncated": False,
                "content": path.read_text(encoding="utf-8", errors="replace"),
            }
        )
    return files


def _load_research_directions(public_dir: Path, *, public_files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_path = {item["path"]: item for item in public_files}
    manifest_item = by_path.get("research_directions/manifest.json")
    if manifest_item is None:
        return []
    manifest = json.loads(str(manifest_item.get("content") or "{}"))
    directions: list[dict[str, Any]] = []
    for item in manifest.get("directions") or []:
        if not isinstance(item, dict):
            continue
        direction = dict(item)
        doc_path = direction.get("doc_path")
        if isinstance(doc_path, str):
            doc_file = by_path.get(doc_path)
            if doc_file is not None:
                direction["doc_markdown"] = doc_file.get("content") or ""
        directions.append(direction)
    return directions


def _media_type_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".md":
        return "text/markdown"
    if suffix == ".json":
        return "application/json"
    if suffix == ".pdf":
        return "application/pdf"
    if suffix == ".py":
        return "text/x-python"
    if suffix in {".yaml", ".yml"}:
        return "application/yaml"
    return "text/plain"


def _task_knowledge_inventory(*, task_id: str, public_dir: Path) -> dict[str, Any]:
    root = public_dir / "knowledge"
    workspace_path = "task/knowledge"
    if not root.exists():
        return {
            "available": False,
            "task_id": task_id,
            "source_path": str(root),
            "workspace_path": workspace_path,
            "digest": None,
            "file_count": 0,
            "size_bytes": 0,
            "files": [],
            "manifest": None,
        }
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"task knowledge must be a directory: {root}")
    resolved_root = root.resolve()
    manifest = _load_task_knowledge_manifest(resolved_root)
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in resolved_root.rglob("*") if item.is_file()):
        if path.is_symlink():
            raise PermissionError(f"task knowledge may not contain symlinks: {path}")
        relative = path.relative_to(resolved_root).as_posix()
        files.append(
            {
                "knowledge_file_id": f"task_knowledge_{_safe_token(task_id)}_{_safe_token(relative)}",
                "task_id": task_id,
                "relative_path": relative,
                "source_path": str(path),
                "workspace_path": f"{workspace_path}/{relative}",
                "media_type": _media_type_for(path),
                "digest": _digest_file(path),
                "size_bytes": path.stat().st_size,
                "metadata": _manifest_item_for_path(manifest, relative),
            }
        )
    return {
        "available": True,
        "task_id": task_id,
        "source_path": str(resolved_root),
        "workspace_path": workspace_path,
        "digest": _digest_directory(resolved_root),
        "file_count": len(files),
        "size_bytes": sum(int(item["size_bytes"]) for item in files),
        "files": files,
        "manifest": manifest,
    }


def _load_task_knowledge_manifest(root: Path) -> dict[str, Any] | None:
    path = root / "manifest.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("task knowledge manifest must be a JSON object")
    items = manifest.get("items")
    if items is None:
        return manifest
    if not isinstance(items, list):
        raise ValueError("task knowledge manifest items must be a list when present")
    for raw in items:
        if not isinstance(raw, dict):
            raise ValueError("task knowledge manifest items must be objects")
        relative = raw.get("path")
        if not isinstance(relative, str) or not relative:
            raise ValueError("task knowledge manifest item path is required")
        target = _safe_relative_task_knowledge_path(root, relative)
        if not target.exists():
            raise FileNotFoundError(target)
    return manifest


def _manifest_item_for_path(manifest: dict[str, Any] | None, relative: str) -> dict[str, Any]:
    if not manifest:
        return {}
    for raw in manifest.get("items") or []:
        if isinstance(raw, dict) and raw.get("path") == relative:
            return {key: value for key, value in raw.items() if key != "path"}
    return {}


def _safe_relative_task_knowledge_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ValueError(f"unsafe task knowledge manifest path: {relative}")
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        raise PermissionError(f"task knowledge manifest path escapes {root}: {relative}")
    return target


class ControlPlaneService:
    def __init__(
        self,
        *,
        repository: ControlPlaneRepository,
        artifact_root: Path,
        job_root: Path,
        database_path: Path,
        environment_root: Path | None = None,
    ) -> None:
        self.repository = repository
        self.artifact_root = artifact_root.resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.database_path = database_path.resolve()
        self.environments = EnvironmentService(
            repository=repository,
            environment_root=environment_root or (self.artifact_root.parent / "envs"),
        )
        self.jobs = JobService(
            repository=repository,
            job_root=job_root,
            database_path=database_path,
            environments=self.environments,
        )
        self.evaluations = EvaluationService(
            repository=repository,
            jobs=self.jobs,
            environments=self.environments,
            database_path=database_path,
            artifact_root=self.artifact_root,
        )
        self.telemetry = TelemetryService(repository=repository, telemetry_root=artifact_root.parent / "telemetry")
        self.traces = AgentTraceService(repository=repository, artifact_root=self.artifact_root)
        self.trace_exports = TraceExportService(repository=repository, export_root=artifact_root.parent / "trace_exports")

    def close(self) -> None:
        self.jobs.close()

    def generate_assignments(
        self,
        *,
        experiment_id: str,
        count: int,
        worker_backend: str = "codex-local",
    ) -> list[dict[str, Any]]:
        experiment = self.repository.get_experiment(experiment_id)
        if experiment is None:
            raise KeyError(experiment_id)
        directions = _direction_plan_for_experiment(experiment)
        return self.repository.create_assignments(
            experiment_id=experiment_id,
            count=count,
            worker_backend=worker_backend,
            direction_plan=directions,
        )

    def register_path_artifact(self, payload: dict[str, Any]) -> dict[str, Any]:
        source = Path(payload["path"]).resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        kind = payload.get("kind") or ("directory" if source.is_dir() else "file")
        artifact_id = payload.get("artifact_id") or make_run_id("artifact")
        artifact_dir = self.artifact_root / artifact_id
        if artifact_dir.exists():
            raise FileExistsError(artifact_dir)
        content_path = artifact_dir / "content"
        manifest_path = artifact_dir / "manifest.json"
        artifact_dir.mkdir(parents=True, exist_ok=False)
        if source.is_dir():
            shutil.copytree(source, content_path, ignore=shutil.ignore_patterns(".git"))
            digest = _digest_directory(content_path)
            file_count = _count_files(content_path)
            size_bytes = _size_bytes(content_path)
        else:
            content_path.mkdir(parents=True, exist_ok=True)
            destination = content_path / source.name
            shutil.copy2(source, destination)
            digest = _digest_file(destination)
            file_count = 1
            size_bytes = destination.stat().st_size
            content_path = destination
        storage_provider = payload.get("storage_provider") or payload.get("artifact_store") or "local"
        remote_metadata: dict[str, Any] = {}
        artifact_uri = content_path.as_uri()
        if storage_provider == "s3":
            stored = S3CompatibleObjectStore(
                bucket=payload.get("bucket"),
                prefix=payload.get("prefix"),
                endpoint_url=payload.get("endpoint_url"),
                region_name=payload.get("region_name"),
            ).upload_path(source=content_path, artifact_id=artifact_id)
            artifact_uri = stored.uri
            remote_metadata = {"remote": stored.metadata}
        elif storage_provider != "local":
            raise ValueError(f"unknown artifact storage_provider: {storage_provider}")
        manifest = {
            "artifact_id": artifact_id,
            "kind": kind,
            "source_path": str(source),
            "content_path": str(content_path),
            "uri": artifact_uri,
            "storage_provider": storage_provider,
            "digest": digest,
            "size_bytes": size_bytes,
            "file_count": file_count,
            "metadata": {**(payload.get("metadata") or {}), **remote_metadata},
        }
        atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        record = self.repository.create_artifact(
            {
                "artifact_id": artifact_id,
                "experiment_id": payload.get("experiment_id"),
                "assignment_id": payload.get("assignment_id"),
                "attempt_id": payload.get("attempt_id"),
                "kind": kind,
                "uri": artifact_uri,
                "local_path": str(content_path),
                "digest": digest,
                "metadata": {
                    **(payload.get("metadata") or {}),
                    **remote_metadata,
                    "storage_provider": storage_provider,
                    "source_path": str(source),
                    "manifest_path": str(manifest_path),
                    "size_bytes": size_bytes,
                    "file_count": file_count,
                },
            }
        )
        self.repository.record_event(
            {
                "experiment_id": record.get("experiment_id"),
                "assignment_id": record.get("assignment_id"),
                "event_type": "artifact.registered",
                "summary": f"artifact registered: {artifact_id}",
                "payload": {"artifact_id": artifact_id, "kind": kind, "digest": digest},
            }
        )
        return record

    def create_evaluation(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.evaluations.create(payload)

    def create_attempt(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.repository.create_attempt(payload)

    def update_attempt(self, attempt_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.repository.update_attempt(attempt_id, payload)

    def attempt_context(self, attempt_id: str) -> dict[str, Any]:
        attempt = self.repository.get_attempt(attempt_id)
        if attempt is None:
            raise KeyError(attempt_id)
        evaluations = self.repository.list_evaluations(attempt_id=attempt_id)
        leaderboard_entries = [
            entry
            for evaluation in evaluations
            for entry in [self.repository.get_leaderboard_entry_for_evaluation(evaluation["evaluation_id"])]
            if entry is not None
        ]
        return {
            "attempt": attempt,
            "artifacts": self.repository.list_artifacts(attempt_id=attempt_id),
            "evaluations": evaluations,
            "jobs": self.repository.list_jobs(attempt_id=attempt_id),
            "telemetry_runs": self.repository.list_telemetry_runs(attempt_id=attempt_id),
            "leaderboard_entries": leaderboard_entries,
        }

    def run_analysis(self, experiment_id: str, *, leaderboard_limit: int = 500) -> dict[str, Any]:
        experiment = self.repository.get_experiment(experiment_id)
        if experiment is None:
            raise KeyError(experiment_id)
        assignments = self.repository.list_assignments(experiment_id=experiment_id)
        assignment_ids = {item["assignment_id"] for item in assignments}
        sessions = self.repository.list_sessions(experiment_id=experiment_id)
        attempts = self.repository.list_attempts(experiment_id=experiment_id)
        artifacts = self.repository.list_artifacts(experiment_id=experiment_id)
        evaluations = self.repository.list_evaluations(experiment_id=experiment_id)
        jobs = self.repository.list_jobs(experiment_id=experiment_id)
        telemetry_runs = self.repository.list_telemetry_runs(experiment_id=experiment_id)
        agent_traces = self.repository.list_agent_traces(experiment_id=experiment_id)
        trace_exports = self.repository.list_trace_export_runs(experiment_id=experiment_id)
        leaderboard_entries = self.repository.list_leaderboard_entries(
            experiment_id=experiment_id,
            limit=max(1, int(leaderboard_limit)),
        )
        findings = self.repository.list_findings(task_id=experiment["task_id"])
        notebook_checkpoints = [
            checkpoint
            for assignment_id in sorted(assignment_ids)
            for checkpoint in self.repository.list_notebook_checkpoints(assignment_id=assignment_id)
        ]
        return _build_run_analysis(
            experiment=experiment,
            assignments=assignments,
            sessions=sessions,
            attempts=attempts,
            artifacts=artifacts,
            evaluations=evaluations,
            jobs=jobs,
            telemetry_runs=telemetry_runs,
            agent_traces=agent_traces,
            trace_exports=trace_exports,
            leaderboard_entries=leaderboard_entries,
            findings=findings,
            notebook_checkpoints=notebook_checkpoints,
        )

    def register_agent_trace(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.traces.register_trace_directory(payload)

    def list_agent_traces(self, **filters: Any) -> list[dict[str, Any]]:
        return self.traces.list_traces(**filters)

    def agent_trace_context(self, trace_id: str) -> dict[str, Any]:
        return self.traces.trace_with_manifest(trace_id)

    def agent_trace_commands(self, trace_id: str, *, failed_only: bool = False, semantic_only: bool = False) -> dict[str, Any]:
        return self.traces.trace_commands(trace_id, failed_only=failed_only, semantic_only=semantic_only)

    def agent_trace_events(self, trace_id: str, *, query: str | None = None, limit: int = 200) -> dict[str, Any]:
        return self.traces.trace_events(trace_id, query=query, limit=limit)

    def search_agent_traces(self, *, query: str, **filters: Any) -> dict[str, Any]:
        return self.traces.search_traces(query=query, **filters)

    def create_trace_export(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.trace_exports.create_export(payload)

    def get_trace_export(self, trace_export_id: str) -> dict[str, Any]:
        return self.trace_exports.get_export(trace_export_id)

    def list_trace_exports(self, **filters: Any) -> list[dict[str, Any]]:
        return self.trace_exports.list_exports(**filters)

    def run_evaluation(self, evaluation_id: str) -> dict[str, Any]:
        return self.evaluations.run(evaluation_id)

    def evaluate_entry(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.create_evaluation({**payload, "async": False})

    def ensure_environment(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("environment_type") == "framework" or payload.get("kind") == "framework":
            return self.environments.ensure_framework_environment()
        task_id = payload.get("task_id")
        assignment_id = payload.get("assignment_id")
        assignment = self.repository.get_assignment(assignment_id) if assignment_id else None
        experiment_id = payload.get("experiment_id") or (assignment or {}).get("experiment_id")
        experiment = self.repository.get_experiment(experiment_id) if experiment_id else None
        if not task_id:
            task_id = (assignment or {}).get("task_id")
        if not task_id:
            raise ValueError("task_id or assignment_id is required")
        return self.environments.ensure_task_environment(
            task_id,
            experiment_id=experiment_id,
            provider=_environment_provider_from_payload(payload, experiment=experiment),
            provider_config=_environment_provider_config(payload, experiment=experiment),
        )

    def create_environment_overlay(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.environments.create_overlay(payload)

    def approve_environment_overlay(self, overlay_id: str) -> dict[str, Any]:
        return self.environments.approve_overlay(overlay_id)

    def export_environment_bundle(self, environment_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        environment = self.repository.get_environment(environment_id)
        if environment is None:
            raise KeyError(environment_id)
        export_root = self.artifact_root.parent / "reproducibility_bundles"
        export_root.mkdir(parents=True, exist_ok=True)
        bundle_id = payload.get("bundle_id") or make_run_id("env_bundle")
        staging_dir = export_root / bundle_id
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=False)
        _write_environment_bundle_staging(environment=environment, staging_dir=staging_dir)
        archive_path = export_root / f"{bundle_id}.tar.gz"
        if archive_path.exists():
            archive_path.unlink()
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(staging_dir, arcname=bundle_id)
        digest = _digest_file(archive_path)
        artifact = self.register_path_artifact(
            {
                "experiment_id": payload.get("experiment_id") or environment.get("experiment_id"),
                "kind": "environment_reproducibility_bundle",
                "path": str(archive_path),
                "metadata": {
                    "environment_id": environment_id,
                    "task_id": environment.get("task_id"),
                    "environment_provider": (environment.get("metadata") or {}).get("provider")
                    or (environment.get("spec") or {}).get("provider")
                    or (environment.get("spec") or {}).get("kind"),
                    "bundle_id": bundle_id,
                    "bundle_digest": digest,
                    "staging_dir": str(staging_dir),
                },
            }
        )
        return {
            "environment": environment,
            "bundle_id": bundle_id,
            "archive_path": str(archive_path),
            "digest": digest,
            "artifact": artifact,
        }

    def export_replay_bundle(self, evaluation_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        evaluation = self.repository.get_evaluation(evaluation_id)
        if evaluation is None:
            raise KeyError(evaluation_id)
        request = evaluation.get("request") or {}
        task_id = request.get("task_id") or payload.get("task_id")
        if not task_id:
            raise ValueError("evaluation request has no task_id")
        experiment = self.repository.get_experiment(evaluation.get("experiment_id")) if evaluation.get("experiment_id") else None
        assignment = self.repository.get_assignment(evaluation.get("assignment_id")) if evaluation.get("assignment_id") else None
        attempt = self.repository.get_attempt(evaluation.get("attempt_id")) if evaluation.get("attempt_id") else None
        artifact = self.repository.get_artifact(evaluation["artifact_id"]) if evaluation.get("artifact_id") else None
        environment = self.repository.get_environment(request.get("environment_id")) if request.get("environment_id") else None
        overlay = self.repository.get_environment_overlay(request.get("environment_overlay_id")) if request.get("environment_overlay_id") else None
        framework_environment = self.repository.get_environment(request.get("framework_environment_id")) if request.get("framework_environment_id") else None
        job = self.repository.get_job(evaluation["job_id"]) if evaluation.get("job_id") else None
        traces = self.repository.list_agent_traces(
            experiment_id=evaluation.get("experiment_id"),
            assignment_id=evaluation.get("assignment_id"),
            task_id=task_id,
        )
        network_events = self.repository.list_network_access_events(
            experiment_id=evaluation.get("experiment_id"),
            assignment_id=evaluation.get("assignment_id"),
            limit=200,
        )

        bundle_id = payload.get("bundle_id") or make_run_id("replay_bundle")
        export_root = self.artifact_root.parent / "replay_bundles"
        export_root.mkdir(parents=True, exist_ok=True)
        staging_dir = export_root / bundle_id
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=False)

        candidate = _copy_replay_candidate(artifact=artifact, staging_dir=staging_dir)
        environment_bundle = None
        if environment is not None:
            environment_dir = staging_dir / "environment"
            environment_dir.mkdir(parents=True, exist_ok=True)
            _write_environment_bundle_staging(environment=environment, staging_dir=environment_dir)
            environment_bundle = {
                "path": "environment",
                "environment_id": environment["environment_id"],
                "provider": _environment_provider_record(environment),
            }
        logs = _copy_replay_job_logs(job=job, staging_dir=staging_dir)
        task_context = _task_replay_context(task_id=task_id, state_root=self.artifact_root.parent)
        records = {
            "experiment": experiment,
            "assignment": assignment,
            "attempt": attempt,
            "evaluation": evaluation,
            "candidate_artifact": artifact,
            "environment": environment,
            "environment_overlay": overlay,
            "framework_environment": framework_environment,
            "job": job,
            "agent_traces": traces,
            "network_access_events": network_events,
        }
        for name, value in records.items():
            _write_json(staging_dir / "records" / f"{name}.json", value)
        manifest = {
            "schema_version": "agentic_opt.replay_bundle.v1",
            "bundle_id": bundle_id,
            "source_evaluation_id": evaluation_id,
            "source_evaluation_status": evaluation.get("status"),
            "task_id": task_id,
            "experiment_id": evaluation.get("experiment_id"),
            "assignment_id": evaluation.get("assignment_id"),
            "attempt_id": evaluation.get("attempt_id"),
            "kind": evaluation.get("kind") or request.get("kind"),
            "candidate": candidate,
            "environment": {
                "environment_id": request.get("environment_id"),
                "environment_overlay_id": request.get("environment_overlay_id"),
                "provider": request.get("environment_provider"),
                "kind": request.get("environment_kind"),
                "lock": request.get("environment_lock"),
                "runner": request.get("runner"),
                "bundle": environment_bundle,
            },
            "framework_environment": {
                "environment_id": request.get("framework_environment_id"),
                "lock": request.get("framework_environment_lock"),
                "runner": request.get("framework_runner"),
            },
            "evaluation": {
                "evaluation_id": evaluation_id,
                "request": request,
                "status": evaluation.get("status"),
                "valid": evaluation.get("valid"),
                "score": evaluation.get("score"),
                "result": evaluation.get("result"),
                "public_feedback": evaluation.get("public_feedback"),
            },
            "task_context": task_context,
            "job_logs": logs,
            "record_paths": {name: f"records/{name}.json" for name in records},
        }
        _write_json(staging_dir / "manifest.json", manifest)
        _write_replay_readme(staging_dir=staging_dir, manifest=manifest)

        archive_path = export_root / f"{bundle_id}.tar.gz"
        if archive_path.exists():
            archive_path.unlink()
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(staging_dir, arcname=bundle_id)
        digest = _digest_file(archive_path)
        artifact_record = self.register_path_artifact(
            {
                "experiment_id": evaluation.get("experiment_id"),
                "assignment_id": evaluation.get("assignment_id"),
                "attempt_id": evaluation.get("attempt_id"),
                "kind": "replay_bundle",
                "path": str(archive_path),
                "metadata": {
                    "bundle_id": bundle_id,
                    "bundle_digest": digest,
                    "source_evaluation_id": evaluation_id,
                    "source_artifact_id": evaluation.get("artifact_id"),
                    "task_id": task_id,
                    "environment_id": request.get("environment_id"),
                    "environment_overlay_id": request.get("environment_overlay_id"),
                    "environment_provider": request.get("environment_provider"),
                    "framework_environment_id": request.get("framework_environment_id"),
                    "staging_dir": str(staging_dir),
                },
            }
        )
        return {
            "bundle_id": bundle_id,
            "source_evaluation": evaluation,
            "archive_path": str(archive_path),
            "digest": digest,
            "manifest": manifest,
            "artifact": artifact_record,
        }

    def replay_bundle(self, payload: dict[str, Any]) -> dict[str, Any]:
        artifact_id = payload.get("artifact_id") or payload.get("bundle_artifact_id")
        bundle_path = payload.get("bundle_path") or payload.get("path")
        artifact = self.repository.get_artifact(str(artifact_id)) if artifact_id else None
        if artifact is not None:
            bundle_path = artifact.get("local_path")
        if not bundle_path:
            raise ValueError("artifact_id or bundle_path is required")
        archive_path = Path(str(bundle_path)).resolve()
        if not archive_path.exists():
            raise FileNotFoundError(archive_path)
        run_id = payload.get("replay_run_id") or make_run_id("replay")
        replay_root = self.artifact_root.parent / "replay_runs" / run_id
        if replay_root.exists():
            shutil.rmtree(replay_root)
        replay_root.mkdir(parents=True, exist_ok=False)
        with tarfile.open(archive_path, "r:gz") as archive:
            _safe_extract_tar(archive, replay_root)
        bundle_root = _replay_bundle_root(replay_root)
        manifest = json.loads((bundle_root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "agentic_opt.replay_bundle.v1":
            raise ValueError(f"unsupported replay bundle schema: {manifest.get('schema_version')}")
        candidate = manifest.get("candidate") or {}
        entry_path, workspace_root = _replay_candidate_paths(bundle_root=bundle_root, candidate=candidate)
        _verify_replay_candidate_digest(bundle_root=bundle_root, candidate=candidate)
        self._restore_replay_environment_records(manifest, bundle_root=bundle_root)
        original_request = ((manifest.get("evaluation") or {}).get("request") or {})
        task_id = str(payload.get("task_id") or manifest.get("task_id") or original_request.get("task_id"))
        if not task_id:
            raise ValueError("replay bundle has no task_id")
        publish_leaderboard = bool(payload.get("publish_leaderboard", False))
        replay_payload = {
            "experiment_id": payload.get("experiment_id") or manifest.get("experiment_id") or original_request.get("experiment_id"),
            "assignment_id": payload.get("assignment_id") or manifest.get("assignment_id") or original_request.get("assignment_id"),
            "attempt_id": payload.get("attempt_id") or manifest.get("attempt_id") or original_request.get("attempt_id"),
            "task_id": task_id,
            "kind": payload.get("kind") or manifest.get("kind") or original_request.get("kind") or "submit",
            "entry_path": str(entry_path),
            "workspace_root": str(workspace_root),
            "environment_id": payload.get("environment_id") or ((manifest.get("environment") or {}).get("environment_id")),
            "environment_overlay_id": payload.get("environment_overlay_id") or ((manifest.get("environment") or {}).get("environment_overlay_id")),
            "environment_provider": payload.get("environment_provider") or ((manifest.get("environment") or {}).get("provider")),
            "async": bool(payload.get("async", False)),
            "snapshot_candidate": bool(payload.get("snapshot_candidate", True)),
            "publish_leaderboard": publish_leaderboard,
            "count_budget": bool(payload.get("count_budget", publish_leaderboard)),
            "expected_task_context_digest": ((manifest.get("task_context") or {}).get("digest")),
            "replay": {
                "run_id": run_id,
                "bundle_artifact_id": artifact_id,
                "bundle_path": str(archive_path),
                "bundle_digest": artifact.get("digest") if artifact else _digest_file(archive_path),
                "source_evaluation_id": manifest.get("source_evaluation_id"),
            },
        }
        if not replay_payload["experiment_id"]:
            raise ValueError("replay requires experiment_id in payload or bundle")
        replay_evaluation = self.create_evaluation(replay_payload)
        return {
            "replay_run_id": run_id,
            "bundle_artifact": artifact,
            "bundle_path": str(archive_path),
            "extracted_path": str(bundle_root),
            "manifest": manifest,
            "evaluation": replay_evaluation,
        }

    def _restore_replay_environment_records(self, manifest: dict[str, Any], *, bundle_root: Path) -> None:
        records = manifest.get("record_paths") or {}
        environment_path = bundle_root / str(records.get("environment") or "")
        if environment_path.exists():
            environment = json.loads(environment_path.read_text(encoding="utf-8"))
            if isinstance(environment, dict) and environment.get("environment_id") and self.repository.get_environment(environment["environment_id"]) is None:
                self.repository.upsert_environment(environment)
        framework_path = bundle_root / str(records.get("framework_environment") or "")
        if framework_path.exists():
            framework_environment = json.loads(framework_path.read_text(encoding="utf-8"))
            if (
                isinstance(framework_environment, dict)
                and framework_environment.get("environment_id")
                and self.repository.get_environment(framework_environment["environment_id"]) is None
            ):
                self.repository.upsert_environment(framework_environment)
        overlay_path = bundle_root / str(records.get("environment_overlay") or "")
        if overlay_path.exists():
            overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
            if isinstance(overlay, dict) and overlay.get("overlay_id") and self.repository.get_environment_overlay(overlay["overlay_id"]) is None:
                self.repository.create_environment_overlay(overlay)

    def checkout_incumbent(self, payload: dict[str, Any]) -> dict[str, Any]:
        experiment_id = payload.get("experiment_id")
        task_id = payload.get("task_id")
        direction_id = payload.get("direction_id")
        destination = Path(payload["destination_path"]).resolve()
        force = bool(payload.get("force"))
        incumbent = self.repository.get_incumbent(
            experiment_id=experiment_id,
            task_id=task_id,
            direction_id=direction_id,
        )
        if incumbent is None:
            raise KeyError("incumbent not found")
        artifact_id = incumbent.get("artifact_id")
        if not artifact_id:
            raise ValueError("incumbent has no artifact_id to checkout")
        artifact = self.repository.get_artifact(artifact_id)
        if artifact is None:
            raise KeyError(artifact_id)
        local_path = Path(artifact["local_path"]).resolve()
        if destination.exists():
            if not force:
                raise FileExistsError(destination)
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        if local_path.is_dir():
            shutil.copytree(local_path, destination)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, destination)
        return {
            "incumbent": incumbent,
            "artifact": artifact,
            "destination_path": str(destination),
            "entry_relative_path": (artifact.get("metadata") or {}).get("entry_relative_path"),
        }

    def context_for_assignment(self, assignment_id: str) -> dict[str, Any]:
        context = _worker_visible_context(self.repository.context_for_assignment(assignment_id))
        assignment = context["assignment"]
        context["task_knowledge"] = self.task_knowledge_inventory(task_id=assignment["task_id"])
        context["network_policy"] = self.network_policy({"assignment_id": assignment_id})
        return context

    def publish_shared_tool(self, payload: dict[str, Any]) -> dict[str, Any]:
        source = Path(payload["path"]).resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        name = payload.get("name") or source.stem
        artifact = self.register_path_artifact(
            {
                "experiment_id": payload.get("experiment_id"),
                "assignment_id": payload.get("assignment_id"),
                "kind": "shared_tool",
                "path": str(source),
                "metadata": {
                    "tool_name": name,
                    "tool_description": payload.get("description") or "",
                    "entrypoint": payload.get("entrypoint"),
                    "session_id": payload.get("session_id"),
                    "agent_id": payload.get("agent_id"),
                },
            }
        )
        record = self.repository.create_shared_tool(
            {
                "name": name,
                "description": payload.get("description") or "",
                "task_id": payload.get("task_id"),
                "experiment_id": payload.get("experiment_id"),
                "assignment_id": payload.get("assignment_id"),
                "session_id": payload.get("session_id"),
                "agent_id": payload.get("agent_id"),
                "scope": payload.get("scope") or "task",
                "artifact_id": artifact["artifact_id"],
                "entrypoint": payload.get("entrypoint"),
                "version": payload.get("version") or "1",
                "digest": artifact.get("digest"),
                "runtime_requirements": payload.get("runtime_requirements") or [],
                "metadata": {
                    **(payload.get("metadata") or {}),
                    "artifact": {
                        "uri": artifact.get("uri"),
                        "local_path": artifact.get("local_path"),
                    },
                },
            }
        )
        return {**record, "artifact": artifact}

    def checkout_shared_tool(self, tool_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        tool = self.repository.get_shared_tool(tool_id)
        if tool is None:
            raise KeyError(tool_id)
        artifact = self.repository.get_artifact(tool["artifact_id"])
        if artifact is None:
            raise KeyError(tool["artifact_id"])
        destination = Path(payload["destination_path"]).resolve()
        _copy_materialized_path(
            source=Path(artifact["local_path"]).resolve(),
            destination=destination,
            force=bool(payload.get("force")),
            read_only=False,
        )
        return {"tool": tool, "artifact": artifact, "destination_path": str(destination)}

    def bootstrap_workspace(self, assignment_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        assignment = self.repository.get_assignment(assignment_id)
        if assignment is None:
            raise KeyError(assignment_id)
        experiment = self.repository.get_experiment(assignment["experiment_id"])
        if experiment is None:
            raise KeyError(assignment["experiment_id"])
        workspace_root = Path(payload["workspace_root"]).resolve()
        task = get_task(assignment["task_id"])
        spec = candidate_spec_for(task)
        entry_path = Path(payload.get("entry_path") or (workspace_root / spec.workspace_entrypoint)).resolve()
        workspace_root.mkdir(parents=True, exist_ok=True)
        seed = self.resolve_workspace_seed(assignment_id)
        seed_materialized = dict(seed)
        if seed.get("artifact_id"):
            artifact = self.repository.get_artifact(str(seed["artifact_id"]))
            if artifact is None:
                seed_materialized = {
                    "kind": "public_seed",
                    "source": "fallback",
                    "reason": f"seed artifact not found: {seed['artifact_id']}",
                    "materialized": False,
                    "destination_path": str(entry_path),
                }
            elif not artifact.get("local_path"):
                seed_materialized = {
                    "kind": "public_seed",
                    "source": "fallback",
                    "reason": f"seed artifact has no local_path: {seed['artifact_id']}",
                    "materialized": False,
                    "destination_path": str(entry_path),
                }
            else:
                _materialize_candidate_artifact(
                    artifact=artifact,
                    workspace_root=workspace_root,
                    entry_path=entry_path,
                )
                seed_materialized = {
                    **seed_materialized,
                    "artifact": artifact,
                    "materialized": True,
                    "destination_path": str(entry_path),
                }
        else:
            seed_materialized = {
                **seed_materialized,
                "materialized": False,
                "destination_path": str(entry_path),
            }

        checked_out_tools = self.checkout_top_shared_tools(
            assignment_id,
            workspace_root=workspace_root,
            limit=_auto_checkout_tool_limit(assignment=assignment, experiment=experiment, payload=payload),
        )
        context_snapshot = self.context_for_assignment(assignment_id)
        task_context = ensure_task_context_snapshot(task_id=assignment["task_id"], state_root=self.artifact_root.parent)
        bootstrap = {
            "assignment_id": assignment_id,
            "experiment_id": assignment["experiment_id"],
            "task_id": assignment["task_id"],
            "session_id": payload.get("session_id"),
            "workspace_root": str(workspace_root),
            "entry_path": str(entry_path),
            "workspace_seed": seed_materialized,
            "checked_out_tools": checked_out_tools,
            "context_snapshot": context_snapshot,
            "task_contract": task_contract(assignment["task_id"]),
            "task_context": task_context,
        }
        event_payload = {key: value for key, value in bootstrap.items() if key not in {"context_snapshot", "task_contract"}}
        self.repository.record_event(
            {
                "experiment_id": assignment["experiment_id"],
                "assignment_id": assignment_id,
                "session_id": payload.get("session_id"),
                "task_id": assignment["task_id"],
                "agent_id": assignment["agent_id"],
                "event_type": "worker.workspace.bootstrapped",
                "summary": _workspace_bootstrap_summary(event_payload),
                "payload": event_payload,
            }
        )
        return bootstrap

    def resolve_workspace_seed(self, assignment_id: str) -> dict[str, Any]:
        assignment = self.repository.get_assignment(assignment_id)
        if assignment is None:
            raise KeyError(assignment_id)
        seed_policy = assignment.get("workspace_seed") or {}
        if seed_policy.get("artifact_id"):
            artifact_id = str(seed_policy["artifact_id"])
            artifact = self.repository.get_artifact(artifact_id)
            return {
                "kind": "artifact",
                "source": "assignment_workspace_seed",
                "artifact_id": artifact_id,
                "reason": None if artifact is not None else f"artifact not found: {artifact_id}",
            }
        mode = str(seed_policy.get("mode") or "auto")
        experiment_id = assignment["experiment_id"]
        direction_id = assignment.get("direction_id")
        fallbacks: list[dict[str, Any]] = []
        if mode in {"public_seed", "public", "fresh"}:
            return {
                "kind": "public_seed",
                "source": "assignment_workspace_seed",
                "reason": f"workspace_seed mode is {mode}",
            }
        if direction_id and mode not in {"global_incumbent", "global"}:
            direction_incumbent = self.repository.get_incumbent(experiment_id=experiment_id, direction_id=direction_id)
            if direction_incumbent is not None and direction_incumbent.get("artifact_id"):
                return _seed_from_incumbent(
                    kind="direction_incumbent",
                    source="leaderboard",
                    incumbent=direction_incumbent,
                    fallback_chain=fallbacks,
                )
            fallbacks.append(
                {
                    "kind": "direction_incumbent",
                    "direction_id": direction_id,
                    "reason": "no direction incumbent artifact",
                }
            )
        if mode not in {"direction_incumbent", "direction"}:
            global_incumbent = self.repository.get_incumbent(experiment_id=experiment_id)
            if global_incumbent is not None and global_incumbent.get("artifact_id"):
                return _seed_from_incumbent(
                    kind="global_incumbent",
                    source="leaderboard",
                    incumbent=global_incumbent,
                    fallback_chain=fallbacks,
                )
            fallbacks.append({"kind": "global_incumbent", "reason": "no global incumbent artifact"})
        return {
            "kind": "public_seed",
            "source": "public_task_seed",
            "reason": "no usable incumbent artifact",
            "fallback_chain": fallbacks,
        }

    def checkout_top_shared_tools(
        self,
        assignment_id: str,
        *,
        workspace_root: Path,
        limit: int,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        assignment = self.repository.get_assignment(assignment_id)
        if assignment is None:
            raise KeyError(assignment_id)
        tools = self.repository.list_shared_tools(
            task_id=assignment["task_id"],
            experiment_id=assignment["experiment_id"],
        )[:limit]
        checked_out: list[dict[str, Any]] = []
        used_names: set[str] = set()
        for tool in tools:
            name = _safe_token(str(tool["name"]))
            destination_name = name
            if destination_name in used_names:
                destination_name = f"{name}_{_safe_token(str(tool['tool_id']))}"
            used_names.add(destination_name)
            destination = workspace_root / "shared_tools" / destination_name
            result = self.checkout_shared_tool(
                str(tool["tool_id"]),
                {"destination_path": str(destination), "force": True},
            )
            checked_out.append(
                {
                    "tool_id": tool["tool_id"],
                    "name": tool["name"],
                    "description": tool.get("description") or "",
                    "entrypoint": tool.get("entrypoint"),
                    "destination_path": result["destination_path"],
                    "artifact_id": tool.get("artifact_id"),
                }
            )
        return checked_out

    def task_knowledge_inventory(self, *, task_id: str) -> dict[str, Any]:
        task = get_task(task_id)
        return _task_knowledge_inventory(task_id=task_id, public_dir=task.public_dir)

    def network_policy(self, payload: dict[str, Any]) -> dict[str, Any]:
        assignment = self.repository.get_assignment(payload["assignment_id"]) if payload.get("assignment_id") else None
        session = self.repository.get_session(payload["session_id"]) if payload.get("session_id") else None
        experiment_id = payload.get("experiment_id") or (assignment or session or {}).get("experiment_id")
        experiment = self.repository.get_experiment(experiment_id) if experiment_id else None
        if experiment is None and assignment is not None:
            experiment = self.repository.get_experiment(assignment["experiment_id"])
        if experiment is None:
            raise ValueError("experiment_id, assignment_id, or session_id is required")
        raw_policy = ((experiment.get("policy") or {}).get("network") or (experiment.get("config") or {}).get("network") or {})
        control_plane = raw_policy.get("control_plane") or raw_policy.get("control_plane_network") or "allow"
        external = raw_policy.get("external_internet")
        if external is None:
            external = "deny" if raw_policy.get("allow_external_internet") is False else "allow"
        provider = (assignment or session or {}).get("worker_backend") or payload.get("worker_backend") or "unknown"
        docker_enforced = provider in {"docker_image", "docker", "local-docker", "local-docker-strict"}
        control_plane_allowed = str(control_plane) == "allow"
        session_relay = ((session or {}).get("details") or {}).get("control_plane_relay") if session else None
        session_network_enforcement = ((session or {}).get("details") or {}).get("network_enforcement") if session else None
        if not isinstance(session_network_enforcement, dict):
            session_network_enforcement = {}
        external_enforced = (
            bool(session_network_enforcement["external_internet_enforced"])
            if "external_internet_enforced" in session_network_enforcement
            else _provider_enforces_external_network(provider)
        )
        policy_weakened = (
            bool(session_network_enforcement["policy_weakened"])
            if "policy_weakened" in session_network_enforcement
            else str(external) == "deny" and not external_enforced
        )
        control_plane_relay_required = (
            bool(session_network_enforcement["control_plane_requires_relay"])
            if "control_plane_requires_relay" in session_network_enforcement
            else docker_enforced and str(external) == "deny" and control_plane_allowed
        )
        control_plane_relay_configured = bool(raw_policy.get("control_plane_relay") or session_relay)
        control_plane_available = control_plane_allowed and (not control_plane_relay_required or control_plane_relay_configured)
        return {
            "experiment_id": experiment["experiment_id"],
            "assignment_id": (assignment or session or {}).get("assignment_id"),
            "session_id": (session or {}).get("session_id") or payload.get("session_id"),
            "task_id": experiment["task_id"],
            "policy": {
                "control_plane": control_plane,
                "external_internet": external,
                "package_indexes": raw_policy.get("package_indexes") or "policy",
                "allowed_hosts": raw_policy.get("allowed_hosts") or ["127.0.0.1", "localhost"],
                "denied_hosts": raw_policy.get("denied_hosts") or [],
                "audit_external_attempts": bool(raw_policy.get("audit_external_attempts", True)),
                "outbound_proxy": raw_policy.get("outbound_proxy") or raw_policy.get("audit_proxy"),
            },
            "enforcement": {
                "worker_backend": provider,
                "control_plane_enforced": True,
                "external_internet_enforced": external_enforced,
                "outbound_audit_proxy_available": docker_enforced and str(external) == "audit",
                "outbound_audit_proxy_mode": session_network_enforcement.get("outbound_audit_mode")
                or ("docker_env_proxy" if docker_enforced and str(external) == "audit" else None),
                "policy_weakened": policy_weakened,
                "enforcement_mode": "docker_network_none" if session_network_enforcement.get("docker_network_mode") == "none" else "docker_network_none" if docker_enforced and str(external) == "deny" else None,
                "control_plane_relay_required": control_plane_relay_required,
                "control_plane_relay_configured": control_plane_relay_configured,
                "control_plane_relay": session_relay or raw_policy.get("control_plane_relay"),
                "control_plane_available": control_plane_available,
                "operationally_ready": control_plane_available and not policy_weakened,
                "reason": session_network_enforcement.get("policy_weakened_reason")
                or (
                    "worker backend exposes coarse network access for control-plane HTTP"
                    if policy_weakened
                    else None
                ),
                "network_enforcement": session_network_enforcement,
            },
        }


def _direction_plan_for_experiment(experiment: dict[str, Any]) -> list[dict[str, Any]]:
    directions = task_contract(experiment["task_id"])["public_context"].get("research_directions") or []
    if not directions:
        return []
    policy = experiment.get("policy") or {}
    config = experiment.get("config") or {}
    raw_ids = (
        ((policy.get("directions") or {}).get("enabled_direction_ids"))
        or policy.get("enabled_direction_ids")
        or config.get("direction_ids")
        or config.get("enabled_direction_ids")
    )
    if not raw_ids:
        return list(directions)
    enabled_ids = {str(item) for item in raw_ids}
    return [direction for direction in directions if str(direction.get("direction_id")) in enabled_ids]


def _safe_token(raw: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._-")
    return token or "item"


def _seed_from_incumbent(
    *,
    kind: str,
    source: str,
    incumbent: dict[str, Any],
    fallback_chain: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "kind": kind,
        "source": source,
        "leaderboard_entry_id": incumbent.get("leaderboard_entry_id"),
        "artifact_id": incumbent.get("artifact_id"),
        "score": incumbent.get("score"),
        "direction_id": incumbent.get("direction_id"),
        "evaluation_id": incumbent.get("evaluation_id"),
        "fallback_chain": list(fallback_chain),
    }


def _auto_checkout_tool_limit(*, assignment: dict[str, Any], experiment: dict[str, Any], payload: dict[str, Any]) -> int:
    if payload.get("shared_tool_limit") is not None:
        return max(0, int(payload["shared_tool_limit"]))
    seed_policy = assignment.get("workspace_seed") or {}
    if seed_policy.get("shared_tool_limit") is not None:
        return max(0, int(seed_policy["shared_tool_limit"]))
    config = experiment.get("config") or {}
    policy = experiment.get("policy") or {}
    shared_tools_config = config.get("shared_tools") or policy.get("shared_tools") or {}
    if isinstance(shared_tools_config, dict) and shared_tools_config.get("auto_checkout_limit") is not None:
        return max(0, int(shared_tools_config["auto_checkout_limit"]))
    return 5


def _workspace_bootstrap_summary(bootstrap: dict[str, Any]) -> str:
    seed = bootstrap.get("workspace_seed") or {}
    seed_kind = seed.get("kind") or "unknown"
    tool_count = len(bootstrap.get("checked_out_tools") or [])
    if seed.get("artifact_id"):
        return f"workspace bootstrapped from {seed_kind} artifact {seed['artifact_id']} with {tool_count} shared tools"
    return f"workspace bootstrapped from {seed_kind} with {tool_count} shared tools"


def _copy_replay_candidate(*, artifact: dict[str, Any] | None, staging_dir: Path) -> dict[str, Any] | None:
    if artifact is None or not artifact.get("local_path"):
        return None
    source = Path(artifact["local_path"]).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    candidate_root = staging_dir / "candidate" / "content"
    candidate_root.parent.mkdir(parents=True, exist_ok=True)
    metadata = artifact.get("metadata") or {}
    if source.is_dir():
        shutil.copytree(source, candidate_root)
        content_relative_path = "candidate/content"
        digest = _digest_directory(candidate_root)
        file_count = _count_files(candidate_root)
        size_bytes = _size_bytes(candidate_root)
        entry_relative_path = metadata.get("entry_relative_path")
    else:
        candidate_root.mkdir(parents=True, exist_ok=True)
        destination = candidate_root / source.name
        shutil.copy2(source, destination)
        content_relative_path = f"candidate/content/{source.name}"
        digest = _digest_file(destination)
        file_count = 1
        size_bytes = destination.stat().st_size
        entry_relative_path = source.name
    return {
        "artifact_id": artifact["artifact_id"],
        "kind": artifact.get("kind"),
        "content_relative_path": content_relative_path,
        "entry_relative_path": entry_relative_path,
        "digest": digest,
        "source_digest": artifact.get("digest"),
        "size_bytes": size_bytes,
        "file_count": file_count,
        "metadata": {
            "entry_relative_path": metadata.get("entry_relative_path"),
            "candidate_root": metadata.get("candidate_root"),
            "storage_provider": metadata.get("storage_provider"),
        },
    }


def _copy_replay_job_logs(*, job: dict[str, Any] | None, staging_dir: Path) -> dict[str, Any]:
    if job is None:
        return {}
    copied: dict[str, Any] = {"job_id": job.get("job_id"), "files": {}}
    logs_dir = staging_dir / "job_logs"
    outputs = job.get("outputs") or {}
    for key in ("stdout_path", "stderr_path", "command_path"):
        raw_path = outputs.get(key)
        if not raw_path:
            continue
        source = Path(str(raw_path))
        if not source.exists() or not source.is_file():
            continue
        logs_dir.mkdir(parents=True, exist_ok=True)
        destination = logs_dir / source.name
        shutil.copy2(source, destination)
        copied["files"][key] = {
            "path": destination.relative_to(staging_dir).as_posix(),
            "digest": _digest_file(destination),
            "size_bytes": destination.stat().st_size,
        }
    return copied


def _task_replay_context(*, task_id: str, state_root: Path | None = None) -> dict[str, Any]:
    task = get_task(task_id)
    public_dir = task.public_dir
    public_files = _read_public_context_files(public_dir)
    snapshot = ensure_task_context_snapshot(task_id=task_id, state_root=state_root) if state_root is not None else None
    return {
        "task_id": task_id,
        "public_dir": str(public_dir),
        "public_digest": _digest_public_context(public_dir),
        "snapshot": snapshot,
        "digest": (snapshot or {}).get("digest"),
        "public_files": [
            {
                "path": item["path"],
                "media_type": item["media_type"],
                "size_bytes": item["size_bytes"],
                "truncated": item["truncated"],
            }
            for item in public_files
        ],
        "task_knowledge": _task_knowledge_inventory(task_id=task_id, public_dir=public_dir),
    }


def _digest_public_context(public_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in public_dir.rglob("*") if item.is_file()):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(public_dir).as_posix()
        digest.update(relative.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_replay_readme(*, staging_dir: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Replay Bundle",
        "",
        f"- bundle_id: `{manifest.get('bundle_id')}`",
        f"- source_evaluation_id: `{manifest.get('source_evaluation_id')}`",
        f"- task_id: `{manifest.get('task_id')}`",
        f"- provider: `{(manifest.get('environment') or {}).get('provider')}`",
        f"- candidate_digest: `{((manifest.get('candidate') or {}).get('digest'))}`",
        "",
        "This archive packages the candidate snapshot, evaluation request/result,",
        "environment records, framework environment metadata, task context digests,",
        "job logs, trace references, and policy-related records needed for replay.",
        "",
    ]
    atomic_write_text(staging_dir / "README.md", "\n".join(lines))


def _safe_extract_tar(archive: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if not target.is_relative_to(destination):
            raise PermissionError(f"replay bundle member escapes destination: {member.name}")
        if member.issym() or member.islnk():
            raise PermissionError(f"replay bundle may not contain links: {member.name}")
    archive.extractall(destination)


def _replay_bundle_root(replay_root: Path) -> Path:
    manifest = replay_root / "manifest.json"
    if manifest.exists():
        return replay_root
    candidates = [item for item in replay_root.iterdir() if item.is_dir() and (item / "manifest.json").exists()]
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one replay bundle manifest under {replay_root}")
    return candidates[0]


def _replay_candidate_paths(*, bundle_root: Path, candidate: dict[str, Any]) -> tuple[Path, Path]:
    content_relative = candidate.get("content_relative_path")
    if not content_relative:
        raise ValueError("replay bundle has no candidate content")
    content_path = (bundle_root / str(content_relative)).resolve()
    if not content_path.is_relative_to(bundle_root.resolve()):
        raise PermissionError(f"candidate content escapes replay bundle: {content_relative}")
    candidate_root = ((candidate.get("metadata") or {}).get("candidate_root"))
    if content_path.is_dir():
        entry_relative = candidate.get("entry_relative_path")
        if not entry_relative:
            raise ValueError("directory candidate replay requires entry_relative_path")
        if candidate_root:
            workspace_root = bundle_root / "replay_workspace"
            destination = (workspace_root / str(candidate_root)).resolve()
            if destination.exists():
                shutil.rmtree(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(content_path, destination)
            entry_path = (destination / str(entry_relative)).resolve()
            if not entry_path.is_relative_to(destination):
                raise PermissionError(f"candidate entry escapes candidate root: {entry_relative}")
            return entry_path, workspace_root
        entry_path = (content_path / str(entry_relative)).resolve()
        if not entry_path.is_relative_to(content_path):
            raise PermissionError(f"candidate entry escapes candidate content: {entry_relative}")
        return entry_path, content_path
    return content_path, content_path.parent


def _verify_replay_candidate_digest(*, bundle_root: Path, candidate: dict[str, Any]) -> None:
    content_relative = candidate.get("content_relative_path")
    expected = candidate.get("digest")
    if not content_relative or not expected:
        raise ValueError("replay candidate digest metadata is incomplete")
    content_path = (bundle_root / str(content_relative)).resolve()
    actual = _digest_directory(content_path) if content_path.is_dir() else _digest_file(content_path)
    if actual != expected:
        raise ValueError(f"replay candidate digest mismatch: expected {expected}, got {actual}")


def _environment_provider_record(environment: dict[str, Any]) -> str | None:
    metadata = environment.get("metadata") or {}
    spec = environment.get("spec") or {}
    return metadata.get("provider") or spec.get("provider") or spec.get("kind")


def _materialize_candidate_artifact(*, artifact: dict[str, Any], workspace_root: Path, entry_path: Path) -> None:
    source = Path(artifact["local_path"]).resolve()
    metadata = artifact.get("metadata") or {}
    if not source.is_dir():
        _copy_materialized_path(source=source, destination=entry_path, force=True, read_only=False)
        return
    candidate_root = metadata.get("candidate_root")
    if candidate_root:
        destination = (workspace_root / str(candidate_root)).resolve()
        if not destination.is_relative_to(workspace_root):
            raise PermissionError(f"candidate root must stay inside workspace: {candidate_root}")
        _copy_materialized_path(source=source, destination=destination, force=True, read_only=False)
        return
    _copy_directory_contents(source=source, destination=entry_path.parent)


def _copy_directory_contents(*, source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for item in sorted(source.rglob("*")):
        relative = item.relative_to(source)
        target = destination / relative
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.copy2(item, target)


def _write_environment_bundle_staging(*, environment: dict[str, Any], staging_dir: Path) -> None:
    metadata = environment.get("metadata") or {}
    lock = environment.get("lock") or {}
    spec = environment.get("spec") or {}
    atomic_write_text(staging_dir / "environment.json", json.dumps(environment, indent=2, sort_keys=True) + "\n")
    atomic_write_text(staging_dir / "lock.json", json.dumps(lock, indent=2, sort_keys=True) + "\n")
    atomic_write_text(staging_dir / "spec.json", json.dumps(spec, indent=2, sort_keys=True) + "\n")
    atomic_write_text(
        staging_dir / "README.md",
        "\n".join(
            [
                "# Environment Reproducibility Bundle",
                "",
                f"- environment_id: `{environment.get('environment_id')}`",
                f"- task_id: `{environment.get('task_id')}`",
                f"- provider: `{metadata.get('provider') or spec.get('provider') or spec.get('kind')}`",
                f"- fingerprint: `{environment.get('fingerprint')}`",
                "",
                "This bundle records the control-plane environment record, lock, spec,",
                "provider metadata, and Docker build/preflight files when available.",
                "It is an export artifact; the SQLite control-plane record remains the",
                "source of semantic state.",
                "",
            ]
        ),
    )
    manifest_path = metadata.get("manifest_path")
    if manifest_path and Path(str(manifest_path)).exists():
        shutil.copy2(Path(str(manifest_path)), staging_dir / "runtime_manifest.json")
    build_context = metadata.get("build_context")
    if build_context and Path(str(build_context)).exists():
        dockerfile = Path(str(build_context)) / "Dockerfile"
        requirements = Path(str(build_context)) / "requirements.txt"
        if dockerfile.exists():
            shutil.copy2(dockerfile, staging_dir / "Dockerfile")
        if requirements.exists():
            shutil.copy2(requirements, staging_dir / "requirements.txt")
    host_root = metadata.get("host_root_path") or environment.get("root_path")
    if host_root and Path(str(host_root)).exists():
        for filename in ("docker_build_stdout.log", "docker_build_stderr.log", "import_preflight.json", "public_seed_preflight.json"):
            source = Path(str(host_root)) / filename
            if source.exists():
                shutil.copy2(source, staging_dir / filename)


_WORKER_HIDDEN_METADATA_KEYS = {
    "auto_continue",
    "budget_exhausted",
    "global_stop_condition",
    "resume",
    "resume_after_strict_safe_scoring_fix",
    "resume_budget_total_evaluator_runs",
    "resume_to_exhaust_eval_budget",
    "stop_condition",
}


def _build_run_analysis(
    *,
    experiment: dict[str, Any],
    assignments: list[dict[str, Any]],
    sessions: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    telemetry_runs: list[dict[str, Any]],
    agent_traces: list[dict[str, Any]],
    trace_exports: list[dict[str, Any]],
    leaderboard_entries: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    notebook_checkpoints: list[dict[str, Any]],
) -> dict[str, Any]:
    evaluations_by_id = {item["evaluation_id"]: item for item in evaluations}
    leaderboard_by_evaluation_id = {item["evaluation_id"]: item for item in leaderboard_entries}
    leaderboard_rank = {
        item["leaderboard_entry_id"]: index + 1
        for index, item in enumerate(sorted(leaderboard_entries, key=lambda entry: (-float(entry["score"]), entry["updated_at"], entry["leaderboard_entry_id"])))
    }

    attempt_nodes: list[dict[str, Any]] = []
    candidate_lineage: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []

    traces_by_observed_attempt = _traces_by_observed_id(agent_traces, "attempt_ids")
    findings_by_attempt = _records_by_mentioned_id(findings, "attempt")
    notebooks_by_session = _records_by_key(notebook_checkpoints, "session_id")
    artifacts_by_attempt = _records_by_key(artifacts, "attempt_id")
    evaluations_by_attempt = _records_by_key(evaluations, "attempt_id")
    jobs_by_attempt = _records_by_key(jobs, "attempt_id")
    telemetry_by_attempt = _records_by_key(telemetry_runs, "attempt_id")
    traces_by_session = _records_by_key(agent_traces, "session_id")

    for attempt in sorted(attempts, key=lambda item: (item["created_at"], item["attempt_id"])):
        attempt_id = attempt["attempt_id"]
        related_evaluations = evaluations_by_attempt.get(attempt_id, [])
        related_leaderboard = [
            leaderboard_by_evaluation_id[evaluation["evaluation_id"]]
            for evaluation in related_evaluations
            if evaluation["evaluation_id"] in leaderboard_by_evaluation_id
        ]
        best_score = max((float(entry["score"]) for entry in related_leaderboard), default=None)
        trace_ids = _unique_ids(
            [item["trace_id"] for item in traces_by_session.get(attempt.get("session_id"), [])]
            + [item["trace_id"] for item in traces_by_observed_attempt.get(attempt_id, [])]
        )
        node = {
            "attempt_id": attempt_id,
            "status": attempt["status"],
            "assignment_id": attempt.get("assignment_id"),
            "session_id": attempt.get("session_id"),
            "agent_id": attempt.get("agent_id"),
            "direction_id": attempt.get("direction_id"),
            "parent_attempt_id": attempt.get("parent_attempt_id"),
            "candidate_artifact_id": attempt.get("candidate_artifact_id"),
            "created_at": attempt["created_at"],
            "updated_at": attempt["updated_at"],
            "artifact_ids": [item["artifact_id"] for item in artifacts_by_attempt.get(attempt_id, [])],
            "evaluation_ids": [item["evaluation_id"] for item in related_evaluations],
            "job_ids": _unique_ids([item["job_id"] for item in jobs_by_attempt.get(attempt_id, [])] + _attached_job_ids(jobs, attempt_id)),
            "telemetry_ids": [item["telemetry_id"] for item in telemetry_by_attempt.get(attempt_id, [])],
            "trace_ids": trace_ids,
            "finding_ids": [item["finding_id"] for item in findings_by_attempt.get(attempt_id, [])],
            "notebook_checkpoint_ids": [item["checkpoint_id"] for item in notebooks_by_session.get(attempt.get("session_id"), [])],
            "leaderboard_entry_ids": [item["leaderboard_entry_id"] for item in related_leaderboard],
            "best_score": best_score,
        }
        attempt_nodes.append(node)
        candidate_lineage.append(
            {
                "attempt_id": attempt_id,
                "parent_attempt_id": attempt.get("parent_attempt_id"),
                "candidate_artifact_id": attempt.get("candidate_artifact_id"),
                "artifact_ids": node["artifact_ids"],
                "evaluation_ids": node["evaluation_ids"],
                "leaderboard_entry_ids": node["leaderboard_entry_ids"],
                "best_score": best_score,
                "trace_ids": trace_ids,
                "created_at": attempt["created_at"],
            }
        )
        if attempt.get("parent_attempt_id"):
            relationships.append(_relationship("attempt", attempt["parent_attempt_id"], "attempt", attempt_id, "parent_attempt"))
        if attempt.get("candidate_artifact_id"):
            relationships.append(_relationship("attempt", attempt_id, "artifact", attempt["candidate_artifact_id"], "candidate_artifact"))

    for artifact in artifacts:
        if artifact.get("attempt_id"):
            relationships.append(_relationship("attempt", artifact["attempt_id"], "artifact", artifact["artifact_id"], "attempt_artifact"))
    for evaluation in evaluations:
        if evaluation.get("attempt_id"):
            relationships.append(_relationship("attempt", evaluation["attempt_id"], "evaluation", evaluation["evaluation_id"], "attempt_evaluation"))
        if evaluation.get("artifact_id"):
            relationships.append(_relationship("artifact", evaluation["artifact_id"], "evaluation", evaluation["evaluation_id"], "evaluated_artifact"))
        if evaluation.get("job_id"):
            relationships.append(_relationship("evaluation", evaluation["evaluation_id"], "job", evaluation["job_id"], "evaluation_job"))
    for job in jobs:
        if job.get("attempt_id"):
            relationships.append(_relationship("attempt", job["attempt_id"], "job", job["job_id"], "launched_job"))
        for attachment in (job.get("details") or {}).get("attachments") or []:
            if attachment.get("attempt_id"):
                relationships.append(
                    _relationship(
                        "attempt",
                        attachment["attempt_id"],
                        "job",
                        job["job_id"],
                        "attached_job",
                        {"attachment_id": attachment.get("attachment_id"), "mode": attachment.get("mode"), "attached_at": attachment.get("attached_at")},
                    )
                )
    for telemetry in telemetry_runs:
        if telemetry.get("attempt_id"):
            relationships.append(_relationship("attempt", telemetry["attempt_id"], "telemetry", telemetry["telemetry_id"], "attempt_telemetry"))
        if telemetry.get("job_id"):
            relationships.append(_relationship("job", telemetry["job_id"], "telemetry", telemetry["telemetry_id"], "job_telemetry"))
        if telemetry.get("artifact_id"):
            relationships.append(_relationship("artifact", telemetry["artifact_id"], "telemetry", telemetry["telemetry_id"], "artifact_telemetry"))
    for trace in agent_traces:
        relationships.append(_relationship("session", trace["session_id"], "trace", trace["trace_id"], "session_trace"))
        if trace.get("artifact_id"):
            relationships.append(_relationship("trace", trace["trace_id"], "artifact", trace["artifact_id"], "trace_artifact"))
        observed = (trace.get("metadata") or {}).get("observed_ids") or {}
        for target_type, observed_key in (
            ("attempt", "attempt_ids"),
            ("artifact", "artifact_ids"),
            ("evaluation", "evaluation_ids"),
            ("job", "job_ids"),
            ("telemetry", "telemetry_ids"),
            ("trace", "trace_ids"),
        ):
            for target_id in observed.get(observed_key, []):
                relationships.append(_relationship("trace", trace["trace_id"], target_type, target_id, "trace_observed_id"))
    for entry in leaderboard_entries:
        relationships.append(_relationship("evaluation", entry["evaluation_id"], "leaderboard_entry", entry["leaderboard_entry_id"], "leaderboard_score"))
        if entry.get("artifact_id"):
            relationships.append(_relationship("artifact", entry["artifact_id"], "leaderboard_entry", entry["leaderboard_entry_id"], "leaderboard_artifact"))

    score_series = [
        {
            "leaderboard_entry_id": entry["leaderboard_entry_id"],
            "evaluation_id": entry["evaluation_id"],
            "attempt_id": (evaluations_by_id.get(entry["evaluation_id"]) or {}).get("attempt_id"),
            "artifact_id": entry.get("artifact_id"),
            "assignment_id": entry.get("assignment_id"),
            "direction_id": entry.get("direction_id"),
            "score": entry["score"],
            "rank": leaderboard_rank.get(entry["leaderboard_entry_id"]),
            "created_at": entry["created_at"],
            "updated_at": entry["updated_at"],
        }
        for entry in sorted(leaderboard_entries, key=lambda item: (item["created_at"], item["leaderboard_entry_id"]))
    ]
    incumbent = max(leaderboard_entries, key=lambda item: float(item["score"]), default=None)

    return {
        "schema_version": "agentic_opt.run_analysis.v1",
        "kind": "human_run_analysis",
        "experiment": experiment,
        "summary": {
            "assignment_count": len(assignments),
            "session_count": len(sessions),
            "attempt_count": len(attempts),
            "evaluation_count": len(evaluations),
            "artifact_count": len(artifacts),
            "job_count": len(jobs),
            "trace_count": len(agent_traces),
            "finding_count": len(findings),
            "notebook_checkpoint_count": len(notebook_checkpoints),
            "best_score": incumbent.get("score") if incumbent else None,
            "incumbent_artifact_id": incumbent.get("artifact_id") if incumbent else None,
        },
        "attempt_graph": {
            "nodes": attempt_nodes,
            "edges": _dedupe_relationships([item for item in relationships if item["source_type"] == "attempt" or item["target_type"] == "attempt"]),
        },
        "score_series": score_series,
        "candidate_lineage": candidate_lineage,
        "relationships": _dedupe_relationships(relationships),
        "entities": {
            "assignments": assignments,
            "sessions": sessions,
            "attempts": attempts,
            "artifacts": artifacts,
            "evaluations": evaluations,
            "jobs": jobs,
            "telemetry_runs": telemetry_runs,
            "agent_traces": agent_traces,
            "trace_exports": trace_exports,
            "leaderboard_entries": leaderboard_entries,
            "findings": findings,
            "notebook_checkpoints": notebook_checkpoints,
        },
        "dashboard_use": {
            "audience": "human",
            "worker_semantic_tool": False,
            "description": "Join model for attempt graph, score curves, candidate lineage, evaluations, traces, artifacts, jobs, findings, and notebooks.",
        },
    }


def _records_by_key(records: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        value = record.get(key)
        if value:
            grouped.setdefault(str(value), []).append(record)
    return grouped


def _traces_by_observed_id(traces: list[dict[str, Any]], observed_key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for trace in traces:
        for observed_id in ((trace.get("metadata") or {}).get("observed_ids") or {}).get(observed_key, []):
            grouped.setdefault(str(observed_id), []).append(trace)
    return grouped


def _records_by_mentioned_id(records: list[dict[str, Any]], prefix: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    needle = f"{prefix}_"
    for record in records:
        haystack = json.dumps(
            {
                "title": record.get("title"),
                "body": record.get("body"),
                "links": record.get("links"),
                "metadata": record.get("metadata"),
            },
            sort_keys=True,
        )
        for token in haystack.replace('"', " ").replace("'", " ").replace(",", " ").split():
            candidate = token.strip(" .:;()[]{}")
            if candidate.startswith(needle):
                grouped.setdefault(candidate, []).append(record)
    return grouped


def _attached_job_ids(jobs: list[dict[str, Any]], attempt_id: str) -> list[str]:
    attached: list[str] = []
    for job in jobs:
        for attachment in (job.get("details") or {}).get("attachments") or []:
            if attachment.get("attempt_id") == attempt_id:
                attached.append(job["job_id"])
    return attached


def _relationship(
    source_type: str,
    source_id: str,
    target_type: str,
    target_id: str,
    kind: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "source_type": source_type,
        "source_id": source_id,
        "target_type": target_type,
        "target_id": target_id,
        "kind": kind,
        "metadata": metadata or {},
    }


def _dedupe_relationships(relationships: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for relationship in relationships:
        metadata_key = json.dumps(relationship.get("metadata") or {}, sort_keys=True)
        key = (
            relationship["source_type"],
            relationship["source_id"],
            relationship["target_type"],
            relationship["target_id"],
            relationship["kind"],
            metadata_key,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(relationship)
    return deduped


def _unique_ids(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in ids:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _worker_visible_context(context: dict[str, Any]) -> dict[str, Any]:
    visible = dict(context)
    visible["assignment"] = _worker_visible_record(context.get("assignment") or {})
    visible["experiment"] = _worker_visible_record(context.get("experiment") or {})
    return visible


def _worker_visible_record(record: dict[str, Any]) -> dict[str, Any]:
    visible = dict(record)
    visible.pop("budget", None)
    visible.pop("config", None)
    if isinstance(visible.get("metadata"), dict):
        visible["metadata"] = {
            key: value
            for key, value in visible["metadata"].items()
            if key not in _WORKER_HIDDEN_METADATA_KEYS and "budget" not in key.lower()
        }
    if isinstance(visible.get("policy"), dict):
        visible["policy"] = {
            key: value
            for key, value in visible["policy"].items()
            if key not in {"jobs", "budget"}
        }
    return visible


def _copy_materialized_path(*, source: Path, destination: Path, force: bool, read_only: bool) -> None:
    if destination.exists():
        if not force:
            raise FileExistsError(destination)
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)
    if read_only:
        _chmod_read_only(destination)


def _chmod_read_only(path: Path) -> None:
    if path.is_dir():
        for item in path.rglob("*"):
            if item.is_file():
                item.chmod(0o444)
        path.chmod(0o555)
        return
    path.chmod(0o444)


def _environment_provider_from_payload(payload: dict[str, Any], *, experiment: dict[str, Any] | None) -> str | None:
    explicit = payload.get("provider") or payload.get("environment_provider") or payload.get("kind")
    if explicit and explicit != "framework":
        return str(explicit)
    config = (experiment or {}).get("config") or {}
    policy = (experiment or {}).get("policy") or {}
    env_config = config.get("environment") if isinstance(config.get("environment"), dict) else {}
    env_policy = policy.get("environments") if isinstance(policy.get("environments"), dict) else {}
    return (
        config.get("environment_provider")
        or env_config.get("provider")
        or env_config.get("kind")
        or env_policy.get("provider")
    )


def _environment_provider_config(payload: dict[str, Any], *, experiment: dict[str, Any] | None) -> dict[str, Any]:
    config = (experiment or {}).get("config") or {}
    env_config = config.get("environment") if isinstance(config.get("environment"), dict) else {}
    docker_config = config.get("docker") if isinstance(config.get("docker"), dict) else {}
    payload_docker = payload.get("docker") if isinstance(payload.get("docker"), dict) else {}
    result: dict[str, Any] = {}
    result.update(env_config)
    result.update(docker_config)
    result.update(payload_docker)
    for key in ("base_image", "image_ref", "image", "build", "platform", "build_timeout_s", "preflight_timeout_s", "provider"):
        if key in payload:
            result[key] = payload[key]
    provider = _environment_provider_from_payload(payload, experiment=experiment)
    if provider:
        result["provider"] = provider
    return result


def _provider_enforces_external_network(provider: str) -> bool:
    return provider in {"docker_image", "docker", "local-docker", "local-docker-strict"}


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


def _count_files(path: Path) -> int:
    if path.is_file():
        return 1
    return sum(1 for item in path.rglob("*") if item.is_file())


def _size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
