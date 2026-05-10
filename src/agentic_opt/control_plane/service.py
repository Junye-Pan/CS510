from __future__ import annotations

import hashlib
import json
import shutil
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
from .telemetry import TelemetryService


def task_contract(task_id: str) -> dict[str, Any]:
    task = get_task(task_id)
    metadata = task.metadata
    spec = candidate_spec_for(task)
    task_md = (task.public_dir / "TASK.md").read_text(encoding="utf-8")
    public_contract = (task.public_dir / "public_contract.md").read_text(encoding="utf-8")
    public_files = _read_public_context_files(task.public_dir)
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
    if suffix in {".yaml", ".yml"}:
        return "application/yaml"
    return "text/plain"


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
        self.jobs = JobService(repository=repository, job_root=job_root, database_path=database_path)
        self.evaluations = EvaluationService(
            repository=repository,
            jobs=self.jobs,
            environments=self.environments,
            database_path=database_path,
            artifact_root=self.artifact_root,
        )
        self.telemetry = TelemetryService(repository=repository, telemetry_root=artifact_root.parent / "telemetry")

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
            shutil.copytree(source, content_path)
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
        if not task_id:
            task_id = (assignment or {}).get("task_id")
        if not task_id:
            raise ValueError("task_id or assignment_id is required")
        return self.environments.ensure_task_environment(
            task_id,
            experiment_id=payload.get("experiment_id") or (assignment or {}).get("experiment_id"),
        )

    def create_environment_overlay(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.environments.create_overlay(payload)

    def approve_environment_overlay(self, overlay_id: str) -> dict[str, Any]:
        return self.environments.approve_overlay(overlay_id)

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
