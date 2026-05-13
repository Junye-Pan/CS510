from __future__ import annotations

import hashlib
import json
import mimetypes
import re
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

    def context_for_assignment(self, assignment_id: str) -> dict[str, Any]:
        context = _worker_visible_context(self.repository.context_for_assignment(assignment_id))
        assignment = context["assignment"]
        context["knowledge_items"] = self.list_knowledge_items(task_id=assignment["task_id"])
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
        bootstrap = {
            "assignment_id": assignment_id,
            "experiment_id": assignment["experiment_id"],
            "task_id": assignment["task_id"],
            "session_id": payload.get("session_id"),
            "workspace_root": str(workspace_root),
            "entry_path": str(entry_path),
            "workspace_seed": seed_materialized,
            "checked_out_tools": checked_out_tools,
        }
        self.repository.record_event(
            {
                "experiment_id": assignment["experiment_id"],
                "assignment_id": assignment_id,
                "session_id": payload.get("session_id"),
                "task_id": assignment["task_id"],
                "agent_id": assignment["agent_id"],
                "event_type": "worker.workspace.bootstrapped",
                "summary": _workspace_bootstrap_summary(bootstrap),
                "payload": bootstrap,
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

    def list_knowledge_items(self, *, task_id: str, query: str | None = None) -> list[dict[str, Any]]:
        self.index_task_knowledge(task_id)
        return self.repository.list_knowledge_items(task_id=task_id, query=query)

    def get_knowledge_item(self, knowledge_id: str) -> dict[str, Any] | None:
        return self.repository.get_knowledge_item(knowledge_id)

    def materialize_knowledge_item(self, knowledge_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        item = self.repository.get_knowledge_item(knowledge_id)
        if item is None:
            raise KeyError(knowledge_id)
        source = Path(item["source_path"]).resolve()
        destination_raw = payload.get("destination_path")
        if destination_raw:
            destination = Path(destination_raw).resolve()
        else:
            workspace_root = payload.get("workspace_root")
            if not workspace_root:
                raise ValueError("destination_path or workspace_root is required")
            destination = Path(workspace_root).resolve() / "knowledge" / knowledge_id / source.name
        _copy_materialized_path(
            source=source,
            destination=destination,
            force=bool(payload.get("force")),
            read_only=True,
        )
        return {"knowledge_item": item, "destination_path": str(destination)}

    def index_task_knowledge(self, task_id: str) -> list[dict[str, Any]]:
        task = get_task(task_id)
        knowledge_root = (task.public_dir / "knowledge").resolve()
        manifest_path = knowledge_root / "manifest.json"
        if not manifest_path.exists():
            return []
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        items = manifest.get("items") or []
        if not isinstance(items, list):
            raise ValueError("knowledge manifest must contain an items list")
        indexed: list[dict[str, Any]] = []
        for raw in items:
            if not isinstance(raw, dict):
                raise ValueError("knowledge manifest items must be objects")
            relative = raw.get("path")
            if not isinstance(relative, str) or not relative:
                raise ValueError("knowledge item path is required")
            source = (knowledge_root / relative).resolve()
            if not source.is_relative_to(knowledge_root):
                raise PermissionError(f"knowledge item must stay inside {knowledge_root}: {relative}")
            if not source.exists():
                raise FileNotFoundError(source)
            local_id = str(raw.get("knowledge_id") or raw.get("id") or Path(relative).stem)
            media_type = str(raw.get("media_type") or mimetypes.guess_type(source.name)[0] or "application/octet-stream")
            record = self.repository.upsert_knowledge_item(
                {
                    "knowledge_id": _stable_knowledge_id(task_id, local_id),
                    "local_id": local_id,
                    "task_id": task_id,
                    "title": str(raw.get("title") or local_id),
                    "kind": str(raw.get("kind") or "reference"),
                    "source_path": str(source),
                    "media_type": media_type,
                    "summary": raw.get("summary"),
                    "scope": "task",
                    "digest": _digest_directory(source) if source.is_dir() else _digest_file(source),
                    "size_bytes": _size_bytes(source),
                    "tags": raw.get("tags") or [],
                    "metadata": {
                        "manifest_path": str(manifest_path),
                        "manifest_item": raw,
                    },
                }
            )
            indexed.append(record)
        return indexed

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
        external_enforced = _provider_enforces_external_network(provider)
        policy_weakened = str(external) == "deny" and not external_enforced
        docker_enforced = provider in {"docker_image", "docker", "local-docker", "local-docker-strict"}
        control_plane_relay_required = docker_enforced and str(external) == "deny" and str(control_plane) == "allow"
        session_relay = ((session or {}).get("details") or {}).get("control_plane_relay") if session else None
        control_plane_relay_configured = bool(raw_policy.get("control_plane_relay") or session_relay)
        control_plane_available = not control_plane_relay_required or control_plane_relay_configured
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
            },
            "enforcement": {
                "worker_backend": provider,
                "control_plane_enforced": True,
                "external_internet_enforced": external_enforced,
                "policy_weakened": policy_weakened,
                "enforcement_mode": "docker_network_none" if docker_enforced and str(external) == "deny" else None,
                "control_plane_relay_required": control_plane_relay_required,
                "control_plane_relay_configured": control_plane_relay_configured,
                "control_plane_available": control_plane_available,
                "operationally_ready": control_plane_available and not policy_weakened,
                "reason": (
                    "worker backend exposes coarse network access for control-plane HTTP"
                    if policy_weakened
                    else None
                ),
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


def _stable_knowledge_id(task_id: str, local_id: str) -> str:
    return f"knowledge_{_safe_token(task_id)}_{_safe_token(local_id)}"


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
