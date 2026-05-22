from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from agentic_opt.common.atomic import atomic_write_text
from agentic_opt.common.config import get_repo_root
from agentic_opt.common.runtime_env import RuntimeEnvironmentError, prepare_task_runtime
from agentic_opt.task_registry import get_task

from .docker_image_policy import docker_image_identity, docker_policy_from_experiment, enforce_docker_image_policy
from .repository import ControlPlaneRepository


class EnvironmentService:
    """Server-owned runtime environment manager.

    Base task environments are immutable records derived from each task's
    runtime spec. Worker overlays are separate records and never silently
    mutate the base environment used for official evaluation.
    """

    def __init__(self, *, repository: ControlPlaneRepository, environment_root: Path) -> None:
        self.repository = repository
        self.environment_root = environment_root.resolve()
        self.environment_root.mkdir(parents=True, exist_ok=True)
        self._task_environment_lock = threading.Lock()

    def ensure_framework_environment(self) -> dict[str, Any]:
        python_path = Path(sys.executable).resolve()
        fingerprint = _fingerprint_json({"kind": "framework", "python": str(python_path), "version": sys.version})
        existing = self.repository.get_environment("env_framework_current")
        if (
            existing is not None
            and existing.get("status") == "ready"
            and existing.get("fingerprint") == fingerprint
            and (existing.get("lock") or {}).get("status") != "unavailable"
        ):
            return existing
        root_path = self.environment_root / "framework" / fingerprint
        root_path.mkdir(parents=True, exist_ok=True)
        return self.repository.upsert_environment(
            {
                "environment_id": "env_framework_current",
                "environment_type": "framework",
                "status": "ready",
                "fingerprint": fingerprint,
                "python_path": str(python_path),
                "root_path": str(root_path),
                "spec": {"kind": "current_process", "python": sys.version},
                "lock": _pip_freeze(python_path),
                "metadata": {"executable": str(python_path)},
            }
        )

    def ensure_task_environment(
        self,
        task_id: str,
        *,
        experiment_id: str | None = None,
        provider: str | None = None,
        provider_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._task_environment_lock:
            task = get_task(task_id)
            provider_name = _normalize_environment_provider(provider or (provider_config or {}).get("provider") or getattr(task.runtime_spec, "kind", "local_venv"))
            if provider_name == "docker_image":
                return self._ensure_docker_task_environment(
                    task=task,
                    experiment_id=experiment_id,
                    provider_config=provider_config or {},
                )
            if provider_name != "local_venv":
                raise RuntimeEnvironmentError(f"unsupported environment provider for {task_id}: {provider_name}")
            prepared = prepare_task_runtime(task, envs_root=self.environment_root / "tasks")
            environment_id = _task_environment_id(task_id=task_id, fingerprint=prepared.fingerprint)
            existing = self.repository.get_environment(environment_id)
            record = self.repository.upsert_environment(
                {
                    "environment_id": environment_id,
                    "environment_type": "task",
                    "status": "ready",
                    "fingerprint": prepared.fingerprint,
                    "python_path": str(prepared.python_path),
                    "root_path": str(prepared.root),
                    "task_id": task_id,
                    "experiment_id": experiment_id,
                    "spec": prepared.spec.to_jsonable(),
                    "lock": _pip_freeze(prepared.python_path),
                    "metadata": {
                        "manifest_path": str(prepared.manifest_path),
                        "venv_dir": str(prepared.venv_dir),
                    },
                }
            )
        if existing is None:
            self.repository.record_event(
                {
                    "experiment_id": experiment_id,
                    "task_id": task_id,
                    "event_type": "environment.ready",
                    "summary": f"task environment ready: {environment_id}",
                    "payload": {"environment_id": environment_id, "fingerprint": prepared.fingerprint},
                }
            )
        return record

    def _ensure_docker_task_environment(
        self,
        *,
        task: Any,
        experiment_id: str | None,
        provider_config: dict[str, Any],
    ) -> dict[str, Any]:
        spec = task.runtime_spec
        task_id = task.metadata.task_id
        experiment = self.repository.get_experiment(experiment_id) if experiment_id else None
        trust_policy = docker_policy_from_experiment(experiment, provider_config)
        default_env = _docker_default_env(provider_config)
        source_digest = _digest_environment_sources(task)
        base_image = str(provider_config.get("base_image") or "python:3.11-slim")
        prebuilt_image = provider_config.get("image_ref") or provider_config.get("image") or provider_config.get("worker_image")
        should_build = not prebuilt_image or bool(provider_config.get("build", True))
        fingerprint = _fingerprint_json(
            {
                "schema": 2,
                "task_id": task_id,
                "provider": "docker_image",
                "base_image": base_image,
                "requirements": list(spec.requirements),
                "required_imports": list(spec.required_imports),
                "verify_public_seed": spec.verify_public_seed,
                "source_digest": source_digest,
                "prebuilt_image": None if should_build else str(prebuilt_image),
                "default_env": default_env,
            }
        )
        environment_id = _task_environment_id(task_id=task_id, fingerprint=fingerprint)
        existing = self.repository.get_environment(environment_id)
        if existing is not None and existing["status"] == "ready":
            return existing

        host_root = self.environment_root / "tasks" / task_id / fingerprint
        build_context = host_root / "build_context"
        host_root.mkdir(parents=True, exist_ok=True)
        image_ref = str(prebuilt_image or provider_config.get("image_ref") or _docker_task_image_ref(task_id=task_id, fingerprint=fingerprint))
        if should_build:
            _prepare_docker_build_context(task=task, build_context=build_context, base_image=base_image)
            dockerfile = build_context / "Dockerfile"
            build_proc = _run_process(
                ["docker", "build", "-f", str(dockerfile), "-t", image_ref, str(build_context)],
                cwd=host_root,
                timeout_s=float(provider_config.get("build_timeout_s") or 1800),
            )
            atomic_write_text(host_root / "docker_build_stdout.log", build_proc.stdout)
            atomic_write_text(host_root / "docker_build_stderr.log", build_proc.stderr)
            if build_proc.returncode != 0:
                failed = self.repository.upsert_environment(
                    {
                        "environment_id": environment_id,
                        "environment_type": "task",
                        "status": "failed",
                        "fingerprint": fingerprint,
                        "python_path": "/usr/local/bin/python",
                        "root_path": str(host_root),
                        "task_id": task_id,
                        "experiment_id": experiment_id,
                        "spec": _docker_environment_spec(spec=spec, provider_config=provider_config, base_image=base_image),
                        "lock": {"status": "failed", "provider": "docker_image", "image_ref": image_ref},
                        "metadata": {"host_root_path": str(host_root), "build_context": str(build_context)},
                    }
                )
                raise RuntimeEnvironmentError(f"docker image build failed for {task_id}: {build_proc.stderr.strip() or build_proc.stdout.strip()}")
        image_info = _inspect_docker_image(image_ref)
        identity = docker_image_identity(image_ref=image_ref, image_info=image_info)
        trust_decision = enforce_docker_image_policy(
            identity=identity,
            policy=trust_policy,
            source="local_build" if should_build else "prebuilt_image",
        )
        _run_docker_import_preflight(image_ref=image_ref, task=task, host_root=host_root, timeout_s=float(provider_config.get("preflight_timeout_s") or 300))
        if spec.verify_public_seed:
            _run_docker_public_seed_preflight(image_ref=image_ref, task=task, host_root=host_root, timeout_s=float(provider_config.get("preflight_timeout_s") or 300))
        manifest = {
            "task_id": task_id,
            "fingerprint": fingerprint,
            "provider": "docker_image",
            "host_root": str(host_root),
            "container_root": "/opt/agentic-opt",
            "container_python": "/usr/local/bin/python",
            "container_tasks_root": "/opt/agentic-opt/tasks",
            "image_ref": image_ref,
            "image_id": image_info.get("Id"),
            "image_digest": _docker_image_digest(image_info),
            "repo_digests": image_info.get("RepoDigests") or [],
            "repo_tags": image_info.get("RepoTags") or [],
            "source_digest": source_digest,
            "default_env": default_env,
            "trust_decision": trust_decision,
            "spec": _docker_environment_spec(spec=spec, provider_config=provider_config, base_image=base_image),
        }
        manifest_path = host_root / "manifest.json"
        atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        lock = {
            "status": "ready",
            "format": "docker-image-lock",
            "provider": "docker_image",
            "image_ref": image_ref,
            "image_id": image_info.get("Id"),
            "image_digest": _docker_image_digest(image_info),
            "repo_digests": image_info.get("RepoDigests") or [],
            "repo_tags": image_info.get("RepoTags") or [],
            "base_image": base_image,
            "requirements": list(spec.requirements),
            "source_digest": source_digest,
            "trust_decision": trust_decision,
            "default_env": default_env,
        }
        record = self.repository.upsert_environment(
            {
                "environment_id": environment_id,
                "environment_type": "task",
                "status": "ready",
                "fingerprint": fingerprint,
                "python_path": "/usr/local/bin/python",
                "root_path": str(host_root),
                "task_id": task_id,
                "experiment_id": experiment_id,
                "spec": _docker_environment_spec(spec=spec, provider_config=provider_config, base_image=base_image),
                "lock": lock,
                "metadata": {
                    "provider": "docker_image",
                    "host_root_path": str(host_root),
                    "manifest_path": str(manifest_path),
                    "build_context": str(build_context) if should_build else None,
                    "container_root": "/opt/agentic-opt",
                    "container_python": "/usr/local/bin/python",
                    "container_src_path": "/opt/agentic-opt/src",
                    "container_tasks_root": "/opt/agentic-opt/tasks",
                    "image_ref": image_ref,
                    "image_id": image_info.get("Id"),
                    "image_digest": _docker_image_digest(image_info),
                    "default_env": default_env,
                    "trust_decision": trust_decision,
                    "runner": {"workdir": "/opt/agentic-opt", "python_path": "/usr/local/bin/python"},
                },
            }
        )
        self.repository.record_event(
            {
                "experiment_id": experiment_id,
                "task_id": task_id,
                "event_type": "environment.ready",
                "summary": f"docker task environment ready: {environment_id}",
                "payload": {"environment_id": environment_id, "fingerprint": fingerprint, "image_ref": image_ref, "image_digest": lock["image_digest"]},
            }
        )
        return record

    def create_overlay(self, payload: dict[str, Any]) -> dict[str, Any]:
        base_environment = self._resolve_base_environment(payload)
        experiment_id = payload.get("experiment_id") or base_environment.get("experiment_id")
        assignment_id = payload.get("assignment_id")
        assignment = self.repository.get_assignment(assignment_id) if assignment_id else None
        if assignment is not None:
            experiment_id = experiment_id or assignment["experiment_id"]
        requirements = _normalize_requirements(payload.get("requirements") or payload.get("pip") or payload.get("requirement"))
        if not requirements:
            raise ValueError("at least one requirement is required for an environment overlay")

        approved = bool(payload.get("approved"))
        policy_decision = self._decide_overlay_policy(
            experiment_id=experiment_id,
            requirements=requirements,
            approved=approved,
        )
        overlay = self.repository.create_environment_overlay(
            {
                "base_environment_id": base_environment["environment_id"],
                "experiment_id": experiment_id,
                "assignment_id": assignment_id,
                "session_id": payload.get("session_id"),
                "status": "preparing" if policy_decision["allowed"] else "blocked",
                "requested_by_agent_id": payload.get("requested_by_agent_id") or (assignment or {}).get("agent_id"),
                "requirements": requirements,
                "reason": payload.get("reason"),
                "approved": approved or bool(policy_decision.get("auto_approved")),
                "policy_decision": policy_decision,
                "metadata": payload.get("metadata") or {},
            }
        )
        if not policy_decision["allowed"]:
            self.repository.record_event(
                {
                    "experiment_id": experiment_id,
                    "assignment_id": assignment_id,
                    "task_id": base_environment.get("task_id"),
                    "agent_id": overlay.get("requested_by_agent_id"),
                    "event_type": "environment.overlay.blocked",
                    "summary": "environment overlay blocked by policy",
                    "payload": {"overlay_id": overlay["overlay_id"], "policy_decision": policy_decision},
                }
            )
            return overlay

        try:
            prepared_overlay = self._prepare_overlay_environment(base_environment=base_environment, overlay=overlay)
        except Exception as exc:
            failed = self.repository.update_environment_overlay(
                overlay["overlay_id"],
                {"status": "failed", "metadata": {"error_type": type(exc).__name__, "error": str(exc)}},
            )
            self.repository.record_event(
                {
                    "experiment_id": experiment_id,
                    "assignment_id": assignment_id,
                    "task_id": base_environment.get("task_id"),
                    "agent_id": overlay.get("requested_by_agent_id"),
                    "event_type": "environment.overlay.failed",
                    "summary": "environment overlay preparation failed",
                    "payload": {"overlay_id": overlay["overlay_id"], "error": str(exc)},
                }
            )
            raise

        ready = self.repository.update_environment_overlay(
            overlay["overlay_id"],
            {
                "status": "ready",
                "python_path": str(prepared_overlay["python_path"]),
                "root_path": str(prepared_overlay["root_path"]),
                "lock": prepared_overlay["lock"],
                "metadata": prepared_overlay["metadata"],
            },
        )
        self.repository.record_event(
            {
                "experiment_id": experiment_id,
                "assignment_id": assignment_id,
                "task_id": base_environment.get("task_id"),
                "agent_id": ready.get("requested_by_agent_id"),
                "event_type": "environment.overlay.ready",
                "summary": f"environment overlay ready: {ready['overlay_id']}",
                "payload": {"overlay_id": ready["overlay_id"], "base_environment_id": base_environment["environment_id"]},
            }
        )
        return ready

    def approve_overlay(self, overlay_id: str) -> dict[str, Any]:
        overlay = self.repository.get_environment_overlay(overlay_id)
        if overlay is None:
            raise KeyError(overlay_id)
        if overlay["status"] == "ready":
            return overlay
        base_environment = self.repository.get_environment(overlay["base_environment_id"])
        if base_environment is None:
            raise KeyError(overlay["base_environment_id"])
        policy_decision = self._decide_overlay_policy(
            experiment_id=overlay.get("experiment_id"),
            requirements=list(overlay.get("requirements") or []),
            approved=True,
        )
        if not policy_decision["allowed"]:
            return self.repository.update_environment_overlay(
                overlay_id,
                {
                    "approved": True,
                    "policy_decision": policy_decision,
                    "metadata": {"approval_error": policy_decision.get("reason")},
                },
            )
        overlay = self.repository.update_environment_overlay(
            overlay_id,
            {"status": "preparing", "approved": True, "policy_decision": policy_decision},
        )
        prepared_overlay = self._prepare_overlay_environment(base_environment=base_environment, overlay=overlay)
        ready = self.repository.update_environment_overlay(
            overlay_id,
            {
                "status": "ready",
                "python_path": str(prepared_overlay["python_path"]),
                "root_path": str(prepared_overlay["root_path"]),
                "lock": prepared_overlay["lock"],
                "metadata": prepared_overlay["metadata"],
            },
        )
        self.repository.record_event(
            {
                "experiment_id": ready.get("experiment_id"),
                "assignment_id": ready.get("assignment_id"),
                "task_id": base_environment.get("task_id"),
                "agent_id": ready.get("requested_by_agent_id"),
                "event_type": "environment.overlay.approved",
                "summary": f"environment overlay approved: {overlay_id}",
                "payload": {"overlay_id": overlay_id, "base_environment_id": base_environment["environment_id"]},
            }
        )
        return ready

    def get_execution_environment(
        self,
        *,
        task_id: str,
        experiment_id: str | None = None,
        environment_id: str | None = None,
        overlay_id: str | None = None,
        allow_overlay: bool = False,
        provider: str | None = None,
        provider_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if overlay_id and allow_overlay:
            overlay = self.repository.get_environment_overlay(overlay_id)
            if overlay is None:
                raise KeyError(overlay_id)
            if overlay["status"] != "ready":
                raise RuntimeEnvironmentError(f"environment overlay is not ready: {overlay_id} status={overlay['status']}")
            base = self.repository.get_environment(overlay["base_environment_id"])
            exports = self.exports_for_overlay(overlay)
            base_provider = _environment_provider(base) if base is not None else "local_venv"
            return {
                "kind": "overlay",
                "environment_id": overlay["base_environment_id"],
                "environment_overlay_id": overlay["overlay_id"],
                "provider": base_provider,
                "python_path": exports.get("AO_ENVIRONMENT_OVERLAY_PYTHON") or exports.get("AO_ENVIRONMENT_PYTHON") or overlay["python_path"],
                "root_path": exports.get("AO_ENVIRONMENT_OVERLAY_ROOT") or exports.get("AO_ENVIRONMENT_ROOT") or overlay["root_path"],
                "exports": exports,
                "record": overlay,
                "base_record": base,
            }
        if environment_id:
            environment = self.repository.get_environment(environment_id)
            if environment is None:
                raise KeyError(environment_id)
            if environment["status"] != "ready":
                raise RuntimeEnvironmentError(f"environment is not ready: {environment_id} status={environment['status']}")
        else:
            environment = self.ensure_task_environment(
                task_id,
                experiment_id=experiment_id,
                provider=provider,
                provider_config=provider_config,
            )
        exports = self.exports_for_environment(environment)
        return {
            "kind": "environment",
            "environment_id": environment["environment_id"],
            "environment_overlay_id": None,
            "provider": _environment_provider(environment),
            "python_path": exports.get("AO_ENVIRONMENT_PYTHON") or environment["python_path"],
            "root_path": exports.get("AO_ENVIRONMENT_ROOT") or environment["root_path"],
            "exports": exports,
            "record": environment,
        }

    def exports_for_environment(self, environment: dict[str, Any]) -> dict[str, str]:
        if _environment_provider(environment) == "docker_image":
            metadata = environment.get("metadata") or {}
            container_root = str(metadata.get("container_root") or "/opt/agentic-opt")
            container_python = str(metadata.get("container_python") or "/usr/local/bin/python")
            container_src = str(metadata.get("container_src_path") or f"{container_root}/src")
            container_tasks = str(metadata.get("container_tasks_root") or f"{container_root}/tasks")
            manifest_path = str(metadata.get("manifest_path") or "")
            exports = {
                **_string_env(metadata.get("default_env")),
                "AO_ENVIRONMENT_ID": environment["environment_id"],
                "AO_ENVIRONMENT_TYPE": environment["environment_type"],
                "AO_ENVIRONMENT_PROVIDER": "docker_image",
                "AO_ENVIRONMENT_ROOT": container_root,
                "AO_ENVIRONMENT_PYTHON": container_python,
                "AO_ENVIRONMENT_FINGERPRINT": environment["fingerprint"],
                "AO_TASK_RUNTIME_ROOT": container_root,
                "AO_TASK_RUNTIME_PYTHON": container_python,
                "AO_TASK_RUNTIME_FINGERPRINT": environment["fingerprint"],
                "PYTHONPATH": container_src,
                "AO_TASKS_ROOTS": container_tasks,
            }
            if manifest_path:
                exports["AO_TASK_RUNTIME_ENV"] = manifest_path
            return exports
        exports = {
            **_string_env((environment.get("metadata") or {}).get("default_env")),
            "AO_ENVIRONMENT_ID": environment["environment_id"],
            "AO_ENVIRONMENT_TYPE": environment["environment_type"],
            "AO_ENVIRONMENT_PROVIDER": _environment_provider(environment),
            "AO_ENVIRONMENT_ROOT": environment["root_path"],
            "AO_ENVIRONMENT_PYTHON": environment["python_path"],
            "AO_ENVIRONMENT_FINGERPRINT": environment["fingerprint"],
        }
        if environment["environment_type"] == "task":
            exports.update(
                {
                    "AO_TASK_RUNTIME_ROOT": environment["root_path"],
                    "AO_TASK_RUNTIME_PYTHON": environment["python_path"],
                    "AO_TASK_RUNTIME_FINGERPRINT": environment["fingerprint"],
                }
            )
            manifest_path = (environment.get("metadata") or {}).get("manifest_path")
            if manifest_path:
                exports["AO_TASK_RUNTIME_ENV"] = str(manifest_path)
        return exports

    def exports_for_overlay(self, overlay: dict[str, Any]) -> dict[str, str]:
        base = self.repository.get_environment(overlay["base_environment_id"])
        exports = self.exports_for_environment(base) if base is not None else {}
        if base is not None and _environment_provider(base) == "docker_image":
            metadata = overlay.get("metadata") or {}
            container_root = str(metadata.get("container_root") or exports.get("AO_ENVIRONMENT_ROOT") or "/opt/agentic-opt")
            container_python = str(metadata.get("container_python") or exports.get("AO_ENVIRONMENT_PYTHON") or "/usr/local/bin/python")
            exports.update(
                {
                    "AO_ENVIRONMENT_OVERLAY_ID": overlay["overlay_id"],
                    "AO_ENVIRONMENT_OVERLAY_ROOT": container_root,
                    "AO_ENVIRONMENT_OVERLAY_PYTHON": container_python,
                    "AO_ENVIRONMENT_PYTHON": container_python,
                    "AO_ENVIRONMENT_PROVIDER": "docker_image",
                }
            )
            return {key: value for key, value in exports.items() if value}
        exports.update(
            {
                "AO_ENVIRONMENT_OVERLAY_ID": overlay["overlay_id"],
                "AO_ENVIRONMENT_OVERLAY_ROOT": overlay["root_path"] or "",
                "AO_ENVIRONMENT_OVERLAY_PYTHON": overlay["python_path"] or "",
                "AO_ENVIRONMENT_PYTHON": overlay["python_path"] or exports.get("AO_ENVIRONMENT_PYTHON", ""),
            }
        )
        return {key: value for key, value in exports.items() if value}

    def _resolve_base_environment(self, payload: dict[str, Any]) -> dict[str, Any]:
        base_environment_id = payload.get("base_environment_id") or payload.get("environment_id")
        if base_environment_id:
            base = self.repository.get_environment(base_environment_id)
            if base is None:
                raise KeyError(base_environment_id)
            return base
        assignment_id = payload.get("assignment_id")
        assignment = self.repository.get_assignment(assignment_id) if assignment_id else None
        task_id = payload.get("task_id") or (assignment or {}).get("task_id")
        if not task_id:
            raise ValueError("base_environment_id, environment_id, task_id, or assignment_id is required")
        return self.ensure_task_environment(task_id, experiment_id=payload.get("experiment_id") or (assignment or {}).get("experiment_id"))

    def _decide_overlay_policy(self, *, experiment_id: str | None, requirements: list[str], approved: bool) -> dict[str, Any]:
        experiment = self.repository.get_experiment(experiment_id) if experiment_id else None
        policy = ((experiment or {}).get("policy") or {}).get("environments") or ((experiment or {}).get("policy") or {}).get("environment") or {}
        if policy.get("allow_worker_overlay") is False:
            return {
                "allowed": False,
                "reason": "worker_overlay_disabled",
                "requirements": requirements,
            }
        requirement_names = [_requirement_name(item) for item in requirements]
        approval_required_for = set(policy.get("require_approval_for") or [])
        if approval_required_for and not approved:
            matched = [name for name in requirement_names if "*" in approval_required_for or name in approval_required_for]
            if matched:
                return {
                    "allowed": False,
                    "reason": "approval_required",
                    "requirements": requirements,
                    "matched": matched,
                }
        auto_approve_pip = policy.get("auto_approve_pip")
        if auto_approve_pip:
            allowed_names = set(auto_approve_pip)
            if "*" not in allowed_names and any(name not in allowed_names for name in requirement_names):
                return {
                    "allowed": False,
                    "reason": "requirement_not_auto_approved",
                    "requirements": requirements,
                    "requirement_names": requirement_names,
                }
            return {
                "allowed": True,
                "auto_approved": True,
                "requirements": requirements,
                "requirement_names": requirement_names,
            }
        return {
            "allowed": True,
            "auto_approved": False,
            "requirements": requirements,
            "requirement_names": requirement_names,
        }

    def _prepare_overlay_environment(self, *, base_environment: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
        if _environment_provider(base_environment) == "docker_image":
            return self._prepare_docker_overlay_environment(base_environment=base_environment, overlay=overlay)
        root_path = self.environment_root / "overlays" / overlay["overlay_id"]
        venv_dir = root_path / "venv"
        python_path = _venv_python(venv_dir)
        root_path.mkdir(parents=True, exist_ok=True)
        if not python_path.exists():
            proc = subprocess.run(
                [base_environment["python_path"], "-m", "venv", str(venv_dir)],
                cwd=str(root_path),
                env=_clean_env(),
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )
            if proc.returncode != 0:
                raise RuntimeEnvironmentError(f"failed to create environment overlay venv: {proc.stderr.strip() or proc.stdout.strip()}")
        base_requirements = list((base_environment.get("spec") or {}).get("requirements") or [])
        requirements = [*base_requirements, *overlay["requirements"]]
        _pip_install(python_path=python_path, requirements=requirements, cwd=root_path)
        lock = _pip_freeze(python_path)
        return {
            "python_path": python_path,
            "root_path": root_path,
            "lock": lock,
            "metadata": {
                "base_environment_id": base_environment["environment_id"],
                "venv_dir": str(venv_dir),
                "requirements_installed": requirements,
            },
        }

    def _prepare_docker_overlay_environment(self, *, base_environment: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
        root_path = self.environment_root / "overlays" / overlay["overlay_id"]
        root_path.mkdir(parents=True, exist_ok=True)
        build_context = root_path / "build_context"
        if build_context.exists():
            shutil.rmtree(build_context)
        build_context.mkdir(parents=True, exist_ok=True)
        requirements = list(overlay["requirements"])
        (build_context / "requirements.txt").write_text("\n".join(requirements) + ("\n" if requirements else ""), encoding="utf-8")
        base_lock = base_environment.get("lock") or {}
        base_image = str(base_lock.get("image_ref") or (base_environment.get("metadata") or {}).get("image_ref"))
        if not base_image:
            raise RuntimeEnvironmentError(f"docker overlay base environment has no image_ref: {base_environment['environment_id']}")
        fingerprint = _fingerprint_json(
            {
                "schema": 1,
                "provider": "docker_image_overlay",
                "base_environment_id": base_environment["environment_id"],
                "base_image_digest": base_lock.get("image_digest") or base_lock.get("image_id"),
                "requirements": requirements,
            }
        )
        image_ref = _docker_overlay_image_ref(overlay_id=overlay["overlay_id"], fingerprint=fingerprint)
        atomic_write_text(
            build_context / "Dockerfile",
            "\n".join(
                [
                    f"FROM {base_image}",
                    "COPY requirements.txt /tmp/agentic-opt-overlay-requirements.txt",
                    "RUN if [ -s /tmp/agentic-opt-overlay-requirements.txt ]; then python -m pip install -r /tmp/agentic-opt-overlay-requirements.txt; fi",
                    "",
                ]
            ),
        )
        proc = _run_process(
            ["docker", "build", "-f", str(build_context / "Dockerfile"), "-t", image_ref, str(build_context)],
            cwd=root_path,
            timeout_s=1800,
        )
        atomic_write_text(root_path / "docker_build_stdout.log", proc.stdout)
        atomic_write_text(root_path / "docker_build_stderr.log", proc.stderr)
        if proc.returncode != 0:
            raise RuntimeEnvironmentError(f"failed to build docker overlay image: {proc.stderr.strip() or proc.stdout.strip()}")
        image_info = _inspect_docker_image(image_ref)
        base_metadata = base_environment.get("metadata") or {}
        lock = {
            "status": "ready",
            "format": "docker-image-overlay-lock",
            "provider": "docker_image",
            "image_ref": image_ref,
            "image_id": image_info.get("Id"),
            "image_digest": _docker_image_digest(image_info),
            "repo_digests": image_info.get("RepoDigests") or [],
            "repo_tags": image_info.get("RepoTags") or [],
            "base_environment_id": base_environment["environment_id"],
            "base_image_ref": base_image,
            "base_image_digest": base_lock.get("image_digest"),
            "requirements": requirements,
        }
        return {
            "python_path": Path(str(base_metadata.get("container_python") or "/usr/local/bin/python")),
            "root_path": root_path,
            "lock": lock,
            "metadata": {
                "provider": "docker_image",
                "base_environment_id": base_environment["environment_id"],
                "host_root_path": str(root_path),
                "build_context": str(build_context),
                "container_root": str(base_metadata.get("container_root") or "/opt/agentic-opt"),
                "container_python": str(base_metadata.get("container_python") or "/usr/local/bin/python"),
                "container_src_path": str(base_metadata.get("container_src_path") or "/opt/agentic-opt/src"),
                "container_tasks_root": str(base_metadata.get("container_tasks_root") or "/opt/agentic-opt/tasks"),
                "image_ref": image_ref,
                "image_id": image_info.get("Id"),
                "image_digest": _docker_image_digest(image_info),
                "requirements_installed": requirements,
            },
        }


def _task_environment_id(*, task_id: str, fingerprint: str) -> str:
    safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id)
    return f"env_task_{safe_task_id}_{fingerprint}"


def _normalize_environment_provider(raw: str | None) -> str:
    value = str(raw or "local_venv")
    if value in {"docker", "docker-image", "docker_image"}:
        return "docker_image"
    if value in {"local", "venv", "local-venv", "local_venv"}:
        return "local_venv"
    return value


def _environment_provider(environment: dict[str, Any] | None) -> str:
    if environment is None:
        return "local_venv"
    metadata = environment.get("metadata") or {}
    spec = environment.get("spec") or {}
    return _normalize_environment_provider(metadata.get("provider") or spec.get("provider") or spec.get("kind") or "local_venv")


def _docker_environment_spec(*, spec: Any, provider_config: dict[str, Any], base_image: str) -> dict[str, Any]:
    return {
        "kind": "docker_image",
        "provider": "docker_image",
        "base_image": base_image,
        "source_runtime_spec": spec.to_jsonable(),
        "requirements": list(spec.requirements),
        "required_imports": list(spec.required_imports),
        "forbidden_shadow_modules": list(spec.forbidden_shadow_modules),
        "verify_public_seed": spec.verify_public_seed,
        "platform": provider_config.get("platform"),
        "default_env": _docker_default_env(provider_config),
    }


def _docker_default_env(provider_config: dict[str, Any]) -> dict[str, str]:
    raw = (
        provider_config.get("default_env")
        or provider_config.get("environment_variables")
        or provider_config.get("env")
        or {}
    )
    return _string_env(raw)


def _string_env(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items() if key and value is not None}


def _docker_task_image_ref(*, task_id: str, fingerprint: str) -> str:
    safe_task = re.sub(r"[^a-z0-9_.-]+", "-", task_id.lower()).strip("-._") or "task"
    return f"agentic-opt/task-{safe_task}:{fingerprint}"


def _docker_overlay_image_ref(*, overlay_id: str, fingerprint: str) -> str:
    safe_overlay = re.sub(r"[^a-z0-9_.-]+", "-", overlay_id.lower()).strip("-._") or "overlay"
    return f"agentic-opt/overlay-{safe_overlay}:{fingerprint}"


def _prepare_docker_build_context(*, task: Any, build_context: Path, base_image: str) -> None:
    if build_context.exists():
        shutil.rmtree(build_context)
    build_context.mkdir(parents=True, exist_ok=True)
    repo_root = get_repo_root()
    _copy_tree(repo_root / "src", build_context / "src")
    task_dst = build_context / "tasks" / task.metadata.task_id
    _copy_tree(task.public_dir.parent, task_dst)
    requirements_path = build_context / "requirements.txt"
    requirements_path.write_text("\n".join(task.runtime_spec.requirements) + ("\n" if task.runtime_spec.requirements else ""), encoding="utf-8")
    atomic_write_text(
        build_context / "Dockerfile",
        "\n".join(
            [
                f"FROM {base_image}",
                "ENV PYTHONDONTWRITEBYTECODE=1",
                "ENV PYTHONUNBUFFERED=1",
                "ENV PYTHONPATH=/opt/agentic-opt/src",
                "ENV AO_TASKS_ROOTS=/opt/agentic-opt/tasks",
                "WORKDIR /opt/agentic-opt",
                "COPY src ./src",
                "COPY tasks ./tasks",
                "COPY requirements.txt /tmp/agentic-opt-task-requirements.txt",
                "RUN python -m pip install --upgrade pip && if [ -s /tmp/agentic-opt-task-requirements.txt ]; then python -m pip install -r /tmp/agentic-opt-task-requirements.txt; fi",
                "",
            ]
        ),
    )


def _copy_tree(source: Path, destination: Path) -> None:
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".mypy_cache", ".ruff_cache")
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, ignore=ignore)


def _digest_environment_sources(task: Any) -> str:
    digest = hashlib.sha256()
    for root in (get_repo_root() / "src", task.public_dir.parent):
        if not root.exists():
            continue
        for file_path in sorted(item for item in root.rglob("*") if item.is_file() and "__pycache__" not in item.parts):
            if file_path.suffix == ".pyc":
                continue
            digest.update(file_path.relative_to(root).as_posix().encode("utf-8"))
            with file_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _inspect_docker_image(image_ref: str) -> dict[str, Any]:
    proc = _run_process(["docker", "image", "inspect", image_ref], cwd=get_repo_root(), timeout_s=60)
    if proc.returncode != 0:
        raise RuntimeEnvironmentError(f"failed to inspect docker image {image_ref}: {proc.stderr.strip() or proc.stdout.strip()}")
    payload = json.loads(proc.stdout)
    if not payload:
        raise RuntimeEnvironmentError(f"docker image inspect returned no image: {image_ref}")
    return dict(payload[0])


def _docker_image_digest(image_info: dict[str, Any]) -> str | None:
    repo_digests = image_info.get("RepoDigests") or []
    if repo_digests:
        raw = str(repo_digests[0])
        return raw.split("@", 1)[1] if "@" in raw else raw
    image_id = image_info.get("Id")
    return str(image_id) if image_id else None


def _run_docker_import_preflight(*, image_ref: str, task: Any, host_root: Path, timeout_s: float) -> None:
    modules = list(task.runtime_spec.required_imports)
    code = (
        "import importlib, json, sys\n"
        "mods = {}\n"
        f"for name in {modules!r}:\n"
        "    mod = importlib.import_module(name)\n"
        "    mods[name] = {'version': getattr(mod, '__version__', None), 'file': getattr(mod, '__file__', None)}\n"
        "print(json.dumps({'python': sys.executable, 'modules': mods}, sort_keys=True))\n"
    )
    proc = _run_process(["docker", "run", "--rm", image_ref, "python", "-c", code], cwd=host_root, timeout_s=timeout_s)
    atomic_write_text(host_root / "import_preflight.json", proc.stdout if proc.returncode == 0 else json.dumps({"error": proc.stderr or proc.stdout}) + "\n")
    if proc.returncode != 0:
        raise RuntimeEnvironmentError(f"docker task image import preflight failed: {proc.stderr.strip() or proc.stdout.strip()}")


def _run_docker_public_seed_preflight(*, image_ref: str, task: Any, host_root: Path, timeout_s: float) -> None:
    candidate_spec = getattr(task.metadata, "candidate_spec", None)
    entry_rel = task.metadata.entrypoint_name if candidate_spec is None else candidate_spec.public_entrypoint.as_posix()
    entry_path = f"/opt/agentic-opt/tasks/{task.metadata.task_id}/public/{entry_rel}"
    code = (
        "import json\n"
        "from pathlib import Path\n"
        "from agentic_opt.task_registry import get_task\n"
        f"task = get_task({task.metadata.task_id!r})\n"
        f"result = task.verify_entry(Path({entry_path!r}))\n"
        "payload = {'valid': bool(result.get('valid')), 'status': result.get('status'), 'feedback': result.get('feedback')}\n"
        "print(json.dumps(payload, sort_keys=True))\n"
        "raise SystemExit(0 if payload['valid'] else 1)\n"
    )
    proc = _run_process(["docker", "run", "--rm", image_ref, "python", "-c", code], cwd=host_root, timeout_s=timeout_s)
    atomic_write_text(host_root / "public_seed_preflight.json", proc.stdout if proc.returncode == 0 else json.dumps({"error": proc.stderr or proc.stdout}) + "\n")
    if proc.returncode != 0:
        raise RuntimeEnvironmentError(f"docker task image public-seed preflight failed: {proc.stderr.strip() or proc.stdout.strip()}")


def _run_process(command: list[str], *, cwd: Path, timeout_s: float) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=str(cwd),
            env=_clean_env(),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except OSError as exc:
        raise RuntimeEnvironmentError(f"failed to execute {' '.join(command[:2])}: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeEnvironmentError(f"timed out executing {' '.join(command[:3])}") from exc


def _fingerprint_json(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _normalize_requirements(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        items = [raw]
    else:
        items = list(raw)
    requirements = [str(item).strip() for item in items if str(item).strip()]
    if any("\n" in item for item in requirements):
        raise ValueError("requirements must be individual pip requirement strings")
    return requirements


def _requirement_name(requirement: str) -> str:
    stripped = requirement.strip()
    if "://" in stripped or stripped.startswith(("git+", "file:")):
        return stripped
    match = re.match(r"([A-Za-z0-9_.-]+)", stripped)
    return match.group(1).lower().replace("_", "-") if match else stripped


def _pip_install(*, python_path: Path, requirements: list[str], cwd: Path) -> None:
    if not requirements:
        return
    proc = subprocess.run(
        [str(python_path), "-m", "pip", "install", *requirements],
        cwd=str(cwd),
        env=_clean_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    if proc.returncode != 0:
        raise RuntimeEnvironmentError(f"failed to install environment overlay requirements: {proc.stderr.strip() or proc.stdout.strip()}")


def _pip_freeze(python_path: Path) -> dict[str, Any]:
    proc = subprocess.run(
        [str(python_path), "-m", "pip", "freeze"],
        cwd=str(get_repo_root()),
        env=_clean_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if proc.returncode != 0:
        fallback = _metadata_freeze(python_path)
        if fallback is not None:
            return {
                **fallback,
                **_freeze_error_metadata(proc.stderr.strip() or proc.stdout.strip()),
            }
        return {"status": "unavailable", "error": proc.stderr.strip() or proc.stdout.strip()}
    return {
        "status": "ready",
        "format": "pip-freeze",
        "requirements": [line for line in proc.stdout.splitlines() if line.strip()],
    }


def _metadata_freeze(python_path: Path) -> dict[str, Any] | None:
    code = (
        "import importlib.metadata as metadata, json\n"
        "items = []\n"
        "for dist in metadata.distributions():\n"
        "    name = dist.metadata.get('Name')\n"
        "    version = dist.version\n"
        "    if name and version:\n"
        "        items.append(f'{name}=={version}')\n"
        "print(json.dumps(sorted(set(items), key=str.lower)))\n"
    )
    proc = subprocess.run(
        [str(python_path), "-c", code],
        cwd=str(get_repo_root()),
        env=_clean_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if proc.returncode != 0:
        return None
    try:
        requirements = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(requirements, list):
        return None
    return {
        "status": "ready",
        "format": "importlib-metadata-freeze",
        "requirements": [str(item) for item in requirements if item],
    }


def _freeze_error_metadata(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    digest = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()
    max_chars = 1200
    excerpt = raw if len(raw) <= max_chars else raw[:max_chars] + f"\n[TRUNCATED {len(raw) - max_chars} chars]"
    return {
        "pip_freeze_error_digest": f"sha256:{digest}",
        "pip_freeze_error_excerpt": excerpt,
    }


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _clean_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    return env
