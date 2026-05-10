from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from agentic_opt.common.config import get_repo_root
from agentic_opt.common.runtime_env import RuntimeEnvironmentError, prepare_task_runtime
from agentic_opt.task_registry import get_task

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

    def ensure_framework_environment(self) -> dict[str, Any]:
        python_path = Path(sys.executable).resolve()
        fingerprint = _fingerprint_json({"kind": "framework", "python": str(python_path), "version": sys.version})
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

    def ensure_task_environment(self, task_id: str, *, experiment_id: str | None = None) -> dict[str, Any]:
        task = get_task(task_id)
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
    ) -> dict[str, Any]:
        if overlay_id and allow_overlay:
            overlay = self.repository.get_environment_overlay(overlay_id)
            if overlay is None:
                raise KeyError(overlay_id)
            if overlay["status"] != "ready":
                raise RuntimeEnvironmentError(f"environment overlay is not ready: {overlay_id} status={overlay['status']}")
            return {
                "kind": "overlay",
                "environment_id": overlay["base_environment_id"],
                "environment_overlay_id": overlay["overlay_id"],
                "python_path": overlay["python_path"],
                "root_path": overlay["root_path"],
                "exports": self.exports_for_overlay(overlay),
                "record": overlay,
            }
        if environment_id:
            environment = self.repository.get_environment(environment_id)
            if environment is None:
                raise KeyError(environment_id)
            if environment["status"] != "ready":
                raise RuntimeEnvironmentError(f"environment is not ready: {environment_id} status={environment['status']}")
        else:
            environment = self.ensure_task_environment(task_id, experiment_id=experiment_id)
        return {
            "kind": "environment",
            "environment_id": environment["environment_id"],
            "environment_overlay_id": None,
            "python_path": environment["python_path"],
            "root_path": environment["root_path"],
            "exports": self.exports_for_environment(environment),
            "record": environment,
        }

    def exports_for_environment(self, environment: dict[str, Any]) -> dict[str, str]:
        exports = {
            "AO_ENVIRONMENT_ID": environment["environment_id"],
            "AO_ENVIRONMENT_TYPE": environment["environment_type"],
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


def _task_environment_id(*, task_id: str, fingerprint: str) -> str:
    safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id)
    return f"env_task_{safe_task_id}_{fingerprint}"


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
        return {"status": "unavailable", "error": proc.stderr.strip() or proc.stdout.strip()}
    return {
        "status": "ready",
        "format": "pip-freeze",
        "requirements": [line for line in proc.stdout.splitlines() if line.strip()],
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
