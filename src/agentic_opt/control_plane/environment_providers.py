from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .docker_runtime import DockerMount, build_docker_run_command
from .process_env import build_subprocess_env, sanitize_env


class EnvironmentProviderError(RuntimeError):
    pass


class DockerImageReferenceError(EnvironmentProviderError):
    pass


@dataclass(frozen=True)
class EnvironmentRunSpec:
    execution: dict[str, Any]
    command: str | list[Any]
    cwd: Path
    env: dict[str, Any] = field(default_factory=dict)
    mounts: list[DockerMount] = field(default_factory=list)
    workdir: str | None = None
    network_policy: dict[str, Any] = field(default_factory=dict)
    requested_network_mode: str | None = None
    requires_control_plane: bool = False
    control_plane_relay_socket: Path | None = None
    control_plane_relay_url: str | None = None
    container_name: str | None = None
    pids_limit: int | None = None
    memory: str | None = None
    cpus: str | None = None
    add_hosts: list[str] = field(default_factory=list)
    require_immutable_image: bool = True
    allow_mutable_image_ref: bool = False


@dataclass(frozen=True)
class EnvironmentRunPlan:
    command: list[str] | str
    cwd: Path
    env: dict[str, str]
    metadata: dict[str, Any]
    network_enforcement: dict[str, Any] = field(default_factory=dict)


class EnvironmentProvider:
    name = "unknown"

    def build_run_plan(self, spec: EnvironmentRunSpec) -> EnvironmentRunPlan:
        raise NotImplementedError


class LocalVenvProvider(EnvironmentProvider):
    name = "local_venv"

    def build_run_plan(self, spec: EnvironmentRunSpec) -> EnvironmentRunPlan:
        env = _merged_execution_env(spec)
        return EnvironmentRunPlan(
            command=spec.command,
            cwd=spec.cwd,
            env=build_subprocess_env(env),
            metadata={
                "provider": self.name,
                "environment_id": spec.execution.get("environment_id"),
                "environment_overlay_id": spec.execution.get("environment_overlay_id"),
                "python_path": spec.execution.get("python_path"),
                "root_path": spec.execution.get("root_path"),
            },
        )


class DockerImageProvider(EnvironmentProvider):
    name = "docker_image"

    def build_run_plan(self, spec: EnvironmentRunSpec) -> EnvironmentRunPlan:
        image = docker_image_reference(
            spec.execution,
            require_immutable=spec.require_immutable_image,
            allow_mutable=spec.allow_mutable_image_ref,
        )
        command, enforcement = build_docker_run_command(
            image=image["reference"],
            command=spec.command,
            mounts=spec.mounts,
            workdir=spec.workdir or docker_workdir(spec.execution),
            network_policy=spec.network_policy,
            requested_network_mode=spec.requested_network_mode,
            requires_control_plane=spec.requires_control_plane,
            control_plane_relay_socket=spec.control_plane_relay_socket,
            control_plane_relay_url=spec.control_plane_relay_url,
            env=_merged_execution_env(spec),
            container_name=spec.container_name,
            pids_limit=spec.pids_limit,
            memory=spec.memory,
            cpus=spec.cpus,
            add_hosts=spec.add_hosts,
        )
        policy_weakened = bool(enforcement.get("policy_weakened") or image.get("policy_weakened"))
        enforcement = {**enforcement, "policy_weakened": policy_weakened}
        return EnvironmentRunPlan(
            command=command,
            cwd=spec.cwd,
            env=build_subprocess_env(),
            metadata={
                "provider": self.name,
                "environment_id": spec.execution.get("environment_id"),
                "environment_overlay_id": spec.execution.get("environment_overlay_id"),
                "image": image,
                "workdir": spec.workdir or docker_workdir(spec.execution),
                "mounts": [docker_mount_to_json(mount) for mount in spec.mounts],
                "network_enforcement": enforcement,
                "resource_limits": {
                    "pids_limit": spec.pids_limit,
                    "memory": spec.memory,
                    "cpus": spec.cpus,
                },
                "policy_weakened": policy_weakened,
            },
            network_enforcement=enforcement,
        )


def provider_for_execution(execution: dict[str, Any]) -> EnvironmentProvider:
    provider = str(execution.get("provider") or "local_venv")
    if provider == "docker_image":
        return DockerImageProvider()
    return LocalVenvProvider()


def _merged_execution_env(spec: EnvironmentRunSpec) -> dict[str, str]:
    return {
        **sanitize_env(spec.execution.get("exports") or {}),
        **sanitize_env(spec.env),
    }


def docker_image_reference(
    execution: dict[str, Any],
    *,
    require_immutable: bool,
    allow_mutable: bool,
) -> dict[str, Any]:
    record = execution.get("record") or {}
    base_record = execution.get("base_record") or {}
    metadata = {**(base_record.get("metadata") or {}), **(record.get("metadata") or {})}
    lock = {**(base_record.get("lock") or {}), **(record.get("lock") or {})}
    repo_digests = [str(item) for item in (lock.get("repo_digests") or []) if item]
    if repo_digests:
        return {
            "reference": repo_digests[0],
            "kind": "repo_digest",
            "immutable": True,
            "image_ref": metadata.get("image_ref") or lock.get("image_ref"),
            "image_digest": lock.get("image_digest"),
            "repo_digests": repo_digests,
            "reproducibility": "registry_digest",
            "policy_weakened": False,
        }
    image_id = metadata.get("image_id") or lock.get("image_id") or lock.get("image_digest")
    if image_id and str(image_id).startswith("sha256:"):
        return {
            "reference": str(image_id),
            "kind": "local_image_id",
            "immutable": True,
            "image_ref": metadata.get("image_ref") or lock.get("image_ref"),
            "image_digest": lock.get("image_digest") or str(image_id),
            "repo_digests": repo_digests,
            "reproducibility": "local_image_id",
            "policy_weakened": False,
        }
    image_ref = metadata.get("image_ref") or lock.get("image_ref")
    if image_ref and (allow_mutable or not require_immutable):
        return {
            "reference": str(image_ref),
            "kind": "mutable_ref",
            "immutable": False,
            "image_ref": str(image_ref),
            "image_digest": lock.get("image_digest"),
            "repo_digests": repo_digests,
            "reproducibility": "mutable_ref",
            "policy_weakened": True,
        }
    raise DockerImageReferenceError(
        f"docker_image environment requires an immutable image id or repo digest: {execution.get('environment_id')}"
    )


def docker_workdir(execution: dict[str, Any]) -> str:
    record = execution.get("record") or {}
    metadata = record.get("metadata") or {}
    if execution.get("kind") == "overlay":
        metadata = {**((execution.get("base_record") or {}).get("metadata") or {}), **metadata}
    return str((metadata.get("runner") or {}).get("workdir") or metadata.get("container_root") or "/opt/agentic-opt")


def docker_runner_metadata(execution: dict[str, Any], image: dict[str, Any] | None = None) -> dict[str, Any]:
    record = execution.get("record") or {}
    metadata = record.get("metadata") or {}
    if execution.get("kind") == "overlay":
        metadata = {**((execution.get("base_record") or {}).get("metadata") or {}), **metadata}
        lock = {**((execution.get("base_record") or {}).get("lock") or {}), **(record.get("lock") or {})}
    else:
        lock = record.get("lock") or {}
    return {
        "provider": execution.get("provider"),
        "image_ref": metadata.get("image_ref") or lock.get("image_ref"),
        "image_digest": metadata.get("image_digest") or lock.get("image_digest"),
        "image_reference": (image or {}).get("reference"),
        "image_reference_kind": (image or {}).get("kind"),
        "workdir": docker_workdir(execution),
        "python_path": execution.get("python_path"),
        "policy_weakened": bool((image or {}).get("policy_weakened")),
    }


def docker_mount_to_json(mount: DockerMount) -> dict[str, Any]:
    return {"source": str(mount.source), "target": mount.target, "read_only": mount.read_only}
