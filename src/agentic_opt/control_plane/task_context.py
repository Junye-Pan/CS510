from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from agentic_opt.common.atomic import atomic_write_text
from agentic_opt.common.ids import make_run_id
from agentic_opt.task_registry import get_task

from .docker_runtime import DockerMount


TASK_CONTEXT_WORKSPACE_PATH = "task"
TASK_CONTEXT_READONLY_TARGETS = (
    "task",
    "task/knowledge",
    "task/public_files",
    "task/research_directions",
)


class TaskContextError(RuntimeError):
    pass


class TaskContextMountConflictError(TaskContextError):
    pass


def ensure_task_context_snapshot(*, task_id: str, state_root: Path) -> dict[str, Any]:
    """Create or reuse a server-owned read-only snapshot of worker-visible task context."""

    state_root = state_root.resolve()
    snapshots_root = state_root / "task_contexts" / _safe_token(task_id)
    staging_parent = state_root / "task_contexts" / "_staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_root = staging_parent / make_run_id("task_context")
    task_root = staging_root / TASK_CONTEXT_WORKSPACE_PATH
    task_root.mkdir(parents=True, exist_ok=False)
    try:
        _materialize_task_context(task_id=task_id, task_root=task_root)
        digest = digest_directory(task_root)
        final_root = snapshots_root / digest.replace(":", "_")
        final_task_root = final_root / TASK_CONTEXT_WORKSPACE_PATH
        if final_task_root.exists():
            shutil.rmtree(staging_root)
        else:
            final_root.parent.mkdir(parents=True, exist_ok=True)
            final_root.mkdir(parents=True, exist_ok=False)
            shutil.move(str(task_root), str(final_task_root))
            _make_tree_read_only(final_task_root)
            _write_snapshot_manifest(task_id=task_id, snapshot_root=final_root, task_root=final_task_root, digest=digest)
            shutil.rmtree(staging_root, ignore_errors=True)
        return task_context_snapshot_metadata(task_id=task_id, snapshot_root=final_root, task_root=final_task_root, digest=digest)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise


def task_context_snapshot_metadata(*, task_id: str, snapshot_root: Path, task_root: Path, digest: str) -> dict[str, Any]:
    inventory = _task_context_inventory(task_root=task_root)
    knowledge_inventory = _read_json(task_root / "knowledge_inventory.json") or {
        "available": False,
        "workspace_path": "task/knowledge",
        "digest": None,
        "file_count": 0,
        "size_bytes": 0,
        "files": [],
        "manifest": None,
    }
    return {
        "task_id": task_id,
        "workspace_path": TASK_CONTEXT_WORKSPACE_PATH,
        "digest": digest,
        "snapshot_root": str(snapshot_root.resolve()),
        "task_path": str(task_root.resolve()),
        "file_count": inventory["file_count"],
        "size_bytes": inventory["size_bytes"],
        "files": inventory["files"],
        "task_knowledge": knowledge_inventory,
    }


def materialize_task_context_snapshot(
    *,
    snapshot: dict[str, Any],
    workspace_root: Path,
    provider: str = "local_venv",
    force: bool = True,
) -> dict[str, Any]:
    """Mirror a canonical snapshot into workspace/task when a provider cannot mount it."""

    destination = workspace_root.resolve() / TASK_CONTEXT_WORKSPACE_PATH
    expected_digest = str(snapshot.get("digest") or "")
    if destination.exists():
        verification = verify_task_context_path(task_path=destination, expected_digest=expected_digest)
        if verification["ok"]:
            return _local_enforcement_metadata(snapshot=snapshot, workspace_task_path=destination, provider=provider, verification=verification)
        if not force:
            raise TaskContextError(f"workspace task context already exists with unexpected digest: {destination}")
        _make_tree_writable(destination)
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    source = Path(str(snapshot["task_path"])).resolve()
    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    _make_tree_read_only(destination)
    verification = verify_task_context_path(task_path=destination, expected_digest=expected_digest)
    if not verification["ok"]:
        raise TaskContextError(f"materialized task context digest mismatch: {verification}")
    return _local_enforcement_metadata(snapshot=snapshot, workspace_task_path=destination, provider=provider, verification=verification)


def verify_task_context_path(*, task_path: Path, expected_digest: str | None) -> dict[str, Any]:
    task_path = task_path.resolve()
    if not task_path.exists():
        return {"ok": False, "status": "missing", "task_path": str(task_path), "expected_digest": expected_digest}
    if not task_path.is_dir():
        return {"ok": False, "status": "not_directory", "task_path": str(task_path), "expected_digest": expected_digest}
    actual = digest_directory(task_path)
    return {
        "ok": bool(expected_digest) and actual == expected_digest,
        "status": "matched" if expected_digest and actual == expected_digest else "mismatch",
        "task_path": str(task_path),
        "expected_digest": expected_digest,
        "actual_digest": actual,
    }


def verify_task_context_snapshot(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    if not snapshot:
        return {"ok": False, "status": "missing_snapshot"}
    task_path = snapshot.get("task_path")
    return verify_task_context_path(task_path=Path(str(task_path)), expected_digest=str(snapshot.get("digest") or ""))


def verify_request_task_context(request: dict[str, Any]) -> dict[str, Any]:
    snapshot = request.get("task_context") or {}
    snapshot_check = verify_task_context_snapshot(snapshot)
    workspace_check: dict[str, Any] | None = None
    workspace_root = request.get("workspace_root")
    if workspace_root:
        workspace_task_path = Path(str(workspace_root)).resolve() / TASK_CONTEXT_WORKSPACE_PATH
        if workspace_task_path.exists():
            workspace_check = verify_task_context_path(
                task_path=workspace_task_path,
                expected_digest=str(snapshot.get("digest") or ""),
            )
    ok = bool(snapshot_check.get("ok")) and (workspace_check is None or bool(workspace_check.get("ok")))
    return {"ok": ok, "snapshot": snapshot_check, "workspace": workspace_check}


def docker_task_context_mount(*, snapshot: dict[str, Any], workspace_root: Path) -> DockerMount:
    return DockerMount(
        source=Path(str(snapshot["task_path"])).resolve(),
        target=str((workspace_root.resolve() / TASK_CONTEXT_WORKSPACE_PATH)),
        read_only=True,
    )


def append_docker_task_context_mount(
    *,
    mounts: list[DockerMount],
    snapshot: dict[str, Any],
    workspace_root: Path,
) -> list[DockerMount]:
    mount = docker_task_context_mount(snapshot=snapshot, workspace_root=workspace_root)
    reject_writable_task_context_mounts(mounts=mounts, protected_targets=[mount.target])
    return [*mounts, mount]


def reject_writable_task_context_mounts(*, mounts: list[DockerMount], protected_targets: list[str]) -> None:
    protected = [Path(target).as_posix().rstrip("/") for target in protected_targets if target]
    protected.extend(TASK_CONTEXT_READONLY_TARGETS)
    for mount in mounts:
        if mount.read_only:
            continue
        target = Path(str(mount.target)).as_posix().rstrip("/")
        for protected_target in protected:
            protected_target = Path(protected_target).as_posix().rstrip("/")
            if target == protected_target or target.startswith(f"{protected_target}/"):
                raise TaskContextMountConflictError(
                    f"writable Docker mount targets read-only task context: {mount.target}"
                )


def local_task_context_enforcement(
    *,
    snapshot: dict[str, Any],
    workspace_root: Path | None = None,
    provider: str = "local_venv",
    verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    workspace_task_path = (workspace_root.resolve() / TASK_CONTEXT_WORKSPACE_PATH) if workspace_root else None
    return _local_enforcement_metadata(
        snapshot=snapshot,
        workspace_task_path=workspace_task_path,
        provider=provider,
        verification=verification,
    )


def docker_task_context_enforcement(*, snapshot: dict[str, Any], workspace_root: Path) -> dict[str, Any]:
    mount = docker_task_context_mount(snapshot=snapshot, workspace_root=workspace_root)
    return {
        "provider": "docker_image",
        "workspace_path": TASK_CONTEXT_WORKSPACE_PATH,
        "digest": snapshot.get("digest"),
        "snapshot_task_path": snapshot.get("task_path"),
        "container_task_path": mount.target,
        "mechanism": "docker_readonly_bind_mount",
        "provider_enforced_readonly": True,
        "policy_weakened": False,
        "mount": {"source": str(mount.source), "target": mount.target, "read_only": mount.read_only},
    }


def digest_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        if file_path.is_symlink():
            raise PermissionError(f"task context may not contain symlinks: {file_path}")
        digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _materialize_task_context(*, task_id: str, task_root: Path) -> None:
    task = get_task(task_id)
    public_dir = task.public_dir.resolve()
    if public_dir.is_symlink() or not public_dir.is_dir():
        raise ValueError(f"task public dir must be a directory: {public_dir}")
    _copy_required_text(public_dir / "TASK.md", task_root / "TASK.md")
    _copy_required_text(public_dir / "public_contract.md", task_root / "public_contract.md")
    _copy_public_files(public_dir=public_dir, destination=task_root / "public_files")
    _copy_optional_tree(public_dir / "research_directions", task_root / "research_directions")
    knowledge_inventory = _copy_knowledge(public_dir=public_dir, destination=task_root / "knowledge")
    atomic_write_text(task_root / "knowledge_inventory.json", json.dumps(knowledge_inventory, indent=2, sort_keys=True) + "\n")
    manifest = {
        "task_id": task_id,
        "workspace_path": TASK_CONTEXT_WORKSPACE_PATH,
        "source": "task_package_public",
        "contains": [
            "TASK.md",
            "public_contract.md",
            "public_files/",
            "knowledge/",
            "knowledge_inventory.json",
            "research_directions/",
        ],
    }
    atomic_write_text(task_root / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _copy_required_text(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_public_files(*, public_dir: Path, destination: Path) -> None:
    allowed_suffixes = {".md", ".txt", ".json", ".yaml", ".yml"}
    destination.mkdir(parents=True, exist_ok=True)
    for source in sorted(public_dir.rglob("*")):
        if source.is_symlink():
            raise PermissionError(f"task public context may not contain symlinks: {source}")
        if not source.is_file() or source.suffix.lower() not in allowed_suffixes:
            continue
        relative = source.relative_to(public_dir)
        if relative.parts and relative.parts[0] == "knowledge":
            continue
        target = _safe_child(destination, relative.as_posix())
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _copy_knowledge(*, public_dir: Path, destination: Path) -> dict[str, Any]:
    source_root = public_dir / "knowledge"
    destination.mkdir(parents=True, exist_ok=True)
    manifest = None
    if not source_root.exists():
        return _knowledge_inventory(source_root=source_root, destination=destination, manifest=manifest)
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError(f"task knowledge must be a directory: {source_root}")
    manifest = _load_knowledge_manifest(source_root.resolve())
    _copy_optional_tree(source_root, destination)
    return _knowledge_inventory(source_root=source_root.resolve(), destination=destination, manifest=manifest)


def _copy_optional_tree(source_root: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if not source_root.exists():
        return
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError(f"task context path must be a directory: {source_root}")
    resolved_source = source_root.resolve()
    for source in sorted(resolved_source.rglob("*")):
        if source.is_symlink():
            raise PermissionError(f"task context may not contain symlinks: {source}")
        relative = source.relative_to(resolved_source)
        target = _safe_child(destination, relative.as_posix())
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def _load_knowledge_manifest(root: Path) -> dict[str, Any] | None:
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
        target = _safe_child(root, relative).resolve()
        if not target.is_relative_to(root):
            raise PermissionError(f"task knowledge manifest path escapes {root}: {relative}")
        if not target.exists():
            raise FileNotFoundError(target)
    return manifest


def _knowledge_inventory(*, source_root: Path, destination: Path, manifest: dict[str, Any] | None) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in destination.rglob("*") if item.is_file()):
        relative = path.relative_to(destination).as_posix()
        files.append(
            {
                "relative_path": relative,
                "workspace_path": f"task/knowledge/{relative}",
                "digest": digest_file(path),
                "size_bytes": path.stat().st_size,
                "read_only": True,
            }
        )
    return {
        "available": source_root.exists(),
        "source_path": str(source_root),
        "workspace_path": "task/knowledge",
        "digest": digest_directory(destination) if files else None,
        "file_count": len(files),
        "size_bytes": sum(int(item["size_bytes"]) for item in files),
        "files": files,
        "manifest": manifest,
    }


def _task_context_inventory(*, task_root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in task_root.rglob("*") if item.is_file()):
        relative = path.relative_to(task_root).as_posix()
        files.append(
            {
                "relative_path": relative,
                "workspace_path": f"task/{relative}",
                "digest": digest_file(path),
                "size_bytes": path.stat().st_size,
                "read_only": not bool(path.stat().st_mode & 0o222),
            }
        )
    return {"file_count": len(files), "size_bytes": sum(int(item["size_bytes"]) for item in files), "files": files}


def _write_snapshot_manifest(*, task_id: str, snapshot_root: Path, task_root: Path, digest: str) -> None:
    manifest = {
        "task_id": task_id,
        "schema_version": "agentic_opt.task_context_snapshot.v1",
        "digest": digest,
        "task_path": str(task_root.resolve()),
        "workspace_path": TASK_CONTEXT_WORKSPACE_PATH,
        "inventory": _task_context_inventory(task_root=task_root),
    }
    atomic_write_text(snapshot_root / "snapshot.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _local_enforcement_metadata(
    *,
    snapshot: dict[str, Any],
    workspace_task_path: Path | None,
    provider: str,
    verification: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "provider": provider,
        "workspace_path": TASK_CONTEXT_WORKSPACE_PATH,
        "digest": snapshot.get("digest"),
        "snapshot_task_path": snapshot.get("task_path"),
        "workspace_task_path": str(workspace_task_path.resolve()) if workspace_task_path else None,
        "mechanism": "snapshot_copy_chmod_digest_guard",
        "provider_enforced_readonly": False,
        "policy_weakened": True,
        "policy_weakened_reason": "local same-uid processes can bypass chmod/App Server subdirectory policy; digest guard detects mutation",
        "verification": verification,
    }


def _make_tree_read_only(root: Path) -> None:
    for item in sorted(root.rglob("*"), reverse=True):
        if item.is_file():
            item.chmod(item.stat().st_mode & ~0o222)
        elif item.is_dir():
            item.chmod(item.stat().st_mode & ~0o222)
    root.chmod(root.stat().st_mode & ~0o222)


def _make_tree_writable(root: Path) -> None:
    if root.is_symlink():
        return
    for item in root.rglob("*"):
        try:
            if item.is_dir():
                item.chmod(item.stat().st_mode | 0o700)
            else:
                item.chmod(item.stat().st_mode | 0o600)
        except FileNotFoundError:
            continue
    root.chmod(root.stat().st_mode | 0o700)


def _safe_child(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ValueError(f"unsafe task context path: {relative}")
    target = (root / path).resolve()
    root_resolved = root.resolve()
    if not target.is_relative_to(root_resolved):
        raise PermissionError(f"task context path escapes {root}: {relative}")
    return target


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _safe_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)[:120] or "task"
