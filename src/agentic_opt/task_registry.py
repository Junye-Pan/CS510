from __future__ import annotations

import importlib
import importlib.util
import os
import re
import sys
import types
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

from agentic_opt.task_api import DEFAULT_ENTRYPOINT_NAME, TaskProtocol, candidate_spec_for


_TASK_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_EXTERNAL_TASK_PACKAGE = "agentic_opt_external_tasks"


class TaskLoadError(RuntimeError):
    pass


def get_task(task_id: str) -> TaskProtocol:
    module = get_task_module(task_id)
    factory = getattr(module, "create_task", None)
    if not callable(factory):
        raise TaskLoadError(f"{module.__name__} must define create_task()")
    task = factory()
    _validate_task(task_id=task_id, task=task)
    return task


def get_task_module(task_id: str) -> ModuleType:
    if not _TASK_ID_RE.fullmatch(task_id):
        raise KeyError(f"invalid task_id: {task_id!r}")

    external = _load_external_task_module(task_id)
    if external is not None:
        return external

    raise KeyError(f"Unknown task_id: {task_id}")


def task_search_roots() -> list[Path]:
    roots: list[Path] = []
    env_value = os.environ.get("AO_TASKS_ROOTS") or os.environ.get("AO_TASKS_ROOT")
    if env_value:
        roots.extend(Path(item).expanduser() for item in env_value.split(os.pathsep) if item)
    roots.append(_repo_root() / "tasks")
    return _dedupe_paths(roots)


def _load_external_task_module(task_id: str) -> ModuleType | None:
    for root in task_search_roots():
        task_dir = root / task_id
        task_file = task_dir / "task.py"
        if not task_file.exists():
            continue
        return _load_task_file(
            task_id=task_id,
            tasks_root=root.resolve(),
            task_dir=task_dir.resolve(),
            task_file=task_file.resolve(),
        )
    return None


def _load_task_file(*, task_id: str, tasks_root: Path, task_dir: Path, task_file: Path) -> ModuleType:
    if tasks_root.name.isidentifier():
        return _load_importable_task_module(task_id=task_id, tasks_root=tasks_root)

    _ensure_package(_EXTERNAL_TASK_PACKAGE)
    package_name = f"{_EXTERNAL_TASK_PACKAGE}.{task_id}"
    _ensure_package(package_name, path=task_dir)
    module_name = f"{package_name}.task"
    previous = sys.modules.pop(module_name, None)
    try:
        spec = importlib.util.spec_from_file_location(module_name, task_file)
        if spec is None or spec.loader is None:
            raise TaskLoadError(f"Could not load task module from {task_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception:
        sys.modules.pop(module_name, None)
        if previous is not None:
            sys.modules[module_name] = previous
        raise


def _load_importable_task_module(*, task_id: str, tasks_root: Path) -> ModuleType:
    module_name = f"{tasks_root.name}.{task_id}.task"
    task_file = tasks_root / task_id / "task.py"
    existing = sys.modules.get(module_name)
    if existing is not None and _module_file(existing) == task_file.resolve():
        return existing

    parent = str(tasks_root.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    _ensure_package_search_path(tasks_root.name, tasks_root.resolve())
    importlib.invalidate_caches()
    for existing_name in list(sys.modules):
        if existing_name == module_name or existing_name.startswith(f"{tasks_root.name}.{task_id}."):
            sys.modules.pop(existing_name, None)
    return importlib.import_module(module_name)


def _ensure_package_search_path(package_name: str, path: Path) -> None:
    package = sys.modules.get(package_name)
    package_path = getattr(package, "__path__", None)
    if package is None or package_path is None:
        return
    paths = list(package_path)
    raw_path = str(path)
    if raw_path not in paths:
        paths.insert(0, raw_path)
        package.__path__ = paths  # type: ignore[attr-defined]


def _ensure_package(name: str, *, path: Path | None = None) -> ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__package__ = name
        module.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = module
    if path is not None:
        paths = list(getattr(module, "__path__", []))
        raw_path = str(path)
        if raw_path not in paths:
            paths.insert(0, raw_path)
            module.__path__ = paths  # type: ignore[attr-defined]
    return module


def _validate_task(*, task_id: str, task: Any) -> None:
    metadata = getattr(task, "metadata", None)
    if metadata is None:
        raise TaskLoadError(f"task {task_id} does not expose metadata")
    if getattr(metadata, "task_id", None) != task_id:
        raise TaskLoadError(
            f"task metadata task_id mismatch: requested {task_id!r}, got {getattr(metadata, 'task_id', None)!r}"
        )
    spec = candidate_spec_for(task)
    entrypoint_name = getattr(metadata, "entrypoint_name", DEFAULT_ENTRYPOINT_NAME)
    if getattr(metadata, "candidate_spec", None) is not None and entrypoint_name != DEFAULT_ENTRYPOINT_NAME:
        raise TaskLoadError("tasks should set either metadata.entrypoint_name or metadata.candidate_spec, not both")
    if getattr(metadata, "candidate_spec", None) is None and spec.entrypoint_name != entrypoint_name:
        raise TaskLoadError("metadata.entrypoint_name did not map to candidate spec")
    public_dir = Path(getattr(task, "public_dir"))
    for relative in ("TASK.md", "public_contract.md"):
        if not (public_dir / relative).exists():
            raise TaskLoadError(f"task {task_id} missing public/{relative}")
    if not (public_dir / spec.public_entrypoint).exists():
        raise TaskLoadError(f"task {task_id} missing public/{spec.public_entrypoint.as_posix()}")
    for method_name in ("verify_entry", "probe_entry", "evaluate_entry"):
        if not callable(getattr(task, method_name, None)):
            raise TaskLoadError(f"task {task_id} missing callable {method_name}()")


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(resolved)
    return result


def _module_file(module: ModuleType) -> Path | None:
    path = getattr(module, "__file__", None)
    if not path:
        return None
    return Path(path).resolve()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
