from __future__ import annotations

import importlib
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType


@lru_cache(maxsize=1)
def import_scipy() -> ModuleType:
    repo_root = _repo_root()
    repo_numpy = _repo_numpy_compat_path()
    previous_sys_path = list(sys.path)
    existing_numpy = sys.modules.get("numpy")
    removed_repo_numpy = None
    if existing_numpy is not None and _module_file(existing_numpy) == repo_numpy:
        removed_repo_numpy = sys.modules.pop("numpy", None)

    try:
        sys.path[:] = [
            entry
            for entry in previous_sys_path
            if _resolve_sys_path_entry(entry) != repo_root
        ]
        importlib.import_module("numpy")
        return importlib.import_module("scipy")
    finally:
        sys.path[:] = previous_sys_path
        if removed_repo_numpy is not None and "numpy" not in sys.modules:
            sys.modules["numpy"] = removed_repo_numpy


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _repo_numpy_compat_path() -> Path:
    return (Path(__file__).resolve().parent / "numpy_compat.py").resolve()


def _module_file(module: ModuleType) -> Path | None:
    path = getattr(module, "__file__", None)
    if not path:
        return None
    return Path(path).resolve()


def _resolve_sys_path_entry(entry: str) -> Path:
    if not entry:
        return Path.cwd().resolve()
    return Path(entry).expanduser().resolve()
