from __future__ import annotations

import importlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType


def import_numpy() -> ModuleType:
    real_numpy = _try_import_real_numpy()
    if real_numpy is not None:
        return real_numpy
    return _load_repo_numpy_compat()


@lru_cache(maxsize=1)
def _load_repo_numpy_compat() -> ModuleType:
    compat_path = _repo_numpy_compat_path()
    spec = importlib.util.spec_from_file_location("agentic_opt_repo_numpy_compat", compat_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load numpy compatibility shim from {compat_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _try_import_real_numpy() -> ModuleType | None:
    repo_root = _repo_root()
    repo_numpy = _repo_numpy_compat_path()
    previous_sys_path = list(sys.path)
    existing_numpy = sys.modules.get("numpy")
    removed_repo_numpy = False

    if existing_numpy is not None and _module_file(existing_numpy) == repo_numpy:
        removed_repo_numpy = True
        sys.modules.pop("numpy", None)

    try:
        sys.path[:] = [
            entry
            for entry in previous_sys_path
            if _resolve_sys_path_entry(entry) != repo_root
        ]
        try:
            return importlib.import_module("numpy")
        except ModuleNotFoundError:
            return None
    finally:
        sys.path[:] = previous_sys_path
        if removed_repo_numpy and "numpy" not in sys.modules and existing_numpy is not None:
            sys.modules["numpy"] = existing_numpy


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
