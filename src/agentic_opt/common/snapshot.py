from __future__ import annotations

import fnmatch
import shutil
from pathlib import Path


EXCLUDED_NAMES = {
    ".git",
    ".agents",
    ".run",
    ".runtime_ipc",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".venv",
    ".DS_Store",
    "AGENTS.md",
    "DIRECTION.md",
    "WORKLOG.md",
    "bin",
    "venv",
    "node_modules",
    "reference",
    "artifacts",
    "findings",
    "local_tools",
    "shared_tools",
    "knowledge_base",
    "tmp",
    "results",
    "private",
}

EXCLUDED_GLOBS = {"*.pyc"}


def should_exclude(relative_path: Path) -> bool:
    parts = relative_path.parts
    if any(part in EXCLUDED_NAMES for part in parts):
        return True
    as_posix = relative_path.as_posix()
    return any(fnmatch.fnmatch(as_posix, pattern) for pattern in EXCLUDED_GLOBS)


def copy_snapshot(source_dir: Path, destination_dir: Path) -> list[str]:
    copied: list[str] = []
    destination_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(source_dir.rglob("*")):
        relative = path.relative_to(source_dir)
        if should_exclude(relative):
            continue
        target = destination_dir / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(relative.as_posix())
    return copied


def export_snapshot(source_dir: Path, destination_dir: Path, *, force: bool = False) -> None:
    if destination_dir.exists():
        if not force:
            raise FileExistsError(f"destination already exists: {destination_dir}")
        if destination_dir.is_dir():
            shutil.rmtree(destination_dir)
        else:
            destination_dir.unlink()
    shutil.copytree(source_dir, destination_dir)
