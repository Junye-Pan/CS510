from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from agentic_opt.common.config import get_repo_root


def runs_root() -> Path:
    return get_repo_root() / "runs"


def default_environment_manifest_path() -> Path:
    return runs_root() / "models" / "qwen35_4b_environment_manifest.json"


def default_download_manifest_path() -> Path:
    return runs_root() / "models" / "qwen35_4b_manifest.json"


def default_model_manifest_path() -> Path:
    complete = default_environment_manifest_path()
    return complete if complete.exists() else default_download_manifest_path()


def new_run_dir(kind: str) -> Path:
    safe_kind = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in kind).strip("._-") or "run"
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    base = runs_root() / f"{safe_kind}_{stamp}"
    path = base
    suffix = 1
    while path.exists():
        suffix += 1
        path = runs_root() / f"{safe_kind}_{stamp}_{suffix}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
