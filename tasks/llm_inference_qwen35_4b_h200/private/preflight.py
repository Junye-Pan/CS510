from __future__ import annotations

import importlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .environment_manifest import (
    load_environment_manifest,
    manifest_public_view,
    resolve_model_path_from_manifest,
    validate_environment_manifest,
)
from .run_artifacts import default_model_manifest_path


LIVE_ENABLE_ENV = "AO_LLM_KERNEL_ENABLE_LIVE"
ENV_MANIFEST_ENV = "AO_LLM_KERNEL_ENV_MANIFEST"
REQUIRE_MODEL_ENV = "AO_LLM_KERNEL_REQUIRE_MODEL"
ENFORCE_READONLY_MODEL_ENV = "AO_LLM_KERNEL_ENFORCE_READONLY_MODEL"
PROBE_MODEL_SMOKE_ENV = "AO_LLM_KERNEL_PROBE_MODEL_SMOKE"


def live_verifier_enabled() -> bool:
    return os.environ.get(LIVE_ENABLE_ENV) in {"1", "true", "TRUE", "yes", "on"}


def model_preflight_required() -> bool:
    return os.environ.get(REQUIRE_MODEL_ENV) in {"1", "true", "TRUE", "yes", "on"}


def probe_model_smoke_enabled() -> bool:
    return os.environ.get(PROBE_MODEL_SMOKE_ENV) in {"1", "true", "TRUE", "yes", "on"}


def readonly_model_required() -> bool:
    return os.environ.get(ENFORCE_READONLY_MODEL_ENV) in {"1", "true", "TRUE", "yes", "on"}


def run_live_preflight() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    manifest_path = os.environ.get(ENV_MANIFEST_ENV)
    if not manifest_path and model_preflight_required() and default_model_manifest_path().exists():
        manifest_path = str(default_model_manifest_path())
    manifest: dict[str, Any] = {}
    require_model = model_preflight_required()
    if manifest_path:
        try:
            manifest = load_environment_manifest(Path(manifest_path))
            checks.append({"name": "environment_manifest", "status": "passed", "message": manifest_path})
        except Exception as exc:
            checks.append({"name": "environment_manifest", "status": "failed", "message": str(exc)})
    else:
        checks.append(
            {
                "name": "environment_manifest",
                "status": "failed" if require_model else "skipped",
                "message": f"{ENV_MANIFEST_ENV} is unset",
            }
        )

    imports = ("torch", "triton", "vllm") if require_model else ("torch", "triton")
    checks.extend(_check_python_imports(imports))
    if manifest:
        checks.extend(
            validate_environment_manifest(
                manifest,
                require_model=require_model,
                require_runtime=require_model,
                require_baselines=require_model,
                enforce_readonly_model=readonly_model_required(),
            )
        )
    else:
        checks.append(_check_h200())
        checks.append(_check_model_path(manifest, required=require_model))
    valid = all(item["status"] in {"passed", "skipped"} for item in checks)
    return {
        "valid": valid,
        "checks": checks,
        "manifest": manifest_public_view(manifest) if manifest else manifest,
        "model_preflight_required": require_model,
    }


def _check_python_imports(modules: tuple[str, ...]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", None)
            checks.append({"name": f"import_{module_name}", "status": "passed", "message": version})
        except Exception as exc:
            checks.append({"name": f"import_{module_name}", "status": "failed", "message": str(exc)})
    return checks


def _check_h200() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except Exception as exc:
        return {"name": "h200_gpu", "status": "failed", "message": str(exc)}
    if proc.returncode != 0:
        return {"name": "h200_gpu", "status": "failed", "message": proc.stderr.strip() or proc.stdout.strip()}
    output = proc.stdout.strip()
    if "H200" not in output:
        return {"name": "h200_gpu", "status": "failed", "message": output or "no GPU output"}
    return {"name": "h200_gpu", "status": "passed", "message": output}


def _check_model_path(manifest: dict[str, Any], *, required: bool) -> dict[str, Any]:
    model = manifest.get("model") if isinstance(manifest, dict) else None
    local_path = model.get("local_path") if isinstance(model, dict) else None
    if local_path is None:
        local_path = manifest.get("local_path") if isinstance(manifest, dict) else None
    if not isinstance(local_path, str) or not local_path:
        return {
            "name": "model_path",
            "status": "failed" if required else "skipped",
            "message": "model.local_path missing from manifest",
        }
    path = Path(local_path)
    if not path.exists():
        return {"name": "model_path", "status": "failed", "message": f"{local_path} does not exist"}
    if readonly_model_required() and os.access(path, os.W_OK):
        return {"name": "model_path", "status": "failed", "message": f"{local_path} is writable"}
    writable_note = " (writable dev copy)" if os.access(path, os.W_OK) else ""
    return {"name": "model_path", "status": "passed", "message": local_path + writable_note}


def resolve_model_path() -> Path:
    manifest_path = os.environ.get(ENV_MANIFEST_ENV)
    if manifest_path is None and default_model_manifest_path().exists():
        manifest_path = str(default_model_manifest_path())
    if manifest_path is None:
        raise FileNotFoundError(f"{ENV_MANIFEST_ENV} is unset and {default_model_manifest_path()} does not exist")
    manifest = load_environment_manifest(Path(manifest_path))
    return resolve_model_path_from_manifest(manifest)
