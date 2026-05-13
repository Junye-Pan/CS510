from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .run_artifacts import default_model_manifest_path, runs_root


ENVIRONMENT_MANIFEST_SCHEMA = "agentic_opt.llm_kernel_environment.v1"
MODEL_NAME = "qwen-3.5-4b"
MODEL_REPO_ID = "Qwen/Qwen3.5-4B"
FRAMEWORK_NAME = "vllm"
GPU_CLASS = "H200"
TASK_TARGET_DTYPE = "fp16"
SERVING_DTYPE = "bfloat16"
EVAL_BASELINE_NAME = "qwen35_4b_vllm_eval_suite_baseline.json"
SMOKE_BASELINE_NAME = "qwen35_4b_vllm_smoke_baseline.json"
LOGITS_BASELINE_NAME = "qwen35_4b_logits_distribution_baseline.json"


def load_environment_manifest(path: Path | None = None) -> dict[str, Any]:
    manifest_path = path or default_model_manifest_path()
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = normalize_environment_manifest(raw)
    manifest["_manifest_path"] = str(manifest_path)
    return manifest


def normalize_environment_manifest(raw: dict[str, Any]) -> dict[str, Any]:
    if raw.get("schema") == ENVIRONMENT_MANIFEST_SCHEMA:
        return _with_defaults(raw)

    files = list(raw.get("files") or [])
    revision = raw.get("revision")
    local_path = raw.get("local_path") or raw.get("snapshot_path")
    return _with_defaults(
        {
            "schema": ENVIRONMENT_MANIFEST_SCHEMA,
            "model": {
                "name": MODEL_NAME,
                "repo_id": raw.get("repo_id") or MODEL_REPO_ID,
                "local_path": local_path,
                "snapshot_path": raw.get("snapshot_path") or local_path,
                "revision": revision,
                "digest": raw.get("digest"),
                "size_bytes": raw.get("size_bytes"),
                "file_count": raw.get("file_count"),
            },
            "tokenizer": {
                "revision": raw.get("tokenizer_revision") or revision,
                "files": _tokenizer_file_entries(files),
            },
            "framework": {
                "name": FRAMEWORK_NAME,
                "version": raw.get("vllm_version"),
                "commit": raw.get("vllm_commit"),
            },
            "runtime": {
                "gpu": GPU_CLASS,
                "target_dtype": TASK_TARGET_DTYPE,
                "serving_dtype": SERVING_DTYPE,
                "python": raw.get("python"),
                "torch": raw.get("torch"),
                "triton": raw.get("triton"),
                "cuda": raw.get("cuda"),
                "driver": raw.get("driver"),
            },
            "baseline": {
                "smoke_artifact_path": str(runs_root() / "baselines" / SMOKE_BASELINE_NAME),
                "eval_suite_artifact_path": str(runs_root() / "baselines" / EVAL_BASELINE_NAME),
                "logits_distribution_artifact_path": str(runs_root() / "baselines" / LOGITS_BASELINE_NAME),
            },
        }
    )


def build_current_environment_manifest(model_manifest_path: Path | None = None) -> dict[str, Any]:
    manifest = load_environment_manifest(model_manifest_path)
    manifest.pop("_manifest_path", None)
    runtime = current_runtime_identity()
    framework = manifest.setdefault("framework", {})
    framework.update(
        {
            "name": FRAMEWORK_NAME,
            "version": runtime.get("vllm"),
            "commit": framework.get("commit"),
        }
    )
    manifest["runtime"] = {
        **(manifest.get("runtime") or {}),
        "python": runtime.get("python"),
        "python_executable": runtime.get("python_executable"),
        "torch": runtime.get("torch"),
        "triton": runtime.get("triton"),
        "cuda": runtime.get("cuda"),
        "driver": runtime.get("driver"),
        "gpu": runtime.get("gpu") or GPU_CLASS,
        "target_dtype": TASK_TARGET_DTYPE,
        "serving_dtype": SERVING_DTYPE,
    }
    manifest["baseline"] = _baseline_identity(manifest.get("baseline") or {})
    return manifest


def validate_environment_manifest(
    manifest: dict[str, Any],
    *,
    require_model: bool,
    require_runtime: bool,
    require_baselines: bool,
    enforce_readonly_model: bool,
) -> list[dict[str, Any]]:
    checks = [_check_schema(manifest)]
    checks.extend(_check_model(manifest, required=require_model, enforce_readonly=enforce_readonly_model))
    checks.extend(_check_tokenizer(manifest, required=require_model))
    if require_runtime:
        checks.extend(_check_runtime(manifest))
    if require_baselines:
        checks.extend(_check_baselines(manifest))
    return checks


def current_runtime_identity() -> dict[str, Any]:
    versions = {name: _module_version(name) for name in ("torch", "triton", "vllm")}
    gpu = _nvidia_smi_identity()
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": versions.get("torch"),
        "triton": versions.get("triton"),
        "vllm": versions.get("vllm"),
        "cuda": _torch_cuda_version(),
        "gpu": gpu.get("name"),
        "driver": gpu.get("driver"),
        "gpu_memory_mib": gpu.get("memory_mib"),
    }


def resolve_model_path_from_manifest(manifest: dict[str, Any]) -> Path:
    model = manifest.get("model") if isinstance(manifest, dict) else None
    local_path = model.get("local_path") if isinstance(model, dict) else None
    if not isinstance(local_path, str) or not local_path:
        raise ValueError("model.local_path missing from environment manifest")
    return Path(local_path)


def manifest_public_view(manifest: dict[str, Any]) -> dict[str, Any]:
    public = {key: value for key, value in manifest.items() if not str(key).startswith("_")}
    return public


def _with_defaults(manifest: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(manifest)
    normalized["schema"] = ENVIRONMENT_MANIFEST_SCHEMA
    model = dict(normalized.get("model") or {})
    model.setdefault("name", MODEL_NAME)
    model.setdefault("repo_id", MODEL_REPO_ID)
    normalized["model"] = model
    tokenizer = dict(normalized.get("tokenizer") or {})
    tokenizer.setdefault("revision", model.get("revision"))
    tokenizer.setdefault("files", [])
    normalized["tokenizer"] = tokenizer
    framework = dict(normalized.get("framework") or {})
    framework.setdefault("name", FRAMEWORK_NAME)
    normalized["framework"] = framework
    runtime = dict(normalized.get("runtime") or {})
    runtime.setdefault("gpu", GPU_CLASS)
    runtime.setdefault("target_dtype", TASK_TARGET_DTYPE)
    runtime.setdefault("serving_dtype", SERVING_DTYPE)
    normalized["runtime"] = runtime
    baseline = dict(normalized.get("baseline") or {})
    baseline.setdefault("smoke_artifact_path", str(runs_root() / "baselines" / SMOKE_BASELINE_NAME))
    baseline.setdefault("eval_suite_artifact_path", str(runs_root() / "baselines" / EVAL_BASELINE_NAME))
    baseline.setdefault("logits_distribution_artifact_path", str(runs_root() / "baselines" / LOGITS_BASELINE_NAME))
    normalized["baseline"] = baseline
    return normalized


def _tokenizer_file_entries(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tokenizer_names = {
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
    }
    entries: list[dict[str, Any]] = []
    for item in files:
        path = item.get("path")
        if not isinstance(path, str):
            continue
        if Path(path).name in tokenizer_names:
            entries.append({"path": path, "size": item.get("size")})
    return entries


def _check_schema(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "environment_manifest_schema",
        "status": "passed" if manifest.get("schema") == ENVIRONMENT_MANIFEST_SCHEMA else "failed",
        "message": None
        if manifest.get("schema") == ENVIRONMENT_MANIFEST_SCHEMA
        else f"schema must be {ENVIRONMENT_MANIFEST_SCHEMA}",
    }


def _check_model(
    manifest: dict[str, Any],
    *,
    required: bool,
    enforce_readonly: bool,
) -> list[dict[str, Any]]:
    model = manifest.get("model") if isinstance(manifest.get("model"), dict) else {}
    checks: list[dict[str, Any]] = []
    for key in ("name", "repo_id", "revision", "local_path"):
        value = model.get(key)
        checks.append(
            {
                "name": f"model_{key}",
                "status": "passed" if isinstance(value, str) and value else ("failed" if required else "skipped"),
                "message": None if isinstance(value, str) and value else f"model.{key} missing",
            }
        )
    local_path = model.get("local_path")
    if not isinstance(local_path, str) or not local_path:
        return checks
    path = Path(local_path)
    checks.append(
        {
            "name": "model_path_exists",
            "status": "passed" if path.exists() else "failed",
            "message": None if path.exists() else f"{local_path} does not exist",
        }
    )
    if path.exists() and model.get("size_bytes") is not None:
        actual_size = _directory_size(path)
        expected_size = int(model["size_bytes"])
        checks.append(
            {
                "name": "model_size_bytes",
                "status": "passed" if actual_size == expected_size else "failed",
                "message": None if actual_size == expected_size else f"expected {expected_size}, got {actual_size}",
            }
        )
    if enforce_readonly and path.exists():
        writable = os.access(path, os.W_OK)
        checks.append(
            {
                "name": "model_readonly",
                "status": "failed" if writable else "passed",
                "message": f"{local_path} is writable" if writable else None,
            }
        )
    return checks


def _check_tokenizer(manifest: dict[str, Any], *, required: bool) -> list[dict[str, Any]]:
    tokenizer = manifest.get("tokenizer") if isinstance(manifest.get("tokenizer"), dict) else {}
    model = manifest.get("model") if isinstance(manifest.get("model"), dict) else {}
    model_root = Path(model["local_path"]) if isinstance(model.get("local_path"), str) else None
    checks = [
        {
            "name": "tokenizer_revision",
            "status": "passed"
            if isinstance(tokenizer.get("revision"), str) and tokenizer.get("revision")
            else ("failed" if required else "skipped"),
            "message": None
            if isinstance(tokenizer.get("revision"), str) and tokenizer.get("revision")
            else "tokenizer.revision missing",
        }
    ]
    files = tokenizer.get("files")
    if not isinstance(files, list) or not files:
        checks.append(
            {
                "name": "tokenizer_files",
                "status": "failed" if required else "skipped",
                "message": "tokenizer.files missing",
            }
        )
        return checks
    missing: list[str] = []
    if model_root is not None:
        for item in files:
            relative = item.get("path") if isinstance(item, dict) else None
            if isinstance(relative, str) and not (model_root / relative).exists():
                missing.append(relative)
    checks.append(
        {
            "name": "tokenizer_files_exist",
            "status": "passed" if not missing else "failed",
            "message": None if not missing else f"missing tokenizer files: {missing}",
        }
    )
    return checks


def _check_runtime(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    runtime = manifest.get("runtime") if isinstance(manifest.get("runtime"), dict) else {}
    current = current_runtime_identity()
    checks: list[dict[str, Any]] = []
    for key in ("torch", "triton"):
        expected = runtime.get(key)
        actual = current.get(key)
        status = "passed" if not expected or expected == actual else "failed"
        checks.append(
            {
                "name": f"runtime_{key}",
                "status": status,
                "message": actual if status == "passed" else f"expected {expected}, got {actual}",
            }
        )
    framework = manifest.get("framework") if isinstance(manifest.get("framework"), dict) else {}
    expected_vllm = framework.get("version") or runtime.get("vllm")
    actual_vllm = current.get("vllm")
    checks.append(
        {
            "name": "runtime_vllm",
            "status": "passed" if not expected_vllm or expected_vllm == actual_vllm else "failed",
            "message": actual_vllm
            if not expected_vllm or expected_vllm == actual_vllm
            else f"expected {expected_vllm}, got {actual_vllm}",
        }
    )
    gpu_name = str(current.get("gpu") or "")
    checks.append(
        {
            "name": "runtime_gpu",
            "status": "passed" if GPU_CLASS in gpu_name else "failed",
            "message": gpu_name or "no GPU detected",
        }
    )
    expected_driver = runtime.get("driver")
    actual_driver = current.get("driver")
    checks.append(
        {
            "name": "runtime_driver",
            "status": "passed" if not expected_driver or expected_driver == actual_driver else "failed",
            "message": actual_driver
            if not expected_driver or expected_driver == actual_driver
            else f"expected {expected_driver}, got {actual_driver}",
        }
    )
    checks.append(
        {
            "name": "runtime_serving_dtype",
            "status": "passed" if runtime.get("serving_dtype") == SERVING_DTYPE else "failed",
            "message": None if runtime.get("serving_dtype") == SERVING_DTYPE else f"serving_dtype must be {SERVING_DTYPE}",
        }
    )
    checks.append(
        {
            "name": "runtime_target_dtype",
            "status": "passed" if runtime.get("target_dtype") == TASK_TARGET_DTYPE else "failed",
            "message": None
            if runtime.get("target_dtype") == TASK_TARGET_DTYPE
            else f"target_dtype must be {TASK_TARGET_DTYPE}",
        }
    )
    return checks


def _check_baselines(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    baseline = _baseline_identity(manifest.get("baseline") or {})
    checks: list[dict[str, Any]] = []
    for key in ("smoke_artifact_path", "eval_suite_artifact_path", "logits_distribution_artifact_path"):
        path = Path(str(baseline.get(key) or ""))
        checks.append(
            {
                "name": key,
                "status": "passed" if path.exists() else "failed",
                "message": str(path) if path.exists() else f"{path} missing",
            }
        )
        expected_digest = (manifest.get("baseline") or {}).get(f"{key}_sha256")
        if path.exists() and expected_digest:
            actual_digest = _file_sha256(path)
            checks.append(
                {
                    "name": f"{key}_sha256",
                    "status": "passed" if actual_digest == expected_digest else "failed",
                    "message": None if actual_digest == expected_digest else f"expected {expected_digest}, got {actual_digest}",
                }
            )
    return checks


def _baseline_identity(baseline: dict[str, Any]) -> dict[str, Any]:
    result = dict(baseline)
    result.setdefault("smoke_artifact_path", str(runs_root() / "baselines" / SMOKE_BASELINE_NAME))
    result.setdefault("eval_suite_artifact_path", str(runs_root() / "baselines" / EVAL_BASELINE_NAME))
    result.setdefault("logits_distribution_artifact_path", str(runs_root() / "baselines" / LOGITS_BASELINE_NAME))
    for key in ("smoke_artifact_path", "eval_suite_artifact_path", "logits_distribution_artifact_path"):
        path = Path(str(result[key]))
        if path.exists():
            result[f"{key}_sha256"] = _file_sha256(path)
    return result


def _module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    version = getattr(module, "__version__", None)
    return str(version) if version is not None else None


def _torch_cuda_version() -> str | None:
    try:
        torch = importlib.import_module("torch")
    except Exception:
        return None
    version = getattr(getattr(torch, "version", None), "cuda", None)
    return str(version) if version is not None else None


def _nvidia_smi_identity() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except Exception:
        return {}
    if proc.returncode != 0:
        return {}
    first = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 2:
        return {}
    payload: dict[str, Any] = {"name": parts[0], "driver": parts[1]}
    if len(parts) >= 3:
        try:
            payload["memory_mib"] = int(parts[2])
        except ValueError:
            pass
    return payload


def _directory_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"
