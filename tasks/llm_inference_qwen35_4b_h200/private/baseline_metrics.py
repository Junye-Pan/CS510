from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .environment_manifest import load_environment_manifest
from .integrated_workload import workload_signature
from .preflight import ENV_MANIFEST_ENV
from .run_artifacts import default_model_manifest_path, runs_root, write_json


BASELINE_SCHEMA = "agentic_opt.qwen35_4b_vllm_baseline.v1"
BASELINE_ARTIFACT_ENV = "AO_LLM_KERNEL_BASELINE_ARTIFACT"
MIN_TOP_LOGPROB_OVERLAP = 0.8


def default_baseline_artifact_path() -> Path:
    return runs_root() / "baselines" / "qwen35_4b_vllm_smoke_baseline.json"


def resolve_baseline_artifact_path() -> Path:
    raw = os.environ.get(BASELINE_ARTIFACT_ENV)
    return Path(raw).expanduser().resolve() if raw else default_baseline_artifact_path()


def build_baseline_artifact(smoke_result: dict[str, Any]) -> dict[str, Any]:
    model_manifest = _load_model_manifest()
    return {
        "schema": BASELINE_SCHEMA,
        "created_at_unix": time.time(),
        "model_manifest_path": _model_manifest_path(),
        "model": _model_identity(model_manifest),
        "framework": _framework_identity(),
        "workload": smoke_result.get("workload") or workload_signature(
            prompts=tuple(output.get("prompt", "") for output in smoke_result.get("outputs") or ()),
            max_tokens=int(smoke_result.get("max_tokens") or 0),
        ),
        "metrics": {
            "load_and_generate_elapsed_s": smoke_result.get("load_and_generate_elapsed_s"),
            "generate_elapsed_s": smoke_result.get("generate_elapsed_s"),
        },
        "outputs": _baseline_outputs(smoke_result),
    }


def write_baseline_artifact(smoke_result: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    artifact = build_baseline_artifact(smoke_result)
    destination = path or resolve_baseline_artifact_path()
    write_json(destination, artifact)
    artifact["_artifact_path"] = str(destination)
    return artifact


def load_baseline_artifact(path: Path | None = None) -> dict[str, Any]:
    artifact_path = path or resolve_baseline_artifact_path()
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"baseline artifact is missing: {artifact_path}. "
            f"Run the baseline Qwen/vLLM smoke and write {BASELINE_ARTIFACT_ENV} or the default artifact first."
        )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["_artifact_path"] = str(artifact_path)
    validation = validate_baseline_artifact(artifact)
    if not validation["valid"]:
        raise ValueError(validation["error"])
    return artifact


def validate_baseline_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    if artifact.get("schema") != BASELINE_SCHEMA:
        return {"valid": False, "error": f"baseline schema must be {BASELINE_SCHEMA}"}
    expected_workload = workload_signature()
    if artifact.get("workload") != expected_workload:
        return {"valid": False, "error": "baseline workload signature does not match current integrated smoke"}
    outputs = artifact.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != len(expected_workload["prompts"]):
        return {"valid": False, "error": "baseline outputs do not match prompt count"}
    for index, output in enumerate(outputs):
        if output.get("prompt") != expected_workload["prompts"][index]:
            return {"valid": False, "error": f"baseline output {index} prompt mismatch"}
        token_ids = output.get("token_ids")
        if not isinstance(token_ids, list) or not all(isinstance(item, int) for item in token_ids):
            return {"valid": False, "error": f"baseline output {index} token_ids must be an integer list"}
    model_check = _validate_model_identity(artifact.get("model") or {})
    if not model_check["valid"]:
        return model_check
    return {"valid": True, "error": None, "artifact_path": artifact.get("_artifact_path")}


def compare_smoke_to_baseline(smoke_result: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    expected_workload = baseline.get("workload") or {}
    actual_workload = smoke_result.get("workload") or {}
    checks.append(
        _check(
            "workload_signature",
            actual_workload == expected_workload,
            "candidate smoke workload differs from pinned baseline workload",
        )
    )
    baseline_outputs = baseline.get("outputs") or []
    candidate_outputs = smoke_result.get("outputs") or []
    checks.append(
        _check(
            "output_count",
            len(candidate_outputs) == len(baseline_outputs),
            "candidate output count differs from baseline",
        )
    )

    token_matches = 0
    token_total = 0
    text_matches = 0
    text_total = min(len(candidate_outputs), len(baseline_outputs))
    overlaps: list[float] = []
    for index, (candidate, expected) in enumerate(zip(candidate_outputs, baseline_outputs, strict=False)):
        candidate_tokens = list(candidate.get("token_ids") or [])
        expected_tokens = list(expected.get("token_ids") or [])
        token_total += max(len(candidate_tokens), len(expected_tokens))
        token_matches += sum(1 for left, right in zip(candidate_tokens, expected_tokens, strict=False) if left == right)
        checks.append(
            _check(
                f"output_{index}_token_ids",
                candidate_tokens == expected_tokens,
                f"output {index} generated token ids differ from pinned baseline",
            )
        )
        candidate_text = candidate.get("generated_text")
        expected_text = expected.get("generated_text")
        if candidate_text == expected_text:
            text_matches += 1
        checks.append(
            _check(
                f"output_{index}_text",
                candidate_text == expected_text,
                f"output {index} generated text differs from pinned baseline",
            )
        )
        overlap = _top_logprob_overlap(candidate.get("top_logprob_token_ids"), expected.get("top_logprob_token_ids"))
        if overlap is not None:
            overlaps.append(overlap)
            checks.append(
                _check(
                    f"output_{index}_top_logprob_overlap",
                    overlap >= MIN_TOP_LOGPROB_OVERLAP,
                    f"output {index} top-logprob token set overlap {overlap:.3f} below {MIN_TOP_LOGPROB_OVERLAP:.3f}",
                )
            )

    token_match_rate = float(token_matches / token_total) if token_total else 0.0
    text_match_rate = float(text_matches / text_total) if text_total else 0.0
    min_overlap = min(overlaps) if overlaps else None
    valid = all(item["status"] == "passed" for item in checks)
    return {
        "valid": valid,
        "error": None if valid else _first_failed_message(checks),
        "baseline_artifact_path": baseline.get("_artifact_path"),
        "checks": checks,
        "token_match_rate": token_match_rate,
        "text_match_rate": text_match_rate,
        "min_top_logprob_overlap": min_overlap,
    }


def _baseline_outputs(smoke_result: dict[str, Any]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for output in smoke_result.get("outputs") or []:
        outputs.append(
            {
                "prompt": output.get("prompt"),
                "generated_text": output.get("generated_text") or "",
                "token_ids": list(output.get("token_ids") or []),
                "top_logprob_token_ids": output.get("top_logprob_token_ids") or [],
            }
        )
    return outputs


def _top_logprob_overlap(candidate: Any, baseline: Any) -> float | None:
    if not isinstance(candidate, list) or not isinstance(baseline, list) or not candidate or not baseline:
        return None
    pairs = list(zip(candidate, baseline, strict=False))
    if not pairs:
        return None
    scores: list[float] = []
    for candidate_step, baseline_step in pairs:
        if not isinstance(candidate_step, list) or not isinstance(baseline_step, list) or not baseline_step:
            continue
        candidate_set = {int(item) for item in candidate_step if isinstance(item, int)}
        baseline_set = {int(item) for item in baseline_step if isinstance(item, int)}
        if baseline_set:
            scores.append(len(candidate_set & baseline_set) / len(baseline_set))
    return min(scores) if scores else None


def _load_model_manifest() -> dict[str, Any]:
    path = Path(_model_manifest_path())
    if not path.exists():
        return {}
    return load_environment_manifest(path)


def _model_manifest_path() -> str:
    return os.environ.get(ENV_MANIFEST_ENV) or str(default_model_manifest_path())


def _model_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    model = manifest.get("model") if isinstance(manifest.get("model"), dict) else manifest
    tokenizer = manifest.get("tokenizer") if isinstance(manifest.get("tokenizer"), dict) else {}
    return {
        "name": model.get("name"),
        "repo_id": model.get("repo_id"),
        "revision": model.get("revision"),
        "tokenizer_revision": tokenizer.get("revision"),
        "local_path": model.get("local_path") or manifest.get("local_path"),
        "size_bytes": model.get("size_bytes") or manifest.get("size_bytes"),
    }


def _validate_model_identity(identity: dict[str, Any]) -> dict[str, Any]:
    current = _model_identity(_load_model_manifest())
    for key in ("repo_id", "revision"):
        if current.get(key) and identity.get(key) != current.get(key):
            return {"valid": False, "error": f"baseline model {key} does not match current model manifest"}
    return {"valid": True, "error": None}


def _framework_identity() -> dict[str, Any]:
    identity: dict[str, Any] = {}
    for module_name in ("vllm", "torch", "triton"):
        try:
            module = __import__(module_name)
            identity[module_name] = getattr(module, "__version__", None)
        except Exception:
            identity[module_name] = None
    return identity


def _check(name: str, passed: bool, message: str) -> dict[str, Any]:
    return {"name": name, "status": "passed" if passed else "failed", "message": None if passed else message}


def _first_failed_message(checks: list[dict[str, Any]]) -> str | None:
    for check in checks:
        if check.get("status") == "failed":
            return str(check.get("message") or check.get("name"))
    return None
