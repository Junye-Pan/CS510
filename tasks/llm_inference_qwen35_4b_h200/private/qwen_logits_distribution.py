from __future__ import annotations

import hashlib
import json
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any

from .baseline_metrics import _framework_identity, _load_model_manifest, _model_identity, _model_manifest_path
from .integrated_workload import DEFAULT_SEED, DEFAULT_TEMPERATURE, DEFAULT_TOP_P
from .preflight import resolve_model_path
from .qwen_vllm_smoke import _shutdown_llm
from .run_artifacts import new_run_dir, runs_root, write_json
from .vllm_plugin_runtime import prepare_candidate_rmsnorm_plugin, summarize_rmsnorm_trace


LOGITS_DISTRIBUTION_SCHEMA = "agentic_opt.qwen35_4b_logits_distribution_baseline.v1"
LOGITS_DISTRIBUTION_NAME = "qwen35_4b_logits_distribution_prefill_decode_margin_long_v1"
LOGITS_BASELINE_ARTIFACT_ENV = "AO_LLM_KERNEL_LOGITS_BASELINE_ARTIFACT"
LOGITS_WRITE_BASELINE_ENV = "AO_LLM_KERNEL_WRITE_LOGITS_BASELINE"

DEFAULT_DISTRIBUTION_THRESHOLDS = {
    "max_kl_divergence": 5.0e-3,
    "max_total_variation": 3.0e-2,
    "max_centered_logit_l2": 10.0,
    "max_centered_logit_linf": 5.0e-1,
    "max_centered_logit_rmse": 3.0e-2,
    "require_argmax_match": True,
    "require_selected_token_match": True,
}

CASE_DISTRIBUTION_THRESHOLD_OVERRIDES = {
    ("decode_long_context", "decode"): {
        "max_centered_logit_l2": 20.0,
        "max_centered_logit_rmse": 4.0e-2,
    },
}


def default_logits_distribution_baseline_artifact_path() -> Path:
    return runs_root() / "baselines" / "qwen35_4b_logits_distribution_baseline.json"


def default_logits_distribution_tensor_archive_path() -> Path:
    return runs_root() / "baselines" / "qwen35_4b_logits_distribution_baseline.npz"


def resolve_logits_distribution_baseline_artifact_path() -> Path:
    raw = os.environ.get(LOGITS_BASELINE_ARTIFACT_ENV)
    return Path(raw).expanduser().resolve() if raw else default_logits_distribution_baseline_artifact_path()


def logits_distribution_workload_signature() -> dict[str, Any]:
    return {
        "name": LOGITS_DISTRIBUTION_NAME,
        "cases": [
            {
                "id": case["id"],
                "phase": case["phase"],
                "position_kind": case["position_kind"],
                "max_tokens": case["max_tokens"],
                "selected_count": case["selected_count"],
                "prompt_sha256": _sha256_text(case["prompt"]),
            }
            for case in logits_distribution_cases()
        ],
        "sampling": {
            "temperature": DEFAULT_TEMPERATURE,
            "top_p": DEFAULT_TOP_P,
            "seed": DEFAULT_SEED,
            "full_logprobs": True,
        },
    }


def logits_distribution_cases() -> list[dict[str, Any]]:
    long_context = (
        "Kernel optimization distribution check. The long-context request repeats stable "
        "technical text to exercise prefill, position handling, KV-cache writes, and the "
        "first decode logits after a nontrivial context. "
    )
    return [
        {
            "id": "prefill_short_margin",
            "phase": "prefill",
            "position_kind": "prompt",
            "prompt": "Validate RMSNorm logits drift carefully.",
            "max_tokens": 1,
            "selected_count": 2,
        },
        {
            "id": "prefill_numeric_margin",
            "phase": "prefill",
            "position_kind": "prompt",
            "prompt": "Small margins can flip token choices in low precision.",
            "max_tokens": 1,
            "selected_count": 2,
        },
        {
            "id": "decode_short",
            "phase": "decode",
            "position_kind": "decode",
            "prompt": "Write one sentence about validating final logits.",
            "max_tokens": 4,
            "selected_count": 2,
        },
        {
            "id": "decode_long_context",
            "phase": "decode",
            "position_kind": "decode",
            "prompt": long_context * 24 + "Return the most fragile correctness signal.",
            "max_tokens": 2,
            "selected_count": 1,
        },
    ]


def run_qwen_logits_distribution_probe(
    *,
    run_dir: Path | None = None,
    candidate_manifest_path: Path | None = None,
    require_candidate_rmsnorm: bool | None = None,
    selection_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if run_dir is None:
        run_dir = new_run_dir("qwen_logits_distribution")
    started = time.time()
    model_path = resolve_model_path()
    os.environ.setdefault("HF_HOME", str(runs_root() / "huggingface"))
    os.environ.setdefault("HF_XET_CACHE", str(runs_root() / "huggingface" / "xet"))
    os.environ.setdefault("VLLM_CACHE_ROOT", str(runs_root() / "vllm_cache"))
    os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
    os.environ.setdefault("VLLM_MOE_USE_DEEP_GEMM", "0")
    os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")
    os.environ.setdefault("VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER", "0")

    payload: dict[str, Any]
    plugin_info: dict[str, Any] | None = None
    llm: Any | None = None
    if require_candidate_rmsnorm is None:
        require_candidate_rmsnorm = candidate_manifest_path is not None
    try:
        if candidate_manifest_path is not None:
            plugin_info = prepare_candidate_rmsnorm_plugin(
                run_dir=run_dir,
                manifest_path=candidate_manifest_path,
            )
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=str(model_path),
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=2048,
            max_logprobs=-1,
            gpu_memory_utilization=0.55,
            enforce_eager=True,
            disable_log_stats=True,
            gdn_prefill_backend="triton",
        )
        records: list[dict[str, Any]] = []
        vectors: dict[str, Any] = {}
        missing_selections: list[str] = []
        for case in logits_distribution_cases():
            sampling = _sampling_params(SamplingParams, case=case)
            output = llm.generate([case["prompt"]], sampling)[0]
            case_records, case_vectors, case_missing = _records_for_case(
                case=case,
                output=output,
                selection_records=selection_records,
            )
            records.extend(case_records)
            vectors.update(case_vectors)
            missing_selections.extend(case_missing)

        archive_path = run_dir / "qwen_logits_distribution_vectors.npz"
        archive = _write_tensor_archive(archive_path, vectors)
        _shutdown_llm(llm)
        llm = None
        apply_summary = (
            summarize_rmsnorm_trace(Path(plugin_info["trace_path"]))
            if plugin_info is not None
            else None
        )
        candidate_used = bool(
            apply_summary is not None and int(apply_summary.get("candidate_calls") or 0) > 0
        )
        payload = {
            "ok": not (require_candidate_rmsnorm and not candidate_used) and not missing_selections,
            "schema": "agentic_opt.qwen35_4b_logits_distribution_probe_result.v1",
            "model_path": str(model_path),
            "run_dir": str(run_dir),
            "probe": logits_distribution_workload_signature(),
            "elapsed_s": time.time() - started,
            "records": records,
            "tensor_archive": archive,
            "thresholds": dict(DEFAULT_DISTRIBUTION_THRESHOLDS),
            "candidate_rmsnorm_plugin": plugin_info,
            "candidate_rmsnorm_required": bool(require_candidate_rmsnorm),
            "candidate_rmsnorm_used_in_vllm": candidate_used,
            "vllm_rmsnorm_apply": apply_summary,
        }
        if require_candidate_rmsnorm and not candidate_used:
            payload["error"] = "candidate RMSNorm was not invoked by the vLLM apply path"
        if missing_selections:
            payload["error"] = f"logits distribution probe missed baseline-selected positions: {missing_selections}"
        if (
            payload["ok"]
            and candidate_manifest_path is None
            and os.environ.get(LOGITS_WRITE_BASELINE_ENV) in {"1", "true", "TRUE", "yes", "on"}
        ):
            baseline = write_logits_distribution_baseline_artifact(payload)
            payload["baseline_artifact_path"] = baseline.get("_artifact_path")
    except Exception as exc:
        if llm is not None:
            _shutdown_llm(llm)
        apply_summary = (
            summarize_rmsnorm_trace(Path(plugin_info["trace_path"]))
            if plugin_info is not None
            else None
        )
        payload = {
            "ok": False,
            "schema": "agentic_opt.qwen35_4b_logits_distribution_probe_result.v1",
            "model_path": str(model_path),
            "run_dir": str(run_dir),
            "probe": logits_distribution_workload_signature(),
            "elapsed_s": time.time() - started,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "candidate_rmsnorm_plugin": plugin_info,
            "candidate_rmsnorm_required": bool(require_candidate_rmsnorm),
            "candidate_rmsnorm_used_in_vllm": bool(
                apply_summary is not None and int(apply_summary.get("candidate_calls") or 0) > 0
            ),
            "vllm_rmsnorm_apply": apply_summary,
        }
    write_json(run_dir / "qwen_logits_distribution.json", payload)
    return payload


def build_logits_distribution_baseline_artifact(
    probe_result: dict[str, Any],
    *,
    archive: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_manifest = _load_model_manifest()
    tensor_archive = archive or probe_result.get("tensor_archive") or {}
    return {
        "schema": LOGITS_DISTRIBUTION_SCHEMA,
        "created_at_unix": time.time(),
        "model_manifest_path": _model_manifest_path(),
        "model": _model_identity(model_manifest),
        "framework": _framework_identity(),
        "probe": probe_result.get("probe") or logits_distribution_workload_signature(),
        "thresholds": dict(DEFAULT_DISTRIBUTION_THRESHOLDS),
        "tensor_archive": tensor_archive,
        "records": _baseline_records(probe_result),
        "metrics": _baseline_record_metrics(probe_result.get("records") or []),
    }


def write_logits_distribution_baseline_artifact(
    probe_result: dict[str, Any],
    path: Path | None = None,
) -> dict[str, Any]:
    destination = path or resolve_logits_distribution_baseline_artifact_path()
    archive_destination = destination.with_suffix(".npz")
    vectors = _vectors_from_payload(probe_result)
    archive = _write_tensor_archive(archive_destination, vectors)
    artifact = build_logits_distribution_baseline_artifact(probe_result, archive=archive)
    write_json(destination, artifact)
    artifact["_artifact_path"] = str(destination)
    return artifact


def load_logits_distribution_baseline_artifact(path: Path | None = None) -> dict[str, Any]:
    artifact_path = path or resolve_logits_distribution_baseline_artifact_path()
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"logits distribution baseline artifact is missing: {artifact_path}. "
            f"Run the Qwen/vLLM logits distribution baseline and write "
            f"{LOGITS_BASELINE_ARTIFACT_ENV} or the default artifact first."
        )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["_artifact_path"] = str(artifact_path)
    validation = validate_logits_distribution_baseline_artifact(artifact)
    if not validation["valid"]:
        raise ValueError(validation["error"])
    return artifact


def validate_logits_distribution_baseline_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    if artifact.get("schema") != LOGITS_DISTRIBUTION_SCHEMA:
        return {"valid": False, "error": f"baseline schema must be {LOGITS_DISTRIBUTION_SCHEMA}"}
    if artifact.get("probe") != logits_distribution_workload_signature():
        return {"valid": False, "error": "baseline probe signature does not match current logits distribution probe"}
    records = artifact.get("records")
    if not isinstance(records, list) or not records:
        return {"valid": False, "error": "baseline logits distribution records are missing"}
    archive = artifact.get("tensor_archive") or {}
    archive_path = _resolve_archive_path(archive, artifact_path=artifact.get("_artifact_path"))
    if not archive_path.exists():
        return {"valid": False, "error": f"logits tensor archive is missing: {archive_path}"}
    expected_digest = archive.get("sha256")
    if expected_digest:
        actual_digest = _file_sha256(archive_path)
        if actual_digest != expected_digest:
            return {
                "valid": False,
                "error": f"logits tensor archive sha256 mismatch: expected {expected_digest}, got {actual_digest}",
            }
    keys = set(archive.get("keys") or [])
    if keys:
        missing = [record.get("vector_key") for record in records if record.get("vector_key") not in keys]
        if missing:
            return {"valid": False, "error": f"logits tensor archive missing record keys: {missing}"}
    return {"valid": True, "error": None, "artifact_path": artifact.get("_artifact_path")}


def compare_logits_distribution_to_baseline(
    probe_result: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    thresholds = dict(DEFAULT_DISTRIBUTION_THRESHOLDS)
    thresholds.update(baseline.get("thresholds") or {})
    checks.append(
        _check(
            "probe_signature",
            probe_result.get("probe") == baseline.get("probe"),
            "candidate logits distribution probe differs from pinned baseline probe",
        )
    )
    baseline_records = list(baseline.get("records") or [])
    candidate_records = list(probe_result.get("records") or [])
    checks.append(
        _check(
            "record_count",
            len(candidate_records) == len(baseline_records),
            "candidate logits distribution record count differs from pinned baseline",
        )
    )
    baseline_vectors = _vectors_from_payload(baseline)
    candidate_vectors = _vectors_from_payload(probe_result)
    record_metrics: list[dict[str, Any]] = []
    for index, (candidate, expected) in enumerate(zip(candidate_records, baseline_records, strict=False)):
        identity_ok = _record_identity(candidate) == _record_identity(expected)
        checks.append(
            _check(
                f"record_{index}_identity",
                identity_ok,
                f"logits distribution record {index} identity differs from pinned baseline",
            )
        )
        selected_token_ok = candidate.get("selected_token_id") == expected.get("selected_token_id")
        if bool(thresholds.get("require_selected_token_match", True)):
            checks.append(
                _check(
                    f"record_{index}_selected_token",
                    selected_token_ok,
                    f"logits distribution record {index} selected token differs from pinned baseline",
                )
            )
        candidate_key = str(candidate.get("vector_key") or "")
        baseline_key = str(expected.get("vector_key") or "")
        if candidate_key not in candidate_vectors or baseline_key not in baseline_vectors:
            checks.append(
                _check(
                    f"record_{index}_vector",
                    False,
                    f"logits distribution record {index} vector is missing from tensor archive",
                )
            )
            continue
        metrics = logprob_distribution_metrics(
            baseline_vectors[baseline_key],
            candidate_vectors[candidate_key],
        )
        metrics.update(
            {
                "record_id": expected.get("id"),
                "case_id": expected.get("case_id"),
                "phase": expected.get("phase"),
                "position_kind": expected.get("position_kind"),
                "token_index": expected.get("token_index"),
                "baseline_selected_token_id": expected.get("selected_token_id"),
                "candidate_selected_token_id": candidate.get("selected_token_id"),
                "baseline_top1_margin": expected.get("top1_margin"),
                "candidate_top1_margin": candidate.get("top1_margin"),
            }
        )
        record_metrics.append(metrics)
        record_thresholds = _thresholds_for_record(metrics=metrics, thresholds=thresholds)
        metrics["thresholds"] = record_thresholds
        checks.extend(_metric_checks(index=index, metrics=metrics, thresholds=record_thresholds))

    valid = all(item["status"] == "passed" for item in checks)
    return {
        "valid": valid,
        "error": None if valid else _first_failed_message(checks),
        "baseline_artifact_path": baseline.get("_artifact_path"),
        "checks": checks,
        "thresholds": thresholds,
        "threshold_overrides": _serializable_threshold_overrides(),
        "records": record_metrics,
        "aggregate": _aggregate_distribution_metrics(record_metrics),
    }


def logprob_distribution_metrics(baseline_logprobs: Any, candidate_logprobs: Any) -> dict[str, Any]:
    left = _finite_float_list(baseline_logprobs)
    right = _finite_float_list(candidate_logprobs)
    if len(left) != len(right) or not left:
        return {
            "valid": False,
            "error": f"logprob vector length mismatch: baseline={len(left)}, candidate={len(right)}",
        }
    logp = _renormalized_logprobs(left)
    logq = _renormalized_logprobs(right)
    p = [math.exp(value) for value in logp]
    q = [math.exp(value) for value in logq]
    kl = sum(pi * (lp - lq) for pi, lp, lq in zip(p, logp, logq, strict=False) if pi > 0.0)
    tv = 0.5 * sum(abs(pi - qi) for pi, qi in zip(p, q, strict=False))
    deltas = [lq - lp for lp, lq in zip(logp, logq, strict=False)]
    l2 = math.sqrt(sum(delta * delta for delta in deltas))
    linf = max(abs(delta) for delta in deltas)
    rmse = l2 / math.sqrt(len(deltas))
    baseline_top = _top_two(logp)
    candidate_top = _top_two(logq)
    return {
        "valid": True,
        "error": None,
        "vocab_size": len(logp),
        "kl_divergence": float(max(kl, 0.0)),
        "total_variation": float(tv),
        "centered_logit_l2": float(l2),
        "centered_logit_linf": float(linf),
        "centered_logit_rmse": float(rmse),
        "baseline_argmax_token_id": baseline_top[0],
        "candidate_argmax_token_id": candidate_top[0],
        "argmax_match": baseline_top[0] == candidate_top[0],
        "baseline_top1_margin": float(baseline_top[2]),
        "candidate_top1_margin": float(candidate_top[2]),
    }


def selection_records_from_baseline(baseline: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": record.get("case_id"),
            "phase": record.get("phase"),
            "position_kind": record.get("position_kind"),
            "token_index": record.get("token_index"),
        }
        for record in baseline.get("records") or []
    ]


def _records_for_case(
    *,
    case: dict[str, Any],
    output: Any,
    selection_records: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    if case["phase"] == "prefill":
        steps = getattr(output, "prompt_logprobs", None) or []
        token_ids = list(getattr(output, "prompt_token_ids", None) or [])
    else:
        completion = output.outputs[0] if getattr(output, "outputs", None) else None
        steps = getattr(completion, "logprobs", None) or []
        token_ids = list(getattr(completion, "token_ids", None) or [])
    available: list[tuple[int, list[float], dict[str, Any]]] = []
    for token_index, step in enumerate(steps):
        vector = _dense_logprob_vector(step)
        if not vector:
            continue
        top1, top2, margin = _top_two(vector)
        selected_token_id = token_ids[token_index] if token_index < len(token_ids) else None
        available.append(
            (
                token_index,
                vector,
                {
                    "selected_token_id": selected_token_id,
                    "top1_token_id": top1,
                    "top2_token_id": top2,
                    "top1_margin": margin,
                },
            )
        )
    selected_indices = _selected_indices(case=case, available=available, selection_records=selection_records)
    missing: list[str] = []
    records: list[dict[str, Any]] = []
    vectors: dict[str, Any] = {}
    by_index = {token_index: (vector, stats) for token_index, vector, stats in available}
    for token_index in selected_indices:
        item = by_index.get(token_index)
        if item is None:
            missing.append(f"{case['id']}:{case['phase']}:{token_index}")
            continue
        vector, stats = item
        record_index = len(records)
        vector_key = f"{case['id']}__{case['phase']}__{token_index}"
        record = {
            "id": vector_key,
            "case_id": case["id"],
            "phase": case["phase"],
            "position_kind": case["position_kind"],
            "prompt_sha256": _sha256_text(case["prompt"]),
            "prompt_token_count": len(getattr(output, "prompt_token_ids", None) or []),
            "token_index": token_index,
            "record_index": record_index,
            "vector_key": vector_key,
            "vocab_size": len(vector),
            "logprobs_sha256": _vector_sha256(vector),
            **stats,
        }
        records.append(record)
        vectors[vector_key] = vector
    return records, vectors, missing


def _selected_indices(
    *,
    case: dict[str, Any],
    available: list[tuple[int, list[float], dict[str, Any]]],
    selection_records: list[dict[str, Any]] | None,
) -> list[int]:
    if selection_records is not None:
        return [
            int(record["token_index"])
            for record in selection_records
            if record.get("case_id") == case["id"]
            and record.get("phase") == case["phase"]
            and record.get("position_kind") == case["position_kind"]
            and record.get("token_index") is not None
        ]
    ranked = sorted(available, key=lambda item: (float(item[2]["top1_margin"]), item[0]))
    return [token_index for token_index, _vector, _stats in ranked[: int(case["selected_count"])]]


def _sampling_params(sampling_cls: Any, *, case: dict[str, Any]) -> Any:
    kwargs = {
        "max_tokens": int(case["max_tokens"]),
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "seed": DEFAULT_SEED,
        "ignore_eos": True,
        "logprobs": -1,
    }
    if case["phase"] == "prefill":
        kwargs["prompt_logprobs"] = -1
    return sampling_cls(**kwargs)


def _dense_logprob_vector(step: Any) -> list[float]:
    if not isinstance(step, dict) or not step:
        return []
    entries: list[tuple[int, float]] = []
    max_token_id = -1
    for key, value in step.items():
        try:
            token_id = int(key)
        except (TypeError, ValueError):
            continue
        logprob = _logprob_value(value)
        if logprob is None or not math.isfinite(logprob):
            continue
        entries.append((token_id, logprob))
        max_token_id = max(max_token_id, token_id)
    if max_token_id < 0:
        return []
    vector = [-math.inf] * (max_token_id + 1)
    for token_id, logprob in entries:
        vector[token_id] = logprob
    if any(not math.isfinite(value) for value in vector):
        return []
    return vector


def _logprob_value(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    direct = getattr(value, "logprob", None)
    if isinstance(direct, (int, float)):
        return float(direct)
    if isinstance(value, dict):
        nested = value.get("logprob")
        if isinstance(nested, (int, float)):
            return float(nested)
    return None


def _baseline_records(probe_result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            key: record.get(key)
            for key in (
                "id",
                "case_id",
                "phase",
                "position_kind",
                "prompt_sha256",
                "prompt_token_count",
                "token_index",
                "record_index",
                "vector_key",
                "vocab_size",
                "logprobs_sha256",
                "selected_token_id",
                "top1_token_id",
                "top2_token_id",
                "top1_margin",
            )
        }
        for record in probe_result.get("records") or []
    ]


def _baseline_record_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    margins = [float(record["top1_margin"]) for record in records if isinstance(record.get("top1_margin"), (int, float))]
    return {
        "record_count": len(records),
        "min_top1_margin": min(margins) if margins else None,
        "max_top1_margin": max(margins) if margins else None,
    }


def _metric_checks(index: int, metrics: dict[str, Any], thresholds: dict[str, Any]) -> list[dict[str, Any]]:
    if not metrics.get("valid"):
        return [_check(f"record_{index}_metrics", False, str(metrics.get("error") or "invalid distribution metrics"))]
    checks = [
        _check(
            f"record_{index}_kl",
            float(metrics["kl_divergence"]) <= float(thresholds["max_kl_divergence"]),
            (
                f"logits distribution record {index} KL {float(metrics['kl_divergence']):.6g} exceeds "
                f"{float(thresholds['max_kl_divergence']):.6g}"
            ),
        ),
        _check(
            f"record_{index}_tv",
            float(metrics["total_variation"]) <= float(thresholds["max_total_variation"]),
            (
                f"logits distribution record {index} TV {float(metrics['total_variation']):.6g} exceeds "
                f"{float(thresholds['max_total_variation']):.6g}"
            ),
        ),
        _check(
            f"record_{index}_centered_logit_l2",
            float(metrics["centered_logit_l2"]) <= float(thresholds["max_centered_logit_l2"]),
            (
                f"logits distribution record {index} centered-logit L2 "
                f"{float(metrics['centered_logit_l2']):.6g} exceeds "
                f"{float(thresholds['max_centered_logit_l2']):.6g}"
            ),
        ),
        _check(
            f"record_{index}_centered_logit_linf",
            float(metrics["centered_logit_linf"]) <= float(thresholds["max_centered_logit_linf"]),
            (
                f"logits distribution record {index} centered-logit Linf "
                f"{float(metrics['centered_logit_linf']):.6g} exceeds "
                f"{float(thresholds['max_centered_logit_linf']):.6g}"
            ),
        ),
        _check(
            f"record_{index}_centered_logit_rmse",
            float(metrics["centered_logit_rmse"]) <= float(thresholds["max_centered_logit_rmse"]),
            (
                f"logits distribution record {index} centered-logit RMSE "
                f"{float(metrics['centered_logit_rmse']):.6g} exceeds "
                f"{float(thresholds['max_centered_logit_rmse']):.6g}"
            ),
        ),
    ]
    if bool(thresholds.get("require_argmax_match", True)):
        checks.append(
            _check(
                f"record_{index}_argmax",
                bool(metrics["argmax_match"]),
                f"logits distribution record {index} argmax token differs from pinned baseline",
            )
        )
    return checks


def _thresholds_for_record(metrics: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    record_thresholds = dict(thresholds)
    case_id = str(metrics.get("case_id") or "")
    phase = str(metrics.get("phase") or "")
    record_thresholds.update(CASE_DISTRIBUTION_THRESHOLD_OVERRIDES.get((case_id, phase), {}))
    return record_thresholds


def _serializable_threshold_overrides() -> dict[str, dict[str, float]]:
    return {
        f"{case_id}:{phase}": dict(overrides)
        for (case_id, phase), overrides in CASE_DISTRIBUTION_THRESHOLD_OVERRIDES.items()
    }


def _aggregate_distribution_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = (
        "kl_divergence",
        "total_variation",
        "centered_logit_l2",
        "centered_logit_linf",
        "centered_logit_rmse",
    )
    aggregate: dict[str, Any] = {"record_count": len(records)}
    for key in numeric_keys:
        values = [float(record[key]) for record in records if isinstance(record.get(key), (int, float))]
        aggregate[f"max_{key}"] = max(values) if values else None
        aggregate[f"mean_{key}"] = sum(values) / len(values) if values else None
    aggregate["argmax_match_rate"] = (
        sum(1 for record in records if record.get("argmax_match")) / len(records) if records else 0.0
    )
    return aggregate


def _vectors_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    inline = payload.get("vectors")
    if isinstance(inline, dict):
        return {str(key): value for key, value in inline.items()}
    archive = payload.get("tensor_archive") or {}
    path = _resolve_archive_path(archive, artifact_path=payload.get("_artifact_path"))
    return _load_tensor_archive(path)


def _write_tensor_archive(path: Path, vectors: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.asarray(value, dtype=np.float32) for key, value in vectors.items()}
    np.savez_compressed(path, **arrays)
    return {
        "path": str(path),
        "format": "npz",
        "sha256": _file_sha256(path),
        "keys": sorted(arrays),
    }


def _load_tensor_archive(path: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].astype("float64").tolist() for key in archive.files}


def _resolve_archive_path(archive: dict[str, Any], *, artifact_path: Any | None) -> Path:
    raw_path = archive.get("path")
    if isinstance(raw_path, str) and raw_path:
        path = Path(raw_path)
        if path.is_absolute():
            return path
        if artifact_path:
            return Path(str(artifact_path)).resolve().parent / path
        return path
    if artifact_path:
        return Path(str(artifact_path)).with_suffix(".npz")
    return default_logits_distribution_tensor_archive_path()


def _finite_float_list(values: Any) -> list[float]:
    try:
        result = [float(value) for value in values]
    except TypeError:
        return []
    return result if all(math.isfinite(value) for value in result) else []


def _renormalized_logprobs(values: list[float]) -> list[float]:
    normalizer = _logsumexp(values)
    return [value - normalizer for value in values]


def _logsumexp(values: list[float]) -> float:
    maximum = max(values)
    total = sum(math.exp(value - maximum) for value in values)
    return maximum + math.log(total)


def _top_two(values: list[float]) -> tuple[int | None, int | None, float]:
    if not values:
        return None, None, 0.0
    first_index = max(range(len(values)), key=lambda index: values[index])
    second_index = None
    second_value = -math.inf
    for index, value in enumerate(values):
        if index == first_index:
            continue
        if value > second_value:
            second_value = value
            second_index = index
    margin = values[first_index] - second_value if second_index is not None else math.inf
    return first_index, second_index, float(margin)


def _record_identity(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("case_id"),
        record.get("phase"),
        record.get("position_kind"),
        record.get("prompt_sha256"),
        record.get("token_index"),
    )


def _vector_sha256(values: list[float]) -> str:
    payload = json.dumps([round(float(value), 8) for value in values], separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _check(name: str, passed: bool, message: str) -> dict[str, Any]:
    return {"name": name, "status": "passed" if passed else "failed", "message": None if passed else message}


def _first_failed_message(checks: list[dict[str, Any]]) -> str | None:
    for check in checks:
        if check.get("status") == "failed":
            return str(check.get("message") or check.get("name"))
    return None


def main() -> int:
    candidate_raw = os.environ.get("AO_LLM_KERNEL_LOGITS_CANDIDATE_MANIFEST")
    require_raw = os.environ.get("AO_LLM_KERNEL_LOGITS_REQUIRE_CANDIDATE")
    baseline = None
    selection = None
    if candidate_raw:
        baseline = load_logits_distribution_baseline_artifact()
        selection = selection_records_from_baseline(baseline)
    result = run_qwen_logits_distribution_probe(
        candidate_manifest_path=Path(candidate_raw) if candidate_raw else None,
        require_candidate_rmsnorm=require_raw in {"1", "true", "TRUE", "yes", "on"}
        if require_raw is not None
        else None,
        selection_records=selection,
    )
    if baseline is not None and result.get("ok"):
        result["distribution_correctness"] = compare_logits_distribution_to_baseline(result, baseline)
        if not result["distribution_correctness"].get("valid"):
            result["ok"] = False
            result["error"] = result["distribution_correctness"].get("error")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
