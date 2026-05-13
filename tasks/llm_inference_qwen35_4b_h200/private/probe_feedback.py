from __future__ import annotations

from typing import Any

from .fallback_policy import fallback_thresholds


def build_probe_diagnostics(
    *,
    static_result: dict[str, Any],
    public_workload_shapes: list[dict[str, int]],
    live_result: dict[str, Any] | None = None,
    qwen_smoke: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bundle = static_result.get("bundle") or {}
    implementation_count = int(bundle.get("implementation_count") or 0)
    diagnostics: dict[str, Any] = {
        "static_valid": bool(static_result.get("valid")),
        "implementation_count": implementation_count,
        "claimed_definitions": list(bundle.get("definitions") or []),
        "public_workload_shapes": public_workload_shapes,
        "fallback_policy_thresholds": fallback_thresholds(),
        "dispatch_hit_rate": 1.0 if implementation_count == 0 else 0.0,
        "fallback_count": 0 if implementation_count == 0 else len(public_workload_shapes),
        "candidate_exception_count": 0,
        "latency_deltas_by_family": {},
        "memory_delta": None,
        "bottleneck_hints": [],
    }
    if live_result is not None:
        _merge_live_rmsnorm_diagnostics(diagnostics, live_result)
    if qwen_smoke is not None:
        _merge_qwen_smoke_diagnostics(diagnostics, qwen_smoke)
    if not diagnostics["bottleneck_hints"]:
        diagnostics["bottleneck_hints"] = _default_hints(diagnostics)
    return diagnostics


def _merge_live_rmsnorm_diagnostics(diagnostics: dict[str, Any], live_result: dict[str, Any]) -> None:
    shape_results: list[dict[str, Any]] = []
    for implementation in live_result.get("implementations") or []:
        shape_results.extend(implementation.get("shape_results") or [])
    covered = [item for item in shape_results if item.get("covered")]
    valid_covered = [item for item in covered if item.get("valid")]
    diagnostics["dispatch_hit_rate"] = float(len(covered) / len(shape_results)) if shape_results else 0.0
    diagnostics["fallback_count"] = len([item for item in shape_results if not item.get("covered")])
    diagnostics["candidate_exception_count"] = len([item for item in covered if not item.get("valid")])
    diagnostics["latency_deltas_by_family"] = _latency_by_family(valid_covered)
    diagnostics["live_rmsnorm_summary"] = {
        "valid": bool(live_result.get("valid")),
        "definition": live_result.get("definition"),
        "geomean_speedup": live_result.get("geomean_speedup"),
        "covered_shapes": [item.get("shape") for item in covered],
        "failed_shapes": [item.get("shape") for item in covered if not item.get("valid")],
    }


def _merge_qwen_smoke_diagnostics(diagnostics: dict[str, Any], qwen_smoke: dict[str, Any]) -> None:
    apply_summary = qwen_smoke.get("vllm_rmsnorm_apply") or {}
    candidate_calls = int(apply_summary.get("candidate_calls") or 0)
    fallback_calls = int(apply_summary.get("fallback_calls") or 0)
    total = candidate_calls + fallback_calls
    diagnostics["integrated_smoke"] = {
        "ok": bool(qwen_smoke.get("ok")),
        "candidate_rmsnorm_used_in_vllm": bool(qwen_smoke.get("candidate_rmsnorm_used_in_vllm")),
        "candidate_calls": candidate_calls,
        "fallback_calls": fallback_calls,
        "fallback_rate": float(fallback_calls / total) if total else None,
        "candidate_shapes": apply_summary.get("candidate_shapes") or [],
        "fallback_reasons": apply_summary.get("fallback_reasons") or {},
        "generate_elapsed_s": qwen_smoke.get("generate_elapsed_s"),
        "load_and_generate_elapsed_s": qwen_smoke.get("load_and_generate_elapsed_s"),
    }


def _latency_by_family(shape_results: list[dict[str, Any]]) -> dict[str, Any]:
    families: dict[str, Any] = {}
    for item in shape_results:
        speedup = item.get("speedup")
        baseline_ms = item.get("baseline_ms")
        candidate_ms = item.get("candidate_ms")
        shape = item.get("shape") or {}
        if not isinstance(speedup, (int, float)):
            continue
        family = "decode_like_short_tokens" if int(shape.get("num_tokens") or 0) <= 64 else "prefill_like_many_tokens"
        families[family] = {
            "shape": shape,
            "baseline_ms": baseline_ms,
            "candidate_ms": candidate_ms,
            "speedup": float(speedup),
            "delta_ms": None
            if not isinstance(baseline_ms, (int, float)) or not isinstance(candidate_ms, (int, float))
            else float(candidate_ms - baseline_ms),
        }
    return families


def _default_hints(diagnostics: dict[str, Any]) -> list[str]:
    hints: list[str] = []
    if diagnostics.get("candidate_exception_count"):
        hints.append("candidate produced invalid live RMSNorm outputs on at least one probe shape")
    if diagnostics.get("dispatch_hit_rate", 0.0) < 1.0 and diagnostics.get("implementation_count", 0) > 0:
        hints.append("shape guards do not cover every public/probe RMSNorm shape")
    for family, payload in (diagnostics.get("latency_deltas_by_family") or {}).items():
        if float(payload.get("speedup") or 0.0) < 1.0:
            hints.append(f"{family} RMSNorm probe path is slower than baseline")
    if not hints:
        hints.append("no public probe bottleneck detected")
    return hints
