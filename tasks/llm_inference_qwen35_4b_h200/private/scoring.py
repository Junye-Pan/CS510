from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any


BASELINE_SCORE = 1.0
SMOKE_GENERATE_SCORING_VERSION = "qwen35_4b_vllm_integrated_generate_v1"
SMOKE_GENERATE_SCORE_COMPONENT = "generate_elapsed_speedup"
SMOKE_GENERATE_SCORING_STEPS = (
    "load_pinned_qwen_vllm_smoke_baseline_artifact",
    "run_candidate_through_vllm_general_plugin_apply_path",
    "compare_generated_tokens_text_and_top_logprobs_to_smoke_baseline",
    "score_baseline_generate_elapsed_s_div_candidate_generate_elapsed_s",
)
END_TO_END_SCORING_VERSION = "qwen35_4b_vllm_prefill_decode_mixed_serving_sweeps_v2"
END_TO_END_SCORE_COMPONENT = "prefill_decode_mixed_serving_p90_geomean_speedup"
END_TO_END_SCORING_STEPS = (
    "static_verify_candidate_bundle",
    "run_live_rmsnorm_correctness_and_microbenchmark_diagnostics",
    "load_pinned_qwen_vllm_logits_distribution_baseline_artifact",
    "run_candidate_qwen_vllm_logits_distribution_probe",
    "compare_prefill_decode_margin_and_long_context_logit_distributions",
    "load_pinned_qwen_vllm_eval_suite_baseline_artifact",
    "run_candidate_through_vllm_general_plugin_apply_path",
    "run_prefill_decode_and_mixed_serving_style_batch_concurrency_sweeps",
    "compare_generated_tokens_and_text_to_eval_suite_baseline",
    "enforce_candidate_call_and_fallback_policy",
    "score_geomean_family_serving_p90_request_latency_speedups",
)


def geomean_speedup(values: Iterable[float]) -> float:
    speedups = [float(value) for value in values]
    if not speedups:
        raise ValueError("geomean_speedup requires at least one value")
    if any(value <= 0.0 or not math.isfinite(value) for value in speedups):
        return 0.0
    return float(math.exp(sum(math.log(value) for value in speedups) / len(speedups)))


def end_to_end_generate_score(
    *,
    candidate_smoke: dict[str, Any],
    baseline_artifact: dict[str, Any],
) -> dict[str, Any]:
    baseline_metrics = baseline_artifact.get("metrics") or {}
    baseline_generate = _positive_float(baseline_metrics.get("generate_elapsed_s"))
    candidate_generate = _positive_float(candidate_smoke.get("generate_elapsed_s"))
    if baseline_generate is None:
        return _invalid_end_to_end_score(
            "baseline artifact is missing a positive generate_elapsed_s metric",
            version=SMOKE_GENERATE_SCORING_VERSION,
            score_component=SMOKE_GENERATE_SCORE_COMPONENT,
        )
    if candidate_generate is None:
        return _invalid_end_to_end_score(
            "candidate smoke is missing a positive generate_elapsed_s metric",
            version=SMOKE_GENERATE_SCORING_VERSION,
            score_component=SMOKE_GENERATE_SCORE_COMPONENT,
        )

    score = baseline_generate / candidate_generate
    baseline_load_generate = _positive_float(baseline_metrics.get("load_and_generate_elapsed_s"))
    candidate_load_generate = _positive_float(candidate_smoke.get("load_and_generate_elapsed_s"))
    load_generate_speedup = (
        baseline_load_generate / candidate_load_generate
        if baseline_load_generate is not None and candidate_load_generate is not None
        else None
    )
    return {
        "valid": True,
        "error": None,
        "version": SMOKE_GENERATE_SCORING_VERSION,
        "score_component": SMOKE_GENERATE_SCORE_COMPONENT,
        "score": float(score),
        "baseline_generate_elapsed_s": baseline_generate,
        "candidate_generate_elapsed_s": candidate_generate,
        "baseline_load_and_generate_elapsed_s": baseline_load_generate,
        "candidate_load_and_generate_elapsed_s": candidate_load_generate,
        "load_and_generate_speedup_diagnostic": load_generate_speedup,
        "baseline_artifact_path": baseline_artifact.get("_artifact_path"),
        "scoring_steps": list(SMOKE_GENERATE_SCORING_STEPS),
        "note": (
            "Score uses the timed vLLM generate call for the fixed integrated workload. "
            "Model load time is reported as a diagnostic and is not included in the official score."
        ),
    }


def end_to_end_suite_score(
    *,
    candidate_suite: dict[str, Any],
    baseline_artifact: dict[str, Any],
) -> dict[str, Any]:
    baseline_families = ((baseline_artifact.get("metrics") or {}).get("families") or {})
    candidate_families = {
        family.get("name"): family
        for family in candidate_suite.get("families") or []
        if family.get("name")
    }
    family_speedups: dict[str, float] = {}
    family_metrics: dict[str, dict[str, Any]] = {}
    for family_name in ("prefill", "decode", "mixed"):
        baseline_family = baseline_families.get(family_name) or {}
        candidate_family = candidate_families.get(family_name) or {}
        baseline_elapsed = _positive_float(baseline_family.get("serving_score_s"))
        candidate_elapsed = _positive_float(candidate_family.get("serving_score_s"))
        if baseline_elapsed is None:
            return _invalid_end_to_end_score(f"baseline family {family_name} is missing a positive serving_score_s")
        if candidate_elapsed is None:
            return _invalid_end_to_end_score(f"candidate family {family_name} is missing a positive serving_score_s")
        speedup = baseline_elapsed / candidate_elapsed
        family_speedups[family_name] = float(speedup)
        baseline_metrics = baseline_family.get("serving_metrics") or {}
        candidate_metrics = candidate_family.get("serving_metrics") or {}
        family_metrics[family_name] = {
            "baseline_serving_score_s": baseline_elapsed,
            "candidate_serving_score_s": candidate_elapsed,
            "speedup": float(speedup),
            "baseline_request_latency_s_p50": baseline_metrics.get("request_latency_s_p50"),
            "candidate_request_latency_s_p50": candidate_metrics.get("request_latency_s_p50"),
            "baseline_request_latency_s_p90": baseline_metrics.get("request_latency_s_p90"),
            "candidate_request_latency_s_p90": candidate_metrics.get("request_latency_s_p90"),
            "baseline_ttft_s_p90": baseline_metrics.get("ttft_s_p90"),
            "candidate_ttft_s_p90": candidate_metrics.get("ttft_s_p90"),
            "baseline_tpot_s_p90": baseline_metrics.get("tpot_s_p90"),
            "candidate_tpot_s_p90": candidate_metrics.get("tpot_s_p90"),
            "baseline_generate_elapsed_s": baseline_family.get("generate_elapsed_s"),
            "candidate_generate_elapsed_s": candidate_family.get("generate_elapsed_s"),
            "baseline_output_token_count": baseline_family.get("output_token_count"),
            "candidate_output_token_count": candidate_family.get("output_token_count"),
            "baseline_input_token_count": baseline_family.get("input_token_count"),
            "candidate_input_token_count": candidate_family.get("input_token_count"),
        }
    score = geomean_speedup(family_speedups.values())
    return {
        "valid": True,
        "error": None,
        "version": END_TO_END_SCORING_VERSION,
        "score_component": END_TO_END_SCORE_COMPONENT,
        "score": float(score),
        "family_speedups": family_speedups,
        "family_metrics": family_metrics,
        "baseline_total_generate_elapsed_s": (baseline_artifact.get("metrics") or {}).get("total_generate_elapsed_s"),
        "candidate_total_generate_elapsed_s": candidate_suite.get("total_generate_elapsed_s"),
        "baseline_load_and_generate_elapsed_s": (baseline_artifact.get("metrics") or {}).get(
            "load_and_generate_elapsed_s"
        ),
        "candidate_load_and_generate_elapsed_s": candidate_suite.get("load_and_generate_elapsed_s"),
        "baseline_artifact_path": baseline_artifact.get("_artifact_path"),
        "scoring_steps": list(END_TO_END_SCORING_STEPS),
        "note": (
            "Score is the geometric mean of prefill, decode, and mixed serving-style sweep speedups. "
            "Each family speedup is baseline p90 request latency divided by candidate p90 request latency. "
            "TTFT, TPOT, model load time, and kernel microbenchmarks are diagnostics."
        ),
    }


def _invalid_end_to_end_score(
    error: str,
    *,
    version: str = END_TO_END_SCORING_VERSION,
    score_component: str = END_TO_END_SCORE_COMPONENT,
) -> dict[str, Any]:
    return {
        "valid": False,
        "error": error,
        "version": version,
        "score_component": score_component,
        "score": 0.0,
        "scoring_steps": list(END_TO_END_SCORING_STEPS),
    }


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0.0 or not math.isfinite(parsed):
        return None
    return parsed
