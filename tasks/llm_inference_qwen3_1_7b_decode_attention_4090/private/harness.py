from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any

from .apply_candidate import load_candidate
from .reference import official_decode_attention_fwd
from .verifier_primitives import compare_tensor, geometric_mean
from .workloads import DecodeInputs, build_workloads, materialize_workload, repeat_counts


READONLY_KEYS = ("q", "k_buffer", "v_buffer", "kv_indptr", "kv_indices", "num_kv_splits", "sinks")


def run_decode_suite(
    entry_path: Path,
    *,
    profile: str,
    include_timing: bool,
    warmup_repeats: int | None = None,
    measured_repeats: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        import torch
    except Exception as exc:
        return _failed_import("torch", exc, started)

    if not torch.cuda.is_available():
        return {
            "valid": False,
            "results": [
                {
                    "name": "cuda_available",
                    "valid": False,
                    "error": "CUDA is required for Qwen3 decode attention checks",
                    "metrics": {},
                }
            ],
            "elapsed_s": time.perf_counter() - started,
        }

    candidate = load_candidate(entry_path).modules.decode_attention
    device = torch.device("cuda")
    profile = (profile or "standard").strip().lower()
    default_warmup, default_measured = repeat_counts(profile)
    warmup_repeats = default_warmup if warmup_repeats is None else int(warmup_repeats)
    measured_repeats = default_measured if measured_repeats is None else int(measured_repeats)

    torch.manual_seed(20260529)
    results: list[dict[str, Any]] = []
    for spec in build_workloads(profile):
        try:
            result = _run_workload(
                spec,
                candidate=candidate,
                torch=torch,
                device=device,
                include_timing=include_timing,
                warmup_repeats=warmup_repeats,
                measured_repeats=measured_repeats,
            )
        except Exception as exc:
            result = {
                "name": spec.name,
                "valid": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback_tail": _traceback_tail(),
                "metrics": spec.to_public_json(),
            }
        results.append(result)

    valid = all(bool(result.get("valid")) for result in results)
    speedups = [
        float(result["metrics"]["speedup"])
        for result in results
        if result.get("valid") and "speedup" in result.get("metrics", {})
    ]
    suite_speedup = geometric_mean(speedups) if include_timing and len(speedups) == len(results) else None
    return {
        "valid": valid,
        "profile": profile,
        "include_timing": include_timing,
        "warmup_repeats": warmup_repeats,
        "measured_repeats": measured_repeats,
        "geomean_speedup": suite_speedup,
        "results": results,
        "elapsed_s": time.perf_counter() - started,
    }


def _run_workload(
    spec,
    *,
    candidate,
    torch,
    device,
    include_timing: bool,
    warmup_repeats: int,
    measured_repeats: int,
) -> dict[str, Any]:
    base_inputs = materialize_workload(spec, torch=torch, device=device)
    reference_inputs = base_inputs.clone()
    _invoke_reference(reference_inputs)
    torch.cuda.synchronize()

    candidate_inputs = base_inputs.clone()
    readonly_before = {name: tensor.clone() for name, tensor in candidate_inputs.readonly_tensors().items()}
    fallback_calls = {"correctness": 0}

    def fallback(**overrides):
        fallback_calls["correctness"] += 1
        return _fallback_decode(candidate_inputs, overrides)

    _invoke_candidate(candidate, candidate_inputs, fallback=fallback)
    torch.cuda.synchronize()

    error_stats = compare_tensor(candidate_inputs.o, reference_inputs.o, name=f"{spec.name}.o")
    readonly_stats = _check_readonly_inputs(candidate_inputs, readonly_before, torch=torch)
    valid = bool(error_stats["allclose"] and readonly_stats["valid"])
    metrics: dict[str, Any] = {
        **spec.to_public_json(),
        **error_stats,
        "readonly_inputs_unchanged": readonly_stats["valid"],
        "mutated_readonly_inputs": readonly_stats["mutated"],
        "fallback_calls_correctness": fallback_calls["correctness"],
    }
    error = None
    if not error_stats["allclose"]:
        error = "candidate output differs from SGLang decode_attention_fwd reference"
    if not readonly_stats["valid"]:
        error = f"candidate mutated read-only inputs: {readonly_stats['mutated']}"

    if valid and include_timing:
        reference_timing_inputs = base_inputs.clone()
        candidate_timing_inputs = base_inputs.clone()
        candidate_timing_fallback_calls = {"count": 0}

        def reference_call():
            _fallback_decode(reference_timing_inputs, {})

        def candidate_fallback(**overrides):
            candidate_timing_fallback_calls["count"] += 1
            return _fallback_decode(candidate_timing_inputs, overrides)

        def candidate_call():
            _invoke_candidate(candidate, candidate_timing_inputs, fallback=candidate_fallback)

        ref_ms = _time_cuda_call(
            reference_call,
            torch=torch,
            warmup_repeats=warmup_repeats,
            measured_repeats=measured_repeats,
        )
        candidate_timing_fallback_calls["count"] = 0
        cand_ms = _time_cuda_call(
            candidate_call,
            torch=torch,
            warmup_repeats=warmup_repeats,
            measured_repeats=measured_repeats,
        )
        speedup = ref_ms / cand_ms if cand_ms > 0 else 0.0
        metrics.update(
            {
                "ref_ms": ref_ms,
                "cand_ms": cand_ms,
                "speedup": speedup,
                "fallback_calls_timing": candidate_timing_fallback_calls["count"],
            }
        )

    return {
        "name": spec.name,
        "valid": valid,
        "error": error,
        "metrics": metrics,
    }


def _invoke_reference(inputs: DecodeInputs) -> Any:
    return official_decode_attention_fwd(
        inputs.q,
        inputs.k_buffer,
        inputs.v_buffer,
        inputs.o,
        inputs.kv_indptr,
        inputs.kv_indices,
        inputs.attn_logits,
        inputs.attn_lse,
        inputs.num_kv_splits,
        inputs.max_kv_splits,
        inputs.sm_scale,
        inputs.k_scale,
        inputs.v_scale,
        logit_cap=inputs.logit_cap,
        sinks=inputs.sinks,
        xai_temperature_len=inputs.xai_temperature_len,
        has_mla=inputs.has_mla,
        use_pdl=inputs.use_pdl,
    )


def _invoke_candidate(candidate, inputs: DecodeInputs, *, fallback) -> Any:
    return candidate.run(
        inputs.q,
        inputs.k_buffer,
        inputs.v_buffer,
        inputs.o,
        inputs.kv_indptr,
        inputs.kv_indices,
        inputs.attn_logits,
        inputs.attn_lse,
        inputs.num_kv_splits,
        inputs.max_kv_splits,
        inputs.sm_scale,
        inputs.k_scale,
        inputs.v_scale,
        logit_cap=inputs.logit_cap,
        sinks=inputs.sinks,
        xai_temperature_len=inputs.xai_temperature_len,
        has_mla=inputs.has_mla,
        use_pdl=inputs.use_pdl,
        fallback=fallback,
    )


def _fallback_decode(inputs: DecodeInputs, overrides: dict[str, Any]) -> Any:
    active = DecodeInputs(
        q=overrides.get("q", inputs.q),
        k_buffer=overrides.get("k_buffer", inputs.k_buffer),
        v_buffer=overrides.get("v_buffer", inputs.v_buffer),
        o=overrides.get("o", inputs.o),
        kv_indptr=overrides.get("kv_indptr", inputs.kv_indptr),
        kv_indices=overrides.get("kv_indices", inputs.kv_indices),
        attn_logits=overrides.get("attn_logits", inputs.attn_logits),
        attn_lse=overrides.get("attn_lse", inputs.attn_lse),
        num_kv_splits=overrides.get("num_kv_splits", inputs.num_kv_splits),
        max_kv_splits=overrides.get("max_kv_splits", inputs.max_kv_splits),
        sm_scale=overrides.get("sm_scale", inputs.sm_scale),
        k_scale=overrides.get("k_scale", inputs.k_scale),
        v_scale=overrides.get("v_scale", inputs.v_scale),
        logit_cap=overrides.get("logit_cap", inputs.logit_cap),
        sinks=overrides.get("sinks", inputs.sinks),
        xai_temperature_len=overrides.get("xai_temperature_len", inputs.xai_temperature_len),
        has_mla=overrides.get("has_mla", inputs.has_mla),
        use_pdl=overrides.get("use_pdl", inputs.use_pdl),
    )
    _invoke_reference(active)
    return active.o


def _check_readonly_inputs(inputs: DecodeInputs, before: dict[str, Any], *, torch) -> dict[str, Any]:
    mutated: list[str] = []
    current = inputs.readonly_tensors()
    for name in READONLY_KEYS:
        if name not in before:
            continue
        if name not in current:
            mutated.append(name)
            continue
        if not bool(torch.equal(current[name], before[name])):
            mutated.append(name)
    return {"valid": not mutated, "mutated": mutated}


def _time_cuda_call(callable_obj, *, torch, warmup_repeats: int, measured_repeats: int) -> float:
    for _ in range(max(0, warmup_repeats)):
        callable_obj()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(max(1, measured_repeats)):
        callable_obj()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / float(max(1, measured_repeats)))


def _failed_import(module: str, exc: BaseException, started: float) -> dict[str, Any]:
    return {
        "valid": False,
        "results": [
            {
                "name": f"import_{module}",
                "valid": False,
                "error": f"{type(exc).__name__}: {exc}",
                "metrics": {},
            }
        ],
        "elapsed_s": time.perf_counter() - started,
    }


def _traceback_tail(limit: int = 8) -> list[str]:
    return traceback.format_exc().strip().splitlines()[-limit:]
