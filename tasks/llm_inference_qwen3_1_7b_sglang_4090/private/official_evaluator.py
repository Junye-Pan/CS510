from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .sglang_smoke import (
    MODEL_PATH,
    REQUEST_TIMEOUT_S,
    _compare_token_logprob_pairs,
    _compare_top_logprob_pairs,
    _baseline_cache_root,
    _extract_token_ids,
    _load_stats,
    _merge_stats,
    _pick_free_port,
    _task_runs_root,
    _tail,
    _terminate_process_group,
    _wait_for_port_release,
    _wait_for_server,
)


WORKSPACE_ROOT = Path("/workspace")
DEFAULT_CONTEXT_LENGTH = 1536
DEFAULT_MAX_RUNNING_REQUESTS = 32
DEFAULT_MEM_FRACTION_STATIC = "0.65"
REQUIRED_COUNTERS = {
    "rmsnorm.kernel_hit",
    "fused_add_rmsnorm.kernel_hit",
    "swiglu.kernel_hit",
    "attention.decode.kernel_hit",
    "attention.extend.kernel_hit",
    "sampling.kernel_hit",
}
DIAGNOSTIC_TIMING_WORKLOADS = frozenset(
    {
        "prefill_b4_p512_o1",
        "decode_b8_p256_o16",
        "decode_b16_p128_o16",
    }
)
BASELINE_CACHE_VERSION = "qwen3_1_7b_official_baseline_v2"


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    mode: str
    batch_size: int
    prompt_tokens: int
    max_new_tokens: int
    top_logprobs_num: int = 3

    def payload(self) -> dict[str, Any]:
        input_ids = _make_input_ids(
            batch_size=self.batch_size,
            prompt_tokens=self.prompt_tokens,
            salt=_stable_salt(f"{self.mode}:{self.name}"),
        )
        return {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": self.max_new_tokens,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "logprob_start_len": -1,
            "top_logprobs_num": self.top_logprobs_num,
            "stream": False,
        }


@dataclass(frozen=True)
class EvaluatorPaths:
    run_dir: Path
    stats_dir: Path
    baseline_server_log: Path
    candidate_server_log: Path
    workloads_path: Path
    baseline_results_path: Path
    candidate_results_path: Path
    correctness_path: Path
    report_path: Path


@dataclass(frozen=True)
class BaselineCachePaths:
    cache_dir: Path
    metadata_path: Path
    results_path: Path
    model_info_path: Path
    server_log_path: Path


def run_official_evaluation(entry_path: Path, *, verifier: dict[str, Any] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    if not MODEL_PATH.is_dir():
        return _failed_result(
            "model_missing",
            f"model path does not exist: {MODEL_PATH}",
            started=started,
            verifier=verifier,
        )

    profile = os.environ.get("AO_QWEN3_1_7B_EVAL_PROFILE", "standard").strip().lower() or "standard"
    workloads = _build_workloads(profile)
    warmup_repeats, measured_repeats = _repeat_counts(profile)
    paths = _make_paths()
    paths.workloads_path.write_text(
        json.dumps(
            {
                "profile": profile,
                "warmup_repeats": warmup_repeats,
                "measured_repeats": measured_repeats,
                "workloads": [_workload_json(workload) for workload in workloads],
            },
            indent=2,
            sort_keys=True,
        )
    )
    cache_paths, cache_metadata = _baseline_cache_paths(
        profile=profile,
        workloads=workloads,
        warmup_repeats=warmup_repeats,
        measured_repeats=measured_repeats,
    )
    baseline_cache = {
        "enabled": _baseline_cache_enabled(),
        "hit": False,
        "cache_dir": str(cache_paths.cache_dir),
        "key": cache_metadata["cache_key"],
    }

    baseline_proc: subprocess.Popen | None = None
    candidate_proc: subprocess.Popen | None = None
    port = _pick_free_port()
    try:
        cached_baseline = _load_baseline_cache(cache_paths) if baseline_cache["enabled"] else None
        if cached_baseline is not None:
            baseline_cache["hit"] = True
            baseline_model_info = cached_baseline["baseline_model_info"]
            baseline_results = cached_baseline["baseline_results"]
            paths.baseline_results_path.write_text(json.dumps(baseline_results, indent=2, sort_keys=True))
            paths.baseline_server_log.write_text(
                json.dumps(
                    {
                        "baseline_cache": "hit",
                        "cache_dir": str(cache_paths.cache_dir),
                        "key": cache_metadata["cache_key"],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
        else:
            baseline_proc = _start_server(
                entry_path=entry_path,
                port=port,
                paths=paths,
                inject_candidate=False,
            )
            baseline_model_info = _wait_for_server(
                baseline_proc,
                port=port,
                log_path=paths.baseline_server_log,
            )
            baseline_results = _run_workloads(
                port=port,
                workloads=workloads,
                warmup_repeats=warmup_repeats,
                measured_repeats=measured_repeats,
            )
            paths.baseline_results_path.write_text(json.dumps(baseline_results, indent=2, sort_keys=True))
            if baseline_cache["enabled"]:
                _save_baseline_cache(
                    cache_paths,
                    metadata=cache_metadata,
                    baseline_results=baseline_results,
                    baseline_model_info=baseline_model_info,
                    server_log_path=paths.baseline_server_log,
                )

            _terminate_process_group(baseline_proc)
            baseline_proc = None
            _wait_for_port_release(port)

        port = _pick_free_port(exclude={port})
        candidate_proc = _start_server(
            entry_path=entry_path,
            port=port,
            paths=paths,
            inject_candidate=True,
        )
        candidate_model_info = _wait_for_server(
            candidate_proc,
            port=port,
            log_path=paths.candidate_server_log,
        )
        candidate_results = _run_workloads(
            port=port,
            workloads=workloads,
            warmup_repeats=warmup_repeats,
            measured_repeats=measured_repeats,
        )
        paths.candidate_results_path.write_text(json.dumps(candidate_results, indent=2, sort_keys=True))

        correctness = _compare_runs(workloads, baseline_results, candidate_results)
        stats_payloads = _load_stats(paths.stats_dir)
        merged_stats = _merge_stats(stats_payloads)
        stats_check = _check_stats(merged_stats)
        correctness["stats_valid"] = stats_check["valid"]
        correctness["stats_errors"] = stats_check["errors"]
        correctness["errors"].extend(stats_check["errors"])
        correctness["valid"] = bool(correctness["valid"] and stats_check["valid"])
        paths.correctness_path.write_text(json.dumps(correctness, indent=2, sort_keys=True))

        performance = _compute_performance(workloads, baseline_results, candidate_results)
        correct = bool(correctness["valid"])
        score = float(performance["combined_speedup"]) if correct else 0.0
        result = {
            "score": score,
            "valid": correct,
            "correct": {
                "correct": correct,
                "error": None if correct else "; ".join(correctness["errors"][:8]),
            },
            "metrics": {
                "combined_score": score,
                "combined_speedup": performance["combined_speedup"],
                "all_workload_combined_speedup": performance["all_workload_combined_speedup"],
                "diagnostic_combined_speedup": performance["diagnostic_combined_speedup"],
                "prefill_speedup": performance["mode_scores"].get("prefill"),
                "decode_speedup": performance["mode_scores"].get("decode"),
                "workload_count": len(workloads),
                "hard_gate_workload_count": performance["hard_gate_workload_count"],
                "diagnostic_workload_count": performance["diagnostic_workload_count"],
                "profile": profile,
            },
            "evaluator": {
                "public_details": {
                    "summary": "official prefill/decode evaluator passed"
                    if correct
                    else "official prefill/decode evaluator failed",
                    "score": score,
                    "combined_speedup": performance["combined_speedup"],
                    "all_workload_combined_speedup": performance["all_workload_combined_speedup"],
                    "diagnostic_combined_speedup": performance["diagnostic_combined_speedup"],
                    "prefill_speedup": performance["mode_scores"].get("prefill"),
                    "decode_speedup": performance["mode_scores"].get("decode"),
                    "profile": profile,
                    "workload_count": len(workloads),
                    "hard_gate_workload_count": performance["hard_gate_workload_count"],
                    "diagnostic_workload_count": performance["diagnostic_workload_count"],
                }
            },
            "details": {
                "elapsed_s": time.perf_counter() - started,
                "run_dir": str(paths.run_dir),
                "baseline_server_log": str(paths.baseline_server_log),
                "candidate_server_log": str(paths.candidate_server_log),
                "workloads_path": str(paths.workloads_path),
                "baseline_results_path": str(paths.baseline_results_path),
                "candidate_results_path": str(paths.candidate_results_path),
                "correctness_path": str(paths.correctness_path),
                "baseline_model_info": baseline_model_info,
                "candidate_model_info": candidate_model_info,
                "correctness": correctness,
                "performance": performance,
                "stats": merged_stats,
                "verifier": verifier or {},
                "baseline_cache": baseline_cache,
            },
        }
        paths.report_path.write_text(json.dumps(result, indent=2, sort_keys=True))
        return result
    except Exception as exc:
        return _failed_result(
            "evaluator_exception",
            f"{type(exc).__name__}: {exc}",
            started=started,
            verifier=verifier,
            paths=paths,
        )
    finally:
        if baseline_proc is not None:
            _terminate_process_group(baseline_proc)
        if candidate_proc is not None:
            _terminate_process_group(candidate_proc)
        _wait_for_port_release(port)


def _build_workloads(profile: str) -> list[WorkloadSpec]:
    if profile == "quick":
        return [
            WorkloadSpec("prefill_b1_p128_o1", "prefill", batch_size=1, prompt_tokens=128, max_new_tokens=1),
            WorkloadSpec("decode_b1_p64_o4", "decode", batch_size=1, prompt_tokens=64, max_new_tokens=4),
        ]

    prefill = [
        WorkloadSpec("prefill_b1_p128_o1", "prefill", batch_size=1, prompt_tokens=128, max_new_tokens=1),
        WorkloadSpec("prefill_b1_p512_o1", "prefill", batch_size=1, prompt_tokens=512, max_new_tokens=1),
        WorkloadSpec("prefill_b1_p1024_o1", "prefill", batch_size=1, prompt_tokens=1024, max_new_tokens=1),
        WorkloadSpec("prefill_b2_p512_o1", "prefill", batch_size=2, prompt_tokens=512, max_new_tokens=1),
        WorkloadSpec("prefill_b2_p1024_o1", "prefill", batch_size=2, prompt_tokens=1024, max_new_tokens=1),
        WorkloadSpec("prefill_b4_p256_o1", "prefill", batch_size=4, prompt_tokens=256, max_new_tokens=1),
        WorkloadSpec("prefill_b4_p512_o1", "prefill", batch_size=4, prompt_tokens=512, max_new_tokens=1),
    ]

    if profile == "expanded":
        decode = [
            WorkloadSpec(
                f"decode_b{batch_size}_p{prompt_tokens}_o16",
                "decode",
                batch_size=batch_size,
                prompt_tokens=prompt_tokens,
                max_new_tokens=16,
            )
            for batch_size in (1, 2, 4, 8, 16)
            for prompt_tokens in (64, 256, 512, 1024)
        ]
        return prefill + decode

    decode = [
        WorkloadSpec("decode_b1_p64_o16", "decode", batch_size=1, prompt_tokens=64, max_new_tokens=16),
        WorkloadSpec("decode_b2_p256_o16", "decode", batch_size=2, prompt_tokens=256, max_new_tokens=16),
        WorkloadSpec("decode_b4_p512_o16", "decode", batch_size=4, prompt_tokens=512, max_new_tokens=16),
        WorkloadSpec("decode_b8_p256_o16", "decode", batch_size=8, prompt_tokens=256, max_new_tokens=16),
        WorkloadSpec("decode_b16_p128_o16", "decode", batch_size=16, prompt_tokens=128, max_new_tokens=16),
    ]
    return prefill + decode


def _repeat_counts(profile: str) -> tuple[int, int]:
    if profile == "quick":
        defaults = (0, 1)
    elif profile == "expanded":
        defaults = (1, 3)
    else:
        defaults = (1, 2)
    warmup = int(os.environ.get("AO_QWEN3_1_7B_EVAL_WARMUP_REPEATS", defaults[0]))
    measured = int(os.environ.get("AO_QWEN3_1_7B_EVAL_MEASURED_REPEATS", defaults[1]))
    return max(0, warmup), max(1, measured)


def _start_server(*, entry_path: Path, port: int, paths: EvaluatorPaths, inject_candidate: bool) -> subprocess.Popen:
    env = os.environ.copy()
    sitecustomize_dir = Path(__file__).resolve().parent / "sitecustomize"
    python_paths = [
        str(WORKSPACE_ROOT),
        str(WORKSPACE_ROOT / "src"),
    ]
    if inject_candidate:
        python_paths.insert(0, str(sitecustomize_dir))
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    if inject_candidate:
        env["AO_QWEN3_1_7B_ENABLE_SGLANG_INJECTION"] = "1"
        env["AO_QWEN3_1_7B_CANDIDATE_ENTRY"] = str(entry_path.resolve())
        env["AO_QWEN3_1_7B_STATS_DIR"] = str(paths.stats_dir)
    else:
        env.pop("AO_QWEN3_1_7B_ENABLE_SGLANG_INJECTION", None)
        env.pop("AO_QWEN3_1_7B_CANDIDATE_ENTRY", None)
        env.pop("AO_QWEN3_1_7B_STATS_DIR", None)

    context_length = int(os.environ.get("AO_QWEN3_1_7B_EVAL_CONTEXT_LENGTH", DEFAULT_CONTEXT_LENGTH))
    max_running = int(os.environ.get("AO_QWEN3_1_7B_EVAL_MAX_RUNNING_REQUESTS", DEFAULT_MAX_RUNNING_REQUESTS))
    mem_fraction = os.environ.get("AO_QWEN3_1_7B_EVAL_MEM_FRACTION_STATIC", DEFAULT_MEM_FRACTION_STATIC)
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(MODEL_PATH),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--attention-backend",
        "triton",
        "--sampling-backend",
        "pytorch",
        "--context-length",
        str(context_length),
        "--max-running-requests",
        str(max_running),
        "--mem-fraction-static",
        str(mem_fraction),
        "--disable-radix-cache",
        "--disable-cuda-graph",
        "--trust-remote-code",
    ]
    log_path = paths.candidate_server_log if inject_candidate else paths.baseline_server_log
    log_handle = log_path.open("w")
    try:
        return subprocess.Popen(
            cmd,
            cwd=str(WORKSPACE_ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    finally:
        log_handle.close()


def _run_workloads(
    *,
    port: int,
    workloads: list[WorkloadSpec],
    warmup_repeats: int,
    measured_repeats: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for workload in workloads:
        warmups = [
            _run_workload_once(port=port, workload=workload, repeat_index=repeat_index, phase="warmup")
            for repeat_index in range(warmup_repeats)
        ]
        measured = [
            _run_workload_once(port=port, workload=workload, repeat_index=repeat_index, phase="measured")
            for repeat_index in range(measured_repeats)
        ]
        results[workload.name] = {
            "workload": _workload_json(workload),
            "warmups": warmups,
            "measured": measured,
            "summary": _summarize_measurements(measured),
            "comparison_response": measured[-1]["response"],
        }
    return results


def _run_workload_once(*, port: int, workload: WorkloadSpec, repeat_index: int, phase: str) -> dict[str, Any]:
    payload = workload.payload()
    started = time.perf_counter()
    response = _post_json(port, "/generate", payload, timeout=REQUEST_TIMEOUT_S)
    client_latency_s = time.perf_counter() - started
    _validate_response(workload, response)
    return {
        "phase": phase,
        "repeat_index": repeat_index,
        "client_latency_s": client_latency_s,
        "response_summary": _response_summary(response),
        "response": response,
    }


def _post_json(port: int, path: str, payload: dict[str, Any], *, timeout: float) -> Any:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {path}: {body}") from exc


def _compare_runs(
    workloads: list[WorkloadSpec],
    baseline_results: dict[str, Any],
    candidate_results: dict[str, Any],
) -> dict[str, Any]:
    hard_errors: list[str] = []
    diagnostic_errors: list[str] = []
    workload_results: list[dict[str, Any]] = []
    for workload in workloads:
        baseline = baseline_results[workload.name]["comparison_response"]
        candidate = candidate_results[workload.name]["comparison_response"]
        workload_errors = _compare_response(workload, baseline, candidate)
        hard_gate = _is_hard_gate_workload(workload)
        workload_results.append(
            {
                "name": workload.name,
                "mode": workload.mode,
                "correctness_role": "hard_gate" if hard_gate else "diagnostic_timing",
                "valid": not workload_errors,
                "hard_gate": hard_gate,
                "error_count": len(workload_errors),
                "errors": workload_errors,
            }
        )
        formatted_errors = [f"{workload.name}: {error}" for error in workload_errors]
        if hard_gate:
            hard_errors.extend(formatted_errors)
        else:
            diagnostic_errors.extend(formatted_errors)
    return {
        "valid": not hard_errors,
        "errors": hard_errors,
        "diagnostic_errors": diagnostic_errors,
        "workloads": workload_results,
        "hard_gate_workload_count": sum(1 for workload in workloads if _is_hard_gate_workload(workload)),
        "diagnostic_workload_count": sum(1 for workload in workloads if not _is_hard_gate_workload(workload)),
        "logprob_abs_tol": 2.0e-2,
    }


def _compare_response(workload: WorkloadSpec, baseline: Any, candidate: Any) -> list[str]:
    errors: list[str] = []
    baseline_items = _response_items(baseline)
    candidate_items = _response_items(candidate)
    if len(baseline_items) != workload.batch_size:
        errors.append(f"baseline batch size differs: {len(baseline_items)} != {workload.batch_size}")
    if len(candidate_items) != workload.batch_size:
        errors.append(f"candidate batch size differs: {len(candidate_items)} != {workload.batch_size}")
    if len(baseline_items) != len(candidate_items):
        errors.append(f"response batch size differs: {len(baseline_items)} != {len(candidate_items)}")
        return errors

    for batch_index, (baseline_item, candidate_item) in enumerate(zip(baseline_items, candidate_items)):
        prefix = f"batch[{batch_index}]"
        if baseline_item.get("text") != candidate_item.get("text"):
            errors.append(f"{prefix}: text differs")
        baseline_meta = baseline_item.get("meta_info") or {}
        candidate_meta = candidate_item.get("meta_info") or {}
        for field in ("prompt_tokens", "completion_tokens"):
            if baseline_meta.get(field) != candidate_meta.get(field):
                errors.append(f"{prefix}: {field} differs: {baseline_meta.get(field)} != {candidate_meta.get(field)}")

        baseline_token_logprobs = baseline_meta.get("output_token_logprobs")
        candidate_token_logprobs = candidate_meta.get("output_token_logprobs")
        if not isinstance(baseline_token_logprobs, list) or not isinstance(candidate_token_logprobs, list):
            errors.append(f"{prefix}: output_token_logprobs missing")
        else:
            errors.extend(
                f"{prefix}: {error}"
                for error in _compare_token_logprob_pairs(
                    workload.name,
                    "output_token_logprobs",
                    baseline_token_logprobs,
                    candidate_token_logprobs,
                )
            )

        baseline_top_logprobs = baseline_meta.get("output_top_logprobs")
        candidate_top_logprobs = candidate_meta.get("output_top_logprobs")
        if not isinstance(baseline_top_logprobs, list) or not isinstance(candidate_top_logprobs, list):
            errors.append(f"{prefix}: output_top_logprobs missing")
        else:
            errors.extend(
                f"{prefix}: {error}"
                for error in _compare_top_logprob_pairs(workload.name, baseline_top_logprobs, candidate_top_logprobs)
            )
    return errors


def _check_stats(merged_stats: dict[str, Any]) -> dict[str, Any]:
    counters = merged_stats.get("counters") or {}
    missing = [counter for counter in sorted(REQUIRED_COUNTERS) if int(counters.get(counter, 0)) <= 0]
    exceptions = [
        f"{key}={value}"
        for key, value in sorted(counters.items())
        if str(key).endswith(".exception") and int(value) > 0
    ]
    fallback_reasons = merged_stats.get("fallback_reasons") or {}
    guard_fallbacks = [f"{key}={value}" for key, value in sorted(fallback_reasons.items()) if int(value) > 0]
    errors = [f"required dispatch counter has no hits: {counter}" for counter in missing]
    errors.extend(f"candidate exception counter is nonzero: {item}" for item in exceptions)
    errors.extend(f"task-owned guard fallback occurred: {item}" for item in guard_fallbacks)
    return {"valid": not errors, "errors": errors}


def _compute_performance(
    workloads: list[WorkloadSpec],
    baseline_results: dict[str, Any],
    candidate_results: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    by_mode: dict[str, list[float]] = {}
    all_by_mode: dict[str, list[float]] = {}
    diagnostic_by_mode: dict[str, list[float]] = {}
    hard_gate_speedups: list[float] = []
    all_speedups: list[float] = []
    diagnostic_speedups: list[float] = []
    for workload in workloads:
        baseline_summary = baseline_results[workload.name]["summary"]
        candidate_summary = candidate_results[workload.name]["summary"]
        baseline_latency = float(baseline_summary["avg_client_latency_s"])
        candidate_latency = float(candidate_summary["avg_client_latency_s"])
        speedup = baseline_latency / candidate_latency if candidate_latency > 0 else 0.0
        hard_gate = _is_hard_gate_workload(workload)
        all_by_mode.setdefault(workload.mode, []).append(speedup)
        all_speedups.append(speedup)
        if hard_gate:
            by_mode.setdefault(workload.mode, []).append(speedup)
            hard_gate_speedups.append(speedup)
        else:
            diagnostic_by_mode.setdefault(workload.mode, []).append(speedup)
            diagnostic_speedups.append(speedup)
        rows.append(
            {
                "name": workload.name,
                "mode": workload.mode,
                "correctness_role": "hard_gate" if hard_gate else "diagnostic_timing",
                "batch_size": workload.batch_size,
                "prompt_tokens": workload.prompt_tokens,
                "max_new_tokens": workload.max_new_tokens,
                "baseline_avg_client_latency_s": baseline_latency,
                "candidate_avg_client_latency_s": candidate_latency,
                "speedup": speedup,
                "baseline_tokens_per_s": baseline_summary["avg_total_tokens_per_s"],
                "candidate_tokens_per_s": candidate_summary["avg_total_tokens_per_s"],
            }
        )

    mode_scores = {mode: _geomean(values) for mode, values in sorted(by_mode.items())}
    combined_speedup = _geomean(hard_gate_speedups)
    return {
        "combined_speedup": combined_speedup,
        "all_workload_combined_speedup": _geomean(all_speedups),
        "diagnostic_combined_speedup": _geomean(diagnostic_speedups) if diagnostic_speedups else None,
        "mode_scores": mode_scores,
        "all_mode_scores": {mode: _geomean(values) for mode, values in sorted(all_by_mode.items())},
        "diagnostic_mode_scores": {
            mode: _geomean(values) for mode, values in sorted(diagnostic_by_mode.items())
        },
        "hard_gate_workload_count": len(hard_gate_speedups),
        "diagnostic_workload_count": len(diagnostic_speedups),
        "workloads": rows,
    }


def _summarize_measurements(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(item["client_latency_s"]) for item in measurements]
    prompt_tokens = [int(item["response_summary"]["prompt_tokens"]) for item in measurements]
    completion_tokens = [int(item["response_summary"]["completion_tokens"]) for item in measurements]
    total_tokens = [prompt + completion for prompt, completion in zip(prompt_tokens, completion_tokens)]
    avg_latency = sum(latencies) / len(latencies)
    avg_total_tokens = sum(total_tokens) / len(total_tokens)
    return {
        "avg_client_latency_s": avg_latency,
        "min_client_latency_s": min(latencies),
        "max_client_latency_s": max(latencies),
        "avg_prompt_tokens": sum(prompt_tokens) / len(prompt_tokens),
        "avg_completion_tokens": sum(completion_tokens) / len(completion_tokens),
        "avg_total_tokens_per_s": avg_total_tokens / avg_latency if avg_latency > 0 else 0.0,
        "repeat_count": len(measurements),
    }


def _validate_response(workload: WorkloadSpec, response: Any) -> None:
    items = _response_items(response)
    if len(items) != workload.batch_size:
        raise ValueError(f"{workload.name}: expected batch size {workload.batch_size}, got {len(items)}")
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"{workload.name}[{index}]: response item is not an object")
        if "text" not in item:
            raise ValueError(f"{workload.name}[{index}]: response missing text")
        meta = item.get("meta_info")
        if not isinstance(meta, dict):
            raise ValueError(f"{workload.name}[{index}]: response missing meta_info")
        if int(meta.get("completion_tokens", 0)) != workload.max_new_tokens:
            raise ValueError(
                f"{workload.name}[{index}]: expected {workload.max_new_tokens} completion tokens, "
                f"got {meta.get('completion_tokens')}"
            )
        if int(meta.get("prompt_tokens", 0)) != workload.prompt_tokens:
            raise ValueError(
                f"{workload.name}[{index}]: expected {workload.prompt_tokens} prompt tokens, "
                f"got {meta.get('prompt_tokens')}"
            )
        if not isinstance(meta.get("output_token_logprobs"), list):
            raise ValueError(f"{workload.name}[{index}]: missing output_token_logprobs")
        if not isinstance(meta.get("output_top_logprobs"), list):
            raise ValueError(f"{workload.name}[{index}]: missing output_top_logprobs")


def _response_items(response: Any) -> list[dict[str, Any]]:
    if isinstance(response, list):
        return response
    if isinstance(response, dict):
        return [response]
    raise TypeError(f"unexpected /generate response type: {type(response).__name__}")


def _response_summary(response: Any) -> dict[str, Any]:
    items = _response_items(response)
    prompt_tokens = 0
    completion_tokens = 0
    server_e2e_latencies: list[float] = []
    output_token_ids: list[list[int]] = []
    for item in items:
        meta = item.get("meta_info") or {}
        prompt_tokens += int(meta.get("prompt_tokens", 0))
        completion_tokens += int(meta.get("completion_tokens", 0))
        if meta.get("e2e_latency") is not None:
            server_e2e_latencies.append(float(meta["e2e_latency"]))
        pairs = meta.get("output_token_logprobs")
        output_token_ids.append(_extract_token_ids(pairs) if isinstance(pairs, list) else [])
    return {
        "batch_size": len(items),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "server_e2e_latencies": server_e2e_latencies,
        "output_token_ids": output_token_ids,
    }


def _make_input_ids(*, batch_size: int, prompt_tokens: int, salt: int) -> list[int] | list[list[int]]:
    vocab_window = 120_000
    rows: list[list[int]] = []
    for row_index in range(batch_size):
        row_salt = salt + row_index * 9973
        rows.append([1000 + ((row_salt + token_index * 37) % vocab_window) for token_index in range(prompt_tokens)])
    if batch_size == 1:
        return rows[0]
    return rows


def _stable_salt(text: str) -> int:
    value = 0
    for index, char in enumerate(text):
        value = (value + (index + 1) * ord(char)) % 100_000
    return value


def _workload_json(workload: WorkloadSpec) -> dict[str, Any]:
    return {
        "name": workload.name,
        "mode": workload.mode,
        "correctness_role": "hard_gate" if _is_hard_gate_workload(workload) else "diagnostic_timing",
        "batch_size": workload.batch_size,
        "prompt_tokens": workload.prompt_tokens,
        "max_new_tokens": workload.max_new_tokens,
        "top_logprobs_num": workload.top_logprobs_num,
        "input_salt": _stable_salt(f"{workload.mode}:{workload.name}"),
    }


def _geomean(values: list[float]) -> float:
    positives = [max(float(value), 1.0e-12) for value in values if math.isfinite(float(value))]
    if not positives:
        return 0.0
    return math.exp(sum(math.log(value) for value in positives) / len(positives))


def _is_hard_gate_workload(workload: WorkloadSpec) -> bool:
    return workload.name not in DIAGNOSTIC_TIMING_WORKLOADS


def _make_paths() -> EvaluatorPaths:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = _task_runs_root() / f"qwen3_1_7b_official_eval_{stamp}_{os.getpid()}"
    stats_dir = run_dir / "stats"
    run_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    return EvaluatorPaths(
        run_dir=run_dir,
        stats_dir=stats_dir,
        baseline_server_log=run_dir / "baseline_server.log",
        candidate_server_log=run_dir / "candidate_server.log",
        workloads_path=run_dir / "workloads.json",
        baseline_results_path=run_dir / "baseline_results.json",
        candidate_results_path=run_dir / "candidate_results.json",
        correctness_path=run_dir / "correctness.json",
        report_path=run_dir / "evaluator_report.json",
    )


def _baseline_cache_enabled() -> bool:
    return os.environ.get("AO_QWEN3_1_7B_EVAL_BASELINE_CACHE", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }


def _baseline_cache_paths(
    *,
    profile: str,
    workloads: list[WorkloadSpec],
    warmup_repeats: int,
    measured_repeats: int,
) -> tuple[BaselineCachePaths, dict[str, Any]]:
    context_length = int(os.environ.get("AO_QWEN3_1_7B_EVAL_CONTEXT_LENGTH", DEFAULT_CONTEXT_LENGTH))
    max_running = int(os.environ.get("AO_QWEN3_1_7B_EVAL_MAX_RUNNING_REQUESTS", DEFAULT_MAX_RUNNING_REQUESTS))
    mem_fraction = os.environ.get("AO_QWEN3_1_7B_EVAL_MEM_FRACTION_STATIC", DEFAULT_MEM_FRACTION_STATIC)
    metadata: dict[str, Any] = {
        "cache_version": BASELINE_CACHE_VERSION,
        "profile": profile,
        "model_path": str(MODEL_PATH),
        "python": sys.executable,
        "context_length": context_length,
        "max_running_requests": max_running,
        "mem_fraction_static": mem_fraction,
        "dtype": "bfloat16",
        "attention_backend": "triton",
        "sampling_backend": "pytorch",
        "disable_radix_cache": True,
        "disable_cuda_graph": True,
        "warmup_repeats": warmup_repeats,
        "measured_repeats": measured_repeats,
        "workloads": [_workload_json(workload) for workload in workloads],
    }
    key_material = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    metadata["cache_key"] = hashlib.sha256(key_material).hexdigest()[:24]
    cache_dir = _baseline_cache_root() / "qwen3_1_7b_official_eval" / metadata["cache_key"]
    paths = BaselineCachePaths(
        cache_dir=cache_dir,
        metadata_path=cache_dir / "metadata.json",
        results_path=cache_dir / "baseline_results.json",
        model_info_path=cache_dir / "baseline_model_info.json",
        server_log_path=cache_dir / "baseline_server.log",
    )
    return paths, metadata


def _load_baseline_cache(paths: BaselineCachePaths) -> dict[str, Any] | None:
    if not paths.results_path.is_file() or not paths.model_info_path.is_file() or not paths.metadata_path.is_file():
        return None
    try:
        return {
            "metadata": json.loads(paths.metadata_path.read_text()),
            "baseline_results": json.loads(paths.results_path.read_text()),
            "baseline_model_info": json.loads(paths.model_info_path.read_text()),
        }
    except Exception:
        return None


def _save_baseline_cache(
    paths: BaselineCachePaths,
    *,
    metadata: dict[str, Any],
    baseline_results: dict[str, Any],
    baseline_model_info: dict[str, Any],
    server_log_path: Path,
) -> None:
    paths.cache_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(metadata)
    payload["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    paths.metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    paths.results_path.write_text(json.dumps(baseline_results, indent=2, sort_keys=True))
    paths.model_info_path.write_text(json.dumps(baseline_model_info, indent=2, sort_keys=True))
    try:
        paths.server_log_path.write_text(server_log_path.read_text(errors="replace"))
    except Exception:
        pass


def _failed_result(
    reason: str,
    message: str,
    *,
    started: float,
    verifier: dict[str, Any] | None = None,
    paths: EvaluatorPaths | None = None,
) -> dict[str, Any]:
    details: dict[str, Any] = {
        "reason": reason,
        "elapsed_s": time.perf_counter() - started,
        "verifier": verifier or {},
    }
    if paths is not None:
        details.update(
            {
                "run_dir": str(paths.run_dir),
                "baseline_server_log": str(paths.baseline_server_log),
                "candidate_server_log": str(paths.candidate_server_log),
                "baseline_log_tail": _tail(paths.baseline_server_log),
                "candidate_log_tail": _tail(paths.candidate_server_log),
            }
        )
    return {
        "score": 0.0,
        "valid": False,
        "correct": {"correct": False, "error": message},
        "metrics": {"combined_score": 0.0, "combined_speedup": 0.0},
        "evaluator": {
            "public_details": {
                "summary": "official prefill/decode evaluator failed",
                "reason": reason,
            }
        },
        "details": details,
    }
