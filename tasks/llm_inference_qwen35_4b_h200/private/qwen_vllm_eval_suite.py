from __future__ import annotations

import json
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any

from .baseline_metrics import (
    _framework_identity,
    _load_model_manifest,
    _model_identity,
    _model_manifest_path,
    _validate_model_identity,
)
from .integrated_workload import DEFAULT_SEED, DEFAULT_TEMPERATURE, DEFAULT_TOP_P
from .preflight import resolve_model_path
from .qwen_vllm_smoke import _shutdown_llm
from .run_artifacts import new_run_dir, runs_root, write_json
from .vllm_plugin_runtime import prepare_candidate_rmsnorm_plugin, summarize_rmsnorm_trace


EVAL_SUITE_SCHEMA = "agentic_opt.qwen35_4b_vllm_eval_suite_baseline.v1"
EVAL_SUITE_BASELINE_ARTIFACT_ENV = "AO_LLM_KERNEL_EVAL_BASELINE_ARTIFACT"
EVAL_SUITE_WRITE_BASELINE_ENV = "AO_LLM_KERNEL_WRITE_EVAL_BASELINE"
EVAL_SUITE_NAME = "qwen35_4b_vllm_prefill_decode_mixed_serving_sweeps_v2"
EVAL_SUITE_FAMILIES = ("prefill", "decode", "mixed")


def default_eval_suite_baseline_artifact_path() -> Path:
    return runs_root() / "baselines" / "qwen35_4b_vllm_eval_suite_baseline.json"


def resolve_eval_suite_baseline_artifact_path() -> Path:
    raw = os.environ.get(EVAL_SUITE_BASELINE_ARTIFACT_ENV)
    return Path(raw).expanduser().resolve() if raw else default_eval_suite_baseline_artifact_path()


def evaluation_suite_workload_signature() -> dict[str, Any]:
    return {
        "name": EVAL_SUITE_NAME,
        "families": [
            {
                "name": family["name"],
                "metric": family["metric"],
                "max_tokens": family["max_tokens"],
                "request_count": len(family["requests"]),
                "sweeps": [
                    {
                        "id": sweep["id"],
                        "concurrency": sweep["concurrency"],
                        "repeat_count": sweep["repeat_count"],
                        "request_count": sweep["request_count"],
                    }
                    for sweep in family["sweeps"]
                ],
                "requests": [
                    {
                        "id": request["id"],
                        "prompt_sha256": request["prompt_sha256"],
                    }
                    for request in family["requests"]
                ],
            }
            for family in evaluation_suite_families()
        ],
        "sampling": {
            "temperature": DEFAULT_TEMPERATURE,
            "top_p": DEFAULT_TOP_P,
            "seed": DEFAULT_SEED,
            "ignore_eos": True,
        },
    }


def evaluation_suite_families() -> list[dict[str, Any]]:
    prefill_context = (
        "Qwen kernel optimization benchmark context. "
        "The request is designed to stress prompt ingestion, normalization, attention setup, "
        "KV-cache writes, and the first few decode steps on an H200 GPU. "
    )
    mixed_context = (
        "A production inference trace contains short user turns, retrieval-augmented context, "
        "and medium assistant responses. This synthetic request keeps deterministic text while "
        "covering the mixed latency path. "
    )
    families = [
        {
            "name": "prefill",
            "metric": "serving_request_latency_p90_s",
            "max_tokens": 8,
            "sweeps": [
                {"id": "prefill_c1", "concurrency": 1, "request_count": 1, "repeat_count": 2},
                {"id": "prefill_c2", "concurrency": 2, "request_count": 2, "repeat_count": 2},
            ],
            "requests": [
                {
                    "id": "prefill_512_a",
                    "prompt": prefill_context * 36 + "Summarize the kernel bottleneck in one sentence.",
                },
                {
                    "id": "prefill_512_b",
                    "prompt": prefill_context * 34 + "Name the most important inference latency phase.",
                },
            ],
        },
        {
            "name": "decode",
            "metric": "serving_request_latency_p90_s",
            "max_tokens": 64,
            "sweeps": [
                {"id": "decode_c1", "concurrency": 1, "request_count": 1, "repeat_count": 2},
                {"id": "decode_c2", "concurrency": 2, "request_count": 2, "repeat_count": 2},
            ],
            "requests": [
                {
                    "id": "decode_64_a",
                    "prompt": "Write a compact checklist for validating an optimized RMSNorm kernel.",
                },
                {
                    "id": "decode_64_b",
                    "prompt": "Explain why decode latency matters for interactive inference.",
                },
            ],
        },
        {
            "name": "mixed",
            "metric": "serving_request_latency_p90_s",
            "max_tokens": 32,
            "sweeps": [
                {"id": "mixed_c1", "concurrency": 1, "request_count": 1, "repeat_count": 2},
                {"id": "mixed_c4", "concurrency": 4, "request_count": 4, "repeat_count": 2},
            ],
            "requests": [
                {
                    "id": "mixed_short",
                    "prompt": "Give one reason to pin an inference baseline.",
                },
                {
                    "id": "mixed_medium",
                    "prompt": mixed_context * 8 + "Return a concise engineering note.",
                },
                {
                    "id": "mixed_long",
                    "prompt": mixed_context * 18 + "Identify the likely bottleneck family.",
                },
                {
                    "id": "mixed_chat",
                    "prompt": "User: The model is correct but slow.\nAssistant:",
                },
            ],
        },
    ]
    for family in families:
        for request in family["requests"]:
            request["prompt_sha256"] = _sha256_text(request["prompt"])
    return families


def run_qwen_vllm_eval_suite(
    *,
    run_dir: Path | None = None,
    candidate_manifest_path: Path | None = None,
    require_candidate_rmsnorm: bool | None = None,
) -> dict[str, Any]:
    if run_dir is None:
        run_dir = new_run_dir("qwen_vllm_eval_suite")
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
            gpu_memory_utilization=0.55,
            enforce_eager=True,
            disable_log_stats=True,
            gdn_prefill_backend="triton",
        )
        warmup_started = time.time()
        llm.generate(
            ["Warm up the deterministic Qwen inference benchmark."],
            SamplingParams(
                max_tokens=4,
                temperature=DEFAULT_TEMPERATURE,
                top_p=DEFAULT_TOP_P,
                seed=DEFAULT_SEED,
                ignore_eos=True,
            ),
        )
        warmup_elapsed = time.time() - warmup_started
        family_results: list[dict[str, Any]] = []
        for family in evaluation_suite_families():
            full_sampling = _sampling_params(SamplingParams, max_tokens=int(family["max_tokens"]))
            ttft_sampling = _sampling_params(SamplingParams, max_tokens=1)
            output_records: list[dict[str, Any]] = []
            sweep_records: list[dict[str, Any]] = []
            input_token_count = 0
            output_token_count = 0
            generate_elapsed_total = 0.0
            measured_elapsed_total = 0.0
            request_latency_samples: list[float] = []
            ttft_samples: list[float] = []
            tpot_samples: list[float] = []
            for sweep in family["sweeps"]:
                sweep_requests = _requests_for_sweep(family, sweep)
                prompts = [request["prompt"] for request in sweep_requests]
                for repeat_index in range(int(sweep["repeat_count"])):
                    ttft_started = time.time()
                    ttft_outputs = llm.generate(prompts, ttft_sampling)
                    ttft_elapsed = time.time() - ttft_started
                    generate_started = time.time()
                    outputs = llm.generate(prompts, full_sampling)
                    generate_elapsed = time.time() - generate_started
                    measured_elapsed_total += ttft_elapsed + generate_elapsed
                    generate_elapsed_total += generate_elapsed
                    repeat_input_tokens = 0
                    repeat_output_tokens = 0
                    repeat_records: list[dict[str, Any]] = []
                    for index, (request, output) in enumerate(zip(sweep_requests, outputs, strict=False)):
                        completion = output.outputs[0] if output.outputs else None
                        prompt_token_ids = list(getattr(output, "prompt_token_ids", None) or [])
                        token_ids = list(completion.token_ids) if completion is not None else []
                        repeat_input_tokens += len(prompt_token_ids)
                        repeat_output_tokens += len(token_ids)
                        output_records.append(
                            {
                                "request_id": f"{sweep['id']}:r{repeat_index}:{index}:{request['id']}",
                                "base_request_id": request["id"],
                                "sweep_id": sweep["id"],
                                "repeat_index": repeat_index,
                                "concurrency": int(sweep["concurrency"]),
                                "prompt_sha256": request["prompt_sha256"],
                                "prompt_token_count": len(prompt_token_ids),
                                "generated_text": completion.text if completion is not None else "",
                                "token_ids": token_ids,
                            }
                        )
                        repeat_records.append(output_records[-1])
                    input_token_count += repeat_input_tokens
                    output_token_count += repeat_output_tokens
                    request_count = max(len(sweep_requests), 1)
                    request_latency = generate_elapsed / request_count
                    ttft_per_request = ttft_elapsed / request_count
                    post_first_token_count = max(repeat_output_tokens - request_count, 1)
                    tpot = max(generate_elapsed - ttft_elapsed, 0.0) / post_first_token_count
                    request_latency_samples.extend([request_latency] * request_count)
                    ttft_samples.extend([ttft_per_request] * request_count)
                    tpot_samples.extend([tpot] * request_count)
                    sweep_records.append(
                        {
                            "id": sweep["id"],
                            "repeat_index": repeat_index,
                            "concurrency": int(sweep["concurrency"]),
                            "request_count": request_count,
                            "ttft_proxy_elapsed_s": ttft_elapsed,
                            "generate_elapsed_s": generate_elapsed,
                            "request_latency_s": request_latency,
                            "ttft_proxy_s": ttft_per_request,
                            "tpot_proxy_s": tpot,
                            "input_token_count": repeat_input_tokens,
                            "output_token_count": repeat_output_tokens,
                            "outputs": [
                                {
                                    "request_id": record["request_id"],
                                    "base_request_id": record["base_request_id"],
                                    "prompt_sha256": record["prompt_sha256"],
                                    "token_ids": record["token_ids"],
                                }
                                for record in repeat_records
                            ],
                            "ttft_output_token_count": sum(
                                len(output.outputs[0].token_ids) if output.outputs else 0 for output in ttft_outputs
                            ),
                        }
                    )
            serving_metrics = _serving_metrics(
                request_latency_samples=request_latency_samples,
                ttft_samples=ttft_samples,
                tpot_samples=tpot_samples,
                measured_elapsed_s=measured_elapsed_total,
                generate_elapsed_s=generate_elapsed_total,
                request_count=len(output_records),
                input_token_count=input_token_count,
                output_token_count=output_token_count,
            )
            family_results.append(
                {
                    "name": family["name"],
                    "metric": family["metric"],
                    "max_tokens": int(family["max_tokens"]),
                    "request_count": len(output_records),
                    "base_request_count": len(family["requests"]),
                    "generate_elapsed_s": generate_elapsed_total,
                    "measured_elapsed_s": measured_elapsed_total,
                    "serving_score_s": serving_metrics["request_latency_s_p90"],
                    "serving_metrics": serving_metrics,
                    "input_token_count": input_token_count,
                    "output_token_count": output_token_count,
                    "total_token_count": input_token_count + output_token_count,
                    "output_tokens_per_s": output_token_count / generate_elapsed_total
                    if generate_elapsed_total > 0
                    else None,
                    "sweeps": sweep_records,
                    "outputs": output_records,
                }
            )
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
            "ok": not (require_candidate_rmsnorm and not candidate_used),
            "model_path": str(model_path),
            "run_dir": str(run_dir),
            "suite": evaluation_suite_workload_signature(),
            "load_and_generate_elapsed_s": time.time() - started,
            "warmup_elapsed_s": warmup_elapsed,
            "total_generate_elapsed_s": sum(
                float(family["generate_elapsed_s"]) for family in family_results
            ),
            "total_measured_elapsed_s": sum(
                float(family["measured_elapsed_s"]) for family in family_results
            ),
            "families": family_results,
            "candidate_rmsnorm_plugin": plugin_info,
            "candidate_rmsnorm_required": bool(require_candidate_rmsnorm),
            "candidate_rmsnorm_used_in_vllm": candidate_used,
            "vllm_rmsnorm_apply": apply_summary,
        }
        if require_candidate_rmsnorm and not candidate_used:
            payload["error"] = "candidate RMSNorm was not invoked by the vLLM apply path"
        if (
            payload["ok"]
            and candidate_manifest_path is None
            and os.environ.get(EVAL_SUITE_WRITE_BASELINE_ENV) in {"1", "true", "TRUE", "yes", "on"}
        ):
            baseline = write_eval_suite_baseline_artifact(payload)
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
            "model_path": str(model_path),
            "run_dir": str(run_dir),
            "elapsed_s": time.time() - started,
            "suite": evaluation_suite_workload_signature(),
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "candidate_rmsnorm_plugin": plugin_info,
            "candidate_rmsnorm_required": bool(require_candidate_rmsnorm),
            "candidate_rmsnorm_used_in_vllm": bool(
                apply_summary is not None and int(apply_summary.get("candidate_calls") or 0) > 0
            ),
            "vllm_rmsnorm_apply": apply_summary,
        }
    write_json(run_dir / "qwen_vllm_eval_suite.json", payload)
    return payload


def build_eval_suite_baseline_artifact(suite_result: dict[str, Any]) -> dict[str, Any]:
    model_manifest = _load_model_manifest()
    return {
        "schema": EVAL_SUITE_SCHEMA,
        "created_at_unix": time.time(),
        "model_manifest_path": _model_manifest_path(),
        "model": _model_identity(model_manifest),
        "framework": _framework_identity(),
        "suite": suite_result.get("suite") or evaluation_suite_workload_signature(),
        "metrics": _suite_metrics(suite_result),
        "families": _baseline_families(suite_result),
    }


def write_eval_suite_baseline_artifact(
    suite_result: dict[str, Any],
    path: Path | None = None,
) -> dict[str, Any]:
    artifact = build_eval_suite_baseline_artifact(suite_result)
    destination = path or resolve_eval_suite_baseline_artifact_path()
    write_json(destination, artifact)
    artifact["_artifact_path"] = str(destination)
    return artifact


def load_eval_suite_baseline_artifact(path: Path | None = None) -> dict[str, Any]:
    artifact_path = path or resolve_eval_suite_baseline_artifact_path()
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"evaluation suite baseline artifact is missing: {artifact_path}. "
            f"Run the Qwen/vLLM evaluation suite baseline and write "
            f"{EVAL_SUITE_BASELINE_ARTIFACT_ENV} or the default artifact first."
        )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["_artifact_path"] = str(artifact_path)
    validation = validate_eval_suite_baseline_artifact(artifact)
    if not validation["valid"]:
        raise ValueError(validation["error"])
    return artifact


def validate_eval_suite_baseline_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    if artifact.get("schema") != EVAL_SUITE_SCHEMA:
        return {"valid": False, "error": f"baseline schema must be {EVAL_SUITE_SCHEMA}"}
    expected_suite = evaluation_suite_workload_signature()
    if artifact.get("suite") != expected_suite:
        return {"valid": False, "error": "baseline suite signature does not match current evaluation suite"}
    families = artifact.get("families")
    if not isinstance(families, list) or {item.get("name") for item in families} != set(EVAL_SUITE_FAMILIES):
        return {"valid": False, "error": "baseline families do not match evaluation suite families"}
    for family in families:
        outputs = family.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            return {"valid": False, "error": f"baseline family {family.get('name')} has no outputs"}
        if _positive_float(family.get("serving_score_s")) is None:
            return {"valid": False, "error": f"baseline family {family.get('name')} has invalid serving score"}
        metrics = family.get("serving_metrics") or {}
        for key in ("request_latency_s_p50", "request_latency_s_p90", "ttft_s_p50", "ttft_s_p90", "tpot_s_p50", "tpot_s_p90"):
            if _positive_float(metrics.get(key)) is None:
                return {"valid": False, "error": f"baseline family {family.get('name')} missing {key}"}
    model_check = _validate_model_identity(artifact.get("model") or {})
    if not model_check["valid"]:
        return model_check
    return {"valid": True, "error": None, "artifact_path": artifact.get("_artifact_path")}


def compare_eval_suite_to_baseline(
    suite_result: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    checks.append(
        _check(
            "suite_signature",
            suite_result.get("suite") == baseline.get("suite"),
            "candidate evaluation suite differs from pinned baseline suite",
        )
    )
    baseline_by_family = {family.get("name"): family for family in baseline.get("families") or []}
    candidate_by_family = {family.get("name"): family for family in suite_result.get("families") or []}
    checks.append(
        _check(
            "family_set",
            set(candidate_by_family) == set(baseline_by_family) == set(EVAL_SUITE_FAMILIES),
            "candidate family set differs from pinned baseline suite",
        )
    )
    token_matches = 0
    token_total = 0
    text_matches = 0
    text_total = 0
    for family_name in EVAL_SUITE_FAMILIES:
        candidate_family = candidate_by_family.get(family_name) or {}
        baseline_family = baseline_by_family.get(family_name) or {}
        candidate_outputs = candidate_family.get("outputs") or []
        baseline_outputs = baseline_family.get("outputs") or []
        checks.append(
            _check(
                f"{family_name}_output_count",
                len(candidate_outputs) == len(baseline_outputs),
                f"{family_name} output count differs from pinned baseline",
            )
        )
        for index, (candidate, expected) in enumerate(zip(candidate_outputs, baseline_outputs, strict=False)):
            request_ok = (
                candidate.get("request_id") == expected.get("request_id")
                and candidate.get("prompt_sha256") == expected.get("prompt_sha256")
            )
            checks.append(
                _check(
                    f"{family_name}_{index}_request_identity",
                    request_ok,
                    f"{family_name} output {index} request identity differs from pinned baseline",
                )
            )
            candidate_tokens = list(candidate.get("token_ids") or [])
            expected_tokens = list(expected.get("token_ids") or [])
            token_total += max(len(candidate_tokens), len(expected_tokens))
            token_matches += sum(
                1 for left, right in zip(candidate_tokens, expected_tokens, strict=False) if left == right
            )
            checks.append(
                _check(
                    f"{family_name}_{index}_token_ids",
                    candidate_tokens == expected_tokens,
                    f"{family_name} output {index} generated token ids differ from pinned baseline",
                )
            )
            text_total += 1
            if candidate.get("generated_text") == expected.get("generated_text"):
                text_matches += 1
            checks.append(
                _check(
                    f"{family_name}_{index}_text",
                    candidate.get("generated_text") == expected.get("generated_text"),
                    f"{family_name} output {index} generated text differs from pinned baseline",
                )
            )
    valid = all(item["status"] == "passed" for item in checks)
    return {
        "valid": valid,
        "error": None if valid else _first_failed_message(checks),
        "baseline_artifact_path": baseline.get("_artifact_path"),
        "checks": checks,
        "token_match_rate": float(token_matches / token_total) if token_total else 0.0,
        "text_match_rate": float(text_matches / text_total) if text_total else 0.0,
    }


def _suite_metrics(suite_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "load_and_generate_elapsed_s": suite_result.get("load_and_generate_elapsed_s"),
        "warmup_elapsed_s": suite_result.get("warmup_elapsed_s"),
        "total_generate_elapsed_s": suite_result.get("total_generate_elapsed_s"),
        "total_measured_elapsed_s": suite_result.get("total_measured_elapsed_s"),
        "families": {
            family.get("name"): {
                "generate_elapsed_s": family.get("generate_elapsed_s"),
                "measured_elapsed_s": family.get("measured_elapsed_s"),
                "serving_score_s": family.get("serving_score_s"),
                "serving_metrics": family.get("serving_metrics") or {},
                "input_token_count": family.get("input_token_count"),
                "output_token_count": family.get("output_token_count"),
                "total_token_count": family.get("total_token_count"),
                "output_tokens_per_s": family.get("output_tokens_per_s"),
                "sweep_count": len(family.get("sweeps") or []),
            }
            for family in suite_result.get("families") or []
        },
    }


def _baseline_families(suite_result: dict[str, Any]) -> list[dict[str, Any]]:
    families: list[dict[str, Any]] = []
    for family in suite_result.get("families") or []:
        families.append(
            {
                "name": family.get("name"),
                "metric": family.get("metric"),
                "max_tokens": family.get("max_tokens"),
                "request_count": family.get("request_count"),
                "generate_elapsed_s": family.get("generate_elapsed_s"),
                "measured_elapsed_s": family.get("measured_elapsed_s"),
                "serving_score_s": family.get("serving_score_s"),
                "serving_metrics": family.get("serving_metrics") or {},
                "input_token_count": family.get("input_token_count"),
                "output_token_count": family.get("output_token_count"),
                "total_token_count": family.get("total_token_count"),
                "sweeps": [
                    {
                        "id": sweep.get("id"),
                        "repeat_index": sweep.get("repeat_index"),
                        "concurrency": sweep.get("concurrency"),
                        "request_count": sweep.get("request_count"),
                        "generate_elapsed_s": sweep.get("generate_elapsed_s"),
                        "request_latency_s": sweep.get("request_latency_s"),
                        "ttft_proxy_s": sweep.get("ttft_proxy_s"),
                        "tpot_proxy_s": sweep.get("tpot_proxy_s"),
                    }
                    for sweep in family.get("sweeps") or []
                ],
                "outputs": [
                    {
                        "request_id": output.get("request_id"),
                        "base_request_id": output.get("base_request_id"),
                        "sweep_id": output.get("sweep_id"),
                        "repeat_index": output.get("repeat_index"),
                        "concurrency": output.get("concurrency"),
                        "prompt_sha256": output.get("prompt_sha256"),
                        "prompt_token_count": output.get("prompt_token_count"),
                        "generated_text": output.get("generated_text") or "",
                        "token_ids": list(output.get("token_ids") or []),
                    }
                    for output in family.get("outputs") or []
                ],
            }
        )
    return families


def _check(name: str, passed: bool, message: str) -> dict[str, Any]:
    return {"name": name, "status": "passed" if passed else "failed", "message": None if passed else message}


def _first_failed_message(checks: list[dict[str, Any]]) -> str | None:
    for check in checks:
        if check.get("status") == "failed":
            return str(check.get("message") or check.get("name"))
    return None


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _sampling_params(sampling_cls: Any, *, max_tokens: int) -> Any:
    return sampling_cls(
        max_tokens=max_tokens,
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
        seed=DEFAULT_SEED,
        ignore_eos=True,
    )


def _requests_for_sweep(family: dict[str, Any], sweep: dict[str, Any]) -> list[dict[str, Any]]:
    base_requests = list(family.get("requests") or [])
    request_count = int(sweep.get("request_count") or sweep.get("concurrency") or len(base_requests))
    if not base_requests:
        return []
    return [base_requests[index % len(base_requests)] for index in range(request_count)]


def _serving_metrics(
    *,
    request_latency_samples: list[float],
    ttft_samples: list[float],
    tpot_samples: list[float],
    measured_elapsed_s: float,
    generate_elapsed_s: float,
    request_count: int,
    input_token_count: int,
    output_token_count: int,
) -> dict[str, Any]:
    return {
        "request_latency_s_p50": _percentile(request_latency_samples, 50),
        "request_latency_s_p90": _percentile(request_latency_samples, 90),
        "request_latency_s_max": max(request_latency_samples) if request_latency_samples else None,
        "ttft_s_p50": _percentile(ttft_samples, 50),
        "ttft_s_p90": _percentile(ttft_samples, 90),
        "tpot_s_p50": _percentile(tpot_samples, 50),
        "tpot_s_p90": _percentile(tpot_samples, 90),
        "request_count": request_count,
        "input_token_count": input_token_count,
        "output_token_count": output_token_count,
        "measured_elapsed_s": measured_elapsed_s,
        "generate_elapsed_s": generate_elapsed_s,
        "requests_per_s": request_count / generate_elapsed_s if generate_elapsed_s > 0 else None,
        "output_tokens_per_s": output_token_count / generate_elapsed_s if generate_elapsed_s > 0 else None,
    }


def _percentile(values: list[float], percentile: int) -> float | None:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    if len(finite) == 1:
        return finite[0]
    rank = (len(finite) - 1) * (percentile / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return finite[int(rank)]
    weight = rank - lower
    return finite[lower] * (1.0 - weight) + finite[upper] * weight


def _sha256_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> int:
    candidate_raw = os.environ.get("AO_LLM_KERNEL_EVAL_CANDIDATE_MANIFEST")
    require_raw = os.environ.get("AO_LLM_KERNEL_EVAL_REQUIRE_CANDIDATE")
    result = run_qwen_vllm_eval_suite(
        candidate_manifest_path=Path(candidate_raw) if candidate_raw else None,
        require_candidate_rmsnorm=require_raw in {"1", "true", "TRUE", "yes", "on"}
        if require_raw is not None
        else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
