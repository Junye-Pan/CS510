from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

from .baseline_metrics import write_baseline_artifact
from .integrated_workload import (
    DEFAULT_LOGPROBS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPTS,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    workload_signature,
)
from .preflight import resolve_model_path
from .run_artifacts import new_run_dir, runs_root, write_json
from .vllm_plugin_runtime import (
    prepare_candidate_rmsnorm_plugin,
    summarize_rmsnorm_trace,
)


SMOKE_CANDIDATE_MANIFEST_ENV = "AO_LLM_KERNEL_SMOKE_CANDIDATE_MANIFEST"
SMOKE_REQUIRE_CANDIDATE_ENV = "AO_LLM_KERNEL_SMOKE_REQUIRE_CANDIDATE"
SMOKE_WRITE_BASELINE_ENV = "AO_LLM_KERNEL_WRITE_BASELINE"


def run_qwen_vllm_smoke(
    *,
    run_dir: Path | None = None,
    candidate_manifest_path: Path | None = None,
    require_candidate_rmsnorm: bool | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    prompts: tuple[str, ...] = DEFAULT_PROMPTS,
) -> dict[str, Any]:
    if run_dir is None:
        run_dir = new_run_dir("qwen_vllm_smoke")
    started = time.time()
    model_path = resolve_model_path()
    os.environ.setdefault("HF_HOME", str(runs_root() / "huggingface"))
    os.environ.setdefault("HF_XET_CACHE", str(runs_root() / "huggingface" / "xet"))
    os.environ.setdefault("VLLM_CACHE_ROOT", str(runs_root() / "vllm_cache"))
    # The current H200 environment has vLLM but not optional DeepGEMM/nvcc.
    # Disable those warmup paths so this smoke tests model load/generate rather
    # than optional kernel build tooling.
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
        sampling = SamplingParams(
            max_tokens=max_tokens,
            temperature=DEFAULT_TEMPERATURE,
            top_p=DEFAULT_TOP_P,
            seed=DEFAULT_SEED,
            logprobs=DEFAULT_LOGPROBS,
        )
        generate_started = time.time()
        outputs = llm.generate(list(prompts), sampling)
        generate_elapsed = time.time() - generate_started
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
            "load_and_generate_elapsed_s": time.time() - started,
            "generate_elapsed_s": generate_elapsed,
            "max_tokens": max_tokens,
            "workload": workload_signature(prompts=prompts, max_tokens=max_tokens),
            "candidate_rmsnorm_plugin": plugin_info,
            "candidate_rmsnorm_required": bool(require_candidate_rmsnorm),
            "candidate_rmsnorm_used_in_vllm": candidate_used,
            "vllm_rmsnorm_apply": apply_summary,
            "outputs": [
                {
                    "prompt": out.prompt,
                    "generated_text": out.outputs[0].text if out.outputs else "",
                    "token_ids": list(out.outputs[0].token_ids) if out.outputs else [],
                    "top_logprob_token_ids": _top_logprob_token_ids(out.outputs[0].logprobs)
                    if out.outputs
                    else [],
                }
                for out in outputs
            ],
        }
        if require_candidate_rmsnorm and not candidate_used:
            payload["error"] = "candidate RMSNorm was not invoked by the vLLM apply path"
        if (
            payload["ok"]
            and candidate_manifest_path is None
            and os.environ.get(SMOKE_WRITE_BASELINE_ENV) in {"1", "true", "TRUE", "yes", "on"}
        ):
            baseline = write_baseline_artifact(payload)
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
            "workload": workload_signature(prompts=prompts, max_tokens=max_tokens),
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "candidate_rmsnorm_plugin": plugin_info,
            "candidate_rmsnorm_required": bool(require_candidate_rmsnorm),
            "candidate_rmsnorm_used_in_vllm": bool(
                apply_summary is not None and int(apply_summary.get("candidate_calls") or 0) > 0
            ),
            "vllm_rmsnorm_apply": apply_summary,
        }
    write_json(run_dir / "qwen_vllm_smoke.json", payload)
    return payload


def _top_logprob_token_ids(logprobs: Any) -> list[list[int]]:
    steps: list[list[int]] = []
    if not isinstance(logprobs, list):
        return steps
    for step in logprobs:
        if not isinstance(step, dict):
            steps.append([])
            continue
        token_ids: list[int] = []
        for key in step.keys():
            try:
                token_ids.append(int(key))
            except (TypeError, ValueError):
                continue
        steps.append(token_ids)
    return steps


def _shutdown_llm(llm: Any) -> None:
    shutdown = getattr(llm, "shutdown", None)
    if callable(shutdown):
        shutdown()


def main() -> int:
    candidate_raw = os.environ.get(SMOKE_CANDIDATE_MANIFEST_ENV)
    require_raw = os.environ.get(SMOKE_REQUIRE_CANDIDATE_ENV)
    result = run_qwen_vllm_smoke(
        candidate_manifest_path=Path(candidate_raw) if candidate_raw else None,
        require_candidate_rmsnorm=require_raw in {"1", "true", "TRUE", "yes", "on"}
        if require_raw is not None
        else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
