from __future__ import annotations

import time
import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_opt.common.runtime_env import TaskRuntimeSpec
from agentic_opt.task_api import CandidateSpec, TaskMetadata

from .private.baseline_metrics import compare_smoke_to_baseline, load_baseline_artifact
from .private.fallback_policy import evaluate_apply_trace_policy, fallback_thresholds
from .private.preflight import live_verifier_enabled, model_preflight_required, probe_model_smoke_enabled
from .private.probe_feedback import build_probe_diagnostics
from .private.qwen_logits_distribution import (
    compare_logits_distribution_to_baseline,
    load_logits_distribution_baseline_artifact,
    run_qwen_logits_distribution_probe,
    selection_records_from_baseline,
)
from .private.qwen_vllm_eval_suite import (
    compare_eval_suite_to_baseline,
    load_eval_suite_baseline_artifact,
    run_qwen_vllm_eval_suite,
)
from .private.qwen_vllm_smoke import run_qwen_vllm_smoke
from .private.rmsnorm_live import run_rmsnorm_live_checks
from .private.run_artifacts import default_model_manifest_path, new_run_dir, write_json
from .private.scoring import (
    BASELINE_SCORE,
    end_to_end_suite_score,
    geomean_speedup,
)
from .private.schema import load_manifest
from .private.verifier import EXPECTED_MANIFEST_NAME, StaticVerifier
from .private.workloads import HIDDEN_RMSNORM_SHAPES, HIDDEN_WORKLOAD_FAMILIES, PROBE_WORKLOAD_SHAPES, PUBLIC_WORKLOAD_SHAPES


def _runtime_python_candidates_from_manifest() -> tuple[str, ...]:
    manifest_path = Path(os.environ["AO_LLM_KERNEL_ENV_MANIFEST"]) if os.environ.get("AO_LLM_KERNEL_ENV_MANIFEST") else default_model_manifest_path()
    if not manifest_path.exists():
        return ()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return ()
    runtime = manifest.get("runtime") if isinstance(manifest.get("runtime"), dict) else {}
    python_executable = runtime.get("python_executable")
    if isinstance(python_executable, str) and python_executable:
        return (python_executable,)
    return ()


LLM_KERNEL_RUNTIME = TaskRuntimeSpec(
    python=">=3.11",
    python_candidates=_runtime_python_candidates_from_manifest(),
    requirements=(),
    required_imports=("torch", "triton", "vllm"),
    forbidden_shadow_modules=(),
    system_site_packages=True,
    verify_public_seed=True,
)


@dataclass(frozen=True)
class LLMInferenceQwen35B4H200Task:
    metadata: TaskMetadata = TaskMetadata(
        task_id="llm_inference_qwen35_4b_h200",
        title="Qwen 3.5 4B H200 LLM Kernel Optimization",
        candidate_spec=CandidateSpec(
            candidate_root="candidate",
            public_seed_root="initial_candidate",
            entrypoint_name=EXPECTED_MANIFEST_NAME,
            description=(
                "Directory candidate. Submit candidate/manifest.json plus any declared "
                "kernel source files under candidate/."
            ),
        ),
    )
    runtime_spec: TaskRuntimeSpec = LLM_KERNEL_RUNTIME

    @property
    def public_dir(self) -> Path:
        return Path(__file__).resolve().parent / "public"

    def verify_entry(self, entry_path: Path) -> dict[str, Any]:
        started = time.perf_counter()
        verifier = StaticVerifier()
        result = verifier.verify(entry_path)
        result["elapsed_s"] = time.perf_counter() - started
        return result

    def probe_entry(self, entry_path: Path, *, kind: str) -> dict[str, Any]:
        if kind != "diagnostics":
            raise ValueError(f"Unsupported probe kind for llm_inference_qwen35_4b_h200: {kind}")
        started = time.perf_counter()
        verifier = StaticVerifier()
        result = verifier.verify(entry_path)
        bundle = result.get("bundle") or {}
        implementation_count = int(bundle.get("implementation_count") or 0)
        diagnostics = build_probe_diagnostics(
            static_result=result,
            public_workload_shapes=PUBLIC_WORKLOAD_SHAPES,
        )
        diagnostics.update(
            {
                "kind": kind,
                "live_verifier_enabled": live_verifier_enabled(),
                "official_live_enabled": False,
            }
        )
        if result.get("valid") and implementation_count > 0 and live_verifier_enabled():
            manifest = load_manifest(entry_path)
            run_dir = new_run_dir("llm_kernel_probe")
            live_probe = run_rmsnorm_live_checks(
                entry_path=entry_path,
                manifest=manifest,
                shapes=PROBE_WORKLOAD_SHAPES,
                benchmark=True,
                run_dir=run_dir,
            )
            qwen_probe = None
            integrated_correctness = None
            apply_trace_policy = None
            baseline_artifact_path = None
            if model_preflight_required() and probe_model_smoke_enabled():
                try:
                    baseline_artifact = load_baseline_artifact()
                    baseline_artifact_path = baseline_artifact.get("_artifact_path")
                    qwen_probe = run_qwen_vllm_smoke(
                        run_dir=run_dir,
                        candidate_manifest_path=entry_path,
                        require_candidate_rmsnorm=True,
                    )
                    if qwen_probe.get("ok"):
                        integrated_correctness = compare_smoke_to_baseline(qwen_probe, baseline_artifact)
                        apply_trace_policy = evaluate_apply_trace_policy(
                            qwen_probe.get("vllm_rmsnorm_apply"),
                            candidate_required=True,
                        )
                except Exception as exc:
                    qwen_probe = {"ok": False, "error": str(exc)}
            diagnostics = build_probe_diagnostics(
                static_result=result,
                public_workload_shapes=PUBLIC_WORKLOAD_SHAPES,
                live_result=live_probe,
                qwen_smoke=qwen_probe,
            )
            diagnostics.update(
                {
                    "kind": kind,
                    "live_verifier_enabled": live_verifier_enabled(),
                    "official_live_enabled": True,
                    "integrated_probe_enabled": model_preflight_required() and probe_model_smoke_enabled(),
                    "live_rmsnorm": live_probe,
                    "integrated_correctness": integrated_correctness,
                    "apply_trace_policy": apply_trace_policy,
                    "baseline_artifact_path": baseline_artifact_path,
                }
            )
            diagnostics["run_dir"] = str(run_dir)
            write_json(run_dir / "probe_summary.json", diagnostics)
        return {
            "ok": True,
            "valid": bool(result.get("valid")),
            "score": BASELINE_SCORE if result.get("valid") and implementation_count == 0 else None,
            "kind": kind,
            "feedback": result.get("feedback") or {},
            "diagnostics": diagnostics,
            "elapsed_s": time.perf_counter() - started,
        }

    def evaluate_entry(self, entry_path: Path) -> dict[str, Any]:
        started = time.perf_counter()
        verifier = StaticVerifier()
        verification = verifier.verify(entry_path)
        if not verification.get("valid"):
            return {
                "score": 0.0,
                "valid": False,
                "correct": {"correct": False, "error": (verification.get("feedback") or {}).get("error")},
                "metrics": {"combined_score": 0.0, "official_live_enabled": False},
                "evaluator": {"score": 0.0, "public_details": verification.get("feedback") or {}},
                "verifier": verification,
                "elapsed_s": time.perf_counter() - started,
            }

        bundle = verification.get("bundle") or {}
        implementation_count = int(bundle.get("implementation_count") or 0)
        if implementation_count > 0 and not live_verifier_enabled():
            message = (
                "Candidate declares optimized implementations, but the H200 live verifier is not enabled. "
                "Static validation passed; official speed scoring requires AO_LLM_KERNEL_ENABLE_LIVE=1."
            )
            return {
                "score": 0.0,
                "valid": False,
                "correct": {"correct": False, "error": message},
                "metrics": {
                    "combined_score": 0.0,
                    "official_live_enabled": False,
                    "implementation_count": implementation_count,
                    "fallback_rate": 1.0,
                },
                "evaluator": {
                    "score": 0.0,
                    "public_details": {
                        "error": message,
                        "static_valid": True,
                        "implementation_count": implementation_count,
                    },
                },
                "verifier": verification,
                "elapsed_s": time.perf_counter() - started,
            }

        if implementation_count > 0 and live_verifier_enabled():
            manifest = load_manifest(entry_path)
            run_dir = new_run_dir("llm_kernel_evaluate")
            live_result = run_rmsnorm_live_checks(
                entry_path=entry_path,
                manifest=manifest,
                shapes=HIDDEN_RMSNORM_SHAPES,
                benchmark=True,
                run_dir=run_dir,
            )
            if not live_result.get("valid"):
                message = str(live_result.get("error") or "live RMSNorm evaluation failed")
                return {
                    "score": 0.0,
                    "valid": False,
                    "correct": {"correct": False, "error": message},
                    "metrics": {
                        "combined_score": 0.0,
                        "official_live_enabled": True,
                        "implementation_count": implementation_count,
                        "live_rmsnorm": live_result,
                        "run_dir": str(run_dir),
                    },
                    "evaluator": {
                        "score": 0.0,
                        "public_details": {"error": message, "static_valid": True},
                    },
                    "verifier": verification,
                    "elapsed_s": time.perf_counter() - started,
                }
            rmsnorm_microbenchmark_speedup = float(live_result.get("geomean_speedup") or 0.0)
            logits_distribution_probe = None
            logits_distribution_correctness = None
            logits_baseline_artifact = None
            qwen_eval_suite = None
            integrated_correctness = None
            apply_trace_policy = None
            baseline_artifact = None
            if not model_preflight_required():
                message = (
                    "End-to-end scoring requires the Qwen/vLLM model preflight. "
                    "Set AO_LLM_KERNEL_REQUIRE_MODEL=1 for official submit."
                )
                return {
                    "score": 0.0,
                    "valid": False,
                    "correct": {"correct": False, "error": message},
                    "metrics": {
                        "combined_score": 0.0,
                        "official_live_enabled": True,
                        "model_smoke_required": False,
                        "implementation_count": implementation_count,
                        "live_rmsnorm": live_result,
                        "live_rmsnorm_geomean_speedup": rmsnorm_microbenchmark_speedup,
                        "run_dir": str(run_dir),
                    },
                    "evaluator": {
                        "score": 0.0,
                        "public_details": {
                            "error": message,
                            "run_dir": str(run_dir),
                            "live_rmsnorm_geomean_speedup": rmsnorm_microbenchmark_speedup,
                        },
                    },
                    "verifier": verification,
                    "elapsed_s": time.perf_counter() - started,
                }
            try:
                logits_baseline_artifact = load_logits_distribution_baseline_artifact()
                baseline_artifact = load_eval_suite_baseline_artifact()
            except Exception as exc:
                message = f"pinned Qwen/vLLM baseline artifact is not available or invalid: {exc}"
                return {
                    "score": 0.0,
                    "valid": False,
                    "correct": {"correct": False, "error": message},
                    "metrics": {
                        "combined_score": 0.0,
                        "official_live_enabled": True,
                        "model_smoke_required": True,
                        "implementation_count": implementation_count,
                        "live_rmsnorm": live_result,
                        "live_rmsnorm_geomean_speedup": rmsnorm_microbenchmark_speedup,
                        "baseline_error": message,
                        "run_dir": str(run_dir),
                    },
                    "evaluator": {
                        "score": 0.0,
                        "public_details": {
                            "error": message,
                            "run_dir": str(run_dir),
                            "logits_distribution_ok": False,
                            "candidate_rmsnorm_used_in_vllm": None,
                        },
                    },
                    "verifier": verification,
                    "elapsed_s": time.perf_counter() - started,
                }
            logits_distribution_probe = run_qwen_logits_distribution_probe(
                run_dir=run_dir,
                candidate_manifest_path=entry_path,
                require_candidate_rmsnorm=True,
                selection_records=selection_records_from_baseline(logits_baseline_artifact),
            )
            if not logits_distribution_probe.get("ok"):
                message = str(logits_distribution_probe.get("error") or "Qwen/vLLM logits distribution probe failed")
                return {
                    "score": 0.0,
                    "valid": False,
                    "correct": {"correct": False, "error": message},
                    "metrics": {
                        "combined_score": 0.0,
                        "official_live_enabled": True,
                        "model_smoke_required": True,
                        "implementation_count": implementation_count,
                        "live_rmsnorm": live_result,
                        "live_rmsnorm_geomean_speedup": rmsnorm_microbenchmark_speedup,
                        "logits_distribution_probe": logits_distribution_probe,
                        "logits_distribution_baseline_artifact_path": logits_baseline_artifact.get("_artifact_path"),
                        "run_dir": str(run_dir),
                    },
                    "evaluator": {
                        "score": 0.0,
                        "public_details": {
                            "error": message,
                            "run_dir": str(run_dir),
                            "logits_distribution_ok": False,
                            "candidate_rmsnorm_used_in_vllm": bool(
                                logits_distribution_probe.get("candidate_rmsnorm_used_in_vllm")
                            ),
                        },
                    },
                    "verifier": verification,
                    "elapsed_s": time.perf_counter() - started,
                }
            logits_distribution_correctness = compare_logits_distribution_to_baseline(
                logits_distribution_probe,
                logits_baseline_artifact,
            )
            if not logits_distribution_correctness.get("valid"):
                message = str(
                    logits_distribution_correctness.get("error")
                    or "Qwen/vLLM logits distribution drift exceeded tolerance"
                )
                return {
                    "score": 0.0,
                    "valid": False,
                    "correct": {"correct": False, "error": message},
                    "metrics": {
                        "combined_score": 0.0,
                        "official_live_enabled": True,
                        "model_smoke_required": True,
                        "implementation_count": implementation_count,
                        "live_rmsnorm": live_result,
                        "live_rmsnorm_geomean_speedup": rmsnorm_microbenchmark_speedup,
                        "logits_distribution_probe": logits_distribution_probe,
                        "logits_distribution_correctness": logits_distribution_correctness,
                        "logits_distribution_baseline_artifact_path": logits_baseline_artifact.get("_artifact_path"),
                        "run_dir": str(run_dir),
                    },
                    "evaluator": {
                        "score": 0.0,
                        "public_details": {
                            "error": message,
                            "run_dir": str(run_dir),
                            "logits_distribution_ok": False,
                            "logits_distribution_aggregate": logits_distribution_correctness.get("aggregate"),
                            "candidate_rmsnorm_used_in_vllm": bool(
                                logits_distribution_probe.get("candidate_rmsnorm_used_in_vllm")
                            ),
                        },
                    },
                    "verifier": verification,
                    "elapsed_s": time.perf_counter() - started,
                }
            qwen_eval_suite = run_qwen_vllm_eval_suite(
                run_dir=run_dir,
                candidate_manifest_path=entry_path,
                require_candidate_rmsnorm=True,
            )
            if not qwen_eval_suite.get("ok"):
                message = str(qwen_eval_suite.get("error") or "Qwen/vLLM evaluation suite failed")
                return {
                    "score": 0.0,
                    "valid": False,
                    "correct": {"correct": False, "error": message},
                    "metrics": {
                        "combined_score": 0.0,
                        "official_live_enabled": True,
                        "model_smoke_required": True,
                        "implementation_count": implementation_count,
                        "live_rmsnorm": live_result,
                        "live_rmsnorm_geomean_speedup": rmsnorm_microbenchmark_speedup,
                        "logits_distribution_probe": logits_distribution_probe,
                        "logits_distribution_correctness": logits_distribution_correctness,
                        "logits_distribution_baseline_artifact_path": logits_baseline_artifact.get("_artifact_path"),
                        "qwen_vllm_eval_suite": qwen_eval_suite,
                        "run_dir": str(run_dir),
                    },
                    "evaluator": {
                        "score": 0.0,
                        "public_details": {"error": message, "run_dir": str(run_dir)},
                    },
                    "verifier": verification,
                    "elapsed_s": time.perf_counter() - started,
                }
            integrated_correctness = compare_eval_suite_to_baseline(qwen_eval_suite, baseline_artifact)
            apply_trace_policy = evaluate_apply_trace_policy(
                qwen_eval_suite.get("vllm_rmsnorm_apply"),
                candidate_required=True,
            )
            if not integrated_correctness.get("valid") or not apply_trace_policy.get("valid"):
                message = str(
                    integrated_correctness.get("error")
                    or apply_trace_policy.get("error")
                    or "integrated Qwen/vLLM correctness or fallback policy failed"
                )
                return {
                    "score": 0.0,
                    "valid": False,
                    "correct": {"correct": False, "error": message},
                    "metrics": {
                        "combined_score": 0.0,
                        "official_live_enabled": True,
                        "model_smoke_required": True,
                        "implementation_count": implementation_count,
                        "live_rmsnorm": live_result,
                        "live_rmsnorm_geomean_speedup": rmsnorm_microbenchmark_speedup,
                        "logits_distribution_probe": logits_distribution_probe,
                        "logits_distribution_correctness": logits_distribution_correctness,
                        "logits_distribution_baseline_artifact_path": logits_baseline_artifact.get("_artifact_path"),
                        "qwen_vllm_eval_suite": qwen_eval_suite,
                        "integrated_correctness": integrated_correctness,
                        "apply_trace_policy": apply_trace_policy,
                        "baseline_artifact_path": baseline_artifact.get("_artifact_path"),
                        "run_dir": str(run_dir),
                    },
                    "evaluator": {
                        "score": 0.0,
                        "public_details": {"error": message, "run_dir": str(run_dir)},
                    },
                    "verifier": verification,
                    "elapsed_s": time.perf_counter() - started,
                }
            end_to_end_score = end_to_end_suite_score(
                candidate_suite=qwen_eval_suite,
                baseline_artifact=baseline_artifact,
            )
            if not end_to_end_score.get("valid"):
                message = str(end_to_end_score.get("error") or "end-to-end score could not be computed")
                return {
                    "score": 0.0,
                    "valid": False,
                    "correct": {"correct": False, "error": message},
                    "metrics": {
                        "combined_score": 0.0,
                        "official_live_enabled": True,
                        "model_smoke_required": True,
                        "implementation_count": implementation_count,
                        "live_rmsnorm": live_result,
                        "live_rmsnorm_geomean_speedup": rmsnorm_microbenchmark_speedup,
                        "logits_distribution_probe": logits_distribution_probe,
                        "logits_distribution_correctness": logits_distribution_correctness,
                        "logits_distribution_baseline_artifact_path": logits_baseline_artifact.get("_artifact_path"),
                        "qwen_vllm_eval_suite": qwen_eval_suite,
                        "integrated_correctness": integrated_correctness,
                        "apply_trace_policy": apply_trace_policy,
                        "end_to_end_score": end_to_end_score,
                        "baseline_artifact_path": baseline_artifact.get("_artifact_path"),
                        "run_dir": str(run_dir),
                    },
                    "evaluator": {
                        "score": 0.0,
                        "public_details": {"error": message, "run_dir": str(run_dir)},
                    },
                    "verifier": verification,
                    "elapsed_s": time.perf_counter() - started,
                }
            score = float(end_to_end_score["score"])
            metrics = {
                "combined_score": score,
                "end_to_end_score": end_to_end_score,
                "official_live_enabled": True,
                "model_smoke_required": model_preflight_required(),
                "implementation_count": implementation_count,
                "fallback_rate": 0.0 if apply_trace_policy is None else apply_trace_policy.get("fallback_rate"),
                "fallback_policy_thresholds": fallback_thresholds(),
                "live_rmsnorm": live_result,
                "live_rmsnorm_geomean_speedup": rmsnorm_microbenchmark_speedup,
                "logits_distribution_probe": logits_distribution_probe,
                "logits_distribution_correctness": logits_distribution_correctness,
                "qwen_vllm_eval_suite": qwen_eval_suite,
                "integrated_correctness": integrated_correctness,
                "apply_trace_policy": apply_trace_policy,
                "logits_distribution_baseline_artifact_path": None
                if logits_baseline_artifact is None
                else logits_baseline_artifact.get("_artifact_path"),
                "baseline_artifact_path": None if baseline_artifact is None else baseline_artifact.get("_artifact_path"),
                "run_dir": str(run_dir),
            }
            write_json(run_dir / "evaluate_summary.json", metrics)
            return {
                "score": score,
                "valid": True,
                "correct": {"correct": True, "error": None},
                "metrics": metrics,
                "evaluator": {
                    "score": score,
                    "public_details": {
                        "official_live_enabled": True,
                        "definition": live_result.get("definition"),
                        "score_component": end_to_end_score.get("score_component"),
                        "scoring_steps": end_to_end_score.get("scoring_steps"),
                        "end_to_end_suite_speedup": score,
                        "family_speedups": end_to_end_score.get("family_speedups"),
                        "baseline_total_generate_elapsed_s": end_to_end_score.get("baseline_total_generate_elapsed_s"),
                        "candidate_total_generate_elapsed_s": end_to_end_score.get("candidate_total_generate_elapsed_s"),
                        "live_rmsnorm_geomean_speedup": rmsnorm_microbenchmark_speedup,
                        "model_smoke_required": model_preflight_required(),
                        "logits_distribution_ok": None
                        if logits_distribution_correctness is None
                        else bool(logits_distribution_correctness.get("valid")),
                        "logits_distribution_aggregate": None
                        if logits_distribution_correctness is None
                        else logits_distribution_correctness.get("aggregate"),
                        "qwen_vllm_eval_suite_ok": None
                        if qwen_eval_suite is None
                        else bool(qwen_eval_suite.get("ok")),
                        "integrated_correctness_ok": None
                        if integrated_correctness is None
                        else bool(integrated_correctness.get("valid")),
                        "apply_trace_policy_ok": None
                        if apply_trace_policy is None
                        else bool(apply_trace_policy.get("valid")),
                        "candidate_rmsnorm_used_in_vllm": None
                        if qwen_eval_suite is None
                        else bool(qwen_eval_suite.get("candidate_rmsnorm_used_in_vllm")),
                        "fallback_rate": None
                        if apply_trace_policy is None
                        else apply_trace_policy.get("fallback_rate"),
                        "logits_distribution_baseline_artifact_path": None
                        if logits_baseline_artifact is None
                        else logits_baseline_artifact.get("_artifact_path"),
                        "baseline_artifact_path": None
                        if baseline_artifact is None
                        else baseline_artifact.get("_artifact_path"),
                        "run_dir": str(run_dir),
                    },
                },
                "verifier": verification,
                "elapsed_s": time.perf_counter() - started,
            }

        family_speedups = {family: BASELINE_SCORE for family in HIDDEN_WORKLOAD_FAMILIES}
        score = geomean_speedup(family_speedups.values())
        return {
            "score": score,
            "valid": True,
            "correct": {"correct": True, "error": None},
            "metrics": {
                "combined_score": score,
                "official_live_enabled": False,
                "implementation_count": implementation_count,
                "fallback_rate": 1.0,
                "family_speedups": family_speedups,
            },
            "evaluator": {
                "score": score,
                "public_details": {
                    "baseline_only": implementation_count == 0,
                    "official_live_enabled": False,
                    "message": "Baseline fallback candidate evaluated at the fixed baseline score.",
                },
            },
            "verifier": verification,
            "elapsed_s": time.perf_counter() - started,
        }


def create_task() -> LLMInferenceQwen35B4H200Task:
    return LLMInferenceQwen35B4H200Task()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Qwen 3.5 4B H200 kernel task directly.")
    parser.add_argument("mode", choices=("verify", "probe", "evaluate"))
    parser.add_argument("--entry", required=True, type=Path)
    parser.add_argument("--probe-kind", default="diagnostics")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    task = create_task()
    if args.mode == "verify":
        result = task.verify_entry(args.entry)
    elif args.mode == "probe":
        result = task.probe_entry(args.entry, kind=args.probe_kind)
    else:
        result = task.evaluate_entry(args.entry)
    if args.summary:
        result = _summary_result(result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("valid", True) else 1


def _summary_result(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics") or {}
    public_details = (result.get("evaluator") or {}).get("public_details") or {}
    end_to_end = metrics.get("end_to_end_score") or {}
    return {
        "valid": result.get("valid"),
        "score": result.get("score"),
        "correct": result.get("correct"),
        "score_component": public_details.get("score_component") or end_to_end.get("score_component"),
        "family_speedups": public_details.get("family_speedups") or end_to_end.get("family_speedups"),
        "logits_distribution_ok": public_details.get("logits_distribution_ok"),
        "logits_distribution_aggregate": public_details.get("logits_distribution_aggregate"),
        "candidate_rmsnorm_used_in_vllm": public_details.get("candidate_rmsnorm_used_in_vllm"),
        "fallback_rate": public_details.get("fallback_rate") or metrics.get("fallback_rate"),
        "run_dir": public_details.get("run_dir") or metrics.get("run_dir"),
        "error": (result.get("correct") or {}).get("error") or public_details.get("error"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
