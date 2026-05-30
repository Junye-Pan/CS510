from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .harness import run_decode_suite
from .verifier_primitives import geometric_mean


def run_official_evaluation(entry_path: Path, *, verifier: dict[str, Any] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    profile = os.environ.get("AO_QWEN3_1_7B_DECODE_ATTENTION_EVAL_PROFILE", "standard").strip().lower() or "standard"
    paths = _make_paths(profile)
    try:
        suite = run_decode_suite(entry_path, profile=profile, include_timing=True)
        correct = bool(suite.get("valid"))
        speedups = [
            float(result["metrics"]["speedup"])
            for result in suite.get("results", [])
            if result.get("valid") and "speedup" in result.get("metrics", {})
        ]
        score = geometric_mean(speedups) if correct and speedups else 0.0
        failures = [
            {
                "name": result.get("name"),
                "error": result.get("error"),
                "traceback_tail": result.get("traceback_tail"),
            }
            for result in suite.get("results", [])
            if not result.get("valid")
        ]
        result = {
            "score": score,
            "valid": correct,
            "correct": {
                "correct": correct,
                "error": None if correct else "; ".join(str(item.get("error")) for item in failures[:6]),
            },
            "metrics": {
                "combined_score": score,
                "decode_attention_speedup": score,
                "workload_count": len(suite.get("results", [])),
                "profile": profile,
            },
            "evaluator": {
                "public_details": {
                    "summary": "official decode attention evaluator passed"
                    if correct
                    else "official decode attention evaluator failed",
                    "score": score,
                    "decode_attention_speedup": score,
                    "profile": profile,
                    "workload_count": len(suite.get("results", [])),
                    "table": _public_table(suite.get("results", [])),
                    "failures": failures,
                }
            },
            "details": {
                "elapsed_s": time.perf_counter() - started,
                "run_dir": str(paths["run_dir"]),
                "report_path": str(paths["report_path"]),
                "suite": suite,
                "verifier": verifier or {},
            },
        }
        paths["report_path"].write_text(json.dumps(result, indent=2, sort_keys=True))
        return result
    except Exception as exc:
        result = {
            "score": 0.0,
            "valid": False,
            "correct": {"correct": False, "error": f"{type(exc).__name__}: {exc}"},
            "metrics": {
                "combined_score": 0.0,
                "decode_attention_speedup": 0.0,
                "workload_count": 0,
                "profile": profile,
            },
            "evaluator": {
                "public_details": {
                    "summary": "official decode attention evaluator crashed",
                    "score": 0.0,
                    "profile": profile,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            },
            "details": {
                "elapsed_s": time.perf_counter() - started,
                "run_dir": str(paths["run_dir"]),
                "report_path": str(paths["report_path"]),
                "verifier": verifier or {},
            },
        }
        paths["report_path"].write_text(json.dumps(result, indent=2, sort_keys=True))
        return result


def _public_table(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for result in results:
        metrics = result.get("metrics", {})
        table.append(
            {
                "name": result.get("name"),
                "valid": bool(result.get("valid")),
                "ref_ms": _round_optional(metrics.get("ref_ms")),
                "cand_ms": _round_optional(metrics.get("cand_ms")),
                "speedup": _round_optional(metrics.get("speedup")),
                "max_abs_error": _round_optional(metrics.get("max_abs_error")),
                "matched_ratio": _round_optional(metrics.get("matched_ratio")),
                "fallback_calls": metrics.get("fallback_calls_timing", metrics.get("fallback_calls_correctness")),
            }
        )
    return table


def _round_optional(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return round(float(value), 6)
    return value


def _make_paths(profile: str) -> dict[str, Path]:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    runs_root = Path(os.environ.get("AO_TASK_RUNS_ROOT", "/workspace/runs"))
    run_dir = runs_root / f"qwen3_1_7b_decode_attention_eval_{profile}_{stamp}_{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_dir,
        "report_path": run_dir / "report.json",
    }
