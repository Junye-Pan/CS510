from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentic_opt.tasks.ppbench_harness_golden300.private.benchmark_loader import load_public_records
from agentic_opt.tasks.ppbench_harness_golden300.task import (
    EVALUATE_CONCURRENCY,
    EVALUATE_TIMEOUT_S,
    PROBE_CONCURRENCY,
    PROBE_TIMEOUT_S,
    _aggregate_outcomes,
    _run_one,
)


TASK_ID = "ppbench_harness_golden300"


def _json_default(value: Any) -> str:
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    temp_path.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=_json_default))
        handle.write("\n")
        handle.flush()


def _summarize_probe(payload: dict[str, Any]) -> dict[str, Any]:
    diagnostics = payload.get("diagnostics") or {}
    return {
        "split": (payload.get("feedback") or {}).get("public_details", {}).get("split"),
        "resolved": int(diagnostics.get("solved_count") or 0),
        "total": int(diagnostics.get("total_count") or 0),
        "legal": int(diagnostics.get("legal_count") or 0),
        "invalid": int(diagnostics.get("invalid_count") or 0),
        "timeouts": int(diagnostics.get("timeout_count") or 0),
        "crashes": int(diagnostics.get("crash_count") or 0),
        "model_calls": int(diagnostics.get("model_calls") or 0),
        "elapsed_s": float(payload.get("elapsed_s") or 0.0),
        "solved_by_type": diagnostics.get("solved_by_type") or {},
        "attempted_by_type": diagnostics.get("attempted_by_type") or {},
    }


def _summarize_evaluate(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics") or {}
    details = (payload.get("evaluator") or {}).get("public_details") or {}
    return {
        "split": details.get("split"),
        "resolved": int(metrics.get("solved_count") or 0),
        "total": int(metrics.get("total_count") or 0),
        "score": float(payload.get("score") or 0.0),
        "legal": int(metrics.get("legal_count") or 0),
        "invalid": int(metrics.get("invalid_count") or 0),
        "timeouts": int(metrics.get("timeout_count") or 0),
        "crashes": int(metrics.get("crash_count") or 0),
        "model_calls": int(metrics.get("model_calls") or 0),
        "elapsed_s": float(metrics.get("elapsed_s") or 0.0),
        "mean_runtime_s": float(metrics.get("mean_runtime_s") or 0.0),
        "solved_by_type": metrics.get("solved_by_type") or {},
        "attempted_by_type": metrics.get("attempted_by_type") or {},
    }


def _split_config(mode: str) -> tuple[str, float, int, bool]:
    if mode == "probe":
        return "probe10", PROBE_TIMEOUT_S, PROBE_CONCURRENCY, True
    return "private50", EVALUATE_TIMEOUT_S, EVALUATE_CONCURRENCY, False


def _load_records(mode: str) -> list[dict[str, Any]]:
    split, _, _, _ = _split_config(mode)
    return load_public_records(split=split, limit=None)


def _run_streamed(
    *,
    mode: str,
    entry: Path,
    out_dir: Path,
) -> dict[str, Any]:
    split, timeout_s, max_workers, include_private_trace = _split_config(mode)
    records = _load_records(mode)
    instance_dir = out_dir / f"{mode}_instances"
    jsonl_path = out_dir / f"{mode}_instances.jsonl"
    progress_path = out_dir / f"{mode}_progress.json"

    instance_dir.mkdir(parents=True, exist_ok=True)
    if jsonl_path.exists():
        jsonl_path.unlink()

    started = time.perf_counter()
    outcomes: list[dict[str, Any] | None] = [None] * len(records)
    completed = 0

    def run_index(index: int, record: dict[str, Any]) -> dict[str, Any]:
        outcome = _run_one(
            entry_path=entry,
            puzzle=record,
            timeout_s=timeout_s,
            phase="probe" if mode == "probe" else "evaluate",
            include_private_trace=include_private_trace,
        )
        return {"index": index, "outcome": outcome}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(run_index, index, record): index
            for index, record in enumerate(records)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            completed += 1
            try:
                result = future.result()
                outcome = result["outcome"]
            except Exception as exc:
                outcome = {
                    "puzzle_type": records[index].get("puzzle_type"),
                    "status": "crash",
                    "valid_contract": False,
                    "legal": False,
                    "solved": False,
                    "runtime_s": 0.0,
                    "failure_reason": f"runner error: {type(exc).__name__}: {exc}",
                    "replay": None,
                    "model_name": "",
                    "llm_capable": False,
                    "model_calls": 0,
                }
                if include_private_trace:
                    outcome["puzzle_id"] = records[index].get("puzzle_id")
            outcomes[index] = outcome
            event = {
                "mode": mode,
                "split": split,
                "index": index,
                "completed": completed,
                "total": len(records),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "outcome": outcome,
            }
            _write_json(instance_dir / f"{index:03d}.json", event)
            _append_jsonl(jsonl_path, event)

            finished = [item for item in outcomes if item is not None]
            partial_metrics = _aggregate_outcomes(finished, include_records=False)
            progress = {
                "mode": mode,
                "split": split,
                "completed": completed,
                "total": len(records),
                "elapsed_s": time.perf_counter() - started,
                "partial": partial_metrics,
                "latest": {
                    "index": index,
                    "puzzle_type": outcome.get("puzzle_type"),
                    "status": outcome.get("status"),
                    "legal": outcome.get("legal"),
                    "solved": outcome.get("solved"),
                    "runtime_s": outcome.get("runtime_s"),
                    "model_calls": outcome.get("model_calls"),
                },
            }
            _write_json(progress_path, progress)
            print(
                json.dumps(
                    {
                        "event": "instance_finished",
                        "mode": mode,
                        "split": split,
                        "completed": completed,
                        "total": len(records),
                        "index": index,
                        "puzzle_type": outcome.get("puzzle_type"),
                        "status": outcome.get("status"),
                        "legal": outcome.get("legal"),
                        "solved": outcome.get("solved"),
                        "runtime_s": outcome.get("runtime_s"),
                        "model_calls": outcome.get("model_calls"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    final_outcomes = [item for item in outcomes if item is not None]
    metrics = _aggregate_outcomes(final_outcomes, include_records=(mode == "probe"))
    if mode == "evaluate":
        metrics["evaluator_concurrency"] = max_workers
        metrics["elapsed_s"] = time.perf_counter() - started
        score = int(metrics["solved_count"])
        return {
            "score": float(score),
            "valid": True,
            "correct": {"correct": True, "error": None},
            "metrics": metrics,
            "evaluator": {
                "score": float(score),
                "public_details": {
                    "split": split,
                    "private_details_redacted": True,
                    "solved_count": score,
                    "total_count": metrics["total_count"],
                    "legal_count": metrics["legal_count"],
                    "invalid_count": metrics["invalid_count"],
                    "timeout_count": metrics["timeout_count"],
                    "crash_count": metrics["crash_count"],
                    "concurrency": max_workers,
                    "fixed_model_name": metrics["fixed_model_name"],
                },
            },
            "extra": {
                "split": split,
                "private_details_redacted": True,
                "aggregate": metrics,
            },
        }

    return {
        "ok": True,
        "kind": "diagnostics",
        "feedback": {
            "error": None,
            "public_details": {
                "split": split,
                "solved_count": metrics["solved_count"],
                "legal_count": metrics["legal_count"],
                "invalid_count": metrics["invalid_count"],
                "timeout_count": metrics["timeout_count"],
                "concurrency": max_workers,
                "fixed_model_name": metrics["fixed_model_name"],
            },
        },
        "diagnostics": metrics,
        "elapsed_s": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["probe", "evaluate"])
    parser.add_argument("--entry", type=Path, default=Path("candidate/initial.py"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    entry = args.entry if args.entry.is_absolute() else root / args.entry
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir

    started = time.perf_counter()
    payload: dict[str, Any]
    if args.mode == "probe":
        payload = _run_streamed(mode="probe", entry=entry, out_dir=out_dir)
        summary = _summarize_probe(payload)
        result_path = out_dir / "probe.json"
    else:
        payload = _run_streamed(mode="evaluate", entry=entry, out_dir=out_dir)
        summary = _summarize_evaluate(payload)
        result_path = out_dir / "evaluate_private50.json"

    envelope = {
        "mode": args.mode,
        "task_id": TASK_ID,
        "entry": str(entry),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "wall_elapsed_s": time.perf_counter() - started,
        "env": {
            "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL"),
            "MODEL_NAME": os.environ.get("MODEL_NAME"),
            "AO_PPBENCH_MAX_MODEL_CALLS": os.environ.get("AO_PPBENCH_MAX_MODEL_CALLS"),
            "AO_PPBENCH_PROBE_CONCURRENCY": os.environ.get("AO_PPBENCH_PROBE_CONCURRENCY"),
            "AO_PPBENCH_EVALUATE_CONCURRENCY": os.environ.get("AO_PPBENCH_EVALUATE_CONCURRENCY"),
        },
        "summary": summary,
        "payload": payload,
    }
    _write_json(result_path, envelope)
    _write_json(out_dir / f"{args.mode}_summary.json", {"summary": summary})
    print(json.dumps({"result_path": str(result_path), "summary": summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
