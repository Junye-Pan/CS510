from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any


PPBENCH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PPBENCH_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"
RUNTIME_PYTHON = (
    REPO_ROOT
    / ".ao_envs"
    / "ppbench_harness_golden300"
    / "1f967a70124d6c99"
    / "venv"
    / "bin"
    / "python"
)


def sanitize_import_path() -> None:
    """Avoid repo-root shadow modules such as numpy.py while keeping src importable."""

    cleaned: list[str] = []
    for item in sys.path:
        raw = item or os.getcwd()
        try:
            resolved = Path(raw).resolve()
        except OSError:
            cleaned.append(item)
            continue
        if resolved == REPO_ROOT:
            continue
        cleaned.append(item)
    sys.path[:] = cleaned
    src = str(SRC_ROOT)
    if src not in sys.path:
        sys.path.insert(0, src)


sanitize_import_path()

from agentic_opt.tasks.ppbench_harness_golden300.private.benchmark_loader import (  # noqa: E402
    load_public_records,
)
from agentic_opt.tasks.ppbench_harness_golden300.private.replay import replay_moves  # noqa: E402
from ppbench import Puzzle  # noqa: E402
from ppbench.benchmarks.harness import run_strategy  # noqa: E402
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings  # noqa: E402
from pydantic_ai.providers.openai import AsyncOpenAI, OpenAIProvider  # noqa: E402


MODEL_NAME = "gpt-5.2"
DEFAULT_BASE_URL = "https://chatgpt.com/backend-api/codex"


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--limit", type=int, default=None, help="Optional first-N limit for smoke runs.")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--reasoning-effort", default="high", choices=["low", "medium", "high", "xhigh"])
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--request-timeout-s", type=float, default=1800.0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Skip indexes already present in results.jsonl.")
    parser.add_argument(
        "--retry-invalid",
        action="store_true",
        help="With --resume, rerun indexes whose latest row is invalid.",
    )
    return parser


def build_model(*, model_name: str, reasoning_effort: str, timeout_s: float) -> OpenAIResponsesModel:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is required")
    base_url = os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=5)
    provider = OpenAIProvider(openai_client=client)
    return OpenAIResponsesModel(
        model_name,
        provider=provider,
        settings=OpenAIResponsesModelSettings(
            openai_reasoning_effort=reasoning_effort,
            openai_store=False,
            timeout=timeout_s,
        ),
    )


def load_private50_records(limit: int | None) -> list[dict[str, Any]]:
    return load_public_records(split="private50", limit=limit)


def latest_rows(results_path: Path) -> dict[int, dict[str, Any]]:
    if not results_path.exists():
        return {}
    latest: dict[int, dict[str, Any]] = {}
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item.get("index"), int):
            latest[item["index"]] = item
    return latest


def completed_indexes(results_path: Path, *, retry_invalid: bool) -> set[int]:
    done: set[int] = set()
    for index, item in latest_rows(results_path).items():
        if retry_invalid and item.get("invalid"):
            continue
        done.add(index)
    return done


def summarize_results(results_path: Path, *, total_count: int, strategy_name: str, model_label: str) -> dict[str, Any]:
    latest = latest_rows(results_path)
    rows = [latest[index] for index in sorted(latest)]
    solved_count = sum(1 for row in rows if row.get("solved"))
    legal_count = sum(1 for row in rows if row.get("legal"))
    invalid_count = sum(1 for row in rows if row.get("invalid"))
    timeout_count = sum(1 for row in rows if row.get("timeout"))
    crash_count = sum(1 for row in rows if row.get("crash"))
    total_runtime_s = sum(float(row.get("duration_s") or 0.0) for row in rows)
    attempted_by_type: dict[str, int] = {}
    solved_by_type: dict[str, int] = {}
    for row in rows:
        puzzle_type = str(row.get("puzzle_type") or "unknown")
        attempted_by_type[puzzle_type] = attempted_by_type.get(puzzle_type, 0) + 1
        if row.get("solved"):
            solved_by_type[puzzle_type] = solved_by_type.get(puzzle_type, 0) + 1
    return {
        "strategy": strategy_name,
        "model": model_label,
        "split": "private50",
        "private_details_redacted": True,
        "completed_count": len(rows),
        "total_count": total_count,
        "solved_count": solved_count,
        "accuracy": solved_count / len(rows) if rows else 0.0,
        "legal_count": legal_count,
        "invalid_count": invalid_count,
        "timeout_count": timeout_count,
        "crash_count": crash_count,
        "total_runtime_s": total_runtime_s,
        "mean_runtime_s": total_runtime_s / len(rows) if rows else 0.0,
        "model_requests": sum(int(row.get("model_requests") or 0) for row in rows),
        "move_count": sum(int(row.get("move_count") or 0) for row in rows),
        "attempted_by_type": attempted_by_type,
        "solved_by_type": solved_by_type,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


async def run_private50_baseline(
    *,
    strategy_factory: Any,
    strategy_name: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"
    records = load_private50_records(args.limit)
    done = completed_indexes(results_path, retry_invalid=bool(args.retry_invalid)) if args.resume else set()
    model_label = f"codex/{args.model}@{args.reasoning_effort}"
    semaphore = asyncio.Semaphore(max(1, int(args.concurrency)))
    lock = asyncio.Lock()

    async def one(index: int, record: dict[str, Any]) -> None:
        if index in done:
            return
        async with semaphore:
            started = time.perf_counter()
            row: dict[str, Any]
            try:
                puzzle = Puzzle.from_url(str(record["puzzlink_url"]))
                model = build_model(
                    model_name=args.model,
                    reasoning_effort=args.reasoning_effort,
                    timeout_s=args.request_timeout_s,
                )
                detailed = await run_strategy(
                    strategy_factory(),
                    puzzle,
                    model,
                    model_label,
                    storage=None,
                    max_retries=args.max_retries,
                    request_timeout=args.request_timeout_s,
                )
                moves = list(detailed.summary.parsed_moves or [])
                replay = replay_moves(puzzlink_url=str(record["puzzlink_url"]), moves=moves)
                error_type = detailed.summary.error_type
                row = {
                    "index": index,
                    "puzzle_type": record.get("puzzle_type"),
                    "solved": bool(replay.complete),
                    "legal": bool(replay.legal),
                    "invalid": bool((not replay.legal) or error_type),
                    "timeout": bool(error_type and "Timeout" in error_type),
                    "crash": bool(error_type and "Timeout" not in error_type),
                    "duration_s": time.perf_counter() - started,
                    "model_requests": int(detailed.summary.total_requests or 0),
                    "move_count": len(moves),
                    "error_type": error_type,
                    "replay_engine_available": bool(replay.engine_available),
                    "failure_reason_redacted": replay.error[:240] if replay.error else None,
                }
            except Exception as exc:
                row = {
                    "index": index,
                    "puzzle_type": record.get("puzzle_type"),
                    "solved": False,
                    "legal": False,
                    "invalid": True,
                    "timeout": "Timeout" in type(exc).__name__,
                    "crash": "Timeout" not in type(exc).__name__,
                    "duration_s": time.perf_counter() - started,
                    "model_requests": 0,
                    "move_count": 0,
                    "error_type": f"{type(exc).__name__}: {str(exc)[:240]}",
                    "replay_engine_available": None,
                    "failure_reason_redacted": None,
                }
            async with lock:
                with results_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                summary = summarize_results(
                    results_path,
                    total_count=len(records),
                    strategy_name=strategy_name,
                    model_label=model_label,
                )
                write_json(summary_path, summary)
                print(
                    f"[{strategy_name}] {summary['completed_count']}/{len(records)} "
                    f"solved={summary['solved_count']} legal={summary['legal_count']} "
                    f"invalid={summary['invalid_count']}",
                    flush=True,
                )

    await asyncio.gather(*(one(index, record) for index, record in enumerate(records)))
    return summarize_results(
        results_path,
        total_count=len(records),
        strategy_name=strategy_name,
        model_label=model_label,
    )
