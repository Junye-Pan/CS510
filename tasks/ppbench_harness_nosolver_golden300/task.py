from __future__ import annotations

import ast
import builtins
import concurrent.futures
import importlib.util
import json
import multiprocessing
import os
import py_compile
import re
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_opt.common.config import get_repo_root
from agentic_opt.common.runtime_env import TaskRuntimeSpec
from agentic_opt.common.snapshot import copy_snapshot
from agentic_opt.task_api import CandidateSpec, TaskMetadata

from .private.benchmark_loader import PPBenchDependencyError, dependency_status, load_public_records, smoke_records
from .private.replay import replay_moves


FIXED_MODEL_NAME = "gpt-5.2"
DEFAULT_BASE_URL = "https://chatgpt.com/backend-api/codex"
VERIFY_TIMEOUT_S = float(os.environ.get("AO_PPBENCH_VERIFY_TIMEOUT_S", "30"))
PROBE_TIMEOUT_S = float(os.environ.get("AO_PPBENCH_PROBE_TIMEOUT_S", "900"))
EVALUATE_TIMEOUT_S = float(os.environ.get("AO_PPBENCH_EVALUATE_TIMEOUT_S", "900"))
PROBE_CONCURRENCY = int(os.environ.get("AO_PPBENCH_PROBE_CONCURRENCY", "4"))
EVALUATE_CONCURRENCY = int(os.environ.get("AO_PPBENCH_EVALUATE_CONCURRENCY", "10"))
DISALLOWED_TRANSPORT_IMPORT_ROOTS = {"urllib", "requests", "httpx", "openai", "aiohttp"}
DISALLOWED_TRANSPORT_MARKERS = (
    "/responses",
    "text/event-stream",
    "urlopen",
    "requests.",
    "httpx.",
    "aiohttp.",
)
DISALLOWED_FAMILY_SOLVER_NAME_RE = re.compile(
    r"(^_?solve_(?!puzzle$).+|.*(?:solver|backtrack|exact_cover|pattern_search|constraint_search).*)",
    re.IGNORECASE,
)
DISALLOWED_FAMILY_SOLVER_TEXT_MARKERS = (
    "complete deterministic solver",
    "puzzle-family solver",
    "family-specific solver",
    "backtracking solver",
    "exact cover",
)
REQUIRED_TRANSPORT_MARKERS = (
    "pydantic_ai",
    "OpenAIResponsesModel",
    "OpenAIResponsesModelSettings",
    "AsyncOpenAI",
    "OpenAIProvider",
)
_SPAWN_PREP_LOCK = threading.Lock()

PPBENCH_RUNTIME = TaskRuntimeSpec(
    python=">=3.11,<3.12",
    requirements=(),
    required_imports=(),
    forbidden_shadow_modules=("ppbench", "agentic_opt"),
    verify_public_seed=True,
)


@dataclass(frozen=True)
class PPBenchHarnessTask:
    metadata: TaskMetadata = TaskMetadata(
        task_id="ppbench_harness_nosolver_golden300",
        title="PPBench General LLM Harness (No New Family Solvers)",
        candidate_spec=CandidateSpec(
            entrypoint_name="initial.py",
            candidate_root=".",
            description=(
                "Task-shaped PPBench harness package rooted at the workspace root; "
                "initial.py is the entrypoint and helper files next to it are "
                "included in submitted snapshots."
            ),
        ),
    )
    runtime_spec: TaskRuntimeSpec = PPBENCH_RUNTIME

    @property
    def public_dir(self) -> Path:
        return Path(__file__).resolve().parent / "public"

    def verify_entry(self, entry_path: Path) -> dict[str, Any]:
        started = time.perf_counter()
        checks: list[dict[str, Any]] = []
        if not entry_path.exists():
            return _verify_failure(
                checks,
                started=started,
                name="entrypoint_exists",
                message=f"{entry_path.name} missing",
            )
        checks.append({"name": "entrypoint_exists", "status": "passed", "message": None})
        try:
            _syntax_check(entry_path)
        except py_compile.PyCompileError as exc:
            return _verify_failure(checks, started=started, name="py_compile", message=str(exc))
        checks.append({"name": "py_compile", "status": "passed", "message": None})

        transport_error = _api_transport_policy_error(entry_path)
        if transport_error is not None:
            return _verify_failure(
                checks,
                started=started,
                name="fixed_api_transport_policy",
                message=transport_error,
            )
        checks.append({"name": "fixed_api_transport_policy", "status": "passed", "message": None})

        solver_policy_error = _family_solver_policy_error(entry_path)
        if solver_policy_error is not None:
            return _verify_failure(
                checks,
                started=started,
                name="no_new_family_solver_policy",
                message=solver_policy_error,
            )
        checks.append({"name": "no_new_family_solver_policy", "status": "passed", "message": None})

        dep = dependency_status()
        checks.append(
            {
                "name": "ppbench_dependency",
                "status": "passed" if dep["available"] else "warning",
                "message": dep["error"],
            }
        )

        records = smoke_records()
        invalid_reasons: list[str] = []
        for record in records:
            outcome = _run_one(
                entry_path=entry_path,
                puzzle=record,
                timeout_s=VERIFY_TIMEOUT_S,
                phase="verify",
                include_private_trace=True,
            )
            if not outcome["valid_contract"]:
                invalid_reasons.append(f"{record['puzzle_id']}: {outcome['failure_reason']}")
                continue
            if not outcome["llm_capable"]:
                invalid_reasons.append(f"{record['puzzle_id']}: result meta must declare llm_capable=true")
                continue
            if outcome["model_name"] != FIXED_MODEL_NAME:
                invalid_reasons.append(
                    f"{record['puzzle_id']}: result meta model_name must be {FIXED_MODEL_NAME!r}"
                )
                continue
            if not outcome["legal"]:
                invalid_reasons.append(f"{record['puzzle_id']}: illegal move trace: {outcome['failure_reason']}")

        if invalid_reasons:
            checks.append(
                {
                    "name": "smoke_contract_and_move_legality",
                    "status": "failed",
                    "message": invalid_reasons[0],
                }
            )
            return {
                "status": "failed",
                "valid": False,
                "checks": checks,
                "feedback": {
                    "error": invalid_reasons[0],
                    "public_details": {
                        "invalid_reasons": invalid_reasons[:5],
                        "fixed_model_name": FIXED_MODEL_NAME,
                    },
                },
                "elapsed_s": time.perf_counter() - started,
            }

        checks.append(
            {
                "name": "smoke_contract_and_move_legality",
                "status": "passed",
                "message": f"{len(records)} smoke puzzle(s) returned schema-valid legal traces",
            }
        )
        return {
            "status": "passed",
            "valid": True,
            "checks": checks,
            "feedback": {
                "error": None,
                    "public_details": {
                        "smoke_puzzles": len(records),
                        "ppbench_dependency_available": dep["available"],
                        "ppbench_data_source": dep.get("source"),
                        "fixed_model_name": FIXED_MODEL_NAME,
                    },
                },
            "elapsed_s": time.perf_counter() - started,
        }

    def probe_entry(self, entry_path: Path, *, kind: str) -> dict[str, Any]:
        started = time.perf_counter()
        if kind not in {"diagnostics", "probe10"}:
            raise ValueError(f"Unsupported probe kind for ppbench_harness_nosolver_golden300: {kind}")
        try:
            records = load_public_records(split="probe10", limit=_env_limit("AO_PPBENCH_PROBE_LIMIT"))
        except PPBenchDependencyError as exc:
            return {
                "ok": True,
                "kind": kind,
                "feedback": {
                    "error": str(exc),
                    "public_details": {
                        "dependency_available": False,
                        "install_hint": "Install the `ppbench` package in the task runtime environment.",
                    },
                },
                "diagnostics": {
                    "dependency_available": False,
                    "solved_count": 0,
                    "legal_count": 0,
                    "invalid_count": 0,
                    "timeout_count": 0,
                    "crash_count": 0,
                    "records": [],
                },
                "elapsed_s": time.perf_counter() - started,
            }
        outcomes = _run_records(
            entry_path=entry_path,
            records=records,
            timeout_s=PROBE_TIMEOUT_S,
            phase="probe",
            include_private_trace=True,
            max_workers=PROBE_CONCURRENCY,
        )
        diagnostics = _aggregate_outcomes(outcomes, include_records=True)
        return {
            "ok": True,
            "kind": kind,
            "feedback": {
                "error": None,
                "public_details": {
                    "split": "probe10",
                    "solved_count": diagnostics["solved_count"],
                    "legal_count": diagnostics["legal_count"],
                    "invalid_count": diagnostics["invalid_count"],
                    "timeout_count": diagnostics["timeout_count"],
                    "concurrency": PROBE_CONCURRENCY,
                    "data_source": dependency_status().get("source"),
                    "fixed_model_name": FIXED_MODEL_NAME,
                },
            },
            "diagnostics": diagnostics,
            "elapsed_s": time.perf_counter() - started,
        }

    def evaluate_entry(self, entry_path: Path) -> dict[str, Any]:
        started = time.perf_counter()
        records = load_public_records(split="private50", limit=None)
        outcomes = _run_records(
            entry_path=entry_path,
            records=records,
            timeout_s=EVALUATE_TIMEOUT_S,
            phase="evaluate",
            include_private_trace=False,
            max_workers=EVALUATE_CONCURRENCY,
        )
        metrics = _aggregate_outcomes(outcomes, include_records=False)
        metrics["evaluator_concurrency"] = EVALUATE_CONCURRENCY
        score = int(metrics["solved_count"])
        elapsed_s = time.perf_counter() - started
        metrics["elapsed_s"] = elapsed_s
        return {
            "score": float(score),
            "valid": True,
            "correct": {"correct": True, "error": None},
            "metrics": metrics,
            "evaluator": {
                "score": float(score),
                "public_details": {
                    "split": "private50",
                    "private_details_redacted": True,
                    "solved_count": score,
                    "total_count": metrics["total_count"],
                    "legal_count": metrics["legal_count"],
                    "invalid_count": metrics["invalid_count"],
                    "timeout_count": metrics["timeout_count"],
                    "crash_count": metrics["crash_count"],
                    "concurrency": EVALUATE_CONCURRENCY,
                    "data_source": dependency_status().get("source"),
                    "fixed_model_name": FIXED_MODEL_NAME,
                },
            },
            "extra": {
                "split": "private50",
                "private_details_redacted": True,
                "aggregate": metrics,
            },
        }


def create_task() -> PPBenchHarnessTask:
    return PPBenchHarnessTask()


def _verify_failure(
    checks: list[dict[str, Any]],
    *,
    started: float,
    name: str,
    message: str,
) -> dict[str, Any]:
    checks.append({"name": name, "status": "failed", "message": message})
    return {
        "status": "failed",
        "valid": False,
        "checks": checks,
        "feedback": {"error": message, "public_details": {}},
        "elapsed_s": time.perf_counter() - started,
    }


def _syntax_check(program_path: Path) -> None:
    with tempfile.NamedTemporaryFile(suffix=".pyc", delete=False) as handle:
        bytecode_path = handle.name
    try:
        py_compile.compile(str(program_path), cfile=bytecode_path, doraise=True)
    finally:
        if os.path.exists(bytecode_path):
            os.unlink(bytecode_path)


def _api_transport_policy_error(entry_path: Path) -> str | None:
    source_root = entry_path.resolve().parent
    python_files = sorted(path for path in source_root.rglob("*.py") if "__pycache__" not in path.parts)
    combined_source_parts: list[str] = []
    for path in python_files:
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"candidate Python file is not UTF-8 text: {path.name}"
        combined_source_parts.append(source)
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            return f"could not parse candidate Python file {path.name}: {exc}"
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    if root in DISALLOWED_TRANSPORT_IMPORT_ROOTS:
                        return (
                            "candidate must preserve the task-provided pydantic_ai streamed "
                            f"Codex transport; direct import {alias.name!r} is not allowed"
                        )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                root = module.split(".", 1)[0]
                if root in DISALLOWED_TRANSPORT_IMPORT_ROOTS:
                    return (
                        "candidate must preserve the task-provided pydantic_ai streamed "
                        f"Codex transport; direct import from {module!r} is not allowed"
                    )
    combined_source = "\n".join(combined_source_parts)
    for marker in REQUIRED_TRANSPORT_MARKERS:
        if marker not in combined_source:
            return (
                "candidate must preserve the task-provided pydantic_ai streamed Codex transport; "
                f"missing required transport marker {marker!r}"
            )
    for marker in DISALLOWED_TRANSPORT_MARKERS:
        if marker in combined_source:
            return (
                "candidate must not replace the task-provided pydantic_ai transport with a "
                f"manual HTTP/Responses client; found marker {marker!r}"
            )
    return None


def _family_solver_policy_error(entry_path: Path) -> str | None:
    source_root = entry_path.resolve().parent
    python_files = sorted(path for path in source_root.rglob("*.py") if "__pycache__" not in path.parts)
    for path in python_files:
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return f"candidate Python file is not UTF-8 text: {path.name}"
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            return f"could not parse candidate Python file {path.name}: {exc}"

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = node.name
                if DISALLOWED_FAMILY_SOLVER_NAME_RE.match(name):
                    return (
                        "candidate must not add complete deterministic puzzle-family solvers; "
                        f"{path.name} defines {name!r}. Focus on prompt construction, response "
                        "parsing, repair, generic validation, and model-call policy instead."
                    )
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                lowered = node.value.lower()
                for marker in DISALLOWED_FAMILY_SOLVER_TEXT_MARKERS:
                    if marker in lowered:
                        return (
                            "candidate text indicates a disallowed complete puzzle-family solver "
                            f"({marker!r}) in {path.name}. Optimize the LLM harness rather than "
                            "adding family-specific deterministic solvers."
                        )
    return None


def _run_records(
    *,
    entry_path: Path,
    records: list[dict[str, Any]],
    timeout_s: float,
    phase: str,
    include_private_trace: bool,
    max_workers: int = 1,
) -> list[dict[str, Any]]:
    def run_record(record: dict[str, Any]) -> dict[str, Any]:
        try:
            return _run_one(
                entry_path=entry_path,
                puzzle=record,
                timeout_s=timeout_s,
                phase=phase,
                include_private_trace=include_private_trace,
            )
        except Exception as exc:
            return _outcome(
                puzzle=record,
                status="crash",
                runtime_s=0.0,
                failure_reason=f"host runner error: {type(exc).__name__}: {exc}",
                include_private_trace=include_private_trace,
            )

    if max_workers <= 1 or len(records) <= 1:
        return [run_record(record) for record in records]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_record, record) for record in records]
        return [future.result() for future in futures]


def _run_one(
    *,
    entry_path: Path,
    puzzle: dict[str, Any],
    timeout_s: float,
    phase: str,
    include_private_trace: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="ao_ppbench_candidate_") as tempdir:
        temp_root = Path(tempdir) / "candidate"
        copied = copy_snapshot(entry_path.resolve().parent, temp_root)
        relative_entry = entry_path.name
        if relative_entry not in copied and not (temp_root / relative_entry).exists():
            return _outcome(
                puzzle=puzzle,
                status="crash",
                runtime_s=time.perf_counter() - started,
                failure_reason=f"entrypoint {entry_path.name!r} was not copied into execution sandbox",
                include_private_trace=include_private_trace,
            )
        context = _multiprocessing_context()
        result_q: multiprocessing.Queue = context.Queue()
        with _SPAWN_PREP_LOCK:
            previous_sys_path = list(sys.path)
            previous_pythonpath = os.environ.get("PYTHONPATH")
            _ensure_spawn_can_import_task_package()
            try:
                proc = context.Process(
                    target=_candidate_worker,
                    args=(
                        str(temp_root / relative_entry),
                        _json_compatible(puzzle),
                        _budget_for(timeout_s=timeout_s, phase=phase),
                        result_q,
                    ),
                )
                proc.start()
            finally:
                sys.path[:] = previous_sys_path
                if previous_pythonpath is None:
                    os.environ.pop("PYTHONPATH", None)
                else:
                    os.environ["PYTHONPATH"] = previous_pythonpath
        proc.join(timeout=timeout_s)
        runtime_s = time.perf_counter() - started
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=3)
            if proc.is_alive():
                proc.kill()
            return _outcome(
                puzzle=puzzle,
                status="host_timeout",
                runtime_s=runtime_s,
                failure_reason=f"host timeout after {timeout_s:.1f}s",
                include_private_trace=include_private_trace,
            )
        if result_q.empty():
            return _outcome(
                puzzle=puzzle,
                status="crash",
                runtime_s=runtime_s,
                failure_reason="candidate process returned no result",
                include_private_trace=include_private_trace,
            )
        payload = result_q.get()
    if payload.get("error"):
        return _outcome(
            puzzle=puzzle,
            status="crash",
            runtime_s=runtime_s,
            failure_reason=payload["error"],
            include_private_trace=include_private_trace,
        )
    raw_result = payload.get("result")
    validation = _validate_candidate_result(raw_result)
    if validation["error"]:
        return _outcome(
            puzzle=puzzle,
            status="malformed",
            runtime_s=runtime_s,
            failure_reason=validation["error"],
            include_private_trace=include_private_trace,
            raw_result=raw_result,
        )
    moves = validation["moves"]
    replay = replay_moves(puzzlink_url=str(puzzle["puzzlink_url"]), moves=moves)
    status = validation["status"]
    if not replay.legal:
        status = "illegal"
    solved = bool(replay.complete)
    return _outcome(
        puzzle=puzzle,
        status=status,
        runtime_s=runtime_s,
        valid_contract=True,
        legal=bool(replay.legal),
        solved=solved,
        failure_reason=replay.error,
        replay=replay.to_jsonable(),
        include_private_trace=include_private_trace,
        raw_result=raw_result,
        moves=moves if include_private_trace else None,
        model_name=validation["model_name"],
        llm_capable=validation["llm_capable"],
        model_calls=validation["model_calls"],
    )


def _candidate_worker(
    entry_path_raw: str,
    puzzle: dict[str, Any],
    budget: dict[str, Any],
    result_q: multiprocessing.Queue,
) -> None:
    try:
        os.environ["MODEL_NAME"] = FIXED_MODEL_NAME
        os.environ.setdefault("OPENAI_BASE_URL", DEFAULT_BASE_URL)
        _install_import_blocker({"ppbench", "agentic_opt"})
        module = _load_candidate_module(Path(entry_path_raw))
        solve_puzzle = getattr(module, "solve_puzzle", None)
        if not callable(solve_puzzle):
            raise AttributeError("candidate must expose callable solve_puzzle(puzzle, budget=None)")
        result_q.put({"result": _json_compatible(solve_puzzle(puzzle, budget))})
    except Exception as exc:  # pragma: no cover - subprocess boundary
        result_q.put({"error": f"{type(exc).__name__}: {exc}"})


def _multiprocessing_context() -> Any:
    if sys.platform == "darwin":
        return multiprocessing.get_context("spawn")
    if "fork" in multiprocessing.get_all_start_methods():
        return multiprocessing.get_context("fork")
    return multiprocessing.get_context()


def _repo_paths() -> tuple[Path, Path]:
    repo_root = get_repo_root()
    return repo_root, repo_root / "src"


def _ensure_spawn_can_import_task_package() -> None:
    repo_root, repo_src = _repo_paths()
    for path in (repo_root, repo_src):
        raw = str(path)
        if raw not in sys.path:
            sys.path.insert(0, raw)
    pythonpath = [item for item in os.environ.get("PYTHONPATH", "").split(os.pathsep) if item]
    for path in (repo_src, repo_root):
        raw = str(path)
        if raw not in pythonpath:
            pythonpath.insert(0, raw)
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath)


def _load_candidate_module(entry_path: Path) -> Any:
    previous_sys_path = list(sys.path)
    previous_dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        repo_root, repo_src = _repo_paths()
        sanitized = []
        for item in previous_sys_path:
            resolved = (Path(item).expanduser().resolve() if item else Path.cwd().resolve())
            if resolved in {repo_root, repo_src}:
                continue
            sanitized.append(item)
        sys.path[:] = [str(entry_path.parent), *sanitized]
        spec = importlib.util.spec_from_file_location(f"ppbench_candidate_{uuid.uuid4().hex}", entry_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load candidate from {entry_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = previous_sys_path
        sys.dont_write_bytecode = previous_dont_write


def _install_import_blocker(blocked_roots: set[str]) -> None:
    original_import = builtins.__import__

    def guarded_import(name: str, globals=None, locals=None, fromlist=(), level: int = 0):  # type: ignore[override]
        if level == 0 and name.split(".", 1)[0] in blocked_roots:
            raise ImportError(f"candidate may not import private module {name!r}")
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded_import


def _validate_candidate_result(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {"error": "solve_puzzle must return a dict"}
    status = result.get("status")
    if status not in {"solved", "failed", "timeout"}:
        return {"error": "result.status must be one of solved, failed, timeout"}
    moves = result.get("moves")
    if not isinstance(moves, list) or any(not isinstance(move, str) for move in moves):
        return {"error": "result.moves must be a list[str]"}
    meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
    model_name = str(meta.get("model_name") or "")
    llm_capable = bool(meta.get("llm_capable"))
    model_calls_raw = meta.get("model_calls", 0)
    try:
        model_calls = int(model_calls_raw)
    except Exception:
        model_calls = 0
    return {
        "error": None,
        "status": status,
        "moves": moves,
        "model_name": model_name,
        "llm_capable": llm_capable,
        "model_calls": model_calls,
    }


def _outcome(
    *,
    puzzle: dict[str, Any],
    status: str,
    runtime_s: float,
    failure_reason: str | None,
    include_private_trace: bool,
    valid_contract: bool = False,
    legal: bool = False,
    solved: bool = False,
    replay: dict[str, Any] | None = None,
    raw_result: Any = None,
    moves: list[str] | None = None,
    model_name: str = "",
    llm_capable: bool = False,
    model_calls: int = 0,
) -> dict[str, Any]:
    record = {
        "puzzle_type": puzzle.get("puzzle_type"),
        "status": status,
        "valid_contract": valid_contract,
        "legal": legal,
        "solved": solved,
        "runtime_s": runtime_s,
        "failure_reason": failure_reason,
        "replay": replay,
        "model_name": model_name,
        "llm_capable": llm_capable,
        "model_calls": model_calls,
    }
    if include_private_trace:
        record.update(
            {
                "puzzle_id": puzzle.get("puzzle_id"),
                "moves": moves,
                "raw_result": _json_compatible(raw_result),
            }
        )
    return record


def _aggregate_outcomes(outcomes: list[dict[str, Any]], *, include_records: bool) -> dict[str, Any]:
    total_runtime = sum(float(item.get("runtime_s") or 0.0) for item in outcomes)
    total_count = len(outcomes)
    solved_count = sum(1 for item in outcomes if item.get("solved"))
    legal_count = sum(1 for item in outcomes if item.get("legal"))
    timeout_count = sum(1 for item in outcomes if item.get("status") in {"timeout", "host_timeout"})
    crash_count = sum(1 for item in outcomes if item.get("status") == "crash")
    invalid_count = sum(
        1
        for item in outcomes
        if not item.get("valid_contract") or not item.get("legal") or item.get("status") in {"crash", "host_timeout"}
    )
    solved_by_type: dict[str, int] = {}
    attempted_by_type: dict[str, int] = {}
    for item in outcomes:
        puzzle_type = str(item.get("puzzle_type") or "unknown")
        attempted_by_type[puzzle_type] = attempted_by_type.get(puzzle_type, 0) + 1
        if item.get("solved"):
            solved_by_type[puzzle_type] = solved_by_type.get(puzzle_type, 0) + 1
    aggregate: dict[str, Any] = {
        "dependency_available": True,
        "total_count": total_count,
        "solved_count": solved_count,
        "legal_count": legal_count,
        "invalid_count": invalid_count,
        "timeout_count": timeout_count,
        "crash_count": crash_count,
        "total_runtime_s": total_runtime,
        "mean_runtime_s": total_runtime / total_count if total_count else 0.0,
        "model_calls": sum(int(item.get("model_calls") or 0) for item in outcomes),
        "fixed_model_name": FIXED_MODEL_NAME,
        "solved_by_type": solved_by_type,
        "attempted_by_type": attempted_by_type,
    }
    if include_records:
        aggregate["records"] = outcomes
    return aggregate


def _budget_for(*, timeout_s: float, phase: str) -> dict[str, Any]:
    if phase == "verify":
        max_model_calls = _env_limit("AO_PPBENCH_VERIFY_MODEL_CALLS", default=0)
    else:
        max_model_calls = _env_limit("AO_PPBENCH_MAX_MODEL_CALLS", default=3)
    return {
        "wall_clock_s": timeout_s,
        "phase": phase,
        "max_model_calls": max_model_calls,
        "max_tokens": 16000,
        "request_timeout_s": max(1.0, timeout_s - 15.0),
        "preferred_reasoning_effort": "high",
        "allow_network_search": False,
        "model_name": FIXED_MODEL_NAME,
        "openai_base_url": DEFAULT_BASE_URL,
    }


def _env_limit(name: str, *, default: int | None = None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if value < 0:
        return default
    return value


def _json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
