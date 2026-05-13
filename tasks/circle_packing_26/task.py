from __future__ import annotations

import importlib.util
import html
import multiprocessing
import os
import py_compile
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_opt.common.config import get_repo_root
from agentic_opt.common.numpy_loader import import_numpy
from agentic_opt.common.scipy_loader import import_scipy
from agentic_opt.common.runtime_env import TaskRuntimeSpec, check_declared_dependency_shadowing

from agentic_opt.task_api import CandidateSpec, TaskMetadata


EXPECTED_CIRCLES = 26
DEFAULT_TIMEOUT_S = float(os.environ.get("AO_CIRCLE_PACKING_TIMEOUT_S", "180"))
NEARLY_ACTIVE_TOLERANCE = 1e-3
STRICT_SAFE_MARGIN = 1e-10
_SPAWN_PREP_LOCK = threading.RLock()
CIRCLE_PACKING_RUNTIME = TaskRuntimeSpec(
    python=">=3.11,<3.12",
    requirements=("numpy>=1.23", "scipy>=1.10"),
    required_imports=("numpy", "scipy"),
    forbidden_shadow_modules=("numpy", "scipy"),
)


def _np():
    return import_numpy()


def _scipy():
    return import_scipy()


def _load_program(program_path: Path) -> Any:
    previous = sys.dont_write_bytecode
    previous_sys_path = list(sys.path)
    sys.dont_write_bytecode = True
    try:
        program_parent = str(program_path.resolve().parent)
        check_declared_dependency_shadowing(
            program_dir=program_path.resolve().parent,
            runtime_spec=CIRCLE_PACKING_RUNTIME,
        )
        repo_root = get_repo_root()
        sanitized_sys_path = [
            entry
            for entry in previous_sys_path
            if (Path(entry).expanduser().resolve() if entry else Path.cwd().resolve()) != repo_root
        ]
        sys.path[:] = [program_parent, *sanitized_sys_path]
        spec = importlib.util.spec_from_file_location("circle_packing_26_candidate", program_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {program_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = previous_sys_path
        sys.dont_write_bytecode = previous


def _run_worker(program_path: str, result_q: multiprocessing.Queue, error_q: multiprocessing.Queue) -> None:
    try:
        module = _load_program(Path(program_path))
        if not hasattr(module, "run_packing"):
            raise AttributeError(f"'run_packing' not found in {program_path}")
        run_packing = getattr(module, "run_packing")
        if not callable(run_packing):
            raise TypeError(f"'run_packing' in {program_path} is not callable")
        result_q.put(run_packing())
    except Exception as exc:  # pragma: no cover - process boundary
        error_q.put((type(exc).__name__, str(exc), traceback.format_exc()))


def _run_with_timeout(program_path: Path, *, timeout_s: float) -> tuple[Any, float]:
    context = _multiprocessing_context()
    result_q: multiprocessing.Queue = context.Queue()
    error_q: multiprocessing.Queue = context.Queue()
    proc = context.Process(target=_run_worker, args=(str(program_path), result_q, error_q))
    started = time.perf_counter()
    with _SPAWN_PREP_LOCK:
        previous_sys_path = list(sys.path)
        previous_pythonpath = os.environ.get("PYTHONPATH")
        _ensure_spawn_can_import_task_package()
        try:
            proc.start()
        finally:
            sys.path[:] = previous_sys_path
            if previous_pythonpath is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = previous_pythonpath
    proc.join(timeout=timeout_s)
    elapsed_s = time.perf_counter() - started
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
        raise TimeoutError(f"Execution exceeded timeout of {timeout_s}s")
    if not error_q.empty():
        etype, emsg, tb = error_q.get()
        raise RuntimeError(f"{etype}: {emsg}\n{tb}")
    if result_q.empty():
        raise RuntimeError("Function completed but returned no result")
    # The worker may return real numpy arrays through a multiprocessing queue.
    # Import real numpy in the parent process before unpickling so the repo-root
    # compatibility shim does not shadow numpy.core during deserialization.
    _np()
    return result_q.get(), elapsed_s


def _multiprocessing_context() -> Any:
    if sys.platform == "darwin":
        return multiprocessing.get_context("spawn")
    if "fork" in multiprocessing.get_all_start_methods():
        return multiprocessing.get_context("fork")
    return multiprocessing.get_context()


def _ensure_spawn_can_import_task_package() -> None:
    repo_root = get_repo_root()
    repo_src = repo_root / "src"
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


def _syntax_check(program_path: Path) -> None:
    with tempfile.NamedTemporaryFile(suffix=".pyc", delete=False) as handle:
        bytecode_path = handle.name
    try:
        py_compile.compile(str(program_path), cfile=bytecode_path, doraise=True)
    finally:
        if os.path.exists(bytecode_path):
            os.unlink(bytecode_path)


def _normalize_output(run_output: Any) -> tuple[Any, Any, float]:
    try:
        centers, radii, reported_sum = run_output
    except Exception as exc:
        raise ValueError("run_packing must return (centers, radii, reported_sum)") from exc
    centers = _coerce_array(centers)
    radii = _coerce_array(radii)
    try:
        reported_sum = float(reported_sum)
    except Exception as exc:
        raise ValueError("reported_sum must be numeric") from exc
    return centers, radii, reported_sum


def _coerce_array(value: Any) -> Any:
    np = _np()
    return value if isinstance(value, np.ndarray) else np.array(value)


def _strict_fixed_center_lp(centers: Any) -> dict[str, Any]:
    np = _np()
    scipy = _scipy()
    num_circles = centers.shape[0]
    if num_circles != EXPECTED_CIRCLES:
        raise ValueError(f"strict LP expected {EXPECTED_CIRCLES} circles, got {num_circles}")

    rows: list[Any] = []
    bounds: list[float] = []

    boundary_caps = np.minimum.reduce(
        [
            centers[:, 0],
            centers[:, 1],
            1.0 - centers[:, 0],
            1.0 - centers[:, 1],
        ]
    )
    for index in range(num_circles):
        row = np.zeros(num_circles, dtype=float)
        row[index] = 1.0
        rows.append(row)
        bounds.append(max(0.0, float(boundary_caps[index]) - STRICT_SAFE_MARGIN))

    for first in range(num_circles):
        for second in range(first + 1, num_circles):
            row = np.zeros(num_circles, dtype=float)
            row[first] = 1.0
            row[second] = 1.0
            dist = float(np.sqrt(np.sum((centers[first] - centers[second]) ** 2)))
            rows.append(row)
            bounds.append(max(0.0, dist - STRICT_SAFE_MARGIN))

    result = scipy.optimize.linprog(
        c=-np.ones(num_circles, dtype=float),
        A_ub=np.asarray(rows, dtype=float),
        b_ub=np.asarray(bounds, dtype=float),
        bounds=[(0.0, None)] * num_circles,
        method="highs",
    )
    if not result.success:
        return {
            "score": None,
            "min_boundary_slack": None,
            "min_pair_slack": None,
            "gap": None,
            "error": str(result.message),
        }

    lp_radii = np.asarray(result.x, dtype=float)
    min_boundary_slack = float(np.min(boundary_caps - lp_radii))
    min_pair_slack = None
    if num_circles > 1:
        pair_slacks = [
            float(np.sqrt(np.sum((centers[first] - centers[second]) ** 2))) - float(lp_radii[first] + lp_radii[second])
            for first in range(num_circles)
            for second in range(first + 1, num_circles)
        ]
        min_pair_slack = float(min(pair_slacks))
    score = float(np.sum(lp_radii))
    return {
        "score": score,
        "min_boundary_slack": min_boundary_slack,
        "min_pair_slack": min_pair_slack,
        "gap": None,
        "error": None,
    }


def analyze_output(
    run_output: Any,
    *,
    atol: float = 1e-6,
) -> dict[str, Any]:
    np = _np()
    centers, radii, reported_sum = _normalize_output(run_output)
    diagnostics: dict[str, Any] = {
        "reported_sum": reported_sum,
        "actual_sum": float(np.sum(radii)),
        "sum_mismatch": None,
        "min_boundary_slack": None,
        "min_pair_slack": None,
        "worst_overlap_pair": None,
        "worst_overlap_amount": 0.0,
        "num_negative_radii": 0,
        "num_boundary_violations": 0,
        "num_pair_violations": 0,
        "num_nearly_active_boundary_constraints": 0,
        "num_nearly_active_pair_constraints": 0,
        "conservative_repair_score": 0.0,
        "strict_safe_score": None,
        "strict_safe_score_method": "fixed_center_lp",
        "strict_safe_gap": None,
        "strict_safe_min_boundary_slack": None,
        "strict_safe_min_pair_slack": None,
        "strict_safe_error": None,
        "valid": False,
        "error": None,
        "public_details": {},
        "centers": centers,
        "radii": radii,
    }

    if centers.shape != (EXPECTED_CIRCLES, 2):
        diagnostics["error"] = f"Centers shape incorrect. Expected ({EXPECTED_CIRCLES}, 2), got {centers.shape}."
        return diagnostics
    if radii.shape != (EXPECTED_CIRCLES,):
        diagnostics["error"] = f"Radii shape incorrect. Expected ({EXPECTED_CIRCLES},), got {radii.shape}."
        return diagnostics

    try:
        centers = centers.astype(float, copy=False)
        radii = radii.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        diagnostics["error"] = f"centers and radii must be numeric arrays: {exc}"
        return diagnostics

    diagnostics["centers"] = centers
    diagnostics["radii"] = radii
    actual_sum = float(np.sum(radii))
    diagnostics["actual_sum"] = actual_sum

    if not bool(np.isfinite(reported_sum)):
        diagnostics["error"] = "reported_sum must be finite."
        return diagnostics
    if not bool(np.all(np.isfinite(centers))):
        diagnostics["error"] = "centers must contain only finite values."
        return diagnostics
    if not bool(np.all(np.isfinite(radii))):
        diagnostics["error"] = "radii must contain only finite values."
        return diagnostics

    negative_mask = radii < 0
    diagnostics["num_negative_radii"] = int(np.sum(negative_mask))
    if diagnostics["num_negative_radii"] > 0:
        indices = np.where(negative_mask)[0].tolist()
        diagnostics["error"] = f"Negative radii found for circles at indices: {indices}."
        return diagnostics

    if not np.isclose(actual_sum, reported_sum, atol=atol):
        diagnostics["sum_mismatch"] = float(actual_sum - reported_sum)
        diagnostics["error"] = f"Sum of radii ({actual_sum:.6f}) does not match reported ({reported_sum:.6f})."
        diagnostics["public_details"] = {"actual_sum": actual_sum, "reported_sum": reported_sum}
        return diagnostics

    boundary_slacks: list[float] = []
    boundary_violations = 0
    for index in range(EXPECTED_CIRCLES):
        x_coord, y_coord = centers[index]
        radius = radii[index]
        slack = min(x_coord - radius, y_coord - radius, 1.0 - (x_coord + radius), 1.0 - (y_coord + radius))
        boundary_slacks.append(float(slack))
        if slack < -atol:
            boundary_violations += 1
    diagnostics["num_boundary_violations"] = boundary_violations
    diagnostics["min_boundary_slack"] = float(min(boundary_slacks))
    diagnostics["num_nearly_active_boundary_constraints"] = int(
        sum(slack <= NEARLY_ACTIVE_TOLERANCE for slack in boundary_slacks)
    )

    pair_slacks: list[float] = []
    worst_pair: tuple[int, int] | None = None
    worst_overlap_amount = 0.0
    pair_violations = 0
    for first in range(EXPECTED_CIRCLES):
        for second in range(first + 1, EXPECTED_CIRCLES):
            dist = float(np.sqrt(np.sum((centers[first] - centers[second]) ** 2)))
            slack = dist - float(radii[first] + radii[second])
            pair_slacks.append(slack)
            if slack < -atol:
                pair_violations += 1
                overlap_amount = -slack
                if overlap_amount > worst_overlap_amount:
                    worst_overlap_amount = overlap_amount
                    worst_pair = (first, second)
    diagnostics["num_pair_violations"] = pair_violations
    diagnostics["min_pair_slack"] = float(min(pair_slacks)) if pair_slacks else None
    diagnostics["num_nearly_active_pair_constraints"] = int(
        sum(slack <= NEARLY_ACTIVE_TOLERANCE for slack in pair_slacks)
    )
    diagnostics["worst_overlap_pair"] = list(worst_pair) if worst_pair else None
    diagnostics["worst_overlap_amount"] = float(worst_overlap_amount)

    if pair_violations > 0:
        diagnostics["error"] = (
            f"Circles {worst_pair[0]} and {worst_pair[1]} overlap. "
            f"Overlap amount={worst_overlap_amount:.6f}."
            if worst_pair
            else "Circle overlaps detected."
        )
        diagnostics["public_details"] = {
            "first_bad_pair": list(worst_pair) if worst_pair else None,
            "overlap_amount": float(worst_overlap_amount),
        }
    elif boundary_violations > 0:
        violating_index = next(
            index for index, slack in enumerate(boundary_slacks) if slack < -atol
        )
        diagnostics["error"] = f"Circle {violating_index} is outside the unit square."
        diagnostics["public_details"] = {"circle": violating_index}
    else:
        diagnostics["valid"] = True
        diagnostics["error"] = None
        diagnostics["public_details"] = {
            "num_circles": EXPECTED_CIRCLES,
            "min_slack": float(min(diagnostics["min_boundary_slack"], diagnostics["min_pair_slack"] or 0.0)),
        }

    repair_radii = radii.copy()
    if pair_slacks:
        global_pair_scale = min(
            [1.0] + [
                max(0.0, float(np.sqrt(np.sum((centers[first] - centers[second]) ** 2))) / float(radii[first] + radii[second]))
                for first in range(EXPECTED_CIRCLES)
                for second in range(first + 1, EXPECTED_CIRCLES)
                if float(radii[first] + radii[second]) > 0
            ]
        )
        repair_radii *= min(global_pair_scale, 1.0)
    for index in range(EXPECTED_CIRCLES):
        x_coord, y_coord = centers[index]
        repair_radii[index] = min(repair_radii[index], x_coord, y_coord, 1.0 - x_coord, 1.0 - y_coord)
    diagnostics["conservative_repair_score"] = float(np.sum(np.clip(repair_radii, 0.0, None)))
    strict_safe = _strict_fixed_center_lp(centers)
    diagnostics["strict_safe_score"] = strict_safe["score"]
    diagnostics["strict_safe_min_boundary_slack"] = strict_safe["min_boundary_slack"]
    diagnostics["strict_safe_min_pair_slack"] = strict_safe["min_pair_slack"]
    diagnostics["strict_safe_error"] = strict_safe["error"]
    if strict_safe["score"] is not None:
        diagnostics["strict_safe_gap"] = float(diagnostics["actual_sum"] - strict_safe["score"])
    return diagnostics


def _visualization_payload(centers: Any, radii: Any, *, reported_sum: float) -> dict[str, Any]:
    circles = [
        {
            "index": int(index),
            "x": float(center[0]),
            "y": float(center[1]),
            "r": float(radius),
        }
        for index, (center, radius) in enumerate(zip(centers, radii, strict=True))
    ]
    return {
        "kind": "circle_packing_26",
        "format": "svg",
        "title": "Circle Packing 26",
        "num_circles": len(circles),
        "reported_sum": float(reported_sum),
        "circles": circles,
        "content": _visualization_svg(circles=circles, reported_sum=reported_sum),
    }


def _visualization_svg(*, circles: list[dict[str, float | int]], reported_sum: float, width: int = 560, height: int = 560) -> str:
    padding = 28.0
    inner_width = max(10.0, float(width) - padding * 2.0)
    inner_height = max(10.0, float(height) - padding * 2.0)
    scale = min(inner_width, inner_height)
    max_radius = max(float(circle["r"]) for circle in circles) or 1.0

    def color_for(radius: float) -> str:
        ratio = max(0.0, min(radius / max_radius, 1.0))
        hue = 185.0 - ratio * 140.0
        lightness = 76.0 - ratio * 28.0
        return f"hsl({hue:.0f} 62% {lightness:.0f}%)"

    shapes: list[str] = [
        f'<rect x="{padding:.2f}" y="{padding:.2f}" width="{scale:.2f}" height="{scale:.2f}" '
        'fill="#fbfaf5" stroke="#1f2937" stroke-width="1.5" rx="6" />'
    ]
    for circle in circles:
        radius = float(circle["r"])
        cx = padding + float(circle["x"]) * scale
        cy = padding + (1.0 - float(circle["y"])) * scale
        rendered_radius = max(0.0, radius * scale)
        label = html.escape(f"#{int(circle['index'])} r={radius:.4f}")
        fill = color_for(radius)
        shapes.append(
            f'<g><title>{label}</title><circle cx="{cx:.2f}" cy="{cy:.2f}" r="{rendered_radius:.2f}" '
            f'fill="{fill}" fill-opacity="0.68" stroke="#0f172a" stroke-width="1.0" /></g>'
        )
    shapes.append(
        f'<text x="{padding:.2f}" y="{height - 10:.2f}" fill="#475569" '
        f'font-size="13" font-family="Georgia, serif">reported sum = {float(reported_sum):.12f}</text>'
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        'xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Circle packing view">'
        + "".join(shapes)
        + "</svg>"
    )


@dataclass(frozen=True)
class CirclePacking26Task:
    metadata: TaskMetadata = TaskMetadata(
        task_id="circle_packing_26",
        title="Circle Packing in a Unit Square (n=26)",
        candidate_spec=CandidateSpec(
            entrypoint_name="initial.py",
            description=(
                "Single-file Python candidate. The workspace entrypoint must expose "
                "run_packing() and may contain helper functions in the same file."
            ),
        ),
    )
    runtime_spec: TaskRuntimeSpec = CIRCLE_PACKING_RUNTIME

    @property
    def public_dir(self) -> Path:
        return Path(__file__).resolve().parent / "public"

    def verify_entry(self, entry_path: Path) -> dict[str, Any]:
        started_at = time.perf_counter()
        checks: list[dict[str, Any]] = []
        if not entry_path.exists():
            checks.append({"name": "entrypoint_exists", "status": "failed", "message": f"{entry_path.name} missing"})
            return {
                "status": "failed",
                "valid": False,
                "checks": checks,
                "feedback": {"error": f"{entry_path.name} missing", "public_details": {}},
                "elapsed_s": 0.0,
            }

        checks.append({"name": "entrypoint_exists", "status": "passed", "message": None})
        try:
            _syntax_check(entry_path)
            checks.append({"name": "py_compile", "status": "passed", "message": None})
        except py_compile.PyCompileError as exc:
            message = str(exc)
            checks.append({"name": "py_compile", "status": "failed", "message": message})
            return {
                "status": "failed",
                "valid": False,
                "checks": checks,
                "feedback": {"error": message, "public_details": {}},
                "elapsed_s": time.perf_counter() - started_at,
            }

        try:
            run_output, execution_s = _run_with_timeout(entry_path, timeout_s=DEFAULT_TIMEOUT_S)
        except Exception as exc:
            message = str(exc)
            checks.append({"name": "run_packing", "status": "failed", "message": message})
            return {
                "status": "failed",
                "valid": False,
                "checks": checks,
                "feedback": {"error": message, "public_details": {}},
                "elapsed_s": time.perf_counter() - started_at,
            }

        try:
            diagnostics = analyze_output(run_output)
        except Exception as exc:
            message = str(exc)
            checks.append({"name": "packing_contract", "status": "failed", "message": message})
            return {
                "status": "failed",
                "valid": False,
                "checks": checks,
                "feedback": {"error": message, "public_details": {}},
                "elapsed_s": time.perf_counter() - started_at,
            }
        checks.append(
            {
                "name": "packing_contract",
                "status": "passed" if diagnostics["valid"] else "failed",
                "message": (
                    f"reported_sum={diagnostics['reported_sum']:.6f}; elapsed_s={execution_s:.3f}"
                    if diagnostics["valid"]
                    else diagnostics["error"]
                ),
            }
        )
        return {
            "status": "passed" if diagnostics["valid"] else "failed",
            "valid": bool(diagnostics["valid"]),
            "checks": checks,
            "feedback": {
                "error": diagnostics["error"],
                "public_details": diagnostics["public_details"],
            },
            "elapsed_s": time.perf_counter() - started_at,
            "diagnostics": diagnostics,
        }

    def probe_entry(self, entry_path: Path, *, kind: str) -> dict[str, Any]:
        if kind != "diagnostics":
            raise ValueError(f"Unsupported probe kind for circle_packing_26: {kind}")
        try:
            run_output, execution_s = _run_with_timeout(entry_path, timeout_s=DEFAULT_TIMEOUT_S)
        except Exception as exc:
            return {
                "ok": True,
                "kind": kind,
                "feedback": {"error": str(exc), "public_details": {}},
                "diagnostics": None,
                "elapsed_s": 0.0,
            }
        try:
            diagnostics = analyze_output(run_output)
        except Exception as exc:
            return {
                "ok": True,
                "kind": kind,
                "feedback": {"error": str(exc), "public_details": {}},
                "diagnostics": None,
                "elapsed_s": execution_s,
            }
        public = {
            key: value
            for key, value in diagnostics.items()
            if key
            not in {
                "centers",
                "radii",
            }
        }
        score = diagnostics["strict_safe_score"]
        return {
            "ok": True,
            "valid": bool(diagnostics["valid"]),
            "score": score,
            "kind": kind,
            "feedback": {"error": diagnostics["error"], "public_details": diagnostics["public_details"]},
            "diagnostics": public,
            "elapsed_s": execution_s,
        }

    def evaluate_entry(self, entry_path: Path) -> dict[str, Any]:
        run_output, execution_s = _run_with_timeout(entry_path, timeout_s=DEFAULT_TIMEOUT_S)
        diagnostics = analyze_output(run_output)
        if not diagnostics["valid"]:
            raise ValueError(diagnostics["error"] or "evaluation failed")
        score = diagnostics["strict_safe_score"]
        if score is None:
            raise ValueError(diagnostics["strict_safe_error"] or "strict safe scoring failed")
        np = _np()
        centers = diagnostics["centers"]
        radii = diagnostics["radii"]
        reported_sum = float(diagnostics["reported_sum"])
        actual_sum = float(diagnostics["actual_sum"])
        return {
            "score": score,
            "valid": True,
            "correct": {"correct": True, "error": None},
            "metrics": {
                "combined_score": score,
                "reported_sum": reported_sum,
                "actual_sum": actual_sum,
                "strict_safe_score": score,
                "strict_safe_gap": diagnostics["strict_safe_gap"],
                "strict_safe_min_boundary_slack": diagnostics["strict_safe_min_boundary_slack"],
                "strict_safe_min_pair_slack": diagnostics["strict_safe_min_pair_slack"],
                "execution_time_s": execution_s,
                "min_boundary_slack": diagnostics["min_boundary_slack"],
                "min_pair_slack": diagnostics["min_pair_slack"],
                "num_nearly_active_boundary_constraints": diagnostics["num_nearly_active_boundary_constraints"],
                "num_nearly_active_pair_constraints": diagnostics["num_nearly_active_pair_constraints"],
                "mean_radius": float(np.mean(radii)),
                "max_radius": float(np.max(radii)),
                "min_radius": float(np.min(radii)),
            },
            "evaluator": {
                "score": score,
                "public_details": {
                    "num_circles": EXPECTED_CIRCLES,
                    "strict_safe_score": score,
                    "actual_sum": actual_sum,
                    "reported_sum": reported_sum,
                    "min_slack": float(min(diagnostics["min_boundary_slack"], diagnostics["min_pair_slack"] or 0.0)),
                },
            },
            "extra": {
                "centers": centers,
                "radii": radii,
                "reported_sum": reported_sum,
                "visualization": _visualization_payload(centers, radii, reported_sum=reported_sum),
            },
        }


def create_task() -> CirclePacking26Task:
    return CirclePacking26Task()
