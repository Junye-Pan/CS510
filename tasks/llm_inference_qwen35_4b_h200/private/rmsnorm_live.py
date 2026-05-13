from __future__ import annotations

import hashlib
import importlib.util
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from .schema import BundleManifest, ImplementationSpec
from .run_artifacts import write_json


RMSNORM_DEFINITION = "qwen_rmsnorm_h2560_fp16"
DEFAULT_ATOL = 2e-2
DEFAULT_RTOL = 2e-2
DEFAULT_WARMUP = 8
DEFAULT_ITERS = 40


def run_rmsnorm_live_checks(
    *,
    entry_path: Path,
    manifest: BundleManifest,
    shapes: list[dict[str, int]],
    benchmark: bool,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return {"valid": False, "error": "torch.cuda is not available", "implementations": []}

    candidate_root = entry_path.resolve().parent
    implementations = [
        implementation
        for implementation in manifest.implementations
        if implementation.definition == RMSNORM_DEFINITION
    ]
    results: list[dict[str, Any]] = []
    for implementation in implementations:
        results.append(
            _check_implementation(
                implementation=implementation,
                candidate_root=candidate_root,
                shapes=shapes,
                benchmark=benchmark,
            )
        )
    valid = all(item.get("valid") for item in results)
    best_speedup = 1.0
    speedups: list[float] = []
    for item in results:
        for shape_result in item.get("shape_results") or []:
            speedup = shape_result.get("speedup")
            if isinstance(speedup, (int, float)) and math.isfinite(float(speedup)) and speedup > 0:
                speedups.append(float(speedup))
    if speedups:
        best_speedup = _geomean(speedups)
    payload = {
        "valid": valid,
        "error": None if valid else _first_error(results),
        "definition": RMSNORM_DEFINITION,
        "benchmark": benchmark,
        "geomean_speedup": best_speedup,
        "implementations": results,
    }
    if run_dir is not None:
        write_json(run_dir / ("rmsnorm_benchmark.json" if benchmark else "rmsnorm_correctness.json"), payload)
    return payload


def _check_implementation(
    *,
    implementation: ImplementationSpec,
    candidate_root: Path,
    shapes: list[dict[str, int]],
    benchmark: bool,
) -> dict[str, Any]:
    import torch

    module = _load_module(candidate_root / implementation.entry_path)
    run = getattr(module, implementation.entry_symbol)
    shape_results: list[dict[str, Any]] = []
    for shape in shapes:
        if not _guard_covers(implementation, shape):
            shape_results.append({"shape": shape, "valid": True, "covered": False})
            continue
        torch.manual_seed(1000 + int(shape["num_tokens"]))
        hidden_states = torch.randn(
            (shape["num_tokens"], shape["hidden"]),
            device="cuda",
            dtype=torch.float16,
        )
        weight = torch.randn((shape["hidden"],), device="cuda", dtype=torch.float16)
        reference = _reference_rmsnorm(hidden_states, weight)
        output = torch.empty_like(hidden_states)

        try:
            with torch.no_grad():
                run(hidden_states, weight, output)
            torch.cuda.synchronize()
        except Exception as exc:
            shape_results.append({"shape": shape, "valid": False, "covered": True, "error": str(exc)})
            continue

        valid, error_stats = _check_output(output, reference)
        shape_payload: dict[str, Any] = {
            "shape": shape,
            "valid": valid,
            "covered": True,
            **error_stats,
        }
        if valid and benchmark:
            baseline_ms = _time_call(lambda: _reference_rmsnorm(hidden_states, weight), warmup=DEFAULT_WARMUP, iters=DEFAULT_ITERS)
            candidate_ms = _time_call(lambda: run(hidden_states, weight, output), warmup=DEFAULT_WARMUP, iters=DEFAULT_ITERS)
            shape_payload.update(
                {
                    "baseline_ms": baseline_ms,
                    "candidate_ms": candidate_ms,
                    "speedup": baseline_ms / candidate_ms if candidate_ms > 0.0 else 0.0,
                }
            )
        shape_results.append(shape_payload)

    any_covered = any(item.get("covered") for item in shape_results)
    valid = any_covered and all(item.get("valid") for item in shape_results)
    if not any_covered:
        return {
            "id": implementation.id,
            "valid": False,
            "error": "shape_guard does not cover any live RMSNorm workload shape",
            "shape_results": shape_results,
        }
    return {
        "id": implementation.id,
        "valid": valid,
        "error": None if valid else _first_error(shape_results),
        "shape_results": shape_results,
    }


def _reference_rmsnorm(hidden_states: Any, weight: Any) -> Any:
    import torch

    variance = hidden_states.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    output = hidden_states.to(torch.float32) * torch.rsqrt(variance + 1e-6)
    return (output * weight.to(torch.float32)).to(hidden_states.dtype)


def _check_output(output: Any, reference: Any) -> tuple[bool, dict[str, Any]]:
    import torch

    if tuple(output.shape) != tuple(reference.shape):
        return False, {"error": f"shape mismatch: got {tuple(output.shape)}, expected {tuple(reference.shape)}"}
    if output.dtype != reference.dtype:
        return False, {"error": f"dtype mismatch: got {output.dtype}, expected {reference.dtype}"}
    if torch.isinf(output).any().item() or torch.isnan(output).any().item():
        return False, {"error": "output contains NaN or Inf"}
    abs_error = torch.abs(output.to(torch.float32) - reference.to(torch.float32))
    rel_error = abs_error / (torch.abs(reference.to(torch.float32)) + 1e-8)
    max_abs = float(abs_error.max().item())
    max_rel = float(rel_error.max().item())
    allclose = bool(torch.allclose(output, reference, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL))
    return allclose, {
        "max_absolute_error": max_abs,
        "max_relative_error": max_rel,
        "error": None if allclose else f"allclose failed: max_abs={max_abs:.6g}, max_rel={max_rel:.6g}",
    }


def _time_call(fn: Any, *, warmup: int, iters: int) -> float:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(3):
        started = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0 / iters)
    return float(statistics.median(samples))


def _load_module(path: Path) -> Any:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    module_name = f"ao_llm_kernel_candidate_{digest}"
    previous = sys.modules.pop(module_name, None)
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not import {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception:
        sys.modules.pop(module_name, None)
        if previous is not None:
            sys.modules[module_name] = previous
        raise


def _guard_covers(implementation: ImplementationSpec, shape: dict[str, int]) -> bool:
    guard = implementation.shape_guard
    token_guard = guard.get("num_tokens")
    hidden_guard = guard.get("hidden")
    if hidden_guard != shape["hidden"]:
        return False
    if not isinstance(token_guard, list) or len(token_guard) != 2:
        return False
    return int(token_guard[0]) <= shape["num_tokens"] <= int(token_guard[1])


def _first_error(items: list[dict[str, Any]]) -> str | None:
    for item in items:
        error = item.get("error")
        if error:
            return str(error)
    return None


def _geomean(values: list[float]) -> float:
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        return 0.0
    return float(math.exp(sum(math.log(value) for value in values) / len(values)))
