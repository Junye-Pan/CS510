from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ErrorConfig:
    atol: float = 2.5e-2
    rtol: float = 2.5e-2
    required_matched_ratio: float = 0.999


DEFAULT_ERROR_CONFIG = ErrorConfig()


def check_tensor_contract(candidate, reference, *, name: str) -> None:
    if candidate.shape != reference.shape:
        raise ValueError(f"{name} shape mismatch: {tuple(candidate.shape)} != {tuple(reference.shape)}")
    if candidate.dtype != reference.dtype:
        raise TypeError(f"{name} dtype mismatch: {candidate.dtype} != {reference.dtype}")
    if candidate.device != reference.device:
        raise ValueError(f"{name} device mismatch: {candidate.device} != {reference.device}")
    if not bool(candidate.isfinite().all().item()):
        raise ValueError(f"{name} contains NaN or Inf")


def compute_error_stats(output, reference, config: ErrorConfig = DEFAULT_ERROR_CONFIG) -> dict[str, Any]:
    x = output.to(dtype=reference.dtype).float()
    y = reference.float()

    abs_error = (x - y).abs()
    total_elements = abs_error.numel()
    if total_elements == 0:
        return {
            "max_abs_error": 0.0,
            "max_rel_error": 0.0,
            "matched_ratio": 1.0,
            "exceeds_tolerance": False,
            "allclose": True,
            "numel": 0,
            "abs_tol": config.atol,
            "rel_tol": config.rtol,
            "required_matched_ratio": config.required_matched_ratio,
        }

    rel_error = abs_error / (y.abs() + 1.0e-8)
    exceeds_tol_mask = (abs_error > config.atol) & (rel_error > config.rtol)
    exceeds_count = float(exceeds_tol_mask.sum().item())
    matched_ratio = 1.0 - exceeds_count / float(total_elements)
    matched_ratio = max(0.0, min(1.0, matched_ratio))
    exceeds_tolerance = matched_ratio < config.required_matched_ratio

    max_abs = float(abs_error.max().item())
    max_rel = float(rel_error.max().item())
    return {
        "max_abs_error": _json_float(max_abs),
        "max_rel_error": _json_float(max_rel),
        "matched_ratio": _json_float(matched_ratio),
        "exceeds_tolerance": bool(exceeds_tolerance),
        "allclose": not bool(exceeds_tolerance),
        "numel": int(total_elements),
        "abs_tol": config.atol,
        "rel_tol": config.rtol,
        "required_matched_ratio": config.required_matched_ratio,
    }


def compare_tensor(output, reference, *, name: str, config: ErrorConfig = DEFAULT_ERROR_CONFIG) -> dict[str, Any]:
    check_tensor_contract(output, reference, name=name)
    return compute_error_stats(output, reference, config=config)


def geometric_mean(values: list[float]) -> float:
    finite = [value for value in values if value > 0 and math.isfinite(value)]
    if len(finite) != len(values) or not finite:
        return 0.0
    return float(math.exp(sum(math.log(value) for value in finite) / len(finite)))


def _json_float(value: float) -> float | str:
    if math.isfinite(value):
        return value
    return str(value)
