from __future__ import annotations

import math
import os
from typing import Any


MAX_FALLBACK_RATE_ENV = "AO_LLM_KERNEL_MAX_FALLBACK_RATE"
MIN_CANDIDATE_CALLS_ENV = "AO_LLM_KERNEL_MIN_CANDIDATE_CALLS"
DEFAULT_MAX_FALLBACK_RATE = 0.50
DEFAULT_MIN_CANDIDATE_CALLS = 1


def fallback_thresholds() -> dict[str, Any]:
    return {
        "max_fallback_rate": _float_env(MAX_FALLBACK_RATE_ENV, DEFAULT_MAX_FALLBACK_RATE),
        "min_candidate_calls": _int_env(MIN_CANDIDATE_CALLS_ENV, DEFAULT_MIN_CANDIDATE_CALLS),
    }


def evaluate_apply_trace_policy(
    apply_summary: dict[str, Any] | None,
    *,
    candidate_required: bool,
) -> dict[str, Any]:
    thresholds = fallback_thresholds()
    if apply_summary is None:
        return {
            "valid": not candidate_required,
            "error": "candidate apply trace is missing" if candidate_required else None,
            "thresholds": thresholds,
            "candidate_calls": 0,
            "fallback_calls": 0,
            "fallback_rate": 1.0 if candidate_required else 0.0,
            "fallback_reasons": {},
        }
    candidate_calls = int(apply_summary.get("candidate_calls") or 0)
    fallback_calls = int(apply_summary.get("fallback_calls") or 0)
    total = candidate_calls + fallback_calls
    fallback_rate = float(fallback_calls / total) if total else (1.0 if candidate_required else 0.0)
    checks = [
        {
            "name": "candidate_calls",
            "status": "passed" if candidate_calls >= int(thresholds["min_candidate_calls"]) else "failed",
            "message": None
            if candidate_calls >= int(thresholds["min_candidate_calls"])
            else (
                f"candidate calls {candidate_calls} below minimum "
                f"{int(thresholds['min_candidate_calls'])}"
            ),
        },
        {
            "name": "fallback_rate",
            "status": "passed" if fallback_rate <= float(thresholds["max_fallback_rate"]) else "failed",
            "message": None
            if fallback_rate <= float(thresholds["max_fallback_rate"])
            else (
                f"fallback rate {fallback_rate:.3f} exceeds maximum "
                f"{float(thresholds['max_fallback_rate']):.3f}"
            ),
        },
    ]
    if not candidate_required:
        checks[0]["status"] = "skipped"
        checks[0]["message"] = None
    valid = all(check["status"] in {"passed", "skipped"} for check in checks)
    return {
        "valid": valid,
        "error": None if valid else _first_failed_message(checks),
        "thresholds": thresholds,
        "candidate_calls": candidate_calls,
        "fallback_calls": fallback_calls,
        "fallback_rate": fallback_rate,
        "fallback_reasons": apply_summary.get("fallback_reasons") or {},
        "classes": apply_summary.get("classes") or {},
        "candidate_shapes": apply_summary.get("candidate_shapes") or [],
        "checks": checks,
    }


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if not math.isfinite(value) or value < 0.0:
        return default
    return value


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


def _first_failed_message(checks: list[dict[str, Any]]) -> str | None:
    for check in checks:
        if check.get("status") == "failed":
            return str(check.get("message") or check.get("name"))
    return None
