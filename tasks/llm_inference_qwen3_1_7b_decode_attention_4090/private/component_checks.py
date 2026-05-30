from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .harness import run_decode_suite


def run_component_checks(entry_path: Path) -> dict[str, Any]:
    include_timing = os.environ.get("AO_QWEN3_1_7B_DECODE_ATTENTION_VERIFY_TIMING", "1") != "0"
    return run_decode_suite(
        entry_path,
        profile="public",
        include_timing=include_timing,
        warmup_repeats=3,
        measured_repeats=8,
    )
