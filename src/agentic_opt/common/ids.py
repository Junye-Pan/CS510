from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4


def isoformat_z() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def format_counter_id(prefix: str, value: int, *, width: int = 6) -> str:
    return f"{prefix}_{value:0{width}d}"


def _timestamp_token() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def make_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{_timestamp_token()}_{uuid4().hex[:8]}"


def make_session_id(prefix: str = "session") -> str:
    return f"{prefix}_{_timestamp_token()}_{uuid4().hex[:8]}"


def make_event_id(prefix: str = "event") -> str:
    return f"{prefix}_{_timestamp_token()}_{uuid4().hex[:8]}"
