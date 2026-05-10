from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_opt.common.config import get_repo_root


MOVE_NUMBER_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
MOUSE_BUTTONS = {"left", "right", "leftx2", "rightx2"}


@dataclass(frozen=True)
class ReplayResult:
    legal: bool
    complete: bool
    error: str | None
    engine_available: bool
    move_count: int

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "legal": self.legal,
            "complete": self.complete,
            "error": self.error,
            "engine_available": self.engine_available,
            "move_count": self.move_count,
        }


def replay_moves(*, puzzlink_url: str, moves: list[str]) -> ReplayResult:
    format_error = validate_moves(moves)
    if format_error is not None:
        return ReplayResult(
            legal=False,
            complete=False,
            error=format_error,
            engine_available=False,
            move_count=len(moves),
        )
    try:
        Puzzle = _puzzle_class()
    except Exception:
        return ReplayResult(
            legal=True,
            complete=False,
            error=None,
            engine_available=False,
            move_count=len(moves),
        )
    try:
        puzzle = Puzzle.from_url(puzzlink_url)
        for move in moves:
            puzzle.send_move(move)
        complete = _is_complete(puzzle)
    except Exception as exc:
        return ReplayResult(
            legal=False,
            complete=False,
            error=f"{type(exc).__name__}: {exc}",
            engine_available=True,
            move_count=len(moves),
        )
    return ReplayResult(
        legal=True,
        complete=complete,
        error=None,
        engine_available=True,
        move_count=len(moves),
    )


def validate_moves(moves: list[str]) -> str | None:
    for index, move in enumerate(moves):
        if not isinstance(move, str):
            return f"move {index} is not a string"
        if not move.strip():
            return f"move {index} is empty"
        for segment in move.split(";"):
            error = _validate_move_segment(segment.strip(), index=index)
            if error is not None:
                return error
    return None


def _validate_move_segment(segment: str, *, index: int) -> str | None:
    parts = [part.strip() for part in segment.split(",")]
    if not parts or not parts[0]:
        return f"move {index} has no command"
    command = parts[0]
    if command == "key":
        if len(parts) < 2 or any(not part for part in parts[1:]):
            return f"move {index} has malformed key command"
        return None
    if command == "mouse":
        if len(parts) not in {4, 6}:
            return f"move {index} mouse command must have 4 or 6 comma-separated fields"
        if parts[1] not in MOUSE_BUTTONS:
            return f"move {index} uses unsupported mouse button {parts[1]!r}"
        for raw in parts[2:]:
            if not MOVE_NUMBER_RE.fullmatch(raw):
                return f"move {index} has non-numeric coordinate {raw!r}"
        return None
    return f"move {index} uses unsupported command {command!r}"


def _puzzle_class():
    previous_sys_path = list(sys.path)
    try:
        repo_root = get_repo_root()
        repo_src = repo_root / "src"
        cwd = Path.cwd().resolve()
        sanitized: list[str] = []
        for item in previous_sys_path:
            resolved = (Path(item).expanduser().resolve() if item else cwd)
            if resolved in {repo_root, repo_src, cwd}:
                continue
            sanitized.append(item)
        sys.path[:] = sanitized
        try:
            from ppbench.puzzle import Puzzle
        except Exception:
            from ppbench import Puzzle  # type: ignore
        return Puzzle
    finally:
        sys.path[:] = previous_sys_path


def _is_complete(puzzle: Any) -> bool:
    if hasattr(puzzle, "isComplete"):
        return bool(puzzle.isComplete())
    if hasattr(puzzle, "is_complete"):
        return bool(puzzle.is_complete())
    raise AttributeError("PPBench Puzzle object does not expose isComplete/is_complete")
