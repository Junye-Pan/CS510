from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import time
from typing import Any


MODEL_NAME = "gpt-5.2"
DEFAULT_BASE_URL = "https://chatgpt.com/backend-api/codex"
MAX_BASELINE_MODEL_CALLS = 1
MAX_RETURNED_MOVES = 800
DEFAULT_MOVE_FORMATS = [
    "mouse,left,x,y",
    "mouse,right,x,y",
    "mouse,left,x1,y1,x2,y2",
    "mouse,leftx2,x,y",
    "mouse,rightx2,x,y",
    "key,1",
]
SYSTEM_INSTRUCTIONS = """Solve the supplied PPBench pencil puzzle.
Return a compact JSON object with a replayable final move trace.
Do not browse the web, do not use benchmark answer keys, and do not explain your reasoning.
The only accepted output shape is {"moves":[...],"summary":"short"}."""

MOVE_TOKEN_RE = re.compile(r"(?:mouse|key)\s*,[^\s`\"'{}\[\]]+")
TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
MOVE_PLACEHOLDERS = {"x", "y", "x1", "y1", "x2", "y2"}
MOUSE_BUTTONS = {"left", "right", "leftx2", "rightx2"}
MOVE_KEY_CANDIDATES = ("moves", "move", "answer", "solution", "trace", "steps", "commands", "actions")
SUMMARY_KEY_CANDIDATES = ("summary", "reason", "note", "explanation")


def solve_puzzle(puzzle: dict, budget: dict | None = None) -> dict:
    """Minimum streamed LLM harness for one PPBench puzzle."""

    started = time.perf_counter()
    budget = budget or {}
    phase = str(budget.get("phase") or "unknown")
    allowed_formats = _allowed_move_formats(puzzle)
    primary_button = _primary_mouse_button(allowed_formats)
    max_model_calls = min(int(budget.get("max_model_calls", 1) or 0), MAX_BASELINE_MODEL_CALLS)
    meta = {
        "llm_capable": True,
        "model_name": MODEL_NAME,
        "model_calls": 0,
        "openai_base_url": os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        "phase": phase,
        "harness_style": "minimum_streamed_direct_v1",
        "transport": "pydantic_ai_streamed_responses",
        "used_prompt_context": bool(puzzle.get("prompt_context")),
        "allowed_move_formats": allowed_formats,
        "primary_mouse_button": primary_button,
    }

    if max_model_calls <= 0:
        return _result(
            status="failed",
            moves=[],
            meta={**meta, "reason": "model-call budget is zero"},
            summary="No model call was made because the host budget disabled it.",
        )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return _result(
            status="failed",
            moves=[],
            meta={**meta, "reason": "OPENAI_API_KEY is not set"},
            summary="No configured Codex backend API key.",
        )

    timeout_s = _request_timeout_s(started=started, budget=budget)
    if timeout_s < 30.0:
        return _result(
            status="timeout",
            moves=[],
            meta={**meta, "reason": "insufficient request budget", "request_timeout_s": timeout_s},
            summary="Insufficient request budget.",
        )

    prompt = _build_prompt(
        puzzle=puzzle,
        budget=budget,
        allowed_formats=allowed_formats,
        primary_button=primary_button,
    )
    effort = str(budget.get("preferred_reasoning_effort") or "high")
    raw_text = ""
    errors: list[str] = []
    try:
        raw_text = _call_streamed_model(
            prompt=prompt,
            api_key=api_key,
            timeout_s=timeout_s,
            reasoning_effort=effort,
        )
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")

    moves = _extract_moves(raw_text, allowed_formats, primary_button) if raw_text else []
    summary = _extract_summary(raw_text) or ("Returned model-proposed moves." if moves else "Model returned no moves.")
    return _result(
        status="solved" if moves else "failed",
        moves=moves,
        meta={
            **meta,
            "model_calls": 1,
            "reasoning_effort": effort,
            "request_timeout_s": timeout_s,
            "raw_response_chars": len(raw_text),
            "raw_response_preview": raw_text[:240],
            "parsed_move_count": len(moves),
            "errors": errors[:3],
        },
        summary=summary,
    )


def _result(*, status: str, moves: list[str], meta: dict[str, Any], summary: str) -> dict[str, Any]:
    return {
        "status": status,
        "moves": moves[:MAX_RETURNED_MOVES],
        "meta": meta,
        "summary": summary[:240],
    }


def _build_prompt(
    *,
    puzzle: dict,
    budget: dict,
    allowed_formats: list[str],
    primary_button: str | None,
) -> str:
    context = puzzle.get("prompt_context")
    if not isinstance(context, dict):
        context = {}
    examples = context.get("input_examples")
    if not isinstance(examples, list):
        examples = []
    move_hint = f"mouse,{primary_button},x,y" if primary_button else "mouse,left,x,y"
    public_record = {
        "puzzle_type": puzzle.get("puzzle_type"),
        "width": puzzle.get("width"),
        "height": puzzle.get("height"),
        "puzzlink_url": puzzle.get("puzzlink_url"),
        "encoded_puzzle": context.get("encoded_puzzle"),
    }
    lines = [
        f"Puzzle type: {puzzle.get('puzzle_type') or 'unknown'}",
        "",
        "Return exactly one JSON object and nothing else.",
        f'Preferred schema: {{"moves":["{move_hint}"],"summary":"short"}}',
        f"Allowed replay move shapes: {json.dumps(allowed_formats, ensure_ascii=True)}",
        "Use PPBench pzpr coordinates exactly as described below.",
        "If the model is unsure, still produce the best concrete final trace rather than an empty answer.",
        "",
        "Public puzzle record:",
        json.dumps(public_record, ensure_ascii=True, sort_keys=True),
        "",
        "Rules:",
        _clip_text(context.get("rules_text"), 5000) or "(none)",
        "",
        "Coordinate notes:",
        _clip_text(context.get("coordinate_notes"), 1600) or "(none)",
        "",
        "Move examples for this puzzle family:",
        json.dumps([_clip_jsonish(item, 900) for item in examples[:6]], ensure_ascii=True, sort_keys=True),
        "",
        "Benchmark prompt/context:",
        _clip_text(context.get("official_prompt"), 2500) or "(none)",
        "",
        "Board representation:",
        _clip_text(context.get("board_text"), 9000) or "(none)",
        "",
        "Budget:",
        json.dumps(
            {
                "phase": budget.get("phase"),
                "wall_clock_s": budget.get("wall_clock_s"),
                "request_timeout_s": budget.get("request_timeout_s"),
            },
            ensure_ascii=True,
            sort_keys=True,
        ),
        "",
        "Answer JSON:",
    ]
    return "\n".join(lines)


def _call_streamed_model(
    *,
    prompt: str,
    api_key: str,
    timeout_s: float,
    reasoning_effort: str,
) -> str:
    return asyncio.run(
        _call_streamed_model_async(
            prompt=prompt,
            api_key=api_key,
            timeout_s=timeout_s,
            reasoning_effort=reasoning_effort,
        )
    )


async def _call_streamed_model_async(
    *,
    prompt: str,
    api_key: str,
    timeout_s: float,
    reasoning_effort: str,
) -> str:
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
    from pydantic_ai.providers.openai import AsyncOpenAI, OpenAIProvider

    base_url = os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=5)
    model = OpenAIResponsesModel(
        MODEL_NAME,
        provider=OpenAIProvider(openai_client=client),
        settings=OpenAIResponsesModelSettings(
            openai_reasoning_effort=reasoning_effort,
            openai_store=False,
            timeout=timeout_s,
        ),
    )
    agent = Agent(model, instructions=SYSTEM_INSTRUCTIONS, retries=1)

    async with asyncio.timeout(timeout_s):
        async with agent.iter(prompt) as run:
            async for node in run:
                if node.__class__.__name__ == "ModelRequestNode":
                    async with node.stream(run.ctx) as stream:
                        async for _ in stream:
                            pass
        result = run.result
    output = getattr(result, "output", "")
    return output if isinstance(output, str) else str(output)


def _request_timeout_s(*, started: float, budget: dict) -> float:
    wall_clock_s = float(budget.get("wall_clock_s") or 900.0)
    configured = budget.get("request_timeout_s")
    if configured is None:
        configured = wall_clock_s - 15.0
    remaining = wall_clock_s - (time.perf_counter() - started) - 10.0
    return max(1.0, min(float(configured), remaining))


def _extract_moves(text: str, allowed_formats: list[str], primary_button: str | None) -> list[str]:
    best: list[str] = []
    for candidate in _json_candidates(text):
        data = _loads_relaxed_json(candidate)
        if isinstance(data, dict):
            for key in MOVE_KEY_CANDIDATES:
                if key in data:
                    moves = _sanitize_moves(data[key], allowed_formats, primary_button)
                    if len(moves) > len(best):
                        best = moves
            if not best:
                moves = _sanitize_moves(data, allowed_formats, primary_button)
                if len(moves) > len(best):
                    best = moves
        elif isinstance(data, list):
            moves = _sanitize_moves(data, allowed_formats, primary_button)
            if len(moves) > len(best):
                best = moves
    if not best:
        best = _sanitize_moves(MOVE_TOKEN_RE.findall(text), allowed_formats, primary_button)
    return best[:MAX_RETURNED_MOVES]


def _extract_summary(text: str) -> str:
    for candidate in _json_candidates(text):
        data = _loads_relaxed_json(candidate)
        if not isinstance(data, dict):
            continue
        for key in SUMMARY_KEY_CANDIDATES:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _json_candidates(text: str) -> list[str]:
    stripped = text.strip()
    candidates: list[str] = []
    if stripped:
        candidates.append(stripped)
        fence = _strip_code_fence(stripped)
        if fence != stripped:
            candidates.append(fence)
    starts = [index for index, char in enumerate(text) if char in "[{"]
    for start in starts[:64]:
        candidate = _balanced_json_slice(text, start)
        if candidate:
            candidates.append(candidate)
    return candidates


def _balanced_json_slice(text: str, start: int) -> str | None:
    opening = text[start]
    closing = "}" if opening == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _loads_relaxed_json(candidate: str) -> Any:
    normalized = candidate.strip().strip(";")
    normalized = (
        normalized.replace("\\u201c", '"')
        .replace("\\u201d", '"')
        .replace("\\u2018", "'")
        .replace("\\u2019", "'")
    )
    for attempt in (normalized, TRAILING_COMMA_RE.sub(r"\1", normalized)):
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(attempt)
            except Exception:
                continue
    return None


def _strip_code_fence(text: str) -> str:
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def _sanitize_moves(raw_moves: Any, allowed_formats: list[str], primary_button: str | None) -> list[str]:
    specs = [_format_spec(item) for item in allowed_formats]
    sanitized: list[str] = []
    for raw_move in _flatten_move_candidates(raw_moves, primary_button):
        move = _normalize_move(raw_move)
        if move and _is_legal_move(move, specs):
            sanitized.append(move)
        if len(sanitized) >= MAX_RETURNED_MOVES:
            break
    return sanitized


def _flatten_move_candidates(raw_moves: Any, primary_button: str | None) -> list[str]:
    if isinstance(raw_moves, str):
        return _split_move_text(raw_moves)
    if isinstance(raw_moves, dict):
        direct = _dict_to_move(raw_moves, primary_button)
        if direct:
            return [direct]
        flattened: list[str] = []
        for key in MOVE_KEY_CANDIDATES:
            if key in raw_moves:
                flattened.extend(_flatten_move_candidates(raw_moves[key], primary_button))
        return flattened
    if isinstance(raw_moves, tuple):
        raw_moves = list(raw_moves)
    if not isinstance(raw_moves, list):
        return []
    direct = _sequence_to_move(raw_moves, primary_button)
    if direct:
        return [direct]
    flattened: list[str] = []
    for item in raw_moves:
        flattened.extend(_flatten_move_candidates(item, primary_button))
    return flattened


def _split_move_text(raw_text: str) -> list[str]:
    extracted = MOVE_TOKEN_RE.findall(raw_text)
    if extracted:
        return [_repair_move_text(item) for item in extracted]
    moves: list[str] = []
    for chunk in re.split(r"[\n;|]+", raw_text):
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", chunk.strip())
        if cleaned:
            moves.append(_repair_move_text(cleaned))
    return [move for move in moves if move]


def _dict_to_move(raw_move: dict[str, Any], primary_button: str | None) -> str | None:
    kind = str(raw_move.get("kind") or raw_move.get("type") or raw_move.get("action") or "").strip().lower()
    if kind == "key" or "key" in raw_move:
        key = raw_move.get("key") if raw_move.get("key") is not None else raw_move.get("value")
        return f"key,{key}" if key is not None else None
    button = str(raw_move.get("button") or raw_move.get("side") or primary_button or "left").strip().lower()
    coords = [raw_move.get(key) for key in ("x", "y", "x1", "y1", "x2", "y2") if raw_move.get(key) is not None]
    if len(coords) >= 4:
        return "mouse," + ",".join([button, *(_format_number(item) for item in coords[:4])])
    if len(coords) >= 2:
        return "mouse," + ",".join([button, *(_format_number(item) for item in coords[:2])])
    return None


def _sequence_to_move(raw: list[Any], primary_button: str | None) -> str | None:
    if not raw:
        return None
    if len(raw) >= 2 and str(raw[0]).lower() == "key":
        return f"key,{raw[1]}"
    if len(raw) >= 3 and str(raw[0]).lower() == "mouse":
        return "mouse," + ",".join(str(item).strip() for item in raw[1:])
    if all(_is_number(item) for item in raw):
        button = primary_button or "left"
        if len(raw) >= 4:
            return "mouse," + ",".join([button, *(_format_number(item) for item in raw[:4])])
        if len(raw) >= 2:
            return "mouse," + ",".join([button, *(_format_number(item) for item in raw[:2])])
    return None


def _repair_move_text(value: str) -> str:
    return ",".join(part.strip() for part in str(value).strip().strip("`\"'").split(",") if part.strip())


def _normalize_move(move: str) -> str | None:
    repaired = _repair_move_text(move)
    parts = [part.strip() for part in repaired.split(",")]
    if not parts:
        return None
    if parts[0].lower() == "mouse" and len(parts) >= 4:
        button = parts[1].lower()
        if button not in MOUSE_BUTTONS:
            return None
        coords = [_format_number(part) for part in parts[2:]]
        return ",".join(["mouse", button, *coords])
    if parts[0].lower() == "key" and len(parts) >= 2:
        return f"key,{parts[1]}"
    return None


def _format_spec(move_format: str) -> tuple[str, ...]:
    return tuple(part.strip().lower() for part in move_format.split(",") if part.strip())


def _is_legal_move(move: str, specs: list[tuple[str, ...]]) -> bool:
    parts = tuple(part.strip().lower() for part in move.split(",") if part.strip())
    for spec in specs:
        if len(parts) != len(spec):
            continue
        if all(_part_matches(value, expected) for value, expected in zip(parts, spec)):
            return True
    return False


def _part_matches(value: str, expected: str) -> bool:
    if expected in MOVE_PLACEHOLDERS:
        return _is_number(value)
    return value == expected


def _allowed_move_formats(puzzle: dict) -> list[str]:
    context = puzzle.get("prompt_context")
    formats = context.get("move_format") if isinstance(context, dict) else None
    if not isinstance(formats, list):
        formats = DEFAULT_MOVE_FORMATS
    cleaned = [str(item) for item in formats if isinstance(item, str) and item.strip()]
    return cleaned or DEFAULT_MOVE_FORMATS


def _primary_mouse_button(allowed_formats: list[str]) -> str | None:
    for move_format in allowed_formats:
        parts = [part.strip().lower() for part in move_format.split(",")]
        if len(parts) >= 2 and parts[0] == "mouse" and parts[1] in MOUSE_BUTTONS:
            return parts[1]
    return None


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except Exception:
        return False
    return True


def _format_number(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return str(value).strip()
    if abs(number - round(number)) < 1e-6:
        return str(int(round(number)))
    return f"{number:.6g}"


def _clip_text(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else ""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def _clip_jsonish(value: Any, limit: int) -> Any:
    text = json.dumps(value, ensure_ascii=True, sort_keys=True)
    if len(text) <= limit:
        return value
    return text[:limit] + "...[truncated]"
