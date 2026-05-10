from __future__ import annotations

import json
import os
import re
import sys
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any


ANSWER_FIELD_FRAGMENTS = (
    "answer",
    "solution",
    "moves_full",
    "moves_required",
    "moves_hint",
)

FALLBACK_RULES = {
    "hitori": (
        "Shade cells so that no unshaded number repeats in any row or column. "
        "Shaded cells cannot be orthogonally adjacent, and all unshaded cells "
        "must remain orthogonally connected."
    ),
    "kurodoko": (
        "Shade some cells. Numbered cells stay unshaded and indicate how many "
        "unshaded cells are visible horizontally and vertically, including the "
        "numbered cell, until blocked by a shaded cell or the grid edge. Shaded "
        "cells cannot be orthogonally adjacent, and all unshaded cells are connected."
    ),
    "lits": (
        "Shade exactly four orthogonally connected cells in each region to form "
        "an L, I, T, or S tetromino. All shaded cells form one connected group, "
        "no 2x2 block may be fully shaded, and identical tetromino shapes may "
        "not touch orthogonally across a region border."
    ),
    "norinori": (
        "Shade cells so that each outlined region contains exactly two shaded "
        "cells. Every shaded cell must be orthogonally adjacent to exactly one "
        "other shaded cell, forming shaded domino pairs."
    ),
    "shikaku": (
        "Divide the grid into rectangles. Each rectangle must contain exactly "
        "one clue number, and that number must equal the rectangle's area."
    ),
    "sudoku": (
        "Fill the grid with digits so that every row, every column, and every "
        "marked box contains each required digit exactly once."
    ),
    "tapa": (
        "Shade cells so all shaded cells are connected and no 2x2 block is fully "
        "shaded. Clue cells stay unshaded. Numbers in a clue cell give the "
        "lengths of contiguous shaded runs among its eight neighboring cells, "
        "in cyclic order, with clue order not significant."
    ),
}


class PPBenchDependencyError(RuntimeError):
    pass


def dependency_status() -> dict[str, Any]:
    local_status = _local_bundle_status()
    if local_status["available"]:
        return local_status
    try:
        _load_dataset_function()
    except PPBenchDependencyError as exc:
        return {
            "available": False,
            "source": None,
            "error": str(exc),
        }
    return {
        "available": True,
        "source": "ppbench",
        "error": None,
    }


def load_public_records(*, split: str, limit: int | None = None) -> list[dict[str, Any]]:
    local = _local_bundle_for_split(split)
    if local is not None:
        records = _load_local_bundle_records(local)
        if limit is not None:
            records = records[:limit]
        return records

    records = _load_golden300_records()
    indices = _split_indices(split)
    if limit is not None:
        indices = indices[:limit]
    selected: list[dict[str, Any]] = []
    for index in indices:
        if index >= len(records):
            raise IndexError(f"split index {index} outside golden_300 length {len(records)}")
        selected.append(sanitize_record(records[index], index=index))
    return selected


def smoke_records() -> list[dict[str, Any]]:
    limit = _env_limit("AO_PPBENCH_VERIFY_LIMIT", default=3)
    try:
        return load_public_records(split="probe10", limit=limit)
    except PPBenchDependencyError:
        return [
            _record_with_prompt_context(
                {
                    "puzzle_id": "smoke_missing_ppbench_000",
                    "puzzle_type": "unknown",
                    "puzzlink_url": "ppbench-unavailable://smoke/0",
                    "width": None,
                    "height": None,
                    "metadata": {
                        "purpose": "contract-only smoke puzzle used when ppbench is not installed",
                    },
                }
            ),
            _record_with_prompt_context(
                {
                    "puzzle_id": "smoke_missing_ppbench_001",
                    "puzzle_type": "unknown",
                    "puzzlink_url": "ppbench-unavailable://smoke/1",
                    "width": None,
                    "height": None,
                    "metadata": {
                        "purpose": "contract-only smoke puzzle used when ppbench is not installed",
                    },
                }
            ),
        ]


def sanitize_record(record: dict[str, Any], *, index: int) -> dict[str, Any]:
    stripped = _strip_answer_fields(deepcopy(record))
    puzzlink_url = stripped.get("puzzlink_url") or stripped.get("puzzle_url") or stripped.get("url")
    if not isinstance(puzzlink_url, str) or not puzzlink_url:
        raise ValueError(f"PPBench record {index} does not expose a puzzlink_url")
    puzzle_type = stripped.get("pid") or stripped.get("puzzle_type") or stripped.get("type") or "unknown"
    metadata = stripped.get("metadata") if isinstance(stripped.get("metadata"), dict) else {}
    return _record_with_prompt_context(
        {
            "puzzle_id": f"golden300_{index:03d}",
            "puzzle_type": str(puzzle_type),
            "puzzlink_url": puzzlink_url,
            "width": _optional_int(stripped.get("width") or stripped.get("w")),
            "height": _optional_int(stripped.get("height") or stripped.get("h")),
            "metadata": {
                **{
                    key: value
                    for key, value in stripped.items()
                    if key
                    not in {
                        "puzzlink_url",
                        "puzzle_url",
                        "url",
                        "pid",
                        "puzzle_type",
                        "type",
                        "width",
                        "height",
                        "w",
                        "h",
                        "metadata",
                    }
                    and _is_public_scalar(value)
                },
                **metadata,
            },
        },
        pzpr_url=_optional_string(stripped.get("pzpr_url")),
    )


def _load_local_bundle_records(bundle_root: Path) -> list[dict[str, Any]]:
    public_dir = bundle_root / "public"
    puzzles_path = public_dir / "puzzles.json"
    if not puzzles_path.exists():
        raise PPBenchDependencyError(f"local PPBench bundle is missing {puzzles_path}")
    payload = json.loads(puzzles_path.read_text(encoding="utf-8"))
    raw_records = list(payload["puzzles"])
    return [_sanitize_local_record(record, bundle_root=bundle_root) for record in raw_records]


def _sanitize_local_record(record: dict[str, Any], *, bundle_root: Path) -> dict[str, Any]:
    stripped = _strip_answer_fields(deepcopy(record))
    puzzle_id = str(stripped.get("id") or "")
    if not puzzle_id:
        raise ValueError(f"local PPBench record in {bundle_root.name} lacks id")
    puzzlink_url = stripped.get("puzzlink_url") or stripped.get("pzpr_url")
    if not isinstance(puzzlink_url, str) or not puzzlink_url:
        raise ValueError(f"local PPBench record {puzzle_id} lacks puzzlink_url")
    rank = stripped.get("probe_rank") or stripped.get("subset_rank")
    prompt_markdown = _read_local_prompt(bundle_root=bundle_root, record=stripped)
    metadata = {
        "bundle_id": bundle_root.name,
        "rank": _optional_int(rank),
        "size": stripped.get("size"),
        "pzpr_url": stripped.get("pzpr_url"),
        "ppbench_play_url": stripped.get("ppbench_play_url"),
        "local_play_path": stripped.get("local_play_path"),
        "solve_rate_percent": stripped.get("solve_rate_percent"),
        "difficulty_bin": stripped.get("difficulty_bin"),
        "golden_300_line": stripped.get("golden_300_line"),
        "selection_role": stripped.get("selection_role"),
        "selection_detail": stripped.get("selection_detail"),
    }
    if prompt_markdown is not None:
        metadata["prompt_markdown"] = prompt_markdown
    return _record_with_prompt_context(
        {
            "puzzle_id": f"{bundle_root.name}:{puzzle_id}",
            "puzzle_type": str(stripped.get("type") or stripped.get("pid") or "unknown"),
            "puzzlink_url": puzzlink_url,
            "width": _optional_int(stripped.get("width")),
            "height": _optional_int(stripped.get("height")),
            "metadata": {key: value for key, value in metadata.items() if _is_public_scalar(value)},
        },
        prompt_markdown=prompt_markdown,
        pzpr_url=_optional_string(stripped.get("pzpr_url")),
    )


def _read_local_prompt(*, bundle_root: Path, record: dict[str, Any]) -> str | None:
    rank = _optional_int(record.get("probe_rank") or record.get("subset_rank"))
    puzzle_id = record.get("id")
    if rank is None or not isinstance(puzzle_id, str):
        return None
    path = bundle_root / "public" / "prompts" / f"{rank:03d}_{puzzle_id}.md"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _record_with_prompt_context(
    record: dict[str, Any],
    *,
    prompt_markdown: str | None = None,
    pzpr_url: str | None = None,
) -> dict[str, Any]:
    public_record = dict(record)
    public_record["prompt_context"] = _build_public_prompt_context(
        public_record,
        prompt_markdown=prompt_markdown,
        pzpr_url=pzpr_url,
    )
    return public_record


def _build_public_prompt_context(
    record: dict[str, Any],
    *,
    prompt_markdown: str | None,
    pzpr_url: str | None,
) -> dict[str, Any]:
    puzzle_type = str(record.get("puzzle_type") or "unknown")
    puzzlink_url = str(record.get("puzzlink_url") or "")
    encoded_url = pzpr_url or _encoded_pzpr_url(puzzlink_url)
    rules = _rules_for_type(puzzle_type)
    examples = _examples_for_type(puzzle_type)
    return {
        "kind": "ppbench_public_prompt_context",
        "goal": (
            "Engineer a reusable LLM-based harness that solves the given puzzle "
            "from public puzzle context and returns PPBench replay moves."
        ),
        "puzzle_type": puzzle_type,
        "size": {
            "width": record.get("width"),
            "height": record.get("height"),
        },
        "puzzlink_url": puzzlink_url,
        "pzpr_url": encoded_url,
        "encoded_puzzle": _encoded_payload(encoded_url),
        "board_text": _board_text(record=record, pzpr_url=encoded_url),
        "rules_text": rules,
        "move_format": [
            "mouse,left,x,y",
            "mouse,right,x,y",
            "mouse,left,x1,y1,x2,y2",
            "mouse,leftx2,x,y",
            "mouse,rightx2,x,y",
            "key,1",
        ],
        "coordinate_notes": (
            "PPBench uses pzpr replay coordinates. Cell centers are odd integer "
            "coordinates starting at x=1,y=1 and increasing by 2 for each cell. "
            "Grid lines, edges, and vertices use adjacent even/odd coordinates; "
            "drag moves may include a path of pzpr coordinates."
        ),
        "input_examples": examples,
        "official_prompt": prompt_markdown,
        "privacy_notes": (
            "This context is public-safe and contains no decoded solution moves, "
            "answer key, or private split membership."
        ),
    }


def _board_text(*, record: dict[str, Any], pzpr_url: str | None) -> str:
    lines = [
        "PPBench puzzle record",
        f"type: {record.get('puzzle_type')}",
        f"width: {record.get('width')}",
        f"height: {record.get('height')}",
        f"puzzlink_url: {record.get('puzzlink_url')}",
    ]
    if pzpr_url:
        lines.append(f"pzpr_url: {pzpr_url}")
    payload = _encoded_payload(pzpr_url)
    if payload:
        lines.append(f"encoded_payload: {payload}")
    board_repr = _best_effort_pzpr_board_text(str(record.get("puzzlink_url") or ""))
    if board_repr:
        lines.extend(["puzzle_string_repr:", board_repr])
    return "\n".join(lines)


def _encoded_pzpr_url(puzzlink_url: str) -> str | None:
    if not puzzlink_url:
        return None
    if "?" in puzzlink_url:
        return puzzlink_url.split("?", 1)[1]
    if "://" not in puzzlink_url:
        return puzzlink_url
    return None


def _encoded_payload(pzpr_url: str | None) -> str | None:
    if not pzpr_url:
        return None
    parts = pzpr_url.split("/")
    if len(parts) <= 3:
        return None
    return "/".join(parts[3:])


def _best_effort_pzpr_board_text(puzzlink_url: str) -> str | None:
    if not puzzlink_url or puzzlink_url.startswith("ppbench-unavailable:"):
        return None
    previous_sys_path = list(sys.path)
    try:
        try:
            repo_root = _repo_root()
            repo_src = repo_root / "src"
            cwd = Path.cwd().resolve()
            sys.path[:] = [
                item
                for item in previous_sys_path
                if (Path(item).expanduser().resolve() if item else cwd) not in {repo_root, repo_src, cwd}
            ]
            try:
                from ppbench.puzzle import Puzzle
            except Exception:
                from ppbench import Puzzle  # type: ignore
        except Exception:
            return None
        try:
            puzzle = Puzzle.from_url(puzzlink_url)
            for method_name in ("get_string_repr", "getFileData", "get_file_data"):
                method = getattr(puzzle, method_name, None)
                if callable(method):
                    value = method()
                    if isinstance(value, str) and value.strip():
                        return value
        except Exception:
            return None
        return None
    finally:
        sys.path[:] = previous_sys_path


@lru_cache(maxsize=128)
def _rules_for_type(puzzle_type: str) -> str | None:
    puzzle_type = puzzle_type.strip().lower()
    sample_text = _sample_js_for_type(puzzle_type)
    if sample_text is None:
        return FALLBACK_RULES.get(puzzle_type)
    match = re.search(
        r"addRules\(\s*['\"]"
        + re.escape(puzzle_type)
        + r"['\"]\s*,\s*\[\s*\{\s*rules\s*:\s*\"((?:\\.|[^\"\\])*)\"",
        sample_text,
        flags=re.DOTALL,
    )
    if match is None:
        return FALLBACK_RULES.get(puzzle_type)
    return _decode_js_string(match.group(1))


@lru_cache(maxsize=128)
def _examples_for_type(puzzle_type: str) -> list[dict[str, Any]]:
    sample_text = _sample_js_for_type(puzzle_type)
    if sample_text is None:
        return []
    examples: list[dict[str, Any]] = []
    for match in re.finditer(
        r"\{\s*input\s*:\s*\[(?P<inputs>.*?)\](?:\s*,\s*result\s*:\s*\"(?P<result>(?:\\.|[^\"\\])*)\")?",
        sample_text,
        flags=re.DOTALL,
    ):
        raw_inputs = [
            _decode_js_string(token[1:-1])
            for token in re.findall(r"\"(?:\\.|[^\"\\])*\"", match.group("inputs"))
        ]
        if not raw_inputs:
            continue
        example: dict[str, Any] = {"input": raw_inputs}
        raw_result = match.group("result")
        if raw_result:
            example["result"] = _decode_js_string(raw_result)
        examples.append(example)
        if len(examples) >= 6:
            break
    return examples


@lru_cache(maxsize=128)
def _sample_js_for_type(puzzle_type: str) -> str | None:
    safe_type = puzzle_type.strip().lower()
    if not safe_type or not re.fullmatch(r"[a-z0-9_]+", safe_type):
        return None
    root = _benchmark_root()
    for path in sorted(root.glob(f"*/public/ppbench_site/vendor/pzprjs/dist/js/pzpr-samples/{safe_type}.js")):
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
    return None


def _decode_js_string(value: str) -> str:
    try:
        return json.loads(f'"{value}"')
    except Exception:
        return value.encode("utf-8", errors="replace").decode("unicode_escape", errors="replace")


def _local_bundle_for_split(split: str) -> Path | None:
    root = _benchmark_root()
    if split in {"probe", "probe10", "diagnostics"}:
        candidate = root / "ppbench_probe10"
    elif split in {"private", "private50", "evaluate"}:
        candidate = root / "ppbench_repr50"
    else:
        raise ValueError(f"unknown PPBench split: {split}")
    if (candidate / "public" / "puzzles.json").exists():
        return candidate
    return None


def _local_bundle_status() -> dict[str, Any]:
    probe = _local_bundle_for_split("probe10")
    private = _local_bundle_for_split("private50")
    if probe is None or private is None:
        return {
            "available": False,
            "source": "local_bundle",
            "error": "local PPBench bundles ppbench_probe10 and ppbench_repr50 are required",
        }
    return {
        "available": True,
        "source": "local_bundle",
        "error": None,
        "probe_bundle": str(probe),
        "private_bundle": str(private),
    }


def _benchmark_root() -> Path:
    return _task_root() / "benchmarks"


def _task_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _split_indices(split: str) -> list[int]:
    manifest = _split_manifest()
    if split in {"probe", "probe10", "diagnostics"}:
        return [int(item) for item in manifest["probe10_indices"]]
    if split in {"private", "private50", "evaluate"}:
        return [int(item) for item in manifest["private50_indices"]]
    raise ValueError(f"unknown PPBench split: {split}")


def _split_manifest() -> dict[str, Any]:
    path = Path(__file__).resolve().parent / "split_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_golden300_records() -> list[dict[str, Any]]:
    load_dataset = _load_dataset_function()
    try:
        records = load_dataset("golden_300")
    except TypeError:
        records = load_dataset(name="golden_300")
    if not isinstance(records, list):
        records = list(records)
    normalized = [dict(item) for item in records]
    if len(normalized) < 300:
        raise PPBenchDependencyError(f"expected golden_300 to contain 300 records, got {len(normalized)}")
    return normalized


def _load_dataset_function():
    try:
        from ppbench.dataset import load_dataset
    except Exception as first_exc:
        try:
            from ppbench import load_dataset  # type: ignore
        except Exception as second_exc:
            raise PPBenchDependencyError(
                "PPBench is required for probe/evaluate. Install the `ppbench` package "
                "in the task runtime environment."
            ) from second_exc
        if first_exc:
            return load_dataset
    return load_dataset


def _strip_answer_fields(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in ANSWER_FIELD_FRAGMENTS):
                continue
            result[str(key)] = _strip_answer_fields(item)
        return result
    if isinstance(value, list):
        return [_strip_answer_fields(item) for item in value]
    return value


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _optional_string(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _is_public_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _env_limit(name: str, *, default: int | None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if value <= 0:
        return None
    return value
