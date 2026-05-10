#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PUBLIC_DIR = ROOT / "public"
PRIVATE_DIR = ROOT / "private"

FORBIDDEN_PUBLIC_KEYS = {
    "solution",
    "solution_enc",
    "moves_full",
    "moves_required",
    "solution_moves",
    "ai_times",
    "replay",
    "replays",
}

EXPECTED_DIFFICULTY_BINS = {
    "(20,100]": 3,
    "(10,20]": 3,
    "(5,10]": 3,
    "(2.5,5]": 1,
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def load_manifest() -> dict[str, Any]:
    return _read_json(ROOT / "manifest.json")


def load_puzzles() -> list[dict[str, Any]]:
    payload = _read_json(PUBLIC_DIR / "puzzles.json")
    return list(payload["puzzles"])


def _prompt_path(puzzle: dict[str, Any]) -> Path:
    return PUBLIC_DIR / "prompts" / f"{puzzle['probe_rank']:03d}_{puzzle['id']}.md"


def _puzzle_file_path(puzzle: dict[str, Any]) -> Path:
    return PUBLIC_DIR / "puzzles" / f"{puzzle['probe_rank']:03d}_{puzzle['id']}.json"


def _find_puzzle(puzzles: list[dict[str, Any]], selector: str) -> dict[str, Any]:
    if selector.isdigit():
        rank = int(selector)
        for puzzle in puzzles:
            if puzzle["probe_rank"] == rank:
                return puzzle
    for puzzle in puzzles:
        if puzzle["id"] == selector:
            return puzzle
    raise SystemExit(f"unknown puzzle selector: {selector}")


def _iter_keys(value: Any, path: str = "$"):
    if isinstance(value, dict):
        for key, child in value.items():
            yield path, key
            yield from _iter_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_keys(child, f"{path}[{index}]")


def _validate_public_payload(name: str, value: Any) -> list[str]:
    errors = []
    for path, key in _iter_keys(value):
        if key in FORBIDDEN_PUBLIC_KEYS:
            errors.append(f"{name}: forbidden public key {key!r} at {path}")
    return errors


def cmd_list(args: argparse.Namespace) -> int:
    puzzles = load_puzzles()
    if args.format == "json":
        print(json.dumps(puzzles, indent=2, sort_keys=True))
    elif args.format == "jsonl":
        for puzzle in puzzles:
            print(json.dumps(puzzle, sort_keys=True))
    else:
        print("rank id type size solved solve_rate_bin pzpr_url")
        for puzzle in puzzles:
            solved = f"{puzzle['solved_count']}/{puzzle['model_attempt_count']}"
            print(
                f"{puzzle['probe_rank']:03d} "
                f"{puzzle['id']} "
                f"{puzzle['type']} "
                f"{puzzle['size']} "
                f"{solved} "
                f"{puzzle['difficulty_bin']} "
                f"{puzzle['pzpr_url']}"
            )
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    puzzle = _find_puzzle(load_puzzles(), args.puzzle)
    if args.format == "prompt":
        print(_prompt_path(puzzle).read_text(encoding="utf-8").rstrip())
    else:
        print(json.dumps(puzzle, indent=2, sort_keys=True))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    errors: list[str] = []
    manifest = load_manifest()
    public_manifest = _read_json(PUBLIC_DIR / "manifest.json")
    public_json = _read_json(PUBLIC_DIR / "puzzles.json")
    puzzles = list(public_json["puzzles"])
    puzzles_jsonl = _read_jsonl(PUBLIC_DIR / "puzzles.jsonl")

    if manifest.get("benchmark_id") != "ppbench_probe10":
        errors.append("manifest benchmark_id must be ppbench_probe10")
    if public_manifest.get("benchmark_id") != manifest.get("benchmark_id"):
        errors.append("public manifest benchmark_id does not match root manifest")
    if len(puzzles) != 10:
        errors.append(f"expected 10 public puzzles, found {len(puzzles)}")
    if len(puzzles_jsonl) != len(puzzles):
        errors.append("public/puzzles.jsonl count does not match public/puzzles.json")
    if len({p['id'] for p in puzzles}) != len(puzzles):
        errors.append("public puzzle ids are not unique")
    if puzzles != puzzles_jsonl:
        errors.append("public/puzzles.json and public/puzzles.jsonl differ")

    type_counts: dict[str, int] = {}
    difficulty_bins: dict[str, int] = {}
    for puzzle in puzzles:
        type_counts[puzzle["type"]] = type_counts.get(puzzle["type"], 0) + 1
        difficulty_bins[puzzle["difficulty_bin"]] = difficulty_bins.get(puzzle["difficulty_bin"], 0) + 1

    if sorted(type_counts.values()) != [1] * 10:
        errors.append(f"probe10 must contain 10 distinct types; got {json.dumps(type_counts, sort_keys=True)}")
    if difficulty_bins != EXPECTED_DIFFICULTY_BINS:
        errors.append(
            f"difficulty-bin counts do not match expected target: "
            f"{json.dumps(difficulty_bins, sort_keys=True)}"
        )

    errors.extend(_validate_public_payload("public/puzzles.json", public_json))
    errors.extend(_validate_public_payload("public/puzzles.jsonl", puzzles_jsonl))
    errors.extend(_validate_public_payload("public/manifest.json", public_manifest))

    for puzzle in puzzles:
        puzzle_path = _puzzle_file_path(puzzle)
        prompt_path = _prompt_path(puzzle)
        if not puzzle_path.exists():
            errors.append(f"missing public puzzle file: {puzzle_path.relative_to(ROOT)}")
        elif _read_json(puzzle_path) != puzzle:
            errors.append(f"public puzzle file differs: {puzzle_path.relative_to(ROOT)}")
        if not prompt_path.exists():
            errors.append(f"missing public prompt file: {prompt_path.relative_to(ROOT)}")

    answer_key_path = PRIVATE_DIR / "answer_key.jsonl"
    if answer_key_path.exists():
        answer_records = _read_jsonl(answer_key_path)
        if len(answer_records) != len(puzzles):
            errors.append("private answer key count does not match public puzzle count")
        public_ids = {p["id"] for p in puzzles}
        private_ids = {r.get("id") for r in answer_records}
        if private_ids != public_ids:
            errors.append("private answer key ids do not match public puzzle ids")
        for record in answer_records:
            if not isinstance(record.get("solution_enc"), str) or not record["solution_enc"]:
                errors.append(f"private answer record lacks solution_enc: {record.get('id')}")

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    summary = manifest.get("summary", {})
    print("validation ok")
    print(f"benchmark_id {manifest['benchmark_id']}")
    print(f"puzzle_count {len(puzzles)}")
    print(f"type_counts {json.dumps(summary.get('type_counts', {}), sort_keys=True)}")
    print(f"difficulty_bins {json.dumps(summary.get('difficulty_bin_counts', {}), sort_keys=True)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PPBench probe10 bundle helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List public puzzle instances")
    list_parser.add_argument("--format", choices=["table", "json", "jsonl"], default="table")
    list_parser.set_defaults(func=cmd_list)

    show_parser = subparsers.add_parser("show", help="Show one puzzle by rank or id")
    show_parser.add_argument("puzzle", help="One-based probe rank or puzzle id")
    show_parser.add_argument("--format", choices=["json", "prompt"], default="json")
    show_parser.set_defaults(func=cmd_show)

    validate_parser = subparsers.add_parser("validate", help="Validate bundle integrity")
    validate_parser.set_defaults(func=cmd_validate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
