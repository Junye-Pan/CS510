from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from agentic_opt.adapter.semantic_workspace import SEMANTIC_SKILL_BODIES


def sync_skills(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for child in target.iterdir():
        if not child.is_dir():
            continue
        if child.name in SEMANTIC_SKILL_BODIES:
            continue
        shutil.rmtree(child)
    for skill_name, body in SEMANTIC_SKILL_BODIES.items():
        skill_dir = target / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(body.strip() + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync repo-root semantic worker skill files")
    parser.add_argument(
        "--target",
        type=Path,
        default=Path(".agents") / "skills",
        help="directory to receive the synced skill tree",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sync_skills(args.target.resolve())
    print(args.target.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
