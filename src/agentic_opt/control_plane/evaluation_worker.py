from __future__ import annotations

import argparse
from pathlib import Path

from .repository import ControlPlaneRepository
from .service import ControlPlaneService


def run_evaluation_worker(*, db_path: Path, evaluation_id: str) -> int:
    repository = ControlPlaneRepository(db_path)
    state_root = db_path.resolve().parent
    service = ControlPlaneService(
        repository=repository,
        artifact_root=state_root / "artifacts",
        job_root=state_root / "jobs",
        database_path=db_path,
    )
    record = service.run_evaluation(evaluation_id)
    return 0 if record["status"] == "completed" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m agentic_opt.control_plane.evaluation_worker")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--evaluation-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_evaluation_worker(db_path=args.db, evaluation_id=args.evaluation_id)


if __name__ == "__main__":
    raise SystemExit(main())
