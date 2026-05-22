from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from agentic_opt.common.config import get_repo_root
from agentic_opt.control_plane.repository import ControlPlaneRepository
from agentic_opt.control_plane.service import ControlPlaneService

from .routes_control_plane import register_control_plane_routes
from .routes_ui import register_ui_routes
from .workers import WorkerManager


@dataclass(frozen=True)
class WebContext:
    state_root: Path
    database_path: Path
    control: ControlPlaneRepository
    control_service: ControlPlaneService
    workers: WorkerManager


def create_app(*, state_root: Path, database_path: Path, default_api_url: str | None = None):
    try:
        from flask import Flask, jsonify
    except ImportError as exc:  # pragma: no cover - import depends on environment
        raise RuntimeError("Flask is required for the web backend. Install project dependencies first.") from exc

    repo_root = get_repo_root()
    state_root = state_root.resolve()
    state_root.mkdir(parents=True, exist_ok=True)
    database_path = database_path.resolve()
    control = ControlPlaneRepository(database_path)
    control_service = ControlPlaneService(
        repository=control,
        artifact_root=state_root / "artifacts",
        job_root=state_root / "jobs",
        database_path=database_path,
    )
    workers = WorkerManager(
        repo_root=repo_root,
        state_root=state_root,
        control=control,
        default_api_url=default_api_url,
    )
    ctx = WebContext(
        state_root=state_root,
        database_path=database_path,
        control=control,
        control_service=control_service,
        workers=workers,
    )

    app = Flask(__name__)
    app.config["AO_CONTEXT"] = ctx

    @app.get("/")
    def index():
        return jsonify(
            {
                "name": "agentic-opt control plane",
                "state_root": str(ctx.state_root),
                "database_path": str(ctx.database_path),
                "api": "/api/v1",
            }
        )

    @app.get("/healthz")
    def healthz():
        try:
            experiments = ctx.control.list_experiments()
            return jsonify(
                {
                    "ok": True,
                    "service": "control-plane",
                    "database_path": str(ctx.database_path),
                    "experiment_count": len(experiments),
                }
            )
        except Exception as exc:  # pragma: no cover - defensive health path
            return jsonify({"ok": False, "service": "control-plane", "error": str(exc)}), 503

    register_control_plane_routes(app, ctx)
    register_ui_routes(app, ctx)
    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m agentic_opt.web.app", description="Flask control plane for agentic-opt")
    parser.add_argument("--state-root", type=Path, default=(get_repo_root() / "ao_state"))
    parser.add_argument("--db", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    state_root = args.state_root.resolve()
    db_path = args.db or (state_root / "control.sqlite3")
    app = create_app(
        state_root=state_root,
        database_path=db_path,
        default_api_url=f"http://{args.host}:{args.port}",
    )
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
