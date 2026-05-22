from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from flask import Flask

    from .app import WebContext


def register_ui_routes(app: "Flask", ctx: "WebContext") -> None:
    from flask import render_template

    @app.get("/ui")
    @app.get("/ui/")
    def control_plane_ui():
        return render_template(
            "control_plane.html",
            api_base="/api/v1",
            initial_experiment_id="",
            state_root=str(ctx.state_root),
            database_path=str(ctx.database_path),
        )

    @app.get("/ui/experiments/<experiment_id>")
    def control_plane_experiment_ui(experiment_id: str):
        return render_template(
            "control_plane.html",
            api_base="/api/v1",
            initial_experiment_id=experiment_id,
            state_root=str(ctx.state_root),
            database_path=str(ctx.database_path),
        )
