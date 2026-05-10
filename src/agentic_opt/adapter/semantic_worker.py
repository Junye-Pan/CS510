from __future__ import annotations

import argparse
from pathlib import Path

from agentic_opt.common.ids import make_run_id
from agentic_opt.common.runtime_env import PreparedRuntimeEnv, TaskRuntimeSpec
from agentic_opt.control_plane.client import ControlPlaneClient

from .app_server_client import AppServerClient
from .base import BudgetPolicy, InstructionBundle, WorkspacePolicy
from .codex_adapter import AppServerAdapterConfig, AppServerCodexAdapter
from .semantic_workspace import build_semantic_startup_prompt, prepare_semantic_workspace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one server-first semantic worker assignment")
    parser.add_argument("--assignment-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--codex-binary", default="codex")
    parser.add_argument("--max-turn-wall-time-s", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client = ControlPlaneClient(args.api_url)
    context = client.get("/api/v1/context", {"assignment_id": args.assignment_id})
    assignment = context["assignment"]
    environment = client.post(
        "/api/v1/environments",
        {
            "assignment_id": assignment["assignment_id"],
            "experiment_id": assignment["experiment_id"],
            "task_id": assignment["task_id"],
        },
    )
    runtime_env = _runtime_env_from_environment(environment)
    workspace = prepare_semantic_workspace(
        workspace_root=args.workspace_root,
        api_url=args.api_url,
        assignment=assignment,
        session_id=args.session_id,
        runtime_env=runtime_env,
    )
    client.patch(
        f"/api/v1/sessions/{args.session_id}",
        {
            "status": "running",
            "workspace_path": str(workspace.root),
            "details": {
                "entry_path": str(workspace.entry_path),
                "mode": "dry-run" if args.dry_run else "codex-local",
            },
        },
    )
    client.post(
        "/api/v1/events",
        {
            "experiment_id": assignment["experiment_id"],
            "assignment_id": assignment["assignment_id"],
            "session_id": args.session_id,
            "task_id": assignment["task_id"],
            "agent_id": assignment["agent_id"],
            "event_type": "worker.workspace.prepared",
            "summary": "semantic workspace prepared",
            "payload": {"workspace_root": str(workspace.root), "entry_path": str(workspace.entry_path)},
        },
    )
    if args.dry_run:
        _checkpoint_initial_notebook(client=client, assignment=assignment, session_id=args.session_id, workspace_root=workspace.root)
        client.patch(f"/api/v1/sessions/{args.session_id}", {"status": "completed", "details": {"dry_run": True}})
        return 0

    app_client = AppServerClient(
        codex_binary=args.codex_binary,
        root_cwd=str(workspace.root),
        codex_home=str(workspace.root / ".codex-home"),
        extra_env=workspace.env,
    )
    adapter = AppServerCodexAdapter(
        client=app_client,
        config=AppServerAdapterConfig(codex_binary=args.codex_binary, reasoning_effort="high"),
    )
    run_id = make_run_id(assignment["agent_id"])
    final_status = "completed"
    stop_reason = "turn_completed"
    try:
        session = adapter.start_session(
            session_id=args.session_id,
            task_id=assignment["task_id"],
            agent_id=assignment["agent_id"],
            run_id=run_id,
            workspace=WorkspacePolicy(
                workspace_root=str(workspace.root),
                writable_roots=workspace.writable_roots,
                readable_roots=workspace.readable_roots,
                # The semantic CLI talks to the control plane over localhost.
                # Without App Server network access, ctx/eval/finding/notebook
                # fail before a worker can report durable experiment state.
                allow_network=True,
            ),
            instructions=InstructionBundle(
                task_id=assignment["task_id"],
                startup_prompt=build_semantic_startup_prompt(assignment=assignment, workspace=workspace),
                agents_md_path=str(workspace.agents_md_path),
                skills_root_path=str(workspace.skills_root),
            ),
            budget=BudgetPolicy(
                max_turn_wall_time_s=args.max_turn_wall_time_s,
                max_turns=1,
                max_model_turns=1,
                evaluator_run_budget=(assignment.get("budget") or {}).get("evaluator_runs"),
            ),
        )
        turn = adapter.start_turn(
            session_id=session.session_id,
            kind="autonomous",
            prompt=build_semantic_startup_prompt(assignment=assignment, workspace=workspace),
            budget=BudgetPolicy(max_turn_wall_time_s=args.max_turn_wall_time_s, max_turns=1, max_model_turns=1),
        )
        result = adapter.wait_turn(session_id=session.session_id, turn_id=turn.turn_id, timeout_s=args.max_turn_wall_time_s)
        client.post(
            "/api/v1/events",
            {
                "experiment_id": assignment["experiment_id"],
                "assignment_id": assignment["assignment_id"],
                "session_id": args.session_id,
                "task_id": assignment["task_id"],
                "agent_id": assignment["agent_id"],
                "event_type": "agent.turn.finished",
                "summary": f"semantic agent turn finished with outcome={result.outcome}",
                "payload": {
                    "turn_id": result.turn_id,
                    "outcome": result.outcome,
                    "final_message": result.final_message,
                    "trace": result.trace_bundle.summary,
                },
            },
        )
        if result.outcome not in {"completed", "success", "interrupted"}:
            final_status = "stopped"
            stop_reason = "turn_failed"
        _checkpoint_initial_notebook(client=client, assignment=assignment, session_id=args.session_id, workspace_root=workspace.root)
        adapter.close_session(session_id=session.session_id, final_status=final_status)
    except TimeoutError as exc:
        final_status = "stopped"
        stop_reason = "turn_timeout"
        client.post(
            "/api/v1/events",
            {
                "experiment_id": assignment["experiment_id"],
                "assignment_id": assignment["assignment_id"],
                "session_id": args.session_id,
                "task_id": assignment["task_id"],
                "agent_id": assignment["agent_id"],
                "event_type": "agent.turn.timeout",
                "summary": "semantic agent turn timed out",
                "payload": {"error_type": type(exc).__name__, "message": str(exc)},
            },
        )
        _checkpoint_initial_notebook(client=client, assignment=assignment, session_id=args.session_id, workspace_root=workspace.root)
    except Exception as exc:
        final_status = "failed"
        stop_reason = "worker_exception"
        client.post(
            "/api/v1/events",
            {
                "experiment_id": assignment["experiment_id"],
                "assignment_id": assignment["assignment_id"],
                "session_id": args.session_id,
                "task_id": assignment["task_id"],
                "agent_id": assignment["agent_id"],
                "event_type": "worker.failed",
                "summary": "semantic worker failed",
                "payload": {"error_type": type(exc).__name__, "message": str(exc)},
            },
        )
        raise
    finally:
        app_client.close()
        client.patch(f"/api/v1/sessions/{args.session_id}", {"status": final_status, "details": {"stop_reason": stop_reason}})
    return 0


def _checkpoint_initial_notebook(
    *,
    client: ControlPlaneClient,
    assignment: dict,
    session_id: str,
    workspace_root: Path,
) -> None:
    worklog = workspace_root / "WORKLOG.md"
    client.post(
        "/api/v1/notebook-checkpoints",
        {
            "experiment_id": assignment["experiment_id"],
            "assignment_id": assignment["assignment_id"],
            "session_id": session_id,
            "agent_id": assignment["agent_id"],
            "notebook_uri": worklog.as_uri(),
            "content": worklog.read_text(encoding="utf-8") if worklog.exists() else "",
            "metadata": {"kind": "session_checkpoint"},
        },
    )


def _runtime_env_from_environment(environment: dict) -> PreparedRuntimeEnv:
    spec = environment.get("spec") or {}
    root = Path(environment["root_path"])
    metadata = environment.get("metadata") or {}
    return PreparedRuntimeEnv(
        task_id=environment["task_id"],
        fingerprint=environment["fingerprint"],
        root=root,
        venv_dir=Path(metadata.get("venv_dir") or root / "venv"),
        python_path=Path(environment["python_path"]),
        manifest_path=Path(metadata.get("manifest_path") or root / "manifest.json"),
        spec=TaskRuntimeSpec(
            kind=spec.get("kind") or "local_venv",
            python=spec.get("python") or ">=3.11",
            requirements=tuple(spec.get("requirements") or ()),
            required_imports=tuple(spec.get("required_imports") or ()),
            forbidden_shadow_modules=tuple(spec.get("forbidden_shadow_modules") or ()),
            system_site_packages=bool(spec.get("system_site_packages")),
            verify_public_seed=bool(spec.get("verify_public_seed", True)),
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
