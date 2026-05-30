from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from agentic_opt.common.ids import make_run_id
from agentic_opt.common.runtime_env import PreparedRuntimeEnv, TaskRuntimeSpec
from agentic_opt.control_plane.client import ControlPlaneClient
from agentic_opt.control_plane.process_env import build_subprocess_env
from agentic_opt.control_plane.relay import relay_url, start_relay_process

from .app_server_client import AppServerClient
from .base import BudgetPolicy, InstructionBundle, WorkspacePolicy
from .codex_adapter import AppServerAdapterConfig, AppServerCodexAdapter
from .semantic_workspace import build_semantic_startup_prompt, prepare_semantic_workspace


_PATH_COMPONENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_SHELL_ENV_SECRET_NAMES = (
    "CODEX_HOME",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one server-first semantic worker assignment")
    parser.add_argument("--assignment-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--codex-binary", default=os.environ.get("AO_CODEX_BINARY") or os.environ.get("CODEX_BINARY") or "codex")
    parser.add_argument("--max-turn-wall-time-s", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app_client: AppServerClient | None = None
    proxy_bridge_process: subprocess.Popen[str] | None = None
    workspace_broker_process: subprocess.Popen[str] | None = None
    client = ControlPlaneClient(args.api_url)
    context = client.get("/api/v1/context", {"assignment_id": args.assignment_id})
    assignment = context["assignment"]
    network_policy = client.get(
        "/api/v1/network-policy",
        {"assignment_id": args.assignment_id, "session_id": args.session_id},
    )
    environment = client.post(
        "/api/v1/environments",
        {
            "assignment_id": assignment["assignment_id"],
            "experiment_id": assignment["experiment_id"],
            "task_id": assignment["task_id"],
        },
    )
    runtime_env = _runtime_env_from_environment(environment)
    runtime_env = _container_runtime_env_override(runtime_env)
    bootstrap = client.post(
        f"/api/v1/assignments/{assignment['assignment_id']}/workspace-bootstrap",
        {
            "workspace_root": str(args.workspace_root),
            "session_id": args.session_id,
        },
    )
    semantic_tool_api_url, workspace_broker_process, workspace_broker_metadata = _start_workspace_control_broker(
        workspace_root=args.workspace_root,
        target_url=args.api_url,
    )
    workspace = prepare_semantic_workspace(
        workspace_root=args.workspace_root,
        api_url=semantic_tool_api_url,
        assignment=assignment,
        session_id=args.session_id,
        runtime_env=runtime_env,
        network_policy=network_policy,
        bootstrap=bootstrap,
        extra_env_exports=_environment_default_exports(environment),
    )
    client.patch(
        f"/api/v1/sessions/{args.session_id}",
        {
            "status": "running",
            "pid": os.getpid(),
            "workspace_path": str(workspace.root),
            "details": {
                "entry_path": str(workspace.entry_path),
                "mode": "dry-run" if args.dry_run else assignment.get("worker_backend") or "codex-local",
                "worker_pid": os.getpid(),
                "worker_entrypoint": "agentic_opt.adapter.semantic_worker",
                "network_policy": network_policy,
                "workspace_seed": bootstrap.get("workspace_seed"),
                "checked_out_tools": bootstrap.get("checked_out_tools") or [],
                "semantic_tool_transport": workspace_broker_metadata,
                "task_context": workspace.task_context,
                "task_context_enforcement": workspace.task_context.get("enforcement"),
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
            "payload": {
                "workspace_root": str(workspace.root),
                "entry_path": str(workspace.entry_path),
                "workspace_seed": bootstrap.get("workspace_seed"),
                "checked_out_tools": bootstrap.get("checked_out_tools") or [],
                "semantic_tool_transport": workspace_broker_metadata,
                "task_context": workspace.task_context,
                "task_context_enforcement": workspace.task_context.get("enforcement"),
            },
        },
    )
    if ((network_policy.get("enforcement") or {}).get("policy_weakened")):
        client.post(
            "/api/v1/network-access-events",
            {
                "experiment_id": assignment["experiment_id"],
                "assignment_id": assignment["assignment_id"],
                "session_id": args.session_id,
                "task_id": assignment["task_id"],
                "agent_id": assignment["agent_id"],
                "access_type": "policy",
                "decision": "weakened",
                "reason": (network_policy.get("enforcement") or {}).get("reason"),
                "metadata": {"network_policy": network_policy},
            },
        )
    if args.dry_run:
        _checkpoint_initial_notebook(client=client, assignment=assignment, session_id=args.session_id, workspace_root=workspace.root)
        client.patch(f"/api/v1/sessions/{args.session_id}", {"status": "completed", "details": {"dry_run": True}})
        _terminate_process(workspace_broker_process)
        return 0

    proxy_bridge_process = _start_outbound_proxy_bridge()
    _remove_legacy_workspace_codex_home(workspace.root)
    codex_home = private_codex_home_for_workspace(workspace_root=workspace.root, session_id=args.session_id)
    app_client = AppServerClient(
        codex_binary=args.codex_binary,
        root_cwd=str(workspace.root),
        codex_home=str(codex_home),
        config_overrides=[
            f"shell_environment_policy.exclude={json.dumps(list(_SHELL_ENV_SECRET_NAMES))}",
        ],
        extra_env=workspace.env,
        startup_timeout_s=_app_server_startup_timeout_s(args.max_turn_wall_time_s),
    )
    adapter = AppServerCodexAdapter(
        client=app_client,
        config=AppServerAdapterConfig(
            codex_binary=args.codex_binary,
            model=_codex_model(workspace.env),
            reasoning_effort=_codex_reasoning_effort(workspace.env),
        ),
    )
    run_id = make_run_id(assignment["agent_id"])
    final_status = "completed"
    stop_reason = "turn_completed"
    turn = None
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
                sandbox_mode=_worker_sandbox_mode(workspace.env),
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
        registered_trace = _register_turn_trace(
            client=client,
            assignment=assignment,
            session_id=args.session_id,
            run_id=run_id,
            turn_id=result.turn_id,
            trace_dir=Path(result.trace_bundle.events_path).parent,
            outcome=result.outcome,
            status="completed" if result.outcome in {"completed", "success"} else "partial",
            worker_backend=assignment.get("worker_backend") or "codex-local",
        )
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
                    "registered_trace": registered_trace,
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
        registered_trace = None
        if turn is not None:
            registered_trace = _register_turn_trace(
                client=client,
                assignment=assignment,
                session_id=args.session_id,
                run_id=run_id,
                turn_id=turn.turn_id,
                trace_dir=workspace.root / ".run" / "traces" / run_id / turn.turn_id,
                outcome="timeout",
                status="partial",
                worker_backend=assignment.get("worker_backend") or "codex-local",
            )
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
                "payload": {"error_type": type(exc).__name__, "message": str(exc), "registered_trace": registered_trace},
            },
        )
        _checkpoint_initial_notebook(client=client, assignment=assignment, session_id=args.session_id, workspace_root=workspace.root)
    except Exception as exc:
        final_status = "failed"
        stop_reason = "worker_exception"
        registered_trace = None
        if turn is not None:
            registered_trace = _register_turn_trace(
                client=client,
                assignment=assignment,
                session_id=args.session_id,
                run_id=run_id,
                turn_id=turn.turn_id,
                trace_dir=workspace.root / ".run" / "traces" / run_id / turn.turn_id,
                outcome="worker_exception",
                status="partial",
                worker_backend=assignment.get("worker_backend") or "codex-local",
            )
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
                "payload": {"error_type": type(exc).__name__, "message": str(exc), "registered_trace": registered_trace},
            },
        )
        raise
    finally:
        if app_client is not None:
            app_client.close()
        _terminate_process(proxy_bridge_process)
        _terminate_process(workspace_broker_process)
        client.patch(f"/api/v1/sessions/{args.session_id}", {"status": final_status, "details": {"stop_reason": stop_reason}})
    return 0


def _start_workspace_control_broker(
    *,
    workspace_root: Path,
    target_url: str,
) -> tuple[str, subprocess.Popen[str] | None, dict[str, str | int | bool]]:
    if target_url.startswith("unix://") or os.environ.get("AO_WORKSPACE_CONTROL_BROKER") == "disabled":
        return target_url, None, {
            "enabled": False,
            "transport": "direct-unix" if target_url.startswith("unix://") else "direct",
            "target_url": target_url,
        }
    control_dir = workspace_root.resolve() / ".control"
    control_dir.mkdir(parents=True, exist_ok=True)
    socket_path = _workspace_broker_socket_path(control_dir)
    audit_log_path = control_dir / "control_broker_audit.jsonl"
    process = start_relay_process(
        socket_path=socket_path,
        target_url=target_url,
        env=_worker_process_env(),
        audit_log_path=audit_log_path,
    )
    try:
        _wait_for_unix_socket(socket_path)
    except Exception:
        _terminate_process(process)
        raise
    return relay_url(socket_path), process, {
        "enabled": True,
        "transport": "unix-socket",
        "socket_path": str(socket_path),
        "workspace_control_dir": str(control_dir),
        "relay_url": relay_url(socket_path),
        "target_url": target_url,
        "audit_log_path": str(audit_log_path),
        "pid": process.pid,
    }


def _workspace_broker_socket_path(control_dir: Path) -> Path:
    preferred = control_dir / "control.sock"
    # macOS sockaddr_un paths are commonly limited to 104 bytes. Keep the
    # workspace-local socket when possible, otherwise fall back to a short path.
    if len(str(preferred)) <= 95:
        return preferred
    digest = hashlib.sha1(str(control_dir).encode("utf-8")).hexdigest()[:12]
    token = f"{time.time_ns() & 0xFFFF:x}"
    return Path(tempfile.gettempdir()) / f"ao-cp-{digest}-{os.getpid()}-{token}.sock"


def _start_outbound_proxy_bridge() -> subprocess.Popen[str] | None:
    socket_path = os.environ.get("AO_OUTBOUND_PROXY_SOCKET")
    if not socket_path:
        return None
    port = int(os.environ.get("AO_OUTBOUND_PROXY_BRIDGE_PORT") or "8765")
    proxy_url = f"http://127.0.0.1:{port}"
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.setdefault(name, proxy_url)
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    process = subprocess.Popen(
        [
            os.environ.get("AO_WORKER_RUNTIME_PYTHON") or sys.executable,
            "-m",
            "agentic_opt.worker_tools.proxy_bridge",
            "--socket",
            socket_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        _wait_for_tcp("127.0.0.1", port)
    except Exception:
        _terminate_process(process)
        raise
    return process


def _wait_for_tcp(host: str, port: int, *, timeout_s: float = 5.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.1):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"outbound proxy bridge did not start on {host}:{port}")


def _wait_for_unix_socket(path: Path, *, timeout_s: float = 5.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise TimeoutError(f"workspace control broker did not create socket: {path}")


def _worker_process_env() -> dict[str, str]:
    env = build_subprocess_env()
    src_root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = str(src_root) if not env.get("PYTHONPATH") else f"{src_root}:{env['PYTHONPATH']}"
    return env


def _terminate_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


def _register_turn_trace(
    *,
    client: ControlPlaneClient,
    assignment: dict,
    session_id: str,
    run_id: str,
    turn_id: str,
    trace_dir: Path,
    outcome: str,
    status: str,
    worker_backend: str,
) -> dict | None:
    if not trace_dir.exists():
        return None
    try:
        return client.post(
            "/api/v1/agent-traces",
            {
                "experiment_id": assignment["experiment_id"],
                "assignment_id": assignment["assignment_id"],
                "session_id": session_id,
                "task_id": assignment["task_id"],
                "agent_id": assignment["agent_id"],
                "run_id": run_id,
                "turn_id": turn_id,
                "worker_backend": worker_backend,
                "trace_dir": str(trace_dir),
                "outcome": outcome,
                "status": status,
            },
        )
    except Exception as exc:
        try:
            client.post(
                "/api/v1/events",
                {
                    "experiment_id": assignment["experiment_id"],
                    "assignment_id": assignment["assignment_id"],
                    "session_id": session_id,
                    "task_id": assignment["task_id"],
                    "agent_id": assignment["agent_id"],
                    "event_type": "agent_trace.registration_failed",
                    "summary": "agent trace registration failed",
                    "payload": {
                        "run_id": run_id,
                        "turn_id": turn_id,
                        "trace_dir": str(trace_dir),
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                },
            )
        except Exception:
            pass
        return None


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


def _environment_default_exports(environment: dict) -> dict[str, str]:
    raw = (environment.get("metadata") or {}).get("default_env") or {}
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items() if key and value is not None}


def _codex_model(env: dict[str, str]) -> str | None:
    value = env.get("AO_CODEX_MODEL") or os.environ.get("AO_CODEX_MODEL")
    return str(value).strip() if value else "gpt-5.5"


def _codex_reasoning_effort(env: dict[str, str]) -> str | None:
    value = env.get("AO_CODEX_REASONING_EFFORT") or os.environ.get("AO_CODEX_REASONING_EFFORT")
    return str(value).strip() if value else "xhigh"


def _container_runtime_env_override(runtime_env: PreparedRuntimeEnv) -> PreparedRuntimeEnv:
    python_override = os.environ.get("AO_WORKER_RUNTIME_PYTHON")
    if not python_override:
        return runtime_env
    root = Path(os.environ.get("AO_WORKER_RUNTIME_ROOT") or "/opt/agentic-opt")
    venv_dir = Path(os.environ.get("AO_WORKER_RUNTIME_VENV") or "/usr/local")
    manifest_path = Path(os.environ.get("AO_WORKER_RUNTIME_MANIFEST") or root / "docker_worker_runtime.json")
    return PreparedRuntimeEnv(
        task_id=runtime_env.task_id,
        fingerprint=runtime_env.fingerprint,
        root=root,
        venv_dir=venv_dir,
        python_path=Path(python_override),
        manifest_path=manifest_path,
        spec=runtime_env.spec,
    )


def _app_server_startup_timeout_s(max_turn_wall_time_s: int | None) -> float:
    if max_turn_wall_time_s is None:
        return 60.0
    return float(max(10, min(120, max_turn_wall_time_s)))


def _worker_sandbox_mode(workspace_env: dict[str, str] | None = None) -> str:
    explicit = (
        os.environ.get("AO_CODEX_SANDBOX_MODE")
        or os.environ.get("AO_WORKER_SANDBOX_MODE")
        or (workspace_env or {}).get("AO_CODEX_SANDBOX_MODE")
        or (workspace_env or {}).get("AO_WORKER_SANDBOX_MODE")
    )
    if explicit:
        return explicit
    if os.environ.get("AO_WORKER_RUNTIME_PYTHON") or (workspace_env or {}).get("AO_TASK_RUNTIME_PYTHON"):
        return "danger-full-access"
    return "workspace-write"


def private_codex_home_for_workspace(*, workspace_root: Path, session_id: str) -> Path:
    workspace_root = workspace_root.resolve()
    if workspace_root.parent.parent.name == "workspaces":
        state_root = workspace_root.parent.parent.parent
    else:
        state_root = workspace_root.parent
    return state_root / "provider_state" / "codex_home" / _safe_path_component(session_id)


def _safe_path_component(value: str) -> str:
    safe = _PATH_COMPONENT_RE.sub("_", value).strip("._")
    return safe or "session"


def _remove_legacy_workspace_codex_home(workspace_root: Path) -> None:
    legacy_home = workspace_root / ".codex-home"
    if legacy_home.exists():
        shutil.rmtree(legacy_home)


if __name__ == "__main__":
    raise SystemExit(main())
