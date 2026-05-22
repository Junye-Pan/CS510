from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from agentic_opt.common.atomic import atomic_write_text
from agentic_opt.control_plane.client import ControlPlaneClient


def _env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _client(args: argparse.Namespace) -> ControlPlaneClient:
    return ControlPlaneClient(args.api_url or _env("AO_CONTROL_API_URL"))


def _attempt_id_arg(args: argparse.Namespace) -> str | None:
    return getattr(args, "attempt_id", None) or os.environ.get("AO_ATTEMPT_ID")


def _emit(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _workspace_root() -> Path | None:
    raw = os.environ.get("AO_WORKSPACE_ROOT")
    if not raw:
        return None
    return Path(raw).resolve()


def _workspace_reference(kind: str, *, files: list[str] | None = None, directories: list[str] | None = None, **extra: Any) -> dict[str, Any]:
    root = _workspace_root()
    payload: dict[str, Any] = {
        "kind": kind,
        "workspace_root": str(root) if root else None,
        "files": _resolve_workspace_paths(root, files or []),
        "directories": _resolve_workspace_paths(root, directories or []),
        "note": "Inspect these paths with local file tools such as rg, jq, sed, head, tail, or less.",
    }
    payload.update({key: value for key, value in extra.items() if value is not None})
    return payload


def _resolve_workspace_paths(root: Path | None, relatives: list[str]) -> dict[str, str]:
    if root is None:
        return {relative: relative for relative in relatives}
    return {relative: str(root / relative) for relative in relatives}


def _read_text_arg(text: str | None, file_path: str | None) -> str:
    if text and file_path:
        raise ValueError("pass either --body/--content or --file, not both")
    if text is not None:
        return text
    if file_path is not None:
        return Path(file_path).read_text(encoding="utf-8")
    raise ValueError("text or file path is required")


_CTX_FILE_REFERENCE_COMMANDS = {
    "context",
    "assignment",
    "task",
    "findings",
    "evaluations",
    "attempts",
    "artifacts",
    "jobs",
    "environments",
    "leaderboard",
    "incumbent",
    "telemetry",
    "shared-tools",
    "network",
}


def command_ctx(args: argparse.Namespace) -> int:
    if args.ctx_command in _CTX_FILE_REFERENCE_COMMANDS and _workspace_root() is not None:
        _emit(_ctx_workspace_reference(args))
        return 0
    client = _client(args)
    assignment_id = args.assignment_id or _env("AO_ASSIGNMENT_ID")
    if args.ctx_command == "context":
        _emit(client.get("/api/v1/context", {"assignment_id": assignment_id}))
    elif args.ctx_command == "assignment":
        _emit(client.get(f"/api/v1/assignments/{assignment_id}"))
    elif args.ctx_command == "task":
        _emit(client.get(f"/api/v1/tasks/{args.task_id or _env('AO_TASK_ID')}"))
    elif args.ctx_command == "findings":
        _emit(client.get("/api/v1/findings", {"task_id": args.task_id or _env("AO_TASK_ID"), "query": args.query}))
    elif args.ctx_command == "evaluations":
        _emit(client.get("/api/v1/evaluations", {"assignment_id": assignment_id}))
    elif args.ctx_command == "attempts":
        _emit(client.get("/api/v1/attempts", {"assignment_id": assignment_id, "status": args.status}))
    elif args.ctx_command == "artifacts":
        _emit(client.get("/api/v1/artifacts", {"assignment_id": assignment_id}))
    elif args.ctx_command == "jobs":
        _emit(client.get("/api/v1/jobs", {"assignment_id": assignment_id}))
    elif args.ctx_command == "environments":
        _emit(client.get("/api/v1/environments", {"task_id": args.task_id or _env("AO_TASK_ID")}))
    elif args.ctx_command == "leaderboard":
        _emit(client.get("/api/v1/leaderboard", {"experiment_id": _env("AO_EXPERIMENT_ID"), "limit": args.limit}))
    elif args.ctx_command == "incumbent":
        _emit(
            client.get(
                "/api/v1/incumbent",
                {
                    "experiment_id": _env("AO_EXPERIMENT_ID"),
                    "direction_id": args.direction_id,
                },
            )
        )
    elif args.ctx_command == "telemetry":
        _emit(client.get("/api/v1/telemetry-runs", {"assignment_id": assignment_id}))
    elif args.ctx_command == "shared-tools":
        _emit(
            client.get(
                "/api/v1/shared-tools",
                {
                    "task_id": args.task_id or _env("AO_TASK_ID"),
                    "experiment_id": os.environ.get("AO_EXPERIMENT_ID"),
                    "query": args.query,
                },
            )
        )
    elif args.ctx_command == "network":
        _emit(client.get("/api/v1/network-policy", {"assignment_id": assignment_id, "session_id": os.environ.get("AO_SESSION_ID")}))
    elif args.ctx_command in {"stop", "local-stop", "local_stop"}:
        session_id = _env("AO_SESSION_ID")
        reason = args.reason.strip()
        if not reason:
            raise ValueError("--reason must not be empty")
        local_stop = {
            "source": "worker",
            "scope": "local",
            "session_id": session_id,
            "reason": reason,
        }
        session = client.patch(f"/api/v1/sessions/{session_id}", {"details": {"local_stop_condition": local_stop}})
        client.post(
            "/api/v1/events",
            {
                "experiment_id": _env("AO_EXPERIMENT_ID"),
                "assignment_id": assignment_id,
                "session_id": session_id,
                "task_id": args.task_id or _env("AO_TASK_ID"),
                "agent_id": _env("AO_AGENT_ID"),
                "event_type": "session.local_stop",
                "summary": reason,
                "payload": local_stop,
            },
        )
        _emit(session)
    elif args.ctx_command in {"global-stop", "global_stop"}:
        session_id = _env("AO_SESSION_ID")
        reason = args.reason.strip()
        if not reason:
            raise ValueError("--reason must not be empty")
        if not args.confirm_global_stop:
            raise ValueError("global stop requires --confirm-global-stop")
        global_stop = {
            "source": "worker",
            "scope": "global",
            "session_id": session_id,
            "reason": reason,
        }
        payload = {
            "status": "completed",
            "metadata": {
                "global_stop_condition": global_stop,
            },
        }
        assignment = client.patch(f"/api/v1/assignments/{assignment_id}", payload)
        client.post(
            "/api/v1/events",
            {
                "experiment_id": _env("AO_EXPERIMENT_ID"),
                "assignment_id": assignment_id,
                "session_id": session_id,
                "task_id": args.task_id or _env("AO_TASK_ID"),
                "agent_id": _env("AO_AGENT_ID"),
                "event_type": "assignment.global_stop",
                "summary": reason,
                "payload": global_stop,
            },
        )
        _emit(assignment)
    else:
        raise ValueError(args.ctx_command)
    return 0


def _ctx_workspace_reference(args: argparse.Namespace) -> dict[str, Any]:
    command = args.ctx_command
    if command == "context":
        return _workspace_reference(
            "context",
            files=["context/current_state.json", "context/assignment.json", "context/experiment.json", "context/network_policy.json"],
            directories=["task", "history"],
        )
    if command == "assignment":
        return _workspace_reference("assignment", files=["context/assignment.json", "context/current_state.json"])
    if command == "task":
        return _workspace_reference(
            "task",
            files=["task/TASK.md", "task/public_contract.md", "task/manifest.json"],
            directories=["task/public_files", "task/research_directions"],
        )
    if command == "findings":
        return _workspace_reference("findings", files=["history/findings/index.jsonl"], directories=["history/findings"], query=getattr(args, "query", None))
    if command == "evaluations":
        return _workspace_reference("evaluations", directories=["history/evaluations"])
    if command == "attempts":
        return _workspace_reference("attempts", directories=["history/attempts"], status=getattr(args, "status", None))
    if command == "artifacts":
        return _workspace_reference("artifacts", directories=["history/artifacts", "artifacts"])
    if command == "jobs":
        return _workspace_reference("jobs", directories=["history/jobs"])
    if command == "environments":
        return _workspace_reference("environments", files=["history/environments.jsonl", "history/environment_overlays.jsonl"])
    if command == "leaderboard":
        return _workspace_reference("leaderboard", files=["history/leaderboard.jsonl"], limit=getattr(args, "limit", None))
    if command == "incumbent":
        return _workspace_reference(
            "incumbent",
            files=["history/incumbent.json", "history/direction_incumbent.json"],
            direction_id=getattr(args, "direction_id", None),
        )
    if command == "telemetry":
        return _workspace_reference("telemetry", directories=["history/telemetry"])
    if command == "shared-tools":
        return _workspace_reference(
            "shared-tools",
            directories=["history/shared_tools", "shared_tools"],
            query=getattr(args, "query", None),
        )
    if command == "network":
        return _workspace_reference("network", files=["context/network_policy.json", "history/network/policy.json", "history/network/events.jsonl"])
    raise ValueError(command)


def command_artifact(args: argparse.Namespace) -> int:
    if args.artifact_command == "list" and _workspace_root() is not None:
        _emit(_workspace_reference("artifacts", directories=["history/artifacts", "artifacts"], attempt_id=_attempt_id_arg(args)))
        return 0
    client = _client(args)
    if args.artifact_command == "upload":
        path = Path(args.path).resolve()
        _emit(
            client.post(
                "/api/v1/artifacts",
                {
                    "experiment_id": _env("AO_EXPERIMENT_ID"),
                    "assignment_id": _env("AO_ASSIGNMENT_ID"),
                    "attempt_id": _attempt_id_arg(args),
                    "kind": args.kind,
                    "path": str(path),
                    "metadata": {"note": args.note} if args.note else {},
                },
            )
        )
    elif args.artifact_command == "list":
        _emit(client.get("/api/v1/artifacts", {"assignment_id": _env("AO_ASSIGNMENT_ID"), "attempt_id": _attempt_id_arg(args)}))
    elif args.artifact_command == "checkout-incumbent":
        _emit(
            client.post(
                "/api/v1/incumbent/checkout",
                {
                    "experiment_id": _env("AO_EXPERIMENT_ID"),
                    "task_id": os.environ.get("AO_TASK_ID"),
                    "direction_id": args.direction_id,
                    "destination_path": str(Path(args.destination).resolve()),
                    "force": args.force,
                },
            )
        )
    else:
        raise ValueError(args.artifact_command)
    return 0


def command_eval(args: argparse.Namespace) -> int:
    client = _client(args)
    if args.eval_command == "status":
        _emit(_evaluation_location(client.get(f"/api/v1/evaluations/{args.evaluation_id}")))
        return 0
    if args.eval_command == "wait":
        _emit(_evaluation_location(_wait_for_status(client, f"/api/v1/evaluations/{args.evaluation_id}", args.timeout_s)))
        return 0
    kind = {"verify": "verify", "probe": "probe", "submit": "submit"}[args.eval_command]
    payload = {
        "experiment_id": _env("AO_EXPERIMENT_ID"),
        "assignment_id": _env("AO_ASSIGNMENT_ID"),
        "task_id": args.task_id or _env("AO_TASK_ID"),
        "kind": kind,
        "probe_kind": getattr(args, "kind", None),
        "workspace_root": os.environ.get("AO_WORKSPACE_ROOT"),
        "attempt_id": _attempt_id_arg(args),
    }
    if getattr(args, "entry", None):
        payload["entry_path"] = str(Path(args.entry).resolve())
    if getattr(args, "artifact_id", None):
        payload["artifact_id"] = args.artifact_id
    if getattr(args, "sync", False):
        payload["async"] = False
    if getattr(args, "run_async", False):
        payload["async"] = True
    if getattr(args, "environment_id", None):
        payload["environment_id"] = args.environment_id
    if getattr(args, "environment_overlay_id", None):
        payload["environment_overlay_id"] = args.environment_overlay_id
    _emit(_evaluation_location(client.post("/api/v1/evaluations", payload)))
    return 0


def command_finding(args: argparse.Namespace) -> int:
    if args.finding_command == "search" and _workspace_root() is not None:
        _emit(_workspace_reference("findings", files=["history/findings/index.jsonl"], directories=["history/findings"], query=args.query))
        return 0
    client = _client(args)
    if args.finding_command == "share":
        body = _read_text_arg(args.body, args.file)
        _emit(
            client.post(
                "/api/v1/findings",
                {
                    "experiment_id": _env("AO_EXPERIMENT_ID"),
                    "assignment_id": _env("AO_ASSIGNMENT_ID"),
                    "task_id": args.task_id or _env("AO_TASK_ID"),
                    "finding_type": args.type,
                    "title": args.title,
                    "body": body,
                },
            )
        )
    elif args.finding_command == "search":
        _emit(client.get("/api/v1/findings", {"task_id": args.task_id or _env("AO_TASK_ID"), "query": args.query}))
    else:
        raise ValueError(args.finding_command)
    return 0


def command_notebook(args: argparse.Namespace) -> int:
    if args.notebook_command == "list" and _workspace_root() is not None:
        _emit(_workspace_reference("notebooks", files=["history/notebooks/index.jsonl"], directories=["history/notebooks"]))
        return 0
    client = _client(args)
    if args.notebook_command == "checkpoint":
        content = _read_text_arg(args.content, args.file)
        uri = Path(args.file).resolve().as_uri() if args.file else None
        _emit(
            client.post(
                "/api/v1/notebook-checkpoints",
                {
                    "experiment_id": _env("AO_EXPERIMENT_ID"),
                    "assignment_id": _env("AO_ASSIGNMENT_ID"),
                    "session_id": os.environ.get("AO_SESSION_ID"),
                    "agent_id": _env("AO_AGENT_ID"),
                    "notebook_uri": uri,
                    "content": content,
                    "metadata": {"kind": args.kind},
                },
            )
        )
    elif args.notebook_command == "list":
        _emit(client.get("/api/v1/notebook-checkpoints", {"assignment_id": _env("AO_ASSIGNMENT_ID")}))
    else:
        raise ValueError(args.notebook_command)
    return 0


def command_job(args: argparse.Namespace) -> int:
    if args.job_command == "list" and _workspace_root() is not None:
        _emit(_workspace_reference("jobs", directories=["history/jobs"], attempt_id=_attempt_id_arg(args)))
        return 0
    client = _client(args)
    if args.job_command == "create":
        payload = {
            "experiment_id": _env("AO_EXPERIMENT_ID"),
            "assignment_id": _env("AO_ASSIGNMENT_ID"),
            "session_id": os.environ.get("AO_SESSION_ID"),
            "attempt_id": _attempt_id_arg(args),
            "task_id": _env("AO_TASK_ID"),
            "provider": args.provider,
            "inputs": {"command": args.command},
        }
        if args.cwd:
            payload["cwd"] = str(Path(args.cwd).resolve())
        if args.image:
            payload["image"] = args.image
        if args.environment_id:
            payload["environment_id"] = args.environment_id
        if args.environment_overlay_id:
            payload["environment_overlay_id"] = args.environment_overlay_id
        if args.network_mode:
            payload["network_mode"] = args.network_mode
        if args.requires_control_plane:
            payload["requires_control_plane"] = True
            payload["control_plane_url"] = _env("AO_CONTROL_API_URL")
        if args.template_id:
            payload["template_id"] = args.template_id
        if args.gpu_type_id:
            payload["gpu_type_ids"] = args.gpu_type_id
        if args.gpu_count is not None:
            payload["gpu_count"] = args.gpu_count
        if args.dry_run:
            payload["dry_run"] = True
        if args.env:
            payload["env"] = _parse_key_values(args.env)
        if args.requires_approval:
            payload["requires_approval"] = True
        if args.approved:
            payload["approved"] = True
        if args.estimated_cost_usd is not None:
            payload["estimated_cost"] = {"estimated_usd": args.estimated_cost_usd}
        _emit(
            client.post(
                "/api/v1/jobs",
                payload,
            )
        )
    elif args.job_command == "list":
        _emit(client.get("/api/v1/jobs", {"assignment_id": _env("AO_ASSIGNMENT_ID"), "attempt_id": _attempt_id_arg(args)}))
    elif args.job_command == "status":
        _emit(client.get(f"/api/v1/jobs/{args.job_id}"))
    elif args.job_command == "logs":
        record = client.get(f"/api/v1/jobs/{args.job_id}")
        outputs = record.get("outputs") or {}
        _emit(
            {
                "job_id": args.job_id,
                "status": record.get("status"),
                "stdout_path": outputs.get("stdout_path"),
                "stderr_path": outputs.get("stderr_path"),
                "outputs": outputs,
                "max_bytes_requested": args.max_bytes,
                "note": "Read log files directly with tail, rg, sed, or less; this command does not copy log contents into context.",
            }
        )
    elif args.job_command == "attach":
        payload = {
            "experiment_id": _env("AO_EXPERIMENT_ID"),
            "assignment_id": _env("AO_ASSIGNMENT_ID"),
            "session_id": os.environ.get("AO_SESSION_ID"),
            "attempt_id": _attempt_id_arg(args),
            "agent_id": _env("AO_AGENT_ID"),
            "mode": args.mode,
            "note": args.note,
        }
        _emit(client.post(f"/api/v1/jobs/{args.job_id}/attach", payload))
    elif args.job_command == "cancel":
        _emit(client.post(f"/api/v1/jobs/{args.job_id}/cancel", {}))
    elif args.job_command == "wait":
        _emit(_wait_for_status(client, f"/api/v1/jobs/{args.job_id}", args.timeout_s))
    else:
        raise ValueError(args.job_command)
    return 0


def command_env(args: argparse.Namespace) -> int:
    client = _client(args)
    assignment_id = args.assignment_id or os.environ.get("AO_ASSIGNMENT_ID")
    task_id = args.task_id or os.environ.get("AO_TASK_ID")
    if args.env_command == "status":
        environment_id = args.environment_id or os.environ.get("AO_ENVIRONMENT_ID")
        payload: dict[str, Any] = {}
        if environment_id:
            payload["environment"] = client.get(f"/api/v1/environments/{environment_id}")
        else:
            payload.update(client.get("/api/v1/environments", {"task_id": task_id}))
        if assignment_id:
            payload.update(client.get("/api/v1/environment-overlays", {"assignment_id": assignment_id}))
        _emit(payload)
    elif args.env_command == "ensure":
        _emit(
            client.post(
                "/api/v1/environments",
                {
                    "task_id": task_id,
                    "assignment_id": assignment_id,
                    "experiment_id": os.environ.get("AO_EXPERIMENT_ID"),
                },
            )
        )
    elif args.env_command == "install":
        _emit(
            client.post(
                "/api/v1/environment-overlays",
                {
                    "base_environment_id": args.environment_id or os.environ.get("AO_ENVIRONMENT_ID"),
                    "task_id": task_id,
                    "experiment_id": os.environ.get("AO_EXPERIMENT_ID"),
                    "assignment_id": assignment_id,
                    "session_id": os.environ.get("AO_SESSION_ID"),
                    "requested_by_agent_id": os.environ.get("AO_AGENT_ID"),
                    "requirements": args.pip,
                    "reason": args.reason,
                    "approved": args.approved,
                },
            )
        )
    elif args.env_command == "list-overlays":
        _emit(
            client.get(
                "/api/v1/environment-overlays",
                {
                    "assignment_id": assignment_id,
                    "base_environment_id": args.environment_id or os.environ.get("AO_ENVIRONMENT_ID"),
                    "status": args.status,
                },
            )
        )
    elif args.env_command == "overlay":
        _emit(client.get(f"/api/v1/environment-overlays/{args.overlay_id}"))
    elif args.env_command == "approve":
        _emit(client.post(f"/api/v1/environment-overlays/{args.overlay_id}/approve", {}))
    else:
        raise ValueError(args.env_command)
    return 0


def command_telemetry(args: argparse.Namespace) -> int:
    if args.telemetry_command == "list" and _workspace_root() is not None:
        _emit(_workspace_reference("telemetry", directories=["history/telemetry"], attempt_id=_attempt_id_arg(args)))
        return 0
    client = _client(args)
    if args.telemetry_command == "start":
        payload = {
            "experiment_id": _env("AO_EXPERIMENT_ID"),
            "assignment_id": _env("AO_ASSIGNMENT_ID"),
            "session_id": os.environ.get("AO_SESSION_ID"),
            "attempt_id": _attempt_id_arg(args),
            "provider": args.provider,
            "run_name": args.name,
            "params": _parse_json_arg(args.params),
            "tags": _parse_json_arg(args.tags),
        }
        if args.job_id:
            payload["job_id"] = args.job_id
        if args.tracking_uri:
            payload["tracking_uri"] = args.tracking_uri
        if args.experiment_name:
            payload["experiment_name"] = args.experiment_name
        _emit(client.post("/api/v1/telemetry-runs", payload))
    elif args.telemetry_command == "log-metrics":
        _emit(
            client.post(
                f"/api/v1/telemetry-runs/{args.telemetry_id}/metrics",
                {
                    "step": args.step,
                    "metrics": _parse_metrics(args.metric, args.metrics),
                },
            )
        )
    elif args.telemetry_command == "finish":
        _emit(client.post(f"/api/v1/telemetry-runs/{args.telemetry_id}/finish", {"status": args.status}))
    elif args.telemetry_command == "status":
        record = client.get(f"/api/v1/telemetry-runs/{args.telemetry_id}")
        _emit(
            _workspace_reference(
                "telemetry",
                directories=[f"history/telemetry/{args.telemetry_id}"],
                telemetry_id=args.telemetry_id,
                status=record.get("status"),
                provider=record.get("provider"),
            )
            if _workspace_root() is not None
            else record
        )
    elif args.telemetry_command == "list":
        _emit(client.get("/api/v1/telemetry-runs", {"assignment_id": _env("AO_ASSIGNMENT_ID"), "attempt_id": _attempt_id_arg(args)}))
    else:
        raise ValueError(args.telemetry_command)
    return 0


def command_attempt(args: argparse.Namespace) -> int:
    if args.attempt_command == "list" and _workspace_root() is not None:
        _emit(_workspace_reference("attempts", directories=["history/attempts"], status=args.status))
        return 0
    if args.attempt_command == "show" and _workspace_root() is not None:
        _emit(_workspace_reference("attempt", files=[f"history/attempts/{args.attempt_id}/attempt.json"], attempt_id=args.attempt_id))
        return 0
    client = _client(args)
    if args.attempt_command == "create":
        payload = {
            "experiment_id": args.experiment_id or _env("AO_EXPERIMENT_ID"),
            "assignment_id": args.assignment_id or os.environ.get("AO_ASSIGNMENT_ID"),
            "session_id": args.session_id or os.environ.get("AO_SESSION_ID"),
            "task_id": args.task_id or os.environ.get("AO_TASK_ID"),
            "agent_id": args.agent_id or os.environ.get("AO_AGENT_ID"),
            "direction_id": args.direction_id,
            "parent_attempt_id": args.parent_attempt_id,
            "candidate_artifact_id": args.candidate_artifact_id,
            "status": args.status,
            "metadata": _parse_json_arg(args.metadata),
        }
        _emit(client.post("/api/v1/attempts", payload))
    elif args.attempt_command == "list":
        _emit(
            client.get(
                "/api/v1/attempts",
                {
                    "experiment_id": args.experiment_id or os.environ.get("AO_EXPERIMENT_ID"),
                    "assignment_id": args.assignment_id or os.environ.get("AO_ASSIGNMENT_ID"),
                    "session_id": args.session_id,
                    "task_id": args.task_id or os.environ.get("AO_TASK_ID"),
                    "parent_attempt_id": args.parent_attempt_id,
                    "status": args.status,
                },
            )
        )
    elif args.attempt_command == "show":
        _emit(client.get(f"/api/v1/attempts/{args.attempt_id}"))
    elif args.attempt_command == "update":
        payload: dict[str, Any] = {"metadata": _parse_json_arg(args.metadata)}
        if args.status:
            payload["status"] = args.status
        if args.session_id:
            payload["session_id"] = args.session_id
        if args.candidate_artifact_id:
            payload["candidate_artifact_id"] = args.candidate_artifact_id
        _emit(client.patch(f"/api/v1/attempts/{args.attempt_id}", payload))
    else:
        raise ValueError(args.attempt_command)
    return 0


def command_tool(args: argparse.Namespace) -> int:
    if args.tool_command == "list" and _workspace_root() is not None:
        _emit(_workspace_reference("shared-tools", directories=["history/shared_tools", "shared_tools"], query=args.query, status=args.status))
        return 0
    if args.tool_command == "show" and _workspace_root() is not None:
        _emit(_workspace_reference("shared-tool", files=[f"history/shared_tools/{args.tool_id}/tool.json"], tool_id=args.tool_id))
        return 0
    client = _client(args)
    if args.tool_command == "publish":
        _emit(
            client.post(
                "/api/v1/shared-tools",
                {
                    "path": str(Path(args.path).resolve()),
                    "name": args.name,
                    "description": args.description or "",
                    "task_id": args.task_id or os.environ.get("AO_TASK_ID"),
                    "experiment_id": os.environ.get("AO_EXPERIMENT_ID"),
                    "assignment_id": os.environ.get("AO_ASSIGNMENT_ID"),
                    "session_id": os.environ.get("AO_SESSION_ID"),
                    "agent_id": os.environ.get("AO_AGENT_ID"),
                    "scope": args.scope,
                    "entrypoint": args.entrypoint,
                    "version": args.version,
                    "runtime_requirements": args.runtime_requirement or [],
                },
            )
        )
    elif args.tool_command == "list":
        _emit(
            client.get(
                "/api/v1/shared-tools",
                {
                    "task_id": args.task_id or os.environ.get("AO_TASK_ID"),
                    "experiment_id": os.environ.get("AO_EXPERIMENT_ID"),
                    "query": args.query,
                    "status": args.status,
                },
            )
        )
    elif args.tool_command == "show":
        _emit(client.get(f"/api/v1/shared-tools/{args.tool_id}"))
    elif args.tool_command == "checkout":
        _emit(
            client.post(
                f"/api/v1/shared-tools/{args.tool_id}/checkout",
                {"destination_path": str(Path(args.destination).resolve()), "force": args.force},
            )
        )
    elif args.tool_command == "install":
        tool = client.get(f"/api/v1/shared-tools/{args.tool_id}")
        destination = Path(args.destination).resolve() if args.destination else Path(_env("AO_WORKSPACE_ROOT")) / "shared_tools" / tool["name"]
        _emit(
            client.post(
                f"/api/v1/shared-tools/{args.tool_id}/checkout",
                {"destination_path": str(destination), "force": args.force},
            )
        )
    else:
        raise ValueError(args.tool_command)
    return 0


def command_network(args: argparse.Namespace) -> int:
    if args.network_command in {"status", "policy"}:
        if _workspace_root() is not None:
            _emit(_workspace_reference("network", files=["context/network_policy.json", "history/network/policy.json"]))
            return 0
    elif args.network_command == "events" and _workspace_root() is not None:
        _emit(_workspace_reference("network-events", files=["history/network/events.jsonl"], limit=args.limit))
        return 0
    client = _client(args)
    if args.network_command in {"status", "policy"}:
        _emit(
            client.get(
                "/api/v1/network-policy",
                {
                    "experiment_id": os.environ.get("AO_EXPERIMENT_ID"),
                    "assignment_id": os.environ.get("AO_ASSIGNMENT_ID"),
                    "session_id": os.environ.get("AO_SESSION_ID"),
                },
            )
        )
    elif args.network_command == "events":
        _emit(
            client.get(
                "/api/v1/network-access-events",
                {
                    "experiment_id": os.environ.get("AO_EXPERIMENT_ID"),
                    "assignment_id": os.environ.get("AO_ASSIGNMENT_ID"),
                    "session_id": os.environ.get("AO_SESSION_ID"),
                    "limit": args.limit,
                },
            )
        )
    else:
        raise ValueError(args.network_command)
    return 0


def command_trace(args: argparse.Namespace) -> int:
    client = _client(args)
    if args.trace_command == "list":
        payload = client.get("/api/v1/agent-traces", _trace_filters(args))
        _emit({"agent_traces": [_trace_location(item) for item in payload.get("agent_traces", [])]})
    elif args.trace_command == "show":
        payload = client.get(f"/api/v1/agent-traces/{args.trace_id}")
        _emit(_trace_location(payload["trace"]))
    elif args.trace_command == "commands":
        payload = client.get(f"/api/v1/agent-traces/{args.trace_id}")
        trace = _trace_location(payload["trace"])
        _emit(
            {
                "trace_id": trace["trace_id"],
                "file": trace["files"].get("commands"),
                "format": "jsonl",
                "filter_requested": {
                    "failed_only": args.failed_only,
                    "semantic_only": args.semantic_only,
                },
                "trace": trace,
            }
        )
    elif args.trace_command == "events":
        payload = client.get(f"/api/v1/agent-traces/{args.trace_id}")
        trace = _trace_location(payload["trace"])
        _emit(
            {
                "trace_id": trace["trace_id"],
                "file": trace["files"].get("events"),
                "format": "jsonl",
                "query_requested": args.query,
                "limit_requested": args.limit,
                "trace": trace,
            }
        )
    elif args.trace_command == "search":
        payload = client.get("/api/v1/agent-traces", _trace_filters(args))
        traces = [_trace_location(item) for item in payload.get("agent_traces", [])]
        _emit(
            {
                "query": args.query,
                "agent_traces": traces,
                "search_files": [
                    {
                        "trace_id": trace["trace_id"],
                        "events": trace["files"].get("events"),
                        "commands": trace["files"].get("commands"),
                        "agent_messages": trace["files"].get("agent_messages"),
                        "stdout": trace["files"].get("stdout"),
                    }
                    for trace in traces
                ],
            }
        )
    else:
        raise ValueError(args.trace_command)
    return 0


def _trace_location(trace: dict[str, Any]) -> dict[str, Any]:
    metadata = trace.get("metadata") or {}
    return {
        "trace_id": trace.get("trace_id"),
        "experiment_id": trace.get("experiment_id"),
        "assignment_id": trace.get("assignment_id"),
        "session_id": trace.get("session_id"),
        "task_id": trace.get("task_id"),
        "agent_id": trace.get("agent_id"),
        "run_id": trace.get("run_id"),
        "turn_id": trace.get("turn_id"),
        "status": trace.get("status"),
        "artifact_id": trace.get("artifact_id"),
        "trace_root": trace.get("trace_root"),
        "counts": {
            "event_count": metadata.get("event_count"),
            "command_count": metadata.get("command_count"),
            "semantic_command_count": metadata.get("semantic_command_count"),
            "failed_command_count": metadata.get("failed_command_count"),
            "agent_message_count": metadata.get("agent_message_count"),
        },
        "files": _trace_files(trace),
    }


def _trace_files(trace: dict[str, Any]) -> dict[str, str | None]:
    root = trace.get("trace_root")
    if not root:
        return {
            "manifest": None,
            "events": None,
            "commands": None,
            "agent_messages": None,
            "stdout": None,
        }
    metadata = trace.get("metadata") or {}
    files = metadata.get("files") or {}
    return {
        "manifest": str(Path(root) / "manifest.json"),
        "events": str(Path(root) / (files.get("events") or "events.jsonl")),
        "commands": str(Path(root) / (files.get("commands") or "commands.jsonl")),
        "agent_messages": str(Path(root) / (files.get("agent_messages") or "agent_messages.jsonl")),
        "stdout": str(Path(root) / files["stdout"]) if files.get("stdout") else None,
    }


def _trace_filters(args: argparse.Namespace) -> dict[str, Any]:
    assignment_id = args.assignment_id
    if not getattr(args, "all_assignments", False):
        assignment_id = assignment_id or os.environ.get("AO_ASSIGNMENT_ID")
    return {
        "experiment_id": args.experiment_id or os.environ.get("AO_EXPERIMENT_ID"),
        "assignment_id": assignment_id,
        "session_id": args.session_id,
        "task_id": args.task_id or os.environ.get("AO_TASK_ID"),
        "agent_id": args.agent_id,
        "status": args.status,
        "attempt_id": args.attempt_id,
    }


def _evaluation_location(record: dict[str, Any]) -> dict[str, Any]:
    _materialize_evaluation_record(record)
    evaluation_id = record.get("evaluation_id")
    files: dict[str, str] = {}
    root = _workspace_root()
    if root is not None and evaluation_id:
        evaluation_root = root / "history" / "evaluations" / str(evaluation_id)
        files = {
            "evaluation": str(evaluation_root / "evaluation.json"),
            "request": str(evaluation_root / "request.json"),
            "result": str(evaluation_root / "result.json"),
            "public_feedback": str(evaluation_root / "public_feedback.json"),
            "feedback": str(evaluation_root / "feedback.json"),
        }
    return {
        "evaluation_id": evaluation_id,
        "status": record.get("status"),
        "kind": record.get("kind"),
        "valid": record.get("valid"),
        "score": record.get("score"),
        "artifact_id": record.get("artifact_id"),
        "attempt_id": record.get("attempt_id"),
        "files": files,
        "note": "Use the files when present; fresh status is shown here without dumping full result payloads.",
    }


def _materialize_evaluation_record(record: dict[str, Any]) -> None:
    root = _workspace_root()
    evaluation_id = record.get("evaluation_id")
    if root is None or not evaluation_id:
        return
    evaluations_root = root / "history" / "evaluations"
    evaluation_root = evaluations_root / str(evaluation_id)
    atomic_write_text(evaluation_root / "evaluation.json", _json_text(record))
    feedback = record.get("feedback")
    if feedback is None:
        feedback = record.get("public_feedback")
    for filename, payload in (
        ("request.json", record.get("request")),
        ("result.json", record.get("result")),
        ("public_feedback.json", record.get("public_feedback")),
        ("feedback.json", feedback),
    ):
        atomic_write_text(evaluation_root / filename, _json_text(payload))
    _refresh_evaluations_index(evaluations_root)


def _refresh_evaluations_index(root: Path) -> None:
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/evaluation.json")):
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(loaded, dict):
            records.append(loaded)
    atomic_write_text(root / "index.jsonl", "".join(json.dumps(record, sort_keys=True) + "\n" for record in records))


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _add_trace_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment-id")
    parser.add_argument("--assignment-id")
    parser.add_argument("--session-id")
    parser.add_argument("--task-id")
    parser.add_argument("--agent-id")
    parser.add_argument("--status")
    parser.add_argument("--attempt-id")
    parser.add_argument("--all-assignments", action="store_true")


def _wait_for_status(client: ControlPlaneClient, path: str, timeout_s: float) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    terminal = {"completed", "failed", "cancelled", "stopped"}
    last: dict[str, Any] = {}
    while time.time() < deadline:
        last = client.get(path)
        status = last.get("status")
        if status in terminal:
            return last
        time.sleep(0.25)
    return {"status": "timeout", "last": last}


def _parse_json_arg(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    decoded = json.loads(raw)
    if not isinstance(decoded, dict):
        raise ValueError("expected JSON object")
    return decoded


def _parse_metrics(items: list[str] | None, raw_json: str | None) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if raw_json:
        for key, value in _parse_json_arg(raw_json).items():
            metrics[str(key)] = float(value)
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"metric must use key=value form: {item!r}")
        key, value = item.split("=", 1)
        metrics[key] = float(value)
    if not metrics:
        raise ValueError("at least one metric is required")
    return metrics


def _parse_key_values(items: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"value must use KEY=VALUE form: {item!r}")
        key, value = item.split("=", 1)
        result[key] = value
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="semantic-worker-tool")
    parser.add_argument("--api-url")
    subparsers = parser.add_subparsers(dest="tool", required=True)

    ctx = subparsers.add_parser("ctx")
    ctx.add_argument("--assignment-id")
    ctx.add_argument("--task-id")
    ctx_sub = ctx.add_subparsers(dest="ctx_command", required=True)
    for name in ("context", "assignment", "task", "evaluations", "artifacts", "jobs", "environments", "telemetry"):
        ctx_sub.add_parser(name)
    ctx_attempts = ctx_sub.add_parser("attempts")
    ctx_attempts.add_argument("--status")
    findings = ctx_sub.add_parser("findings")
    findings.add_argument("query", nargs="?")
    ctx_tools = ctx_sub.add_parser("shared-tools")
    ctx_tools.add_argument("query", nargs="?")
    ctx_sub.add_parser("network")
    ctx_stop = ctx_sub.add_parser("stop")
    ctx_stop.add_argument("--reason", required=True)
    ctx_local_stop = ctx_sub.add_parser("local-stop")
    ctx_local_stop.add_argument("--reason", required=True)
    ctx_local_stop_alias = ctx_sub.add_parser("local_stop")
    ctx_local_stop_alias.add_argument("--reason", required=True)
    ctx_global_stop = ctx_sub.add_parser("global-stop")
    ctx_global_stop.add_argument("--reason", required=True)
    ctx_global_stop.add_argument("--confirm-global-stop", action="store_true")
    ctx_global_stop_alias = ctx_sub.add_parser("global_stop")
    ctx_global_stop_alias.add_argument("--reason", required=True)
    ctx_global_stop_alias.add_argument("--confirm-global-stop", action="store_true")
    leaderboard = ctx_sub.add_parser("leaderboard")
    leaderboard.add_argument("--limit", type=int, default=20)
    incumbent = ctx_sub.add_parser("incumbent")
    incumbent.add_argument("--direction-id")
    ctx.set_defaults(func=command_ctx)

    artifact = subparsers.add_parser("artifact")
    artifact_sub = artifact.add_subparsers(dest="artifact_command", required=True)
    upload = artifact_sub.add_parser("upload")
    upload.add_argument("--path", required=True)
    upload.add_argument("--kind", default="generic")
    upload.add_argument("--note")
    upload.add_argument("--attempt-id")
    artifact_list = artifact_sub.add_parser("list")
    artifact_list.add_argument("--attempt-id")
    checkout = artifact_sub.add_parser("checkout-incumbent")
    checkout.add_argument("--destination", required=True)
    checkout.add_argument("--direction-id")
    checkout.add_argument("--force", action="store_true")
    artifact.set_defaults(func=command_artifact)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--task-id")
    eval_sub = eval_parser.add_subparsers(dest="eval_command", required=True)
    for name in ("verify", "submit"):
        item = eval_sub.add_parser(name)
        item.add_argument("--entry")
        item.add_argument("--artifact-id")
        item.add_argument("--environment-id")
        item.add_argument("--environment-overlay-id")
        item.add_argument("--attempt-id")
        item.add_argument("--sync", action="store_true")
        item.add_argument("--async", dest="run_async", action="store_true")
    probe = eval_sub.add_parser("probe")
    probe.add_argument("--entry")
    probe.add_argument("--artifact-id")
    probe.add_argument("--environment-id")
    probe.add_argument("--environment-overlay-id")
    probe.add_argument("--attempt-id")
    probe.add_argument("--kind", default="diagnostics")
    probe.add_argument("--sync", action="store_true")
    probe.add_argument("--async", dest="run_async", action="store_true")
    eval_status = eval_sub.add_parser("status")
    eval_status.add_argument("evaluation_id")
    eval_wait = eval_sub.add_parser("wait")
    eval_wait.add_argument("evaluation_id")
    eval_wait.add_argument("--timeout-s", type=float, default=120.0)
    eval_parser.set_defaults(func=command_eval)

    finding = subparsers.add_parser("finding")
    finding.add_argument("--task-id")
    finding_sub = finding.add_subparsers(dest="finding_command", required=True)
    share = finding_sub.add_parser("share")
    share.add_argument("--type", default="insight")
    share.add_argument("--title", required=True)
    share.add_argument("--body")
    share.add_argument("--file")
    search = finding_sub.add_parser("search")
    search.add_argument("query")
    finding.set_defaults(func=command_finding)

    notebook = subparsers.add_parser("notebook")
    notebook_sub = notebook.add_subparsers(dest="notebook_command", required=True)
    checkpoint = notebook_sub.add_parser("checkpoint")
    checkpoint.add_argument("--file")
    checkpoint.add_argument("--content")
    checkpoint.add_argument("--kind", default="manual")
    notebook_sub.add_parser("list")
    notebook.set_defaults(func=command_notebook)

    job = subparsers.add_parser("job")
    job_sub = job.add_subparsers(dest="job_command", required=True)
    create = job_sub.add_parser("create")
    create.add_argument("--provider", default="local")
    create.add_argument("--command", required=True)
    create.add_argument("--cwd")
    create.add_argument("--image")
    create.add_argument("--environment-id")
    create.add_argument("--environment-overlay-id")
    create.add_argument("--network-mode")
    create.add_argument("--requires-control-plane", action="store_true")
    create.add_argument("--template-id")
    create.add_argument("--gpu-type-id", action="append")
    create.add_argument("--gpu-count", type=int)
    create.add_argument("--env", action="append")
    create.add_argument("--attempt-id")
    create.add_argument("--dry-run", action="store_true")
    create.add_argument("--requires-approval", action="store_true")
    create.add_argument("--approved", action="store_true")
    create.add_argument("--estimated-cost-usd", type=float)
    job_list = job_sub.add_parser("list")
    job_list.add_argument("--attempt-id")
    job_status = job_sub.add_parser("status")
    job_status.add_argument("job_id")
    job_logs = job_sub.add_parser("logs")
    job_logs.add_argument("job_id")
    job_logs.add_argument("--max-bytes", type=int, default=200_000)
    job_attach = job_sub.add_parser("attach")
    job_attach.add_argument("job_id")
    job_attach.add_argument("--attempt-id")
    job_attach.add_argument("--mode", choices=("observe", "continue"), default="observe")
    job_attach.add_argument("--note")
    job_cancel = job_sub.add_parser("cancel")
    job_cancel.add_argument("job_id")
    job_wait = job_sub.add_parser("wait")
    job_wait.add_argument("job_id")
    job_wait.add_argument("--timeout-s", type=float, default=120.0)
    job.set_defaults(func=command_job)

    env_parser = subparsers.add_parser("env")
    env_parser.add_argument("--assignment-id")
    env_parser.add_argument("--task-id")
    env_sub = env_parser.add_subparsers(dest="env_command", required=True)
    env_status = env_sub.add_parser("status")
    env_status.add_argument("--environment-id")
    env_sub.add_parser("ensure")
    env_install = env_sub.add_parser("install")
    env_install.add_argument("--pip", action="append", required=True)
    env_install.add_argument("--environment-id")
    env_install.add_argument("--reason", required=True)
    env_install.add_argument("--approved", action="store_true")
    env_list = env_sub.add_parser("list-overlays")
    env_list.add_argument("--environment-id")
    env_list.add_argument("--status")
    env_overlay = env_sub.add_parser("overlay")
    env_overlay.add_argument("overlay_id")
    env_approve = env_sub.add_parser("approve")
    env_approve.add_argument("overlay_id")
    env_parser.set_defaults(func=command_env)

    telemetry = subparsers.add_parser("telemetry")
    telemetry_sub = telemetry.add_subparsers(dest="telemetry_command", required=True)
    telemetry_start = telemetry_sub.add_parser("start")
    telemetry_start.add_argument("--provider", default="local")
    telemetry_start.add_argument("--name")
    telemetry_start.add_argument("--job-id")
    telemetry_start.add_argument("--attempt-id")
    telemetry_start.add_argument("--params")
    telemetry_start.add_argument("--tags")
    telemetry_start.add_argument("--tracking-uri")
    telemetry_start.add_argument("--experiment-name")
    telemetry_log = telemetry_sub.add_parser("log-metrics")
    telemetry_log.add_argument("telemetry_id")
    telemetry_log.add_argument("--metric", action="append")
    telemetry_log.add_argument("--metrics")
    telemetry_log.add_argument("--step", type=int)
    telemetry_finish = telemetry_sub.add_parser("finish")
    telemetry_finish.add_argument("telemetry_id")
    telemetry_finish.add_argument("--status", default="completed")
    telemetry_status = telemetry_sub.add_parser("status")
    telemetry_status.add_argument("telemetry_id")
    telemetry_list = telemetry_sub.add_parser("list")
    telemetry_list.add_argument("--attempt-id")
    telemetry.set_defaults(func=command_telemetry)

    attempt = subparsers.add_parser("attempt")
    attempt_sub = attempt.add_subparsers(dest="attempt_command", required=True)
    attempt_create = attempt_sub.add_parser("create")
    attempt_create.add_argument("--experiment-id")
    attempt_create.add_argument("--assignment-id")
    attempt_create.add_argument("--session-id")
    attempt_create.add_argument("--task-id")
    attempt_create.add_argument("--agent-id")
    attempt_create.add_argument("--direction-id")
    attempt_create.add_argument("--parent-attempt-id")
    attempt_create.add_argument("--candidate-artifact-id")
    attempt_create.add_argument("--status", default="active")
    attempt_create.add_argument("--metadata")
    attempt_list = attempt_sub.add_parser("list")
    attempt_list.add_argument("--experiment-id")
    attempt_list.add_argument("--assignment-id")
    attempt_list.add_argument("--session-id")
    attempt_list.add_argument("--task-id")
    attempt_list.add_argument("--parent-attempt-id")
    attempt_list.add_argument("--status")
    attempt_show = attempt_sub.add_parser("show")
    attempt_show.add_argument("attempt_id")
    attempt_update = attempt_sub.add_parser("update")
    attempt_update.add_argument("attempt_id")
    attempt_update.add_argument("--status")
    attempt_update.add_argument("--session-id")
    attempt_update.add_argument("--candidate-artifact-id")
    attempt_update.add_argument("--metadata")
    attempt.set_defaults(func=command_attempt)

    tool = subparsers.add_parser("tool")
    tool.add_argument("--task-id")
    tool_sub = tool.add_subparsers(dest="tool_command", required=True)
    tool_publish = tool_sub.add_parser("publish")
    tool_publish.add_argument("--path", required=True)
    tool_publish.add_argument("--name", required=True)
    tool_publish.add_argument("--description")
    tool_publish.add_argument("--scope", default="task")
    tool_publish.add_argument("--entrypoint")
    tool_publish.add_argument("--version", default="1")
    tool_publish.add_argument("--runtime-requirement", action="append")
    tool_list = tool_sub.add_parser("list")
    tool_list.add_argument("query", nargs="?")
    tool_list.add_argument("--status", default="active")
    tool_show = tool_sub.add_parser("show")
    tool_show.add_argument("tool_id")
    tool_checkout = tool_sub.add_parser("checkout")
    tool_checkout.add_argument("tool_id")
    tool_checkout.add_argument("--destination", required=True)
    tool_checkout.add_argument("--force", action="store_true")
    tool_install = tool_sub.add_parser("install")
    tool_install.add_argument("tool_id")
    tool_install.add_argument("--destination")
    tool_install.add_argument("--force", action="store_true")
    tool.set_defaults(func=command_tool)

    network = subparsers.add_parser("network")
    network_sub = network.add_subparsers(dest="network_command", required=True)
    network_sub.add_parser("status")
    network_sub.add_parser("policy")
    network_events = network_sub.add_parser("events")
    network_events.add_argument("--limit", type=int, default=200)
    network.set_defaults(func=command_network)

    trace = subparsers.add_parser("trace")
    trace_sub = trace.add_subparsers(dest="trace_command", required=True)
    trace_list = trace_sub.add_parser("list")
    _add_trace_filter_args(trace_list)
    trace_show = trace_sub.add_parser("show")
    trace_show.add_argument("trace_id")
    trace_commands = trace_sub.add_parser("commands")
    trace_commands.add_argument("trace_id")
    trace_commands.add_argument("--failed-only", action="store_true")
    trace_commands.add_argument("--semantic-only", action="store_true")
    trace_events = trace_sub.add_parser("events")
    trace_events.add_argument("trace_id")
    trace_events.add_argument("--query", "-q")
    trace_events.add_argument("--limit", type=int, default=200)
    trace_search = trace_sub.add_parser("search")
    trace_search.add_argument("query")
    _add_trace_filter_args(trace_search)
    trace.set_defaults(func=command_trace)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
