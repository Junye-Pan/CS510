from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from agentic_opt.control_plane.client import ControlPlaneClient


def _env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _client(args: argparse.Namespace) -> ControlPlaneClient:
    return ControlPlaneClient(args.api_url or _env("AO_CONTROL_API_URL"))


def _emit(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _read_text_arg(text: str | None, file_path: str | None) -> str:
    if text and file_path:
        raise ValueError("pass either --body/--content or --file, not both")
    if text is not None:
        return text
    if file_path is not None:
        return Path(file_path).read_text(encoding="utf-8")
    raise ValueError("text or file path is required")


def command_ctx(args: argparse.Namespace) -> int:
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
    else:
        raise ValueError(args.ctx_command)
    return 0


def command_artifact(args: argparse.Namespace) -> int:
    client = _client(args)
    if args.artifact_command == "upload":
        path = Path(args.path).resolve()
        _emit(
            client.post(
                "/api/v1/artifacts",
                {
                    "experiment_id": _env("AO_EXPERIMENT_ID"),
                    "assignment_id": _env("AO_ASSIGNMENT_ID"),
                    "kind": args.kind,
                    "path": str(path),
                    "metadata": {"note": args.note} if args.note else {},
                },
            )
        )
    elif args.artifact_command == "list":
        _emit(client.get("/api/v1/artifacts", {"assignment_id": _env("AO_ASSIGNMENT_ID")}))
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
        _emit(client.get(f"/api/v1/evaluations/{args.evaluation_id}"))
        return 0
    if args.eval_command == "wait":
        _emit(_wait_for_status(client, f"/api/v1/evaluations/{args.evaluation_id}", args.timeout_s))
        return 0
    kind = {"verify": "verify", "probe": "probe", "submit": "submit"}[args.eval_command]
    payload = {
        "experiment_id": _env("AO_EXPERIMENT_ID"),
        "assignment_id": _env("AO_ASSIGNMENT_ID"),
        "task_id": args.task_id or _env("AO_TASK_ID"),
        "kind": kind,
        "probe_kind": getattr(args, "kind", None),
        "workspace_root": os.environ.get("AO_WORKSPACE_ROOT"),
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
    _emit(
        client.post(
            "/api/v1/evaluations",
            payload,
        )
    )
    return 0


def command_finding(args: argparse.Namespace) -> int:
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
    client = _client(args)
    if args.job_command == "create":
        payload = {
            "experiment_id": _env("AO_EXPERIMENT_ID"),
            "assignment_id": _env("AO_ASSIGNMENT_ID"),
            "session_id": os.environ.get("AO_SESSION_ID"),
            "provider": args.provider,
            "inputs": {"command": args.command},
        }
        if args.cwd:
            payload["cwd"] = str(Path(args.cwd).resolve())
        if args.image:
            payload["image"] = args.image
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
        _emit(client.get("/api/v1/jobs", {"assignment_id": _env("AO_ASSIGNMENT_ID")}))
    elif args.job_command == "status":
        _emit(client.get(f"/api/v1/jobs/{args.job_id}"))
    elif args.job_command == "logs":
        _emit(client.get(f"/api/v1/jobs/{args.job_id}/logs", {"max_bytes": args.max_bytes}))
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
    client = _client(args)
    if args.telemetry_command == "start":
        payload = {
            "experiment_id": _env("AO_EXPERIMENT_ID"),
            "assignment_id": _env("AO_ASSIGNMENT_ID"),
            "session_id": os.environ.get("AO_SESSION_ID"),
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
        _emit(client.get(f"/api/v1/telemetry-runs/{args.telemetry_id}"))
    elif args.telemetry_command == "list":
        _emit(client.get("/api/v1/telemetry-runs", {"assignment_id": _env("AO_ASSIGNMENT_ID")}))
    else:
        raise ValueError(args.telemetry_command)
    return 0


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
    findings = ctx_sub.add_parser("findings")
    findings.add_argument("query", nargs="?")
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
    artifact_sub.add_parser("list")
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
        item.add_argument("--sync", action="store_true")
        item.add_argument("--async", dest="run_async", action="store_true")
    probe = eval_sub.add_parser("probe")
    probe.add_argument("--entry")
    probe.add_argument("--artifact-id")
    probe.add_argument("--environment-id")
    probe.add_argument("--environment-overlay-id")
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
    create.add_argument("--template-id")
    create.add_argument("--gpu-type-id", action="append")
    create.add_argument("--gpu-count", type=int)
    create.add_argument("--env", action="append")
    create.add_argument("--dry-run", action="store_true")
    create.add_argument("--requires-approval", action="store_true")
    create.add_argument("--approved", action="store_true")
    create.add_argument("--estimated-cost-usd", type=float)
    job_sub.add_parser("list")
    job_status = job_sub.add_parser("status")
    job_status.add_argument("job_id")
    job_logs = job_sub.add_parser("logs")
    job_logs.add_argument("job_id")
    job_logs.add_argument("--max-bytes", type=int, default=200_000)
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
    telemetry_sub.add_parser("list")
    telemetry.set_defaults(func=command_telemetry)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
