from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from agentic_opt.control_plane.resources import object_model_schema
from agentic_opt.control_plane.service import task_contract

if TYPE_CHECKING:  # pragma: no cover
    from flask import Flask

    from .app import WebContext


def register_control_plane_routes(app: "Flask", ctx: "WebContext") -> None:
    from flask import jsonify, request

    def _payload() -> dict[str, Any]:
        return request.get_json(force=True, silent=True) or {}

    def _error(exc: Exception, status: int = 400):
        return jsonify({"error": str(exc), "error_type": type(exc).__name__}), status

    @app.get("/api/v1/object-model")
    def control_object_model():
        return jsonify(object_model_schema())

    @app.get("/api/v1/tasks/<task_id>")
    def control_task(task_id: str):
        try:
            return jsonify(task_contract(task_id))
        except Exception as exc:
            return _error(exc, 404)

    @app.get("/api/v1/experiments")
    def control_list_experiments():
        return jsonify({"experiments": ctx.control.list_experiments()})

    @app.post("/api/v1/experiments")
    def control_create_experiment():
        try:
            payload = _payload()
            record = ctx.control.create_experiment(payload)
            assignment_count = int(payload.get("assignment_count") or payload.get("num_workers") or 0)
            assignments = []
            if assignment_count > 0:
                assignments = ctx.control_service.generate_assignments(
                    experiment_id=record["experiment_id"],
                    count=assignment_count,
                    worker_backend=payload.get("worker_backend") or "codex-local",
                )
            return jsonify({"experiment": record, "assignments": assignments}), 201
        except Exception as exc:
            return _error(exc)

    @app.get("/api/v1/experiments/<experiment_id>")
    def control_get_experiment(experiment_id: str):
        experiment = ctx.control.get_experiment(experiment_id)
        if experiment is None:
            return jsonify({"error": "experiment not found"}), 404
        return jsonify(
            {
                "experiment": experiment,
                "assignments": ctx.control.list_assignments(experiment_id=experiment_id),
                "sessions": ctx.control.list_sessions(experiment_id=experiment_id),
                "attempts": ctx.control.list_attempts(experiment_id=experiment_id),
                "agent_traces": ctx.control.list_agent_traces(experiment_id=experiment_id),
                "trace_exports": ctx.control.list_trace_export_runs(experiment_id=experiment_id),
                "jobs": ctx.control.list_jobs(experiment_id=experiment_id),
                "environments": ctx.control.list_environments(experiment_id=experiment_id)
                or ctx.control.list_environments(task_id=experiment["task_id"]),
                "environment_overlays": ctx.control.list_environment_overlays(experiment_id=experiment_id),
                "evaluations": ctx.control.list_evaluations(experiment_id=experiment_id),
                "leaderboard": ctx.control.list_leaderboard_entries(experiment_id=experiment_id, limit=20),
                "incumbent": ctx.control.get_incumbent(experiment_id=experiment_id),
                "telemetry_runs": ctx.control.list_telemetry_runs(experiment_id=experiment_id),
                "artifacts": ctx.control.list_artifacts(experiment_id=experiment_id),
                "shared_tools": ctx.control.list_shared_tools(task_id=experiment["task_id"], experiment_id=experiment_id),
                "task_knowledge": ctx.control_service.task_knowledge_inventory(task_id=experiment["task_id"]),
                "network_policy": ctx.control_service.network_policy({"experiment_id": experiment_id}),
                "network_access_events": ctx.control.list_network_access_events(experiment_id=experiment_id, limit=100),
                "findings": ctx.control.list_findings(experiment_id=experiment_id),
                "events": ctx.control.list_events(experiment_id=experiment_id, limit=100),
            }
        )

    @app.get("/api/v1/experiments/<experiment_id>/analysis")
    def control_get_experiment_analysis(experiment_id: str):
        try:
            limit = int(request.args.get("leaderboard_limit") or 500)
            return jsonify(ctx.control_service.run_analysis(experiment_id, leaderboard_limit=limit))
        except Exception as exc:
            return _error(exc, 404)

    @app.patch("/api/v1/experiments/<experiment_id>")
    def control_update_experiment(experiment_id: str):
        try:
            payload = _payload()
            return jsonify(
                ctx.control.update_experiment_status(
                    experiment_id,
                    payload.get("status") or "updated",
                    metadata=payload.get("metadata"),
                )
            )
        except Exception as exc:
            return _error(exc, 404)

    @app.post("/api/v1/experiments/<experiment_id>/assignments")
    def control_create_assignment(experiment_id: str):
        try:
            payload = {**_payload(), "experiment_id": experiment_id}
            return jsonify(ctx.control.create_assignment(payload)), 201
        except Exception as exc:
            return _error(exc, 404)

    @app.post("/api/v1/experiments/<experiment_id>/assignments/generate")
    def control_generate_assignments(experiment_id: str):
        try:
            payload = _payload()
            assignments = ctx.control_service.generate_assignments(
                experiment_id=experiment_id,
                count=int(payload.get("count") or 1),
                worker_backend=payload.get("worker_backend") or "codex-local",
            )
            return jsonify({"assignments": assignments}), 201
        except Exception as exc:
            return _error(exc, 404)

    @app.get("/api/v1/assignments/<assignment_id>")
    def control_get_assignment(assignment_id: str):
        assignment = ctx.control.get_assignment(assignment_id)
        if assignment is None:
            return jsonify({"error": "assignment not found"}), 404
        return jsonify(ctx.control_service.context_for_assignment(assignment_id))

    @app.patch("/api/v1/assignments/<assignment_id>")
    def control_update_assignment(assignment_id: str):
        try:
            payload = _payload()
            return jsonify(
                ctx.control.update_assignment_status(
                    assignment_id,
                    payload.get("status") or "updated",
                    metadata=payload.get("metadata"),
                )
            )
        except Exception as exc:
            return _error(exc, 404)

    @app.post("/api/v1/assignments/<assignment_id>/sessions")
    def control_create_session(assignment_id: str):
        try:
            return jsonify(ctx.control.create_session({**_payload(), "assignment_id": assignment_id})), 201
        except Exception as exc:
            return _error(exc, 404)

    @app.post("/api/v1/assignments/<assignment_id>/workspace-bootstrap")
    def control_bootstrap_workspace(assignment_id: str):
        try:
            return jsonify(ctx.control_service.bootstrap_workspace(assignment_id, _payload())), 201
        except Exception as exc:
            return _error(exc, 404)

    @app.post("/api/v1/assignments/<assignment_id>/start-local")
    def control_start_local_assignment(assignment_id: str):
        try:
            payload = _payload()
            api_url = payload.get("api_url") or request.url_root.rstrip("/")
            session = ctx.workers.start_control_assignment(
                assignment_id=assignment_id,
                api_url=api_url,
                dry_run=bool(payload.get("dry_run")),
                max_turn_wall_time_s=payload.get("max_turn_wall_time_s"),
            )
            return jsonify(session), 202
        except Exception as exc:
            return _error(exc, 404)

    @app.patch("/api/v1/sessions/<session_id>")
    def control_update_session(session_id: str):
        try:
            return jsonify(ctx.control.update_session(session_id, _payload()))
        except Exception as exc:
            return _error(exc, 404)

    @app.get("/api/v1/context")
    def control_context():
        assignment_id = request.args.get("assignment_id")
        if not assignment_id:
            return jsonify({"error": "assignment_id is required"}), 400
        try:
            return jsonify(ctx.control_service.context_for_assignment(assignment_id))
        except Exception as exc:
            return _error(exc, 404)

    @app.post("/api/v1/environments")
    def control_ensure_environment():
        try:
            return jsonify(ctx.control_service.ensure_environment(_payload())), 201
        except Exception as exc:
            return _error(exc)

    @app.get("/api/v1/environments")
    def control_list_environments():
        return jsonify(
            {
                "environments": ctx.control.list_environments(
                    task_id=request.args.get("task_id"),
                    experiment_id=request.args.get("experiment_id"),
                    environment_type=request.args.get("environment_type"),
                )
            }
        )

    @app.get("/api/v1/environments/<environment_id>")
    def control_get_environment(environment_id: str):
        environment = ctx.control.get_environment(environment_id)
        if environment is None:
            return jsonify({"error": "environment not found"}), 404
        return jsonify(environment)

    @app.post("/api/v1/environments/<environment_id>/export-bundle")
    def control_export_environment_bundle(environment_id: str):
        try:
            return jsonify(ctx.control_service.export_environment_bundle(environment_id, _payload())), 201
        except Exception as exc:
            return _error(exc, 404)

    @app.post("/api/v1/environment-overlays")
    def control_create_environment_overlay():
        try:
            return jsonify(ctx.control_service.create_environment_overlay(_payload())), 201
        except Exception as exc:
            return _error(exc)

    @app.get("/api/v1/environment-overlays")
    def control_list_environment_overlays():
        return jsonify(
            {
                "environment_overlays": ctx.control.list_environment_overlays(
                    base_environment_id=request.args.get("base_environment_id"),
                    experiment_id=request.args.get("experiment_id"),
                    assignment_id=request.args.get("assignment_id"),
                    status=request.args.get("status"),
                )
            }
        )

    @app.get("/api/v1/environment-overlays/<overlay_id>")
    def control_get_environment_overlay(overlay_id: str):
        overlay = ctx.control.get_environment_overlay(overlay_id)
        if overlay is None:
            return jsonify({"error": "environment overlay not found"}), 404
        return jsonify(overlay)

    @app.post("/api/v1/environment-overlays/<overlay_id>/approve")
    def control_approve_environment_overlay(overlay_id: str):
        try:
            return jsonify(ctx.control_service.approve_environment_overlay(overlay_id))
        except Exception as exc:
            return _error(exc, 404)

    @app.post("/api/v1/attempts")
    def control_create_attempt():
        try:
            return jsonify(ctx.control_service.create_attempt(_payload())), 201
        except Exception as exc:
            return _error(exc)

    @app.get("/api/v1/attempts")
    def control_list_attempts():
        return jsonify(
            {
                "attempts": ctx.control.list_attempts(
                    experiment_id=request.args.get("experiment_id"),
                    assignment_id=request.args.get("assignment_id"),
                    session_id=request.args.get("session_id"),
                    task_id=request.args.get("task_id"),
                    parent_attempt_id=request.args.get("parent_attempt_id"),
                    status=request.args.get("status"),
                )
            }
        )

    @app.get("/api/v1/attempts/<attempt_id>")
    def control_get_attempt(attempt_id: str):
        try:
            return jsonify(ctx.control_service.attempt_context(attempt_id))
        except Exception as exc:
            return _error(exc, 404)

    @app.patch("/api/v1/attempts/<attempt_id>")
    def control_update_attempt(attempt_id: str):
        try:
            return jsonify(ctx.control_service.update_attempt(attempt_id, _payload()))
        except Exception as exc:
            return _error(exc, 404)

    @app.get("/api/v1/agent-traces")
    def control_list_agent_traces():
        return jsonify(
            {
                "agent_traces": ctx.control_service.list_agent_traces(
                    experiment_id=request.args.get("experiment_id"),
                    assignment_id=request.args.get("assignment_id"),
                    session_id=request.args.get("session_id"),
                    task_id=request.args.get("task_id"),
                    agent_id=request.args.get("agent_id"),
                    status=request.args.get("status"),
                    attempt_id=request.args.get("attempt_id"),
                )
            }
        )

    @app.post("/api/v1/agent-traces")
    def control_register_agent_trace():
        try:
            return jsonify(ctx.control_service.register_agent_trace(_payload())), 201
        except Exception as exc:
            return _error(exc)

    @app.get("/api/v1/agent-traces/search")
    def control_search_agent_traces():
        try:
            return jsonify(
                ctx.control_service.search_agent_traces(
                    query=request.args.get("q") or request.args.get("query") or "",
                    experiment_id=request.args.get("experiment_id"),
                    assignment_id=request.args.get("assignment_id"),
                    session_id=request.args.get("session_id"),
                    task_id=request.args.get("task_id"),
                    agent_id=request.args.get("agent_id"),
                    status=request.args.get("status"),
                    attempt_id=request.args.get("attempt_id"),
                )
            )
        except Exception as exc:
            return _error(exc)

    @app.get("/api/v1/agent-traces/<trace_id>")
    def control_get_agent_trace(trace_id: str):
        try:
            return jsonify(ctx.control_service.agent_trace_context(trace_id))
        except Exception as exc:
            return _error(exc, 404)

    @app.get("/api/v1/agent-traces/<trace_id>/commands")
    def control_agent_trace_commands(trace_id: str):
        try:
            return jsonify(
                ctx.control_service.agent_trace_commands(
                    trace_id,
                    failed_only=request.args.get("failed_only") in {"1", "true", "yes"},
                    semantic_only=request.args.get("semantic_only") in {"1", "true", "yes"},
                )
            )
        except Exception as exc:
            return _error(exc, 404)

    @app.get("/api/v1/agent-traces/<trace_id>/events")
    def control_agent_trace_events(trace_id: str):
        try:
            return jsonify(
                ctx.control_service.agent_trace_events(
                    trace_id,
                    query=request.args.get("q") or request.args.get("query"),
                    limit=int(request.args.get("limit") or 200),
                )
            )
        except Exception as exc:
            return _error(exc, 404)

    @app.post("/api/v1/trace-exports")
    def control_create_trace_export():
        try:
            return jsonify(ctx.control_service.create_trace_export(_payload())), 201
        except Exception as exc:
            return _error(exc)

    @app.get("/api/v1/trace-exports")
    def control_list_trace_exports():
        return jsonify(
            {
                "trace_exports": ctx.control_service.list_trace_exports(
                    experiment_id=request.args.get("experiment_id"),
                    assignment_id=request.args.get("assignment_id"),
                    session_id=request.args.get("session_id"),
                    task_id=request.args.get("task_id"),
                    agent_id=request.args.get("agent_id"),
                    provider=request.args.get("provider"),
                    status=request.args.get("status"),
                )
            }
        )

    @app.get("/api/v1/trace-exports/<trace_export_id>")
    def control_get_trace_export(trace_export_id: str):
        try:
            return jsonify(ctx.control_service.get_trace_export(trace_export_id))
        except Exception as exc:
            return _error(exc, 404)

    @app.post("/api/v1/artifacts")
    def control_create_artifact():
        try:
            payload = _payload()
            if payload.get("path"):
                return jsonify(ctx.control_service.register_path_artifact(payload)), 201
            return jsonify(ctx.control.create_artifact(payload)), 201
        except Exception as exc:
            return _error(exc)

    @app.get("/api/v1/artifacts")
    def control_list_artifacts():
        return jsonify(
            {
                "artifacts": ctx.control.list_artifacts(
                    experiment_id=request.args.get("experiment_id"),
                    assignment_id=request.args.get("assignment_id"),
                    attempt_id=request.args.get("attempt_id"),
                )
            }
        )

    @app.get("/api/v1/artifacts/<artifact_id>")
    def control_get_artifact(artifact_id: str):
        artifact = ctx.control.get_artifact(artifact_id)
        if artifact is None:
            return jsonify({"error": "artifact not found"}), 404
        return jsonify(artifact)

    @app.post("/api/v1/shared-tools")
    def control_publish_shared_tool():
        try:
            return jsonify(ctx.control_service.publish_shared_tool(_payload())), 201
        except Exception as exc:
            return _error(exc)

    @app.get("/api/v1/shared-tools")
    def control_list_shared_tools():
        return jsonify(
            {
                "shared_tools": ctx.control.list_shared_tools(
                    task_id=request.args.get("task_id"),
                    experiment_id=request.args.get("experiment_id"),
                    query=request.args.get("query"),
                    status=request.args.get("status") or "active",
                )
            }
        )

    @app.get("/api/v1/shared-tools/<tool_id>")
    def control_get_shared_tool(tool_id: str):
        tool = ctx.control.get_shared_tool(tool_id)
        if tool is None:
            return jsonify({"error": "shared tool not found"}), 404
        return jsonify(tool)

    @app.post("/api/v1/shared-tools/<tool_id>/checkout")
    def control_checkout_shared_tool(tool_id: str):
        try:
            return jsonify(ctx.control_service.checkout_shared_tool(tool_id, _payload()))
        except Exception as exc:
            return _error(exc, 404)

    @app.post("/api/v1/evaluations")
    def control_create_evaluation():
        try:
            record = ctx.control_service.create_evaluation(_payload())
            return jsonify(record), 202 if record["status"] in {"queued", "running"} else 201
        except Exception as exc:
            return _error(exc)

    @app.get("/api/v1/evaluations")
    def control_list_evaluations():
        return jsonify(
            {
                "evaluations": ctx.control.list_evaluations(
                    experiment_id=request.args.get("experiment_id"),
                    assignment_id=request.args.get("assignment_id"),
                    attempt_id=request.args.get("attempt_id"),
                )
            }
        )

    @app.get("/api/v1/evaluations/<evaluation_id>")
    def control_get_evaluation(evaluation_id: str):
        evaluation = ctx.control.get_evaluation(evaluation_id)
        if evaluation is None:
            return jsonify({"error": "evaluation not found"}), 404
        if evaluation.get("job_id"):
            try:
                ctx.control_service.jobs.get(evaluation["job_id"])
            except Exception:
                pass
        return jsonify(evaluation)

    @app.post("/api/v1/evaluations/<evaluation_id>/replay-bundle")
    def control_export_replay_bundle(evaluation_id: str):
        try:
            return jsonify(ctx.control_service.export_replay_bundle(evaluation_id, _payload())), 201
        except Exception as exc:
            return _error(exc, 404)

    @app.post("/api/v1/replay")
    def control_replay_bundle():
        try:
            result = ctx.control_service.replay_bundle(_payload())
            evaluation = result["evaluation"]
            return jsonify(result), 202 if evaluation["status"] in {"queued", "running"} else 201
        except Exception as exc:
            return _error(exc)

    @app.get("/api/v1/leaderboard")
    def control_list_leaderboard():
        return jsonify(
            {
                "leaderboard": ctx.control.list_leaderboard_entries(
                    experiment_id=request.args.get("experiment_id"),
                    task_id=request.args.get("task_id"),
                    direction_id=request.args.get("direction_id"),
                    limit=int(request.args.get("limit") or 20),
                )
            }
        )

    @app.get("/api/v1/incumbent")
    def control_get_incumbent():
        incumbent = ctx.control.get_incumbent(
            experiment_id=request.args.get("experiment_id"),
            task_id=request.args.get("task_id"),
            direction_id=request.args.get("direction_id"),
        )
        if incumbent is None:
            return jsonify({"error": "incumbent not found"}), 404
        return jsonify(incumbent)

    @app.post("/api/v1/incumbent/checkout")
    def control_checkout_incumbent():
        try:
            return jsonify(ctx.control_service.checkout_incumbent(_payload()))
        except Exception as exc:
            return _error(exc, 404)

    @app.post("/api/v1/telemetry-runs")
    def control_create_telemetry_run():
        try:
            return jsonify(ctx.control_service.telemetry.create_run(_payload())), 201
        except Exception as exc:
            return _error(exc)

    @app.get("/api/v1/telemetry-runs")
    def control_list_telemetry_runs():
        return jsonify(
            {
                "telemetry_runs": ctx.control.list_telemetry_runs(
                    experiment_id=request.args.get("experiment_id"),
                    assignment_id=request.args.get("assignment_id"),
                    job_id=request.args.get("job_id"),
                    attempt_id=request.args.get("attempt_id"),
                )
            }
        )

    @app.get("/api/v1/telemetry-runs/<telemetry_id>")
    def control_get_telemetry_run(telemetry_id: str):
        telemetry = ctx.control.get_telemetry_run(telemetry_id)
        if telemetry is None:
            return jsonify({"error": "telemetry run not found"}), 404
        return jsonify(telemetry)

    @app.post("/api/v1/telemetry-runs/<telemetry_id>/metrics")
    def control_log_telemetry_metrics(telemetry_id: str):
        try:
            return jsonify(ctx.control_service.telemetry.log_metrics(telemetry_id, _payload()))
        except Exception as exc:
            return _error(exc, 404)

    @app.post("/api/v1/telemetry-runs/<telemetry_id>/finish")
    def control_finish_telemetry_run(telemetry_id: str):
        try:
            return jsonify(ctx.control_service.telemetry.finish_run(telemetry_id, _payload()))
        except Exception as exc:
            return _error(exc, 404)

    @app.post("/api/v1/jobs")
    def control_create_job():
        try:
            record = ctx.control_service.jobs.launch(_payload())
            return jsonify(record), 202 if record["status"] in {"queued", "running"} else 201
        except Exception as exc:
            return _error(exc)

    @app.get("/api/v1/jobs")
    def control_list_jobs():
        return jsonify(
            {
                "jobs": ctx.control.list_jobs(
                    experiment_id=request.args.get("experiment_id"),
                    assignment_id=request.args.get("assignment_id"),
                    attempt_id=request.args.get("attempt_id"),
                )
            }
        )

    @app.get("/api/v1/jobs/<job_id>")
    def control_get_job(job_id: str):
        try:
            return jsonify(ctx.control_service.jobs.get(job_id))
        except Exception as exc:
            return _error(exc, 404)

    @app.get("/api/v1/jobs/<job_id>/logs")
    def control_get_job_logs(job_id: str):
        try:
            max_bytes = int(request.args.get("max_bytes") or 200_000)
            return jsonify(ctx.control_service.jobs.read_logs(job_id, max_bytes=max_bytes))
        except Exception as exc:
            return _error(exc, 404)

    @app.post("/api/v1/jobs/<job_id>/attach")
    def control_attach_job(job_id: str):
        try:
            return jsonify(ctx.control_service.jobs.attach(job_id, _payload()))
        except KeyError as exc:
            return _error(exc, 404)
        except Exception as exc:
            return _error(exc)

    @app.post("/api/v1/jobs/<job_id>/cancel")
    def control_cancel_job(job_id: str):
        try:
            return jsonify(ctx.control_service.jobs.cancel(job_id))
        except Exception as exc:
            return _error(exc, 404)

    @app.post("/api/v1/findings")
    def control_share_finding():
        try:
            return jsonify(ctx.control.share_finding(_payload())), 201
        except Exception as exc:
            return _error(exc)

    @app.get("/api/v1/findings")
    def control_list_findings():
        return jsonify(
            {
                "findings": ctx.control.list_findings(
                    task_id=request.args.get("task_id"),
                    experiment_id=request.args.get("experiment_id"),
                    query=request.args.get("query"),
                )
            }
        )

    @app.post("/api/v1/notebook-checkpoints")
    def control_checkpoint_notebook():
        try:
            return jsonify(ctx.control.checkpoint_notebook(_payload())), 201
        except Exception as exc:
            return _error(exc)

    @app.get("/api/v1/notebook-checkpoints")
    def control_list_notebook_checkpoints():
        assignment_id = request.args.get("assignment_id")
        if not assignment_id:
            return jsonify({"error": "assignment_id is required"}), 400
        return jsonify({"notebook_checkpoints": ctx.control.list_notebook_checkpoints(assignment_id=assignment_id)})

    @app.post("/api/v1/events")
    def control_record_event():
        try:
            return jsonify(ctx.control.record_event(_payload())), 201
        except Exception as exc:
            return _error(exc)

    @app.get("/api/v1/events")
    def control_list_events():
        return jsonify(
            {
                "events": ctx.control.list_events(
                    experiment_id=request.args.get("experiment_id"),
                    assignment_id=request.args.get("assignment_id"),
                    limit=int(request.args.get("limit") or 200),
                )
            }
        )

    @app.get("/api/v1/events/stream")
    def control_stream_events():
        from flask import Response, stream_with_context

        experiment_id = request.args.get("experiment_id")
        assignment_id = request.args.get("assignment_id")
        poll_s = float(request.args.get("poll_s") or 1.0)
        max_events = int(request.args.get("max_events") or 0)

        def generate():
            seen: set[str] = set()
            emitted = 0
            while True:
                events = list(reversed(ctx.control.list_events(experiment_id=experiment_id, assignment_id=assignment_id, limit=100)))
                for event in events:
                    if event["event_id"] in seen:
                        continue
                    seen.add(event["event_id"])
                    emitted += 1
                    yield f"id: {event['event_id']}\n"
                    yield f"event: {event['event_type']}\n"
                    yield "data: " + json.dumps(event, sort_keys=True) + "\n\n"
                    if max_events and emitted >= max_events:
                        return
                time.sleep(poll_s)

        return Response(stream_with_context(generate()), mimetype="text/event-stream")

    @app.get("/api/v1/network-policy")
    def control_network_policy():
        try:
            return jsonify(
                ctx.control_service.network_policy(
                    {
                        "experiment_id": request.args.get("experiment_id"),
                        "assignment_id": request.args.get("assignment_id"),
                        "session_id": request.args.get("session_id"),
                        "worker_backend": request.args.get("worker_backend"),
                    }
                )
            )
        except Exception as exc:
            return _error(exc, 404)

    @app.post("/api/v1/network-access-events")
    def control_record_network_access_event():
        try:
            return jsonify(ctx.control.record_network_access_event(_payload())), 201
        except Exception as exc:
            return _error(exc)

    @app.get("/api/v1/network-access-events")
    def control_list_network_access_events():
        return jsonify(
            {
                "network_access_events": ctx.control.list_network_access_events(
                    experiment_id=request.args.get("experiment_id"),
                    assignment_id=request.args.get("assignment_id"),
                    session_id=request.args.get("session_id"),
                    limit=int(request.args.get("limit") or 200),
                )
            }
        )

    @app.get("/api/v1/sessions/<session_id>/trace")
    def control_session_trace(session_id: str):
        session = ctx.control.get_session(session_id)
        if session is None:
            return jsonify({"error": "session not found"}), 404
        events = [
            item
            for item in ctx.control.list_events(assignment_id=session["assignment_id"], limit=int(request.args.get("limit") or 500))
            if item.get("session_id") in {None, session_id} or item.get("event_type", "").startswith(("job.", "evaluation.", "telemetry."))
        ]
        return jsonify({"session": session, "events": events})
