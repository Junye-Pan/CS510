from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from agentic_opt.adapter.semantic_worker import _start_workspace_control_broker, _terminate_process
from agentic_opt.control_plane.client import ControlPlaneClient, ControlPlaneClientError
from agentic_opt.control_plane.docker_image_policy import docker_image_identity, evaluate_docker_image_policy
from agentic_opt.control_plane.jobs import DockerNetworkPolicyError, JobService, build_local_docker_command
from agentic_opt.control_plane.policy import PolicyService
from agentic_opt.control_plane.relay import ControlPlaneRelayServer, ControlPlaneTCPRelayServer, _validate_relay_path, relay_url, tcp_relay_url
from agentic_opt.control_plane.task_context import ensure_task_context_snapshot, materialize_task_context_snapshot
from agentic_opt.web.app import create_app
from agentic_opt.web.workers import WorkerManager, WorkerProcess, build_docker_worker_command


def _docker_runtime_available() -> bool:
    try:
        proc = subprocess.run(["docker", "info"], capture_output=True, text=True, check=False, timeout=10)
    except Exception:
        return False
    return proc.returncode == 0


class WebBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.tasks_root = self.root / "toy_tasks_bundle"
        self.state_root = self.root / "state"
        self.db_path = self.state_root / "control.sqlite3"
        self._old_tasks_roots = os.environ.get("AO_TASKS_ROOTS")
        os.environ["AO_TASKS_ROOTS"] = str(self.tasks_root)
        self._write_toy_task()
        app = create_app(state_root=self.state_root, database_path=self.db_path)
        self.ctx = app.config["AO_CONTEXT"]
        self.client = app.test_client()

    def tearDown(self) -> None:
        self.ctx.workers.close()
        self.ctx.control_service.close()
        if self._old_tasks_roots is None:
            os.environ.pop("AO_TASKS_ROOTS", None)
        else:
            os.environ["AO_TASKS_ROOTS"] = self._old_tasks_roots
        self.tempdir.cleanup()

    def test_control_plane_ui_shell(self) -> None:
        shell = self.client.get("/ui")
        self.assertEqual(shell.status_code, 200, shell.get_data(as_text=True))
        shell_html = shell.get_data(as_text=True)
        self.assertIn('id="ao-ui"', shell_html)
        self.assertIn('data-api-base="/api/v1"', shell_html)
        self.assertIn("control_plane_ui.css", shell_html)
        self.assertIn("control_plane_ui.js", shell_html)

        css = self.client.get("/static/control_plane_ui.css")
        self.assertEqual(css.status_code, 200)
        self.assertIn(".ao-shell", css.get_data(as_text=True))
        css.close()
        script = self.client.get("/static/control_plane_ui.js")
        self.assertEqual(script.status_code, 200)
        self.assertIn("loadExperiments", script.get_data(as_text=True))
        self.assertIn("Score Budget", script.get_data(as_text=True))
        self.assertIn("stale running", script.get_data(as_text=True))
        script.close()

        created = self.client.post(
            "/api/v1/experiments",
            json={
                "task_id": "toy_eval",
                "mode": "local",
                "budget": {"total_evaluator_runs": 1},
            },
        )
        self.assertEqual(created.status_code, 201, created.get_data(as_text=True))
        experiment_id = created.get_json()["experiment"]["experiment_id"]
        deep_link = self.client.get(f"/ui/experiments/{experiment_id}")
        self.assertEqual(deep_link.status_code, 200, deep_link.get_data(as_text=True))
        self.assertIn(f'data-initial-experiment-id="{experiment_id}"', deep_link.get_data(as_text=True))

    def test_control_plane_semantic_resources(self) -> None:
        health = self.client.get("/healthz")
        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.get_json()["ok"])
        self.assertEqual(health.get_json()["service"], "control-plane")

        model = self.client.get("/api/v1/object-model")
        self.assertEqual(model.status_code, 200)
        self.assertIn("WorkerAssignment", model.get_json()["resources"])
        self.assertIn("Finding", model.get_json()["resources"])
        self.assertIn("TelemetryRun", model.get_json()["resources"])
        self.assertIn("Attempt", model.get_json()["resources"])
        self.assertIn("AgentTraceBundle", model.get_json()["resources"])
        self.assertIn("TraceExportRun", model.get_json()["resources"])
        self.assertIn("Environment", model.get_json()["resources"])
        self.assertIn("EnvironmentOverlay", model.get_json()["resources"])
        self.assertIn("LeaderboardEntry", model.get_json()["resources"])
        self.assertIn("TaskKnowledgeFile", model.get_json()["resources"])
        self.assertNotIn("KnowledgeItem", model.get_json()["resources"])

        created = self.client.post(
            "/api/v1/experiments",
            json={
                "task_id": "toy_eval",
                "mode": "local",
                "budget": {"total_evaluator_runs": 6},
                "policy": {"network": {"external_internet": "deny"}},
                "assignment_count": 2,
            },
        )
        self.assertEqual(created.status_code, 201)
        created_payload = created.get_json()
        experiment_id = created_payload["experiment"]["experiment_id"]
        assignment = created_payload["assignments"][0]
        assignment_id = assignment["assignment_id"]
        self.assertEqual(len(created_payload["assignments"]), 2)

        environment = self.client.post(
            "/api/v1/environments",
            json={"experiment_id": experiment_id, "assignment_id": assignment_id, "task_id": "toy_eval"},
        )
        self.assertEqual(environment.status_code, 201, environment.get_data(as_text=True))
        environment_payload = environment.get_json()
        self.assertEqual(environment_payload["environment_type"], "task")
        self.assertEqual(environment_payload["task_id"], "toy_eval")
        self.assertTrue(Path(environment_payload["python_path"]).exists())

        task = self.client.get("/api/v1/tasks/toy_eval")
        self.assertEqual(task.status_code, 200)
        self.assertEqual(task.get_json()["candidate_contract"]["workspace_entrypoint"], "initial.py")
        knowledge_paths = {item["relative_path"] for item in task.get_json()["task_knowledge"]["files"]}
        self.assertEqual(knowledge_paths, {"manifest.json", "note.md"})
        self.assertTrue(task.get_json()["task_knowledge"]["digest"].startswith("sha256:"))

        context = self.client.get("/api/v1/context", query_string={"assignment_id": assignment_id})
        self.assertEqual(context.status_code, 200)
        self.assertEqual(context.get_json()["assignment"]["assignment_id"], assignment_id)
        self.assertNotIn("budget", context.get_json()["assignment"])
        self.assertNotIn("budget", context.get_json()["experiment"])
        self.assertTrue(context.get_json()["environments"])
        self.assertEqual(context.get_json()["network_policy"]["policy"]["external_internet"], "deny")
        self.assertTrue(context.get_json()["network_policy"]["enforcement"]["policy_weakened"])
        self.assertTrue(context.get_json()["task_knowledge"]["files"])
        docker_policy = self.client.get(
            "/api/v1/network-policy",
            query_string={"experiment_id": experiment_id, "worker_backend": "local-docker"},
        )
        self.assertEqual(docker_policy.status_code, 200)
        docker_enforcement = docker_policy.get_json()["enforcement"]
        self.assertTrue(docker_enforcement["external_internet_enforced"])
        self.assertFalse(docker_enforcement["policy_weakened"])
        self.assertEqual(docker_enforcement["enforcement_mode"], "docker_network_none")
        self.assertTrue(docker_enforcement["control_plane_relay_required"])
        self.assertFalse(docker_enforcement["operationally_ready"])

        session = self.client.post(f"/api/v1/assignments/{assignment_id}/sessions", json={})
        self.assertEqual(session.status_code, 201)
        session_id = session.get_json()["session_id"]
        patched = self.client.patch(f"/api/v1/sessions/{session_id}", json={"status": "completed"})
        self.assertEqual(patched.status_code, 200)
        self.assertEqual(patched.get_json()["status"], "completed")

        rejected_attempt = self.client.post(
            "/api/v1/attempts",
            json={
                "experiment_id": experiment_id,
                "assignment_id": assignment_id,
                "task_id": "toy_eval",
                "summary": "attempts intentionally do not store summaries",
            },
        )
        self.assertEqual(rejected_attempt.status_code, 400)
        self.assertIn("findings or notebook", rejected_attempt.get_json()["error"])
        attempt = self.client.post(
            "/api/v1/attempts",
            json={
                "experiment_id": experiment_id,
                "assignment_id": assignment_id,
                "session_id": session_id,
                "task_id": "toy_eval",
                "agent_id": assignment["agent_id"],
                "status": "active",
                "metadata": {"lineage_note": "structured metadata only"},
            },
        )
        self.assertEqual(attempt.status_code, 201, attempt.get_data(as_text=True))
        attempt_payload = attempt.get_json()
        attempt_id = attempt_payload["attempt_id"]
        self.assertNotIn("summary", attempt_payload)
        attempts = self.client.get("/api/v1/attempts", query_string={"assignment_id": assignment_id})
        self.assertEqual(attempts.status_code, 200)
        self.assertEqual(attempts.get_json()["attempts"][0]["attempt_id"], attempt_id)
        context_after_attempt = self.client.get("/api/v1/context", query_string={"assignment_id": assignment_id})
        self.assertEqual(context_after_attempt.status_code, 200)
        self.assertEqual(context_after_attempt.get_json()["attempts"][0]["attempt_id"], attempt_id)

        trace_dir = self.root / "trace_source" / "run_trace_test" / "turn_trace_test"
        trace_dir.mkdir(parents=True)
        trace_events = [
            {
                "method": "item/started",
                "params": {
                    "startedAtMs": 1000,
                    "item": {"id": "cmd-1", "command": "ctx attempts", "cwd": str(self.root), "source": "exec"},
                },
            },
            {
                "method": "item/commandExecution/outputDelta",
                "params": {"itemId": "cmd-1", "delta": f"observed {attempt_id}\n"},
            },
            {
                "method": "item/completed",
                "params": {
                    "completedAtMs": 1300,
                    "item": {
                        "id": "cmd-1",
                        "command": "ctx attempts",
                        "cwd": str(self.root),
                        "source": "exec",
                        "durationMs": 300,
                        "exitCode": 0,
                        "status": "completed",
                        "aggregatedOutput": f"observed {attempt_id}\n",
                    },
                },
            },
            {
                "method": "item/started",
                "params": {
                    "startedAtMs": 1400,
                    "item": {"id": "cmd-2", "command": "python bad.py", "cwd": str(self.root), "source": "exec"},
                },
            },
            {
                "method": "item/completed",
                "params": {
                    "completedAtMs": 1500,
                    "item": {
                        "id": "cmd-2",
                        "command": "python bad.py",
                        "cwd": str(self.root),
                        "source": "exec",
                        "durationMs": 100,
                        "exitCode": 1,
                        "status": "failed",
                        "aggregatedOutput": "boom\n",
                    },
                },
            },
            {"method": "item/agentMessage/delta", "params": {"delta": "I checked prior attempts."}},
        ]
        (trace_dir / "events.jsonl").write_text(
            "".join(json.dumps(event, sort_keys=True) + "\n" for event in trace_events),
            encoding="utf-8",
        )
        registered_trace = self.client.post(
            "/api/v1/agent-traces",
            json={
                "experiment_id": experiment_id,
                "assignment_id": assignment_id,
                "session_id": session_id,
                "task_id": "toy_eval",
                "agent_id": assignment["agent_id"],
                "run_id": "run_trace_test",
                "turn_id": "turn_trace_test",
                "trace_dir": str(trace_dir),
                "outcome": "completed",
                "status": "completed",
                "metadata": {"registered_by": "test"},
            },
        )
        self.assertEqual(registered_trace.status_code, 201, registered_trace.get_data(as_text=True))
        trace_payload = registered_trace.get_json()
        trace_id = trace_payload["trace_id"]
        self.assertEqual(trace_payload["status"], "completed")
        self.assertEqual(trace_payload["metadata"]["command_count"], 2)
        self.assertEqual(trace_payload["metadata"]["semantic_command_count"], 1)
        self.assertEqual(trace_payload["metadata"]["failed_command_count"], 1)
        self.assertIn(attempt_id, trace_payload["metadata"]["observed_ids"]["attempt_ids"])

        repeated_trace = self.client.post(
            "/api/v1/agent-traces",
            json={
                "experiment_id": experiment_id,
                "assignment_id": assignment_id,
                "session_id": session_id,
                "task_id": "toy_eval",
                "agent_id": assignment["agent_id"],
                "run_id": "run_trace_test",
                "turn_id": "turn_trace_test",
                "trace_dir": str(trace_dir),
            },
        )
        self.assertEqual(repeated_trace.status_code, 201)
        self.assertEqual(repeated_trace.get_json()["trace_id"], trace_id)

        trace_context = self.client.get(f"/api/v1/agent-traces/{trace_id}")
        self.assertEqual(trace_context.status_code, 200, trace_context.get_data(as_text=True))
        self.assertEqual(trace_context.get_json()["trace"]["trace_id"], trace_id)
        self.assertEqual(trace_context.get_json()["manifest"]["run_id"], "run_trace_test")
        self.assertTrue((Path(trace_payload["trace_root"]) / "commands.jsonl").exists())
        trace_artifact = self.client.get(f"/api/v1/artifacts/{trace_payload['artifact_id']}")
        self.assertEqual(trace_artifact.status_code, 200)
        self.assertEqual(trace_artifact.get_json()["kind"], "agent_trace_bundle")

        traces = self.client.get("/api/v1/agent-traces", query_string={"assignment_id": assignment_id})
        self.assertEqual(traces.status_code, 200)
        self.assertEqual(traces.get_json()["agent_traces"][0]["trace_id"], trace_id)
        traces_by_attempt = self.client.get("/api/v1/agent-traces", query_string={"attempt_id": attempt_id})
        self.assertEqual(traces_by_attempt.status_code, 200)
        self.assertEqual(traces_by_attempt.get_json()["agent_traces"][0]["trace_id"], trace_id)
        context_after_trace = self.client.get("/api/v1/context", query_string={"assignment_id": assignment_id})
        self.assertEqual(context_after_trace.status_code, 200)
        self.assertEqual(context_after_trace.get_json()["agent_traces"][0]["trace_id"], trace_id)

        semantic_commands = self.client.get(f"/api/v1/agent-traces/{trace_id}/commands", query_string={"semantic_only": "true"})
        self.assertEqual(semantic_commands.status_code, 200, semantic_commands.get_data(as_text=True))
        self.assertEqual(len(semantic_commands.get_json()["commands"]), 1)
        self.assertEqual(semantic_commands.get_json()["commands"][0]["semantic_tool"], "ctx")

        failed_commands = self.client.get(f"/api/v1/agent-traces/{trace_id}/commands", query_string={"failed_only": "true"})
        self.assertEqual(failed_commands.status_code, 200)
        self.assertEqual(failed_commands.get_json()["commands"][0]["command"], "python bad.py")

        searched = self.client.get("/api/v1/agent-traces/search", query_string={"q": "ctx attempts", "assignment_id": assignment_id})
        self.assertEqual(searched.status_code, 200, searched.get_data(as_text=True))
        self.assertEqual(searched.get_json()["matches"][0]["trace"]["trace_id"], trace_id)

        events = self.client.get(f"/api/v1/agent-traces/{trace_id}/events", query_string={"query": attempt_id, "limit": 5})
        self.assertEqual(events.status_code, 200)
        self.assertTrue(events.get_json()["events"])

        finding = self.client.post(
            "/api/v1/findings",
            json={
                "experiment_id": experiment_id,
                "assignment_id": assignment_id,
                "task_id": "toy_eval",
                "finding_type": "pattern",
                "title": "Reusable pattern",
                "body": f"Finding and pattern are one durable knowledge resource for {attempt_id}.",
            },
        )
        self.assertEqual(finding.status_code, 201)
        findings = self.client.get("/api/v1/findings", query_string={"task_id": "toy_eval", "query": "durable"})
        self.assertEqual(findings.status_code, 200)
        self.assertEqual(findings.get_json()["findings"][0]["finding_type"], "pattern")

        notebook = self.client.post(
            "/api/v1/notebook-checkpoints",
            json={
                "experiment_id": experiment_id,
                "assignment_id": assignment_id,
                "session_id": session_id,
                "agent_id": assignment["agent_id"],
                "content": "# Notebook\n\nControl-plane checkpoint.",
            },
        )
        self.assertEqual(notebook.status_code, 201)
        notebooks = self.client.get("/api/v1/notebook-checkpoints", query_string={"assignment_id": assignment_id})
        self.assertTrue(notebooks.get_json()["notebook_checkpoints"])

        artifact_source = self.root / "artifact.txt"
        artifact_source.write_text("artifact body\n", encoding="utf-8")
        artifact = self.client.post(
            "/api/v1/artifacts",
            json={
                "experiment_id": experiment_id,
                "assignment_id": assignment_id,
                "attempt_id": attempt_id,
                "kind": "text",
                "path": str(artifact_source),
            },
        )
        self.assertEqual(artifact.status_code, 201)
        artifact_payload = artifact.get_json()
        self.assertEqual(artifact_payload["attempt_id"], attempt_id)
        self.assertTrue(artifact_payload["digest"].startswith("sha256:"))
        manifest_path = Path(artifact_payload["metadata"]["manifest_path"])
        self.assertTrue(manifest_path.exists())

        artifact_dir = self.root / "artifact_dir"
        (artifact_dir / ".git").mkdir(parents=True)
        (artifact_dir / ".git" / "config").write_text("[core]\n", encoding="utf-8")
        (artifact_dir / "payload.txt").write_text("payload\n", encoding="utf-8")
        directory_artifact = self.client.post(
            "/api/v1/artifacts",
            json={
                "experiment_id": experiment_id,
                "assignment_id": assignment_id,
                "attempt_id": attempt_id,
                "kind": "directory",
                "path": str(artifact_dir),
            },
        )
        self.assertEqual(directory_artifact.status_code, 201, directory_artifact.get_data(as_text=True))
        directory_artifact_path = Path(directory_artifact.get_json()["local_path"])
        self.assertTrue((directory_artifact_path / "payload.txt").exists())
        self.assertFalse((directory_artifact_path / ".git").exists())

        tool_source = self.root / "tool.py"
        tool_source.write_text("print('tool')\n", encoding="utf-8")
        tool = self.client.post(
            "/api/v1/shared-tools",
            json={
                "experiment_id": experiment_id,
                "assignment_id": assignment_id,
                "session_id": session_id,
                "agent_id": assignment["agent_id"],
                "task_id": "toy_eval",
                "path": str(tool_source),
                "name": "toy-tool",
                "description": "tiny reusable helper",
                "entrypoint": "tool.py",
            },
        )
        self.assertEqual(tool.status_code, 201, tool.get_data(as_text=True))
        tool_payload = tool.get_json()
        self.assertEqual(tool_payload["name"], "toy-tool")
        tools = self.client.get("/api/v1/shared-tools", query_string={"task_id": "toy_eval", "query": "tiny"})
        self.assertEqual(tools.status_code, 200)
        self.assertEqual(tools.get_json()["shared_tools"][0]["tool_id"], tool_payload["tool_id"])
        tool_checkout = self.client.post(
            f"/api/v1/shared-tools/{tool_payload['tool_id']}/checkout",
            json={"destination_path": str(self.root / "checked_out_tool.py")},
        )
        self.assertEqual(tool_checkout.status_code, 200)
        self.assertTrue((self.root / "checked_out_tool.py").exists())

        self.assertEqual(self.client.get("/api/v1/knowledge", query_string={"task_id": "toy_eval"}).status_code, 404)

        network_event = self.client.post(
            "/api/v1/network-access-events",
            json={
                "experiment_id": experiment_id,
                "assignment_id": assignment_id,
                "session_id": session_id,
                "task_id": "toy_eval",
                "agent_id": assignment["agent_id"],
                "destination": "https://example.com",
                "access_type": "http",
                "decision": "denied",
                "reason": "external internet denied",
            },
        )
        self.assertEqual(network_event.status_code, 201)
        network_events = self.client.get("/api/v1/network-access-events", query_string={"assignment_id": assignment_id})
        self.assertEqual(network_events.status_code, 200)
        self.assertEqual(network_events.get_json()["network_access_events"][0]["decision"], "denied")

        job = self.client.post(
            "/api/v1/jobs",
            json={
                "experiment_id": experiment_id,
                "assignment_id": assignment_id,
                "session_id": session_id,
                "attempt_id": attempt_id,
                "provider": "queued-only",
                "inputs": {"command": "echo queued"},
            },
        )
        self.assertEqual(job.status_code, 201, job.get_data(as_text=True))
        self.assertEqual(job.get_json()["attempt_id"], attempt_id)
        jobs = self.client.get("/api/v1/jobs", query_string={"attempt_id": attempt_id})
        self.assertEqual(jobs.get_json()["jobs"][0]["job_id"], job.get_json()["job_id"])
        followup_session = self.client.post(f"/api/v1/assignments/{assignment_id}/sessions", json={})
        self.assertEqual(followup_session.status_code, 201)
        followup_attempt = self.client.post(
            "/api/v1/attempts",
            json={
                "experiment_id": experiment_id,
                "assignment_id": assignment_id,
                "session_id": followup_session.get_json()["session_id"],
                "task_id": "toy_eval",
                "agent_id": assignment["agent_id"],
                "parent_attempt_id": attempt_id,
                "status": "active",
            },
        )
        self.assertEqual(followup_attempt.status_code, 201, followup_attempt.get_data(as_text=True))
        followup_attempt_id = followup_attempt.get_json()["attempt_id"]
        attached_job = self.client.post(
            f"/api/v1/jobs/{job.get_json()['job_id']}/attach",
            json={
                "experiment_id": experiment_id,
                "assignment_id": assignment_id,
                "session_id": followup_session.get_json()["session_id"],
                "attempt_id": followup_attempt_id,
                "agent_id": assignment["agent_id"],
                "mode": "observe",
                "note": "continue analysis in a later session",
            },
        )
        self.assertEqual(attached_job.status_code, 200, attached_job.get_data(as_text=True))
        self.assertEqual(attached_job.get_json()["job"]["attempt_id"], attempt_id)
        self.assertEqual(attached_job.get_json()["attachment"]["attempt_id"], followup_attempt_id)
        attached_job_status = self.client.get(f"/api/v1/jobs/{job.get_json()['job_id']}")
        self.assertEqual(attached_job_status.get_json()["details"]["last_attachment"]["attempt_id"], followup_attempt_id)

        telemetry = self.client.post(
            "/api/v1/telemetry-runs",
            json={
                "experiment_id": experiment_id,
                "assignment_id": assignment_id,
                "session_id": session_id,
                "attempt_id": attempt_id,
                "provider": "local",
                "run_name": "attempt telemetry",
            },
        )
        self.assertEqual(telemetry.status_code, 201, telemetry.get_data(as_text=True))
        self.assertEqual(telemetry.get_json()["attempt_id"], attempt_id)
        telemetry_runs = self.client.get("/api/v1/telemetry-runs", query_string={"attempt_id": attempt_id})
        self.assertEqual(telemetry_runs.get_json()["telemetry_runs"][0]["telemetry_id"], telemetry.get_json()["telemetry_id"])

        entry_path = self.tasks_root / "toy_eval" / "public" / "initial.py"
        candidate_artifact = self.client.post(
            "/api/v1/artifacts",
            json={
                "experiment_id": experiment_id,
                "assignment_id": assignment_id,
                "attempt_id": attempt_id,
                "kind": "candidate",
                "path": str(entry_path),
            },
        )
        self.assertEqual(candidate_artifact.status_code, 201)
        artifact_eval = self.client.post(
            "/api/v1/evaluations",
            json={
                "experiment_id": experiment_id,
                "assignment_id": assignment_id,
                "attempt_id": attempt_id,
                "task_id": "toy_eval",
                "kind": "verify",
                "artifact_id": candidate_artifact.get_json()["artifact_id"],
                "async": False,
            },
        )
        self.assertEqual(artifact_eval.status_code, 201)
        self.assertEqual(artifact_eval.get_json()["request"]["input_kind"], "artifact")
        self.assertEqual(artifact_eval.get_json()["attempt_id"], attempt_id)
        self.assertTrue(artifact_eval.get_json()["valid"])
        patched_attempt = self.client.patch(
            f"/api/v1/attempts/{attempt_id}",
            json={
                "status": "evaluated",
                "candidate_artifact_id": candidate_artifact.get_json()["artifact_id"],
                "metadata": {"verified": True},
            },
        )
        self.assertEqual(patched_attempt.status_code, 200, patched_attempt.get_data(as_text=True))
        self.assertEqual(patched_attempt.get_json()["status"], "evaluated")
        self.assertEqual(patched_attempt.get_json()["candidate_artifact_id"], candidate_artifact.get_json()["artifact_id"])
        attempt_context = self.client.get(f"/api/v1/attempts/{attempt_id}")
        self.assertEqual(attempt_context.status_code, 200, attempt_context.get_data(as_text=True))
        attempt_context_payload = attempt_context.get_json()
        self.assertEqual(attempt_context_payload["attempt"]["attempt_id"], attempt_id)
        self.assertTrue(any(item["artifact_id"] == candidate_artifact.get_json()["artifact_id"] for item in attempt_context_payload["artifacts"]))
        self.assertTrue(any(item["evaluation_id"] == artifact_eval.get_json()["evaluation_id"] for item in attempt_context_payload["evaluations"]))
        self.assertTrue(any(item["job_id"] == job.get_json()["job_id"] for item in attempt_context_payload["jobs"]))

        sync_eval = self.client.post(
            "/api/v1/evaluations",
            json={
                "experiment_id": experiment_id,
                "assignment_id": assignment_id,
                "task_id": "toy_eval",
                "kind": "verify",
                "entry_path": str(entry_path),
                "async": False,
            },
        )
        self.assertEqual(sync_eval.status_code, 201)
        self.assertEqual(sync_eval.get_json()["status"], "completed")
        self.assertTrue(sync_eval.get_json()["valid"])

        old_openai_key = os.environ.get("OPENAI_API_KEY")
        old_vscode_injection = os.environ.get("VSCODE_INJECTION")
        os.environ["OPENAI_API_KEY"] = "should-not-be-persisted"
        os.environ["VSCODE_INJECTION"] = "should-not-be-inherited"
        try:
            async_eval = self.client.post(
                "/api/v1/evaluations",
                json={
                    "experiment_id": experiment_id,
                    "assignment_id": assignment_id,
                    "task_id": "toy_eval",
                    "attempt_id": attempt_id,
                    "kind": "submit",
                    "entry_path": str(entry_path),
                },
            )
        finally:
            if old_openai_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = old_openai_key
            if old_vscode_injection is None:
                os.environ.pop("VSCODE_INJECTION", None)
            else:
                os.environ["VSCODE_INJECTION"] = old_vscode_injection
        self.assertEqual(async_eval.status_code, 202)
        async_eval_id = async_eval.get_json()["evaluation_id"]
        completed_eval = self._wait_for(f"/api/v1/evaluations/{async_eval_id}")
        self.assertEqual(completed_eval["status"], "completed")
        self.assertEqual(completed_eval["score"], 1.0)
        eval_job = self.client.get(f"/api/v1/jobs/{completed_eval['job_id']}")
        self.assertEqual(eval_job.status_code, 200)
        eval_job_env = eval_job.get_json()["inputs"]["env"]
        self.assertIn("AO_ENVIRONMENT_ID", eval_job_env)
        self.assertIn("AO_TASKS_ROOTS", eval_job_env)
        self.assertNotIn("OPENAI_API_KEY", eval_job_env)
        self.assertNotIn("VSCODE_INJECTION", eval_job_env)
        self.assertNotIn("SSH_AUTH_SOCK", eval_job_env)
        analysis = self.client.get(f"/api/v1/experiments/{experiment_id}/analysis")
        self.assertEqual(analysis.status_code, 200, analysis.get_data(as_text=True))
        analysis_payload = analysis.get_json()
        self.assertEqual(analysis_payload["schema_version"], "agentic_opt.run_analysis.v1")
        self.assertFalse(analysis_payload["dashboard_use"]["worker_semantic_tool"])
        self.assertTrue(analysis_payload["score_series"])
        self.assertEqual(analysis_payload["score_series"][0]["attempt_id"], attempt_id)
        self.assertTrue(any(node["attempt_id"] == followup_attempt_id for node in analysis_payload["attempt_graph"]["nodes"]))
        self.assertTrue(any(edge["kind"] == "parent_attempt" for edge in analysis_payload["attempt_graph"]["edges"]))
        self.assertTrue(any(edge["kind"] == "attached_job" for edge in analysis_payload["relationships"]))
        self.assertTrue(any(item["attempt_id"] == attempt_id and item["trace_ids"] for item in analysis_payload["candidate_lineage"]))

        telemetry = self.client.post(
            "/api/v1/telemetry-runs",
            json={
                "experiment_id": experiment_id,
                "assignment_id": assignment_id,
                "session_id": session_id,
                "provider": "local",
                "run_name": "toy training",
                "params": {"lr": 0.1},
            },
        )
        self.assertEqual(telemetry.status_code, 201)
        telemetry_id = telemetry.get_json()["telemetry_id"]
        metrics = self.client.post(
            f"/api/v1/telemetry-runs/{telemetry_id}/metrics",
            json={"step": 1, "metrics": {"loss": 0.25, "acc": 0.75}},
        )
        self.assertEqual(metrics.status_code, 200)
        self.assertEqual(metrics.get_json()["metrics"]["loss"], 0.25)
        finished_telemetry = self.client.post(f"/api/v1/telemetry-runs/{telemetry_id}/finish", json={})
        self.assertEqual(finished_telemetry.status_code, 200)
        self.assertEqual(finished_telemetry.get_json()["status"], "completed")

        job = self.client.post(
            "/api/v1/jobs",
            json={
                "experiment_id": experiment_id,
                "assignment_id": assignment_id,
                "provider": "local",
                "inputs": {"command": [sys.executable, "-c", "print('ok')"]},
            },
        )
        self.assertEqual(job.status_code, 202)
        job_payload = job.get_json()
        job_id = job_payload["job_id"]
        self.assertTrue(job_payload["inputs"]["environment_enforced"])
        self.assertEqual(job_payload["inputs"]["task_id"], "toy_eval")
        self.assertEqual(job_payload["details"]["environment"]["provider"], "local_venv")
        self.assertEqual(job_payload["inputs"]["env"]["AO_ENVIRONMENT_ID"], environment_payload["environment_id"])
        completed_job = self._wait_for(f"/api/v1/jobs/{job_id}")
        self.assertEqual(completed_job["status"], "completed")
        logs = self.client.get(f"/api/v1/jobs/{job_id}/logs")
        self.assertEqual(logs.status_code, 200)
        self.assertIn("ok", logs.get_json()["stdout"])

        shell_job = self.client.post(
            "/api/v1/jobs",
            json={
                "experiment_id": experiment_id,
                "assignment_id": assignment_id,
                "provider": "local",
                "inputs": {
                    "command": "python -c \"import os, sys; print(os.environ.get('AO_ENVIRONMENT_ID')); print(sys.executable)\""
                },
            },
        )
        self.assertEqual(shell_job.status_code, 202, shell_job.get_data(as_text=True))
        shell_job_id = shell_job.get_json()["job_id"]
        completed_shell_job = self._wait_for(f"/api/v1/jobs/{shell_job_id}")
        self.assertEqual(completed_shell_job["status"], "completed")
        shell_logs = self.client.get(f"/api/v1/jobs/{shell_job_id}/logs")
        self.assertEqual(shell_logs.status_code, 200)
        self.assertIn(environment_payload["environment_id"], shell_logs.get_json()["stdout"])
        self.assertIn(str(Path(environment_payload["python_path"]).parent), shell_logs.get_json()["stdout"])

        blocked = self.client.post(
            "/api/v1/jobs",
            json={
                "experiment_id": experiment_id,
                "assignment_id": assignment_id,
                "provider": "local-docker",
                "image": "python:3.11",
                "inputs": {"command": "python -c 'print(1)'"},
                "requires_approval": True,
            },
        )
        self.assertEqual(blocked.status_code, 201)
        blocked_payload = blocked.get_json()
        self.assertEqual(blocked_payload["status"], "blocked")
        self.assertEqual(blocked_payload["details"]["policy_block"]["reason"], "approval_required")

        runpod_job = self.client.post(
            "/api/v1/jobs",
            json={
                "experiment_id": experiment_id,
                "assignment_id": assignment_id,
                "provider": "runpod",
                "template_id": "template_test",
                "inputs": {"command": "python -c 'print(42)'"},
                "dry_run": True,
                "estimated_cost": {"estimated_usd": 0.25},
                "policy": {"auto_approve": True, "auto_approval_cost_cap_usd": 1.0},
            },
        )
        self.assertEqual(runpod_job.status_code, 202)
        runpod_payload = runpod_job.get_json()
        self.assertEqual(runpod_payload["provider"], "runpod")
        self.assertTrue(runpod_payload["details"]["policy_decision"]["auto_approved"])
        self.assertTrue(runpod_payload["details"]["runpod"]["dry_run"])

        over_cap = self.client.post(
            "/api/v1/jobs",
            json={
                "experiment_id": experiment_id,
                "assignment_id": assignment_id,
                "provider": "runpod",
                "template_id": "template_test",
                "inputs": {"command": "python -c 'print(42)'"},
                "dry_run": True,
                "estimated_cost": {"estimated_usd": 2.0},
                "policy": {"auto_approve": True, "auto_approval_cost_cap_usd": 1.0},
            },
        )
        self.assertEqual(over_cap.status_code, 201)
        self.assertEqual(over_cap.get_json()["status"], "blocked")
        self.assertEqual(over_cap.get_json()["details"]["policy_block"]["reason"], "auto_approval_cost_cap_exceeded")

        events = self.client.get("/api/v1/events", query_string={"experiment_id": experiment_id})
        self.assertEqual(events.status_code, 200)
        self.assertTrue(events.get_json()["events"])
        trace = self.client.get(f"/api/v1/sessions/{session_id}/trace")
        self.assertEqual(trace.status_code, 200)
        self.assertEqual(trace.get_json()["session"]["session_id"], session_id)

    def test_workspace_bootstrap_uses_direction_incumbent_and_shared_tools(self) -> None:
        ctx = self.client.application.config["AO_CONTEXT"]
        experiment = ctx.control.create_experiment(
            {
                "task_id": "toy_eval",
                "mode": "local",
                "config": {"shared_tools": {"auto_checkout_limit": 2}},
            }
        )
        assignment = ctx.control.create_assignment(
            {
                "experiment_id": experiment["experiment_id"],
                "task_id": "toy_eval",
                "agent_id": "agent_001",
                "direction_id": "explore_a",
            }
        )

        global_seed = self.root / "global_seed.py"
        global_seed.write_text("def solve():\n    return 'global'\n", encoding="utf-8")
        direction_seed = self.root / "direction_seed.py"
        direction_seed.write_text("def solve():\n    return 'direction'\n", encoding="utf-8")
        global_artifact = self.client.post(
            "/api/v1/artifacts",
            json={
                "experiment_id": experiment["experiment_id"],
                "assignment_id": assignment["assignment_id"],
                "kind": "candidate",
                "path": str(global_seed),
            },
        ).get_json()
        direction_artifact = self.client.post(
            "/api/v1/artifacts",
            json={
                "experiment_id": experiment["experiment_id"],
                "assignment_id": assignment["assignment_id"],
                "kind": "candidate",
                "path": str(direction_seed),
            },
        ).get_json()
        ctx.control.create_leaderboard_entry(
            {
                "experiment_id": experiment["experiment_id"],
                "task_id": "toy_eval",
                "evaluation_id": "eval_global_seed",
                "artifact_id": global_artifact["artifact_id"],
                "score": 10.0,
            }
        )
        ctx.control.create_leaderboard_entry(
            {
                "experiment_id": experiment["experiment_id"],
                "task_id": "toy_eval",
                "assignment_id": assignment["assignment_id"],
                "direction_id": "explore_a",
                "evaluation_id": "eval_direction_seed",
                "artifact_id": direction_artifact["artifact_id"],
                "score": 2.0,
            }
        )

        for index in range(3):
            tool_source = self.root / f"tool_{index}.py"
            tool_source.write_text(f"print('tool {index}')\n", encoding="utf-8")
            tool = self.client.post(
                "/api/v1/shared-tools",
                json={
                    "experiment_id": experiment["experiment_id"],
                    "assignment_id": assignment["assignment_id"],
                    "task_id": "toy_eval",
                    "path": str(tool_source),
                    "name": f"tool-{index}",
                    "description": "reusable helper",
                    "entrypoint": tool_source.name,
                },
            )
            self.assertEqual(tool.status_code, 201, tool.get_data(as_text=True))

        workspace_root = self.root / "workspace_bootstrap"
        bootstrap = self.client.post(
            f"/api/v1/assignments/{assignment['assignment_id']}/workspace-bootstrap",
            json={"workspace_root": str(workspace_root), "session_id": "session_bootstrap"},
        )
        self.assertEqual(bootstrap.status_code, 201, bootstrap.get_data(as_text=True))
        payload = bootstrap.get_json()
        self.assertEqual(payload["workspace_seed"]["kind"], "direction_incumbent")
        self.assertEqual(payload["workspace_seed"]["artifact_id"], direction_artifact["artifact_id"])
        self.assertIn("return 'direction'", (workspace_root / "initial.py").read_text(encoding="utf-8"))
        self.assertEqual(len(payload["checked_out_tools"]), 2)
        for checked_out in payload["checked_out_tools"]:
            self.assertTrue(Path(checked_out["destination_path"]).exists())

    def test_worker_manager_reaper_removes_finished_process(self) -> None:
        ctx = self.client.application.config["AO_CONTEXT"]
        manager = WorkerManager(
            repo_root=Path.cwd(),
            state_root=self.state_root,
            control=ctx.control,
            reaper_interval_s=0.05,
        )
        process = subprocess.Popen(
            [sys.executable, "-c", "pass"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        try:
            with manager._lock:
                manager._assignment_processes["assign_reap"] = WorkerProcess(
                    assignment_id="assign_reap",
                    session_id="session_reap",
                    experiment_id="exp_reap",
                    process=process,
                )
            deadline = time.time() + 5.0
            while time.time() < deadline:
                with manager._lock:
                    active = "assign_reap" in manager._assignment_processes
                if not active:
                    break
                time.sleep(0.05)
            self.assertFalse(active)
            self.assertEqual(process.returncode, 0)
            self.assertIsNone(manager.worker_status("assign_reap"))
        finally:
            manager.close()
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=2.0)

    def test_job_service_reaper_removes_finished_launcher(self) -> None:
        ctx = self.client.application.config["AO_CONTEXT"]
        experiment = ctx.control.create_experiment({"task_id": "toy_eval", "mode": "local"})
        jobs = JobService(
            repository=ctx.control,
            job_root=self.state_root / "direct_jobs",
            database_path=self.db_path,
            reaper_interval_s=0.05,
        )
        try:
            job = jobs.launch(
                {
                    "experiment_id": experiment["experiment_id"],
                    "provider": "local",
                    "inputs": {"command": [sys.executable, "-c", "print('ok')"]},
                }
            )
            job_id = job["job_id"]
            deadline = time.time() + 5.0
            record = None
            active = True
            while time.time() < deadline:
                record = ctx.control.get_job(job_id)
                with jobs._lock:
                    active = job_id in jobs._local_processes
                if record is not None and record["status"] == "completed" and not active:
                    break
                time.sleep(0.05)
            self.assertIsNotNone(record)
            self.assertEqual(record["status"], "completed")
            self.assertFalse(active)
        finally:
            jobs.close()

    def test_experiment_status_rolls_up_from_assignments(self) -> None:
        created = self.client.post(
            "/api/v1/experiments",
            json={"task_id": "toy_eval", "mode": "local", "assignment_count": 1},
        )
        self.assertEqual(created.status_code, 201, created.get_data(as_text=True))
        experiment_id = created.get_json()["experiment"]["experiment_id"]
        assignment_id = created.get_json()["assignments"][0]["assignment_id"]

        session = self.client.post(f"/api/v1/assignments/{assignment_id}/sessions", json={})
        self.assertEqual(session.status_code, 201, session.get_data(as_text=True))
        running_experiment = self.client.get(f"/api/v1/experiments/{experiment_id}")
        self.assertEqual(running_experiment.status_code, 200)
        self.assertEqual(running_experiment.get_json()["experiment"]["status"], "running")

        patched = self.client.patch(f"/api/v1/sessions/{session.get_json()['session_id']}", json={"status": "completed"})
        self.assertEqual(patched.status_code, 200, patched.get_data(as_text=True))
        completed_experiment = self.client.get(f"/api/v1/experiments/{experiment_id}")
        self.assertEqual(completed_experiment.status_code, 200)
        experiment = completed_experiment.get_json()["experiment"]
        self.assertEqual(experiment["status"], "completed")
        self.assertEqual(experiment["metadata"]["status_rollup"]["source"], "control_plane_resource_rollup")

    def test_max_jobs_counts_active_jobs_only(self) -> None:
        ctx = self.client.application.config["AO_CONTEXT"]
        experiment = ctx.control.create_experiment(
            {
                "task_id": "toy_eval",
                "mode": "local",
                "budget": {"max_jobs": 1},
            }
        )
        ctx.control.create_job(
            {
                "experiment_id": experiment["experiment_id"],
                "provider": "local",
                "status": "completed",
                "inputs": {"command": [sys.executable, "-c", "print('done')"]},
            }
        )
        policy = PolicyService(ctx.control)
        allowed_after_completed = policy.decide_job(
            {
                "experiment_id": experiment["experiment_id"],
                "provider": "local",
                "inputs": {"command": [sys.executable, "-c", "print('next')"]},
            }
        )
        self.assertTrue(allowed_after_completed.allowed)

        ctx.control.create_job(
            {
                "experiment_id": experiment["experiment_id"],
                "provider": "local",
                "status": "running",
                "inputs": {"command": [sys.executable, "-c", "print('active')"]},
            }
        )
        blocked_with_active = policy.decide_job(
            {
                "experiment_id": experiment["experiment_id"],
                "provider": "local",
                "inputs": {"command": [sys.executable, "-c", "print('blocked')"]},
            }
        )
        self.assertFalse(blocked_with_active.allowed)
        self.assertEqual(blocked_with_active.reason, "max_jobs_exceeded")

    def test_worker_reaper_auto_continues_assignment_until_evaluator_budget(self) -> None:
        ctx, experiment, assignment, session = self._completed_budgeted_assignment(total_budget=3, assignment_budget=3, used_evals=1)
        manager = WorkerManager(repo_root=self.root, state_root=self.state_root, control=ctx.control, reaper_interval_s=None)
        started = []

        def fake_start_control_assignment(**kwargs):
            started.append(kwargs)
            return ctx.control.create_session(
                {
                    "assignment_id": kwargs["assignment_id"],
                    "status": "running",
                    "details": {"api_url": kwargs["api_url"], "dry_run": kwargs["dry_run"]},
                }
            )

        manager.start_control_assignment = fake_start_control_assignment  # type: ignore[method-assign]
        continuation = manager._maybe_continue_assignment(
            {
                "assignment_id": assignment["assignment_id"],
                "session_id": session["session_id"],
                "experiment_id": experiment["experiment_id"],
                "returncode": 0,
                "api_url": "http://127.0.0.1:5010",
                "dry_run": False,
                "max_turn_wall_time_s": 123,
            }
        )

        self.assertIsNotNone(continuation)
        self.assertEqual(len(started), 1)
        self.assertEqual(started[0]["assignment_id"], assignment["assignment_id"])
        self.assertEqual(started[0]["max_turn_wall_time_s"], 123)
        self.assertEqual(ctx.control.get_assignment(assignment["assignment_id"])["status"], "running")
        self.assertEqual(ctx.control.get_experiment(experiment["experiment_id"])["status"], "running")
        events = ctx.control.list_events(assignment_id=assignment["assignment_id"])
        self.assertTrue(any(event["event_type"] == "assignment.auto_continue" for event in events))

    def test_worker_reaper_marks_budget_exhausted_without_continuation(self) -> None:
        ctx, experiment, assignment, session = self._completed_budgeted_assignment(total_budget=1, assignment_budget=1, used_evals=1)
        manager = WorkerManager(repo_root=self.root, state_root=self.state_root, control=ctx.control, reaper_interval_s=None)

        def fail_start_control_assignment(**kwargs):
            raise AssertionError("budget-exhausted assignment should not continue")

        manager.start_control_assignment = fail_start_control_assignment  # type: ignore[method-assign]
        continuation = manager._maybe_continue_assignment(
            {
                "assignment_id": assignment["assignment_id"],
                "session_id": session["session_id"],
                "experiment_id": experiment["experiment_id"],
                "returncode": 0,
            }
        )

        self.assertIsNone(continuation)
        updated = ctx.control.get_assignment(assignment["assignment_id"])
        self.assertEqual(updated["status"], "completed")
        self.assertEqual(updated["metadata"]["budget_exhausted"]["source"], "worker_reaper")
        events = ctx.control.list_events(assignment_id=assignment["assignment_id"])
        self.assertTrue(any(event["event_type"] == "assignment.budget_exhausted" for event in events))

    def test_worker_reaper_continues_after_local_stop_condition(self) -> None:
        ctx, experiment, assignment, session = self._completed_budgeted_assignment(total_budget=3, assignment_budget=3, used_evals=1)
        session = ctx.control.update_session(
            session["session_id"],
            {
                "details": {
                    "local_stop_condition": {
                        "source": "worker",
                        "scope": "local",
                        "reason": "current basin did not improve",
                    }
                }
            },
        )
        manager = WorkerManager(repo_root=self.root, state_root=self.state_root, control=ctx.control, reaper_interval_s=None)
        started = []

        def fake_start_control_assignment(**kwargs):
            started.append(kwargs)
            return ctx.control.create_session({"assignment_id": kwargs["assignment_id"], "status": "running"})

        manager.start_control_assignment = fake_start_control_assignment  # type: ignore[method-assign]
        continuation = manager._maybe_continue_assignment(
            {
                "assignment_id": assignment["assignment_id"],
                "session_id": session["session_id"],
                "experiment_id": experiment["experiment_id"],
                "returncode": 0,
                "api_url": "http://127.0.0.1:5010",
            }
        )

        self.assertIsNotNone(continuation)
        self.assertEqual(len(started), 1)
        self.assertEqual(ctx.control.get_assignment(assignment["assignment_id"])["status"], "running")

    def test_worker_reaper_respects_global_stop_condition(self) -> None:
        ctx, experiment, assignment, session = self._completed_budgeted_assignment(
            total_budget=3,
            assignment_budget=3,
            used_evals=1,
            metadata={"global_stop_condition": {"source": "worker", "scope": "global", "reason": "converged"}},
        )
        manager = WorkerManager(repo_root=self.root, state_root=self.state_root, control=ctx.control, reaper_interval_s=None)

        def fail_start_control_assignment(**kwargs):
            raise AssertionError("global stop condition should not continue")

        manager.start_control_assignment = fail_start_control_assignment  # type: ignore[method-assign]
        continuation = manager._maybe_continue_assignment(
            {
                "assignment_id": assignment["assignment_id"],
                "session_id": session["session_id"],
                "experiment_id": experiment["experiment_id"],
                "returncode": 0,
            }
        )

        self.assertIsNone(continuation)
        updated = ctx.control.get_assignment(assignment["assignment_id"])
        self.assertEqual(updated["status"], "completed")
        self.assertEqual(updated["metadata"]["global_stop_condition"]["reason"], "converged")

    def test_worker_reaper_auto_continues_after_turn_timeout(self) -> None:
        ctx, experiment, assignment, session = self._completed_budgeted_assignment(
            total_budget=3,
            assignment_budget=3,
            used_evals=1,
            session_status="stopped",
            stop_reason="turn_timeout",
        )
        manager = WorkerManager(repo_root=self.root, state_root=self.state_root, control=ctx.control, reaper_interval_s=None)
        started = []

        def fake_start_control_assignment(**kwargs):
            started.append(kwargs)
            return ctx.control.create_session({"assignment_id": kwargs["assignment_id"], "status": "running"})

        manager.start_control_assignment = fake_start_control_assignment  # type: ignore[method-assign]
        continuation = manager._maybe_continue_assignment(
            {
                "assignment_id": assignment["assignment_id"],
                "session_id": session["session_id"],
                "experiment_id": experiment["experiment_id"],
                "returncode": 0,
                "api_url": "http://127.0.0.1:5010",
            }
        )

        self.assertIsNotNone(continuation)
        self.assertEqual(len(started), 1)
        self.assertEqual(ctx.control.get_assignment(assignment["assignment_id"])["status"], "running")

    def test_worker_reaper_recovers_stale_starting_session(self) -> None:
        ctx = self.client.application.config["AO_CONTEXT"]
        experiment = ctx.control.create_experiment(
            {
                "task_id": "toy_eval",
                "mode": "local",
                "budget": {"total_evaluator_runs": 2},
            }
        )
        assignment = ctx.control.create_assignment(
            {
                "experiment_id": experiment["experiment_id"],
                "task_id": "toy_eval",
                "agent_id": "agent_001",
                "budget": {"evaluator_runs": 2},
            }
        )
        stale = ctx.control.create_session(
            {
                "assignment_id": assignment["assignment_id"],
                "status": "starting",
                "updated_at": "2000-01-01T00:00:00Z",
            }
        )
        manager = WorkerManager(
            repo_root=self.root,
            state_root=self.state_root,
            control=ctx.control,
            reaper_interval_s=None,
            stale_session_grace_s=0,
            default_api_url="http://127.0.0.1:5010",
        )
        started = []

        def fake_start_control_assignment(**kwargs):
            started.append(kwargs)
            return ctx.control.create_session(
                {
                    "assignment_id": kwargs["assignment_id"],
                    "status": "running",
                    "details": {"api_url": kwargs["api_url"], "dry_run": kwargs["dry_run"]},
                }
            )

        manager.start_control_assignment = fake_start_control_assignment  # type: ignore[method-assign]
        try:
            recovered = manager.recover_stale_sessions()
        finally:
            manager.close()

        self.assertEqual(len(recovered), 1)
        self.assertEqual(len(started), 1)
        self.assertEqual(started[0]["assignment_id"], assignment["assignment_id"])
        stale_after = ctx.control.get_session(stale["session_id"])
        self.assertEqual(stale_after["status"], "failed")
        self.assertEqual(stale_after["details"]["stale_session_recovery"]["reason"], "no_managed_worker_process")
        sessions = ctx.control.list_sessions(assignment_id=assignment["assignment_id"])
        self.assertTrue(any(session["status"] == "running" for session in sessions))
        events = ctx.control.list_events(assignment_id=assignment["assignment_id"])
        self.assertTrue(any(event["event_type"] == "worker.session.stale" for event in events))
        self.assertTrue(any(event["event_type"] == "assignment.stale_session_restarted" for event in events))

    def test_worker_reaper_preserves_unmanaged_running_session_with_live_pid(self) -> None:
        ctx = self.client.application.config["AO_CONTEXT"]
        experiment = ctx.control.create_experiment(
            {
                "task_id": "toy_eval",
                "mode": "local",
                "budget": {"total_evaluator_runs": 2},
            }
        )
        assignment = ctx.control.create_assignment(
            {
                "experiment_id": experiment["experiment_id"],
                "task_id": "toy_eval",
                "agent_id": "agent_001",
                "budget": {"evaluator_runs": 2},
            }
        )
        session = ctx.control.create_session(
            {
                "assignment_id": assignment["assignment_id"],
                "status": "running",
                "pid": os.getpid(),
                "updated_at": "2000-01-01T00:00:00Z",
            }
        )
        manager = WorkerManager(
            repo_root=self.root,
            state_root=self.state_root,
            control=ctx.control,
            reaper_interval_s=None,
            stale_session_grace_s=0,
            default_api_url="http://127.0.0.1:5010",
        )
        try:
            recovered = manager.recover_stale_sessions()
        finally:
            manager.close()

        self.assertEqual(recovered, [])
        self.assertEqual(ctx.control.get_session(session["session_id"])["status"], "running")
        events = ctx.control.list_events(assignment_id=assignment["assignment_id"])
        self.assertFalse(any(event["event_type"] == "worker.session.stale" for event in events))

    def test_worker_reaper_recovers_unmanaged_completed_session(self) -> None:
        ctx, experiment, assignment, session = self._completed_budgeted_assignment(total_budget=3, assignment_budget=3, used_evals=1)
        manager = WorkerManager(
            repo_root=self.root,
            state_root=self.state_root,
            control=ctx.control,
            reaper_interval_s=None,
            default_api_url="http://127.0.0.1:5010",
        )
        started = []

        def fake_start_control_assignment(**kwargs):
            started.append(kwargs)
            return ctx.control.create_session({"assignment_id": kwargs["assignment_id"], "status": "running"})

        manager.start_control_assignment = fake_start_control_assignment  # type: ignore[method-assign]
        try:
            recovered = manager.recover_stale_sessions()
        finally:
            manager.close()

        self.assertEqual(len(recovered), 1)
        self.assertEqual(len(started), 1)
        self.assertEqual(started[0]["assignment_id"], assignment["assignment_id"])
        self.assertEqual(ctx.control.get_assignment(assignment["assignment_id"])["status"], "running")
        events = ctx.control.list_events(assignment_id=assignment["assignment_id"])
        self.assertTrue(any(event["event_type"] == "assignment.auto_continue" for event in events))

    def test_evaluation_api_enforces_evaluator_budget(self) -> None:
        created = self.client.post(
            "/api/v1/experiments",
            json={"task_id": "toy_eval", "mode": "local", "budget": {"total_evaluator_runs": 1}, "assignment_count": 1},
        )
        self.assertEqual(created.status_code, 201, created.get_data(as_text=True))
        experiment_id = created.get_json()["experiment"]["experiment_id"]
        assignment_id = created.get_json()["assignments"][0]["assignment_id"]
        entry_path = self.tasks_root / "toy_eval" / "public" / "initial.py"

        first = self.client.post(
            "/api/v1/evaluations",
            json={
                "experiment_id": experiment_id,
                "assignment_id": assignment_id,
                "task_id": "toy_eval",
                "kind": "verify",
                "entry_path": str(entry_path),
                "async": False,
            },
        )
        self.assertEqual(first.status_code, 201, first.get_data(as_text=True))

        submitted = self.client.post(
            "/api/v1/evaluations",
            json={
                "experiment_id": experiment_id,
                "assignment_id": assignment_id,
                "task_id": "toy_eval",
                "kind": "submit",
                "entry_path": str(entry_path),
                "async": False,
            },
        )
        self.assertEqual(submitted.status_code, 201, submitted.get_data(as_text=True))

        second_submit = self.client.post(
            "/api/v1/evaluations",
            json={
                "experiment_id": experiment_id,
                "assignment_id": assignment_id,
                "task_id": "toy_eval",
                "kind": "submit",
                "entry_path": str(entry_path),
                "async": False,
            },
        )
        self.assertEqual(second_submit.status_code, 400)
        self.assertIn("evaluator_budget_exhausted", second_submit.get_json()["error"])
        ctx = self.client.application.config["AO_CONTEXT"]
        self.assertEqual(len(ctx.control.list_evaluations(assignment_id=assignment_id)), 2)
        self.assertEqual(len(ctx.control.list_leaderboard_entries(experiment_id=experiment_id)), 1)

    def test_replay_bundle_roundtrip_local_venv(self) -> None:
        created = self.client.post(
            "/api/v1/experiments",
            json={"task_id": "toy_eval", "mode": "local", "budget": {"total_evaluator_runs": 4}, "assignment_count": 1},
        )
        self.assertEqual(created.status_code, 201, created.get_data(as_text=True))
        experiment_id = created.get_json()["experiment"]["experiment_id"]
        assignment_id = created.get_json()["assignments"][0]["assignment_id"]
        entry_path = self.tasks_root / "toy_eval" / "public" / "initial.py"

        submitted = self.client.post(
            "/api/v1/evaluations",
            json={
                "experiment_id": experiment_id,
                "assignment_id": assignment_id,
                "task_id": "toy_eval",
                "kind": "submit",
                "entry_path": str(entry_path),
                "async": False,
            },
        )
        self.assertEqual(submitted.status_code, 201, submitted.get_data(as_text=True))
        original = submitted.get_json()
        self.assertEqual(original["status"], "completed")
        self.assertEqual(original["score"], 1.0)
        self.assertEqual(original["request"]["environment_provider"], "local_venv")
        self.assertEqual(original["request"]["framework_environment_id"], "env_framework_current")
        self.assertIn("framework_environment_lock", original["request"])

        exported = self.client.post(f"/api/v1/evaluations/{original['evaluation_id']}/replay-bundle", json={})
        self.assertEqual(exported.status_code, 201, exported.get_data(as_text=True))
        bundle = exported.get_json()
        self.assertEqual(bundle["artifact"]["kind"], "replay_bundle")
        self.assertEqual(bundle["manifest"]["environment"]["provider"], "local_venv")
        self.assertEqual(bundle["manifest"]["task_context"]["task_knowledge"]["file_count"], 2)
        archive_path = Path(bundle["artifact"]["local_path"])
        self.assertTrue(archive_path.exists())
        with tarfile.open(archive_path, "r:gz") as archive:
            names = archive.getnames()
        self.assertTrue(any(name.endswith("/manifest.json") for name in names))
        self.assertTrue(any(name.endswith("/candidate/content/initial.py") for name in names))
        self.assertTrue(any(name.endswith("/environment/environment.json") for name in names))

        replayed = self.client.post("/api/v1/replay", json={"artifact_id": bundle["artifact"]["artifact_id"]})
        self.assertEqual(replayed.status_code, 201, replayed.get_data(as_text=True))
        replay_payload = replayed.get_json()
        replay_eval = replay_payload["evaluation"]
        self.assertEqual(replay_eval["status"], "completed")
        self.assertEqual(replay_eval["score"], original["score"])
        self.assertEqual(replay_eval["request"]["environment_provider"], "local_venv")
        self.assertEqual(replay_eval["request"]["replay"]["source_evaluation_id"], original["evaluation_id"])
        self.assertNotEqual(replay_eval["evaluation_id"], original["evaluation_id"])
        leaderboard = self.client.get("/api/v1/leaderboard", query_string={"experiment_id": experiment_id})
        self.assertEqual(leaderboard.status_code, 200, leaderboard.get_data(as_text=True))
        self.assertEqual(len(leaderboard.get_json()["leaderboard"]), 1)

    def test_official_submit_overlay_policy_is_explicit(self) -> None:
        ctx = self.client.application.config["AO_CONTEXT"]
        created = self.client.post(
            "/api/v1/experiments",
            json={"task_id": "toy_eval", "mode": "local", "budget": {"total_evaluator_runs": 4}, "assignment_count": 1},
        )
        self.assertEqual(created.status_code, 201, created.get_data(as_text=True))
        experiment_id = created.get_json()["experiment"]["experiment_id"]
        assignment_id = created.get_json()["assignments"][0]["assignment_id"]
        base_environment = ctx.control_service.ensure_environment(
            {"experiment_id": experiment_id, "assignment_id": assignment_id, "task_id": "toy_eval"}
        )
        overlay = ctx.control.create_environment_overlay(
            {
                "base_environment_id": base_environment["environment_id"],
                "experiment_id": experiment_id,
                "assignment_id": assignment_id,
                "status": "ready",
                "approved": True,
                "python_path": base_environment["python_path"],
                "root_path": base_environment["root_path"],
                "requirements": [],
                "lock": {"status": "ready", "format": "test-overlay-lock"},
                "metadata": {"test_overlay": True},
            }
        )
        entry_path = self.tasks_root / "toy_eval" / "public" / "initial.py"

        denied = self.client.post(
            "/api/v1/evaluations",
            json={
                "experiment_id": experiment_id,
                "assignment_id": assignment_id,
                "task_id": "toy_eval",
                "kind": "submit",
                "entry_path": str(entry_path),
                "environment_overlay_id": overlay["overlay_id"],
                "async": False,
            },
        )
        self.assertEqual(denied.status_code, 400)
        self.assertIn("official_overlay_submit_disabled", denied.get_json()["error"])

        allowed_experiment = ctx.control.create_experiment(
            {
                "task_id": "toy_eval",
                "mode": "local",
                "budget": {"total_evaluator_runs": 4},
                "policy": {"environments": {"allow_official_overlay_submit": True}},
            }
        )
        allowed_assignment = ctx.control.create_assignment(
            {
                "experiment_id": allowed_experiment["experiment_id"],
                "task_id": "toy_eval",
                "agent_id": "agent_overlay",
            }
        )
        allowed_base = ctx.control_service.ensure_environment(
            {
                "experiment_id": allowed_experiment["experiment_id"],
                "assignment_id": allowed_assignment["assignment_id"],
                "task_id": "toy_eval",
            }
        )
        allowed_overlay = ctx.control.create_environment_overlay(
            {
                "base_environment_id": allowed_base["environment_id"],
                "experiment_id": allowed_experiment["experiment_id"],
                "assignment_id": allowed_assignment["assignment_id"],
                "status": "ready",
                "approved": True,
                "python_path": allowed_base["python_path"],
                "root_path": allowed_base["root_path"],
                "requirements": [],
                "lock": {"status": "ready", "format": "test-overlay-lock"},
            }
        )

        submitted = self.client.post(
            "/api/v1/evaluations",
            json={
                "experiment_id": allowed_experiment["experiment_id"],
                "assignment_id": allowed_assignment["assignment_id"],
                "task_id": "toy_eval",
                "kind": "submit",
                "entry_path": str(entry_path),
                "environment_overlay_id": allowed_overlay["overlay_id"],
                "async": False,
            },
        )
        self.assertEqual(submitted.status_code, 201, submitted.get_data(as_text=True))
        evaluation = submitted.get_json()
        self.assertEqual(evaluation["status"], "completed")
        self.assertEqual(evaluation["request"]["environment_overlay_id"], allowed_overlay["overlay_id"])
        self.assertEqual(evaluation["request"]["environment_kind"], "overlay")
        leaderboard = ctx.control.list_leaderboard_entries(experiment_id=allowed_experiment["experiment_id"])
        self.assertEqual(leaderboard[0]["environment_overlay_id"], allowed_overlay["overlay_id"])

    @unittest.skipUnless(_docker_runtime_available(), "Docker runtime is not available")
    def test_replay_bundle_roundtrip_docker_image_provider(self) -> None:
        repo_temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(repo_temp.cleanup)
        old_root = self.root
        old_tasks_root = self.tasks_root
        old_state_root = self.state_root
        old_db_path = self.db_path
        old_tasks_roots_env = os.environ.get("AO_TASKS_ROOTS")
        try:
            self.root = Path(repo_temp.name)
            self.tasks_root = self.root / "toy_tasks_bundle"
            self.state_root = self.root / "state"
            self.db_path = self.state_root / "control.sqlite3"
            os.environ["AO_TASKS_ROOTS"] = str(self.tasks_root)
            self._write_toy_task()
            client = create_app(state_root=self.state_root, database_path=self.db_path).test_client()

            created = client.post(
                "/api/v1/experiments",
                json={
                    "task_id": "toy_eval",
                    "mode": "local",
                    "budget": {"total_evaluator_runs": 4},
                    "assignment_count": 1,
                    "config": {
                        "environment_provider": "docker_image",
                        "environment": {
                            "provider": "docker_image",
                            "base_image": "python:3.11-slim",
                            "build_timeout_s": 1800,
                            "preflight_timeout_s": 300,
                        },
                    },
                    "policy": {"network": {"external_internet": "allow"}},
                },
            )
            self.assertEqual(created.status_code, 201, created.get_data(as_text=True))
            experiment_id = created.get_json()["experiment"]["experiment_id"]
            assignment_id = created.get_json()["assignments"][0]["assignment_id"]
            entry_path = self.tasks_root / "toy_eval" / "public" / "initial.py"

            submitted = client.post(
                "/api/v1/evaluations",
                json={
                    "experiment_id": experiment_id,
                    "assignment_id": assignment_id,
                    "task_id": "toy_eval",
                    "kind": "submit",
                    "entry_path": str(entry_path),
                    "async": False,
                },
            )
            self.assertEqual(submitted.status_code, 201, submitted.get_data(as_text=True))
            original = submitted.get_json()
            self.assertEqual(original["status"], "completed", original.get("public_feedback"))
            self.assertEqual(original["score"], 1.0)
            self.assertEqual(original["request"]["environment_provider"], "docker_image")
            self.assertIn("image_reference", original["request"]["runner"])

            exported = client.post(f"/api/v1/evaluations/{original['evaluation_id']}/replay-bundle", json={})
            self.assertEqual(exported.status_code, 201, exported.get_data(as_text=True))
            bundle = exported.get_json()
            self.assertEqual(bundle["manifest"]["environment"]["provider"], "docker_image")
            self.assertEqual(bundle["artifact"]["kind"], "replay_bundle")

            replayed = client.post("/api/v1/replay", json={"artifact_id": bundle["artifact"]["artifact_id"]})
            self.assertEqual(replayed.status_code, 201, replayed.get_data(as_text=True))
            replay_eval = replayed.get_json()["evaluation"]
            self.assertEqual(replay_eval["status"], "completed", replay_eval.get("public_feedback"))
            self.assertEqual(replay_eval["score"], original["score"])
            self.assertEqual(replay_eval["request"]["environment_provider"], "docker_image")
            self.assertEqual(replay_eval["request"]["replay"]["source_evaluation_id"], original["evaluation_id"])
            leaderboard = client.get("/api/v1/leaderboard", query_string={"experiment_id": experiment_id})
            self.assertEqual(leaderboard.status_code, 200, leaderboard.get_data(as_text=True))
            self.assertEqual(len(leaderboard.get_json()["leaderboard"]), 1)
        finally:
            self.root = old_root
            self.tasks_root = old_tasks_root
            self.state_root = old_state_root
            self.db_path = old_db_path
            if old_tasks_roots_env is None:
                os.environ.pop("AO_TASKS_ROOTS", None)
            else:
                os.environ["AO_TASKS_ROOTS"] = old_tasks_roots_env

    def test_docker_network_enforcement_command_builder(self) -> None:
        command, enforcement = build_local_docker_command(
            image="python:3.11",
            command="python -c 'print(1)'",
            cwd=self.root,
            network_policy={"external_internet": "deny", "control_plane": "allow"},
        )
        self.assertIn("--network", command)
        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertTrue(enforcement["external_internet_enforced"])
        self.assertFalse(enforcement["control_plane_available"])

        with self.assertRaises(DockerNetworkPolicyError) as bridge_error:
            build_local_docker_command(
                image="python:3.11",
                command="python -c 'print(1)'",
                cwd=self.root,
                network_policy={"external_internet": "deny", "control_plane": "allow"},
                requested_network_mode="bridge",
            )
        self.assertEqual(bridge_error.exception.reason, "docker_network_mode_violates_external_deny")

        with self.assertRaises(DockerNetworkPolicyError) as relay_error:
            build_local_docker_command(
                image="python:3.11",
                command="python -c 'print(1)'",
                cwd=self.root,
                network_policy={"external_internet": "deny", "control_plane": "allow"},
                requires_control_plane=True,
            )
        self.assertEqual(relay_error.exception.reason, "docker_control_plane_relay_required")

        relay_socket = self.root / "control.sock"
        relay_socket.write_text("", encoding="utf-8")
        relayed_command, relayed_enforcement = build_local_docker_command(
            image="python:3.11",
            command="python -c 'print(1)'",
            cwd=self.root,
            network_policy={"external_internet": "deny", "control_plane": "allow"},
            requires_control_plane=True,
            control_plane_relay_socket=relay_socket,
        )
        self.assertIn("--network", relayed_command)
        self.assertEqual(relayed_command[relayed_command.index("--network") + 1], "none")
        self.assertIn(f"AO_CONTROL_API_URL=unix:///ao-control/control.sock", relayed_command)
        self.assertTrue(relayed_enforcement["external_internet_enforced"])
        self.assertTrue(relayed_enforcement["control_plane_available"])
        self.assertTrue(relayed_enforcement["control_plane_relay_configured"])
        self.assertEqual(relayed_enforcement["control_plane_relay_transport"], "unix-socket")

        tcp_command, tcp_enforcement = build_local_docker_command(
            image="python:3.11",
            command="python -c 'print(1)'",
            cwd=self.root,
            network_policy={"external_internet": "deny", "control_plane": "allow"},
            requires_control_plane=True,
            control_plane_relay_url="http://host.docker.internal:5123",
        )
        self.assertIn("--network", tcp_command)
        self.assertEqual(tcp_command[tcp_command.index("--network") + 1], "bridge")
        self.assertIn("AO_CONTROL_API_URL=http://host.docker.internal:5123", tcp_command)
        self.assertFalse(tcp_enforcement["external_internet_enforced"])
        self.assertTrue(tcp_enforcement["control_plane_available"])
        self.assertEqual(tcp_enforcement["control_plane_relay_transport"], "tcp")
        self.assertTrue(tcp_enforcement["policy_weakened"])

    def test_docker_command_builder_injects_outbound_audit_proxy(self) -> None:
        command, enforcement = build_local_docker_command(
            image="python:3.11",
            command="python -c 'print(1)'",
            cwd=self.root,
            network_policy={
                "external_internet": "audit",
                "control_plane": "allow",
                "outbound_proxy_url": "http://host.docker.internal:50123",
                "outbound_proxy_no_proxy": "localhost",
            },
            add_hosts=["host.docker.internal:host-gateway"],
        )

        self.assertIn("HTTP_PROXY=http://host.docker.internal:50123", command)
        self.assertIn("HTTPS_PROXY=http://host.docker.internal:50123", command)
        self.assertIn("NO_PROXY=localhost", command)
        self.assertIn("--add-host", command)
        self.assertTrue(enforcement["outbound_audit_proxy_configured"])
        self.assertEqual(enforcement["outbound_audit_mode"], "env_proxy")

    def test_docker_command_builder_supports_unix_outbound_proxy_bridge(self) -> None:
        relay_socket = self.root / "control.sock"
        relay_socket.write_text("", encoding="utf-8")
        proxy_socket = self.root / "outbound.sock"
        proxy_socket.write_text("", encoding="utf-8")
        command, enforcement = build_local_docker_command(
            image="python:3.11",
            command="python -c 'print(1)'",
            cwd=self.root,
            network_policy={
                "external_internet": "audit",
                "control_plane": "allow",
                "outbound_proxy_socket": str(proxy_socket),
                "outbound_proxy_container_socket": "/ao-network/proxy.sock",
                "outbound_proxy_bridge_port": 8765,
                "outbound_proxy_no_proxy": "localhost",
            },
            requires_control_plane=True,
            control_plane_relay_socket=relay_socket,
        )

        self.assertIn("--network", command)
        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertIn(f"{relay_socket.resolve()}:/ao-control/control.sock", command)
        self.assertIn(f"{proxy_socket.resolve()}:/ao-network/proxy.sock", command)
        self.assertIn("AO_OUTBOUND_PROXY_SOCKET=/ao-network/proxy.sock", command)
        self.assertIn("AO_OUTBOUND_PROXY_BRIDGE_PORT=8765", command)
        self.assertIn("HTTP_PROXY=http://127.0.0.1:8765", command)
        self.assertTrue(enforcement["external_internet_enforced"])
        self.assertTrue(enforcement["outbound_audit_proxy_configured"])
        self.assertEqual(enforcement["outbound_audit_proxy_transport"], "unix-socket")
        self.assertEqual(enforcement["outbound_audit_mode"], "unix_proxy_bridge")
        self.assertFalse(enforcement["policy_weakened"])

        with self.assertRaises(DockerNetworkPolicyError) as relay_error:
            build_local_docker_command(
                image="python:3.11",
                command="python -c 'print(1)'",
                cwd=self.root,
                network_policy={
                    "external_internet": "audit",
                    "control_plane": "allow",
                    "outbound_proxy_socket": str(proxy_socket),
                },
                requires_control_plane=True,
                control_plane_relay_url="http://host.docker.internal:50123",
            )
        self.assertEqual(relay_error.exception.reason, "docker_tcp_relay_incompatible_with_proxy_isolation")

    def test_docker_worker_command_builder_uses_relay_and_isolation(self) -> None:
        relay_socket = self.root / "control.sock"
        relay_socket.write_text("", encoding="utf-8")
        assignment = {
            "assignment_id": "assign_docker",
            "experiment_id": "exp_docker",
            "task_id": "toy_eval",
            "agent_id": "agent_001",
        }
        codex_source = self.root / "codex_source"
        codex_source.mkdir()
        workspace_root = self.state_root / "workspaces" / "assign_docker" / "session_docker"
        command, enforcement = build_docker_worker_command(
            image="agentic-opt-worker:test",
            assignment=assignment,
            session_id="session_docker",
            api_url="unix:///ao-control/control.sock",
            workspace_root=workspace_root,
            state_root=self.state_root,
            network_policy={"external_internet": "deny", "control_plane": "allow"},
            control_plane_relay_socket=relay_socket,
            dry_run=True,
            max_turn_wall_time_s=60,
            container_name="agentic-opt-assign-session",
            codex_source_home=codex_source,
        )

        self.assertEqual(command[:3], ["docker", "run", "--rm"])
        self.assertIn("--cap-drop", command)
        self.assertEqual(command[command.index("--cap-drop") + 1], "ALL")
        self.assertIn("--security-opt", command)
        self.assertIn("--network", command)
        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertIn("-v", command)
        self.assertIn(f"{relay_socket.resolve()}:/ao-control/control.sock", command)
        self.assertIn(f"{workspace_root.resolve()}:{workspace_root}", command)
        task_context_mount = enforcement["task_context_enforcement"]["mount"]
        self.assertIn(f"{Path(task_context_mount['source']).resolve()}:{task_context_mount['target']}:ro", command)
        self.assertIn(f"{(self.state_root / 'envs').resolve()}:{self.state_root / 'envs'}:ro", command)
        self.assertIn(f"{codex_source.resolve()}:/ao-codex-source:ro", command)
        self.assertIn("AO_CONTROL_API_URL=unix:///ao-control/control.sock", command)
        self.assertIn("AO_WORKER_RUNTIME_PYTHON=/usr/local/bin/python", command)
        self.assertIn("AO_CODEX_SOURCE_HOME=/ao-codex-source", command)
        self.assertIn("agentic-opt-worker:test", command)
        self.assertIn("--workspace-root", command)
        self.assertIn("--dry-run", command)
        self.assertTrue(enforcement["external_internet_enforced"])
        self.assertTrue(enforcement["control_plane_available"])
        self.assertTrue(enforcement["control_plane_relay_configured"])
        self.assertTrue(enforcement["task_context_enforcement"]["provider_enforced_readonly"])
        self.assertFalse(enforcement["task_context_enforcement"]["policy_weakened"])

    def test_docker_worker_command_builder_supports_tcp_relay_fallback(self) -> None:
        assignment = {
            "assignment_id": "assign_docker",
            "experiment_id": "exp_docker",
            "task_id": "toy_eval",
            "agent_id": "agent_001",
        }
        workspace_root = self.state_root / "workspaces" / "assign_docker" / "session_docker"
        command, enforcement = build_docker_worker_command(
            image="agentic-opt-worker:test",
            assignment=assignment,
            session_id="session_docker",
            api_url="http://host.docker.internal:5123",
            workspace_root=workspace_root,
            state_root=self.state_root,
            network_policy={"external_internet": "deny", "control_plane": "allow"},
            control_plane_relay_url="http://host.docker.internal:5123",
            dry_run=True,
        )

        self.assertIn("--network", command)
        self.assertEqual(command[command.index("--network") + 1], "bridge")
        self.assertIn("--add-host", command)
        self.assertIn("host.docker.internal:host-gateway", command)
        self.assertIn("AO_CONTROL_API_URL=http://host.docker.internal:5123", command)
        self.assertFalse(enforcement["external_internet_enforced"])
        self.assertEqual(enforcement["control_plane_relay_transport"], "tcp")
        self.assertTrue(enforcement["policy_weakened"])

    def test_docker_image_environment_provider_records_digest_and_preflights(self) -> None:
        def fake_run(cmd, **kwargs):
            command = [str(item) for item in cmd]
            if command[:3] == ["docker", "image", "inspect"]:
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    json.dumps(
                        [
                            {
                                "Id": "sha256:imageid",
                                "RepoTags": [command[-1]],
                                "RepoDigests": ["agentic-opt/task-toy_eval@sha256:repo_digest"],
                            }
                        ]
                    ),
                    "",
                )
            if command[:2] == ["docker", "build"]:
                return subprocess.CompletedProcess(cmd, 0, "built\n", "")
            if command[:3] == ["docker", "run", "--rm"]:
                return subprocess.CompletedProcess(cmd, 0, json.dumps({"valid": True}) + "\n", "")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with patch("agentic_opt.control_plane.environments.subprocess.run", side_effect=fake_run):
            response = self.client.post(
                "/api/v1/environments",
                json={"task_id": "toy_eval", "provider": "docker_image", "base_image": "python:3.11-slim"},
            )

        self.assertEqual(response.status_code, 201, response.get_data(as_text=True))
        environment = response.get_json()
        self.assertEqual(environment["status"], "ready")
        self.assertEqual(environment["spec"]["provider"], "docker_image")
        self.assertEqual(environment["lock"]["image_digest"], "sha256:repo_digest")
        self.assertEqual(environment["metadata"]["container_python"], "/usr/local/bin/python")
        self.assertTrue(Path(environment["metadata"]["manifest_path"]).exists())

    def test_docker_image_environment_policy_blocks_untrusted_registry(self) -> None:
        ctx = self.client.application.config["AO_CONTEXT"]
        experiment = ctx.control.create_experiment(
            {
                "task_id": "toy_eval",
                "mode": "local",
                "policy": {"docker_image": {"trusted_registries": ["ghcr.io"]}},
            }
        )

        def fake_run(cmd, **kwargs):
            command = [str(item) for item in cmd]
            if command[:3] == ["docker", "image", "inspect"]:
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    json.dumps(
                        [
                            {
                                "Id": "sha256:imageid",
                                "RepoTags": ["evil.example.com/team/task:latest"],
                                "RepoDigests": ["evil.example.com/team/task@sha256:bad"],
                            }
                        ]
                    ),
                    "",
                )
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with patch("agentic_opt.control_plane.environments.subprocess.run", side_effect=fake_run):
            response = self.client.post(
                "/api/v1/environments",
                json={
                    "experiment_id": experiment["experiment_id"],
                    "task_id": "toy_eval",
                    "provider": "docker_image",
                    "image_ref": "evil.example.com/team/task:latest",
                    "build": False,
                },
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error_type"], "DockerImagePolicyError")

    def test_docker_image_policy_parses_custom_registry(self) -> None:
        identity = docker_image_identity(
            image_ref="ghcr.io/team/task:1.0",
            image_info={
                "Id": "sha256:imageid",
                "RepoTags": ["ghcr.io/team/task:1.0"],
                "RepoDigests": ["ghcr.io/team/task@sha256:repo_digest"],
            },
        )
        self.assertEqual(identity.registries, ("ghcr.io",))
        self.assertEqual(identity.repositories, ("team/task",))

        decision = evaluate_docker_image_policy(
            identity=identity,
            policy={"denied_registries": ["ghcr.io"]},
            source="prebuilt_image",
        )
        self.assertFalse(decision["allowed"])
        self.assertEqual(decision["reason"], "denied_registry")

    def test_docker_image_default_env_reaches_job_and_bundle_export(self) -> None:
        ctx = self.client.application.config["AO_CONTEXT"]
        experiment = ctx.control.create_experiment({"task_id": "toy_eval", "mode": "local"})
        host_root = self.state_root / "envs" / "docker-default-env"
        host_root.mkdir(parents=True)
        (host_root / "manifest.json").write_text('{"ok": true}\n', encoding="utf-8")
        environment = ctx.control.upsert_environment(
            {
                "environment_id": "env_docker_default_env",
                "environment_type": "task",
                "status": "ready",
                "fingerprint": "dockerdefault",
                "python_path": "/usr/local/bin/python",
                "root_path": str(host_root),
                "task_id": "toy_eval",
                "experiment_id": experiment["experiment_id"],
                "spec": {"kind": "docker_image", "provider": "docker_image"},
                "lock": {"image_ref": "agentic-opt/task-toy:default", "image_digest": "sha256:default"},
                "metadata": {
                    "provider": "docker_image",
                    "image_ref": "agentic-opt/task-toy:default",
                    "image_digest": "sha256:default",
                    "manifest_path": str(host_root / "manifest.json"),
                    "default_env": {"AO_TEST_DEFAULT": "enabled"},
                },
            }
        )
        popen_calls: list[list[str]] = []

        class FakeProcess:
            pid = 778

            def poll(self):
                return None

            def wait(self, timeout=None):
                return None

        def fake_popen(cmd, **kwargs):
            popen_calls.append([str(item) for item in cmd])
            return FakeProcess()

        with patch("agentic_opt.control_plane.jobs.subprocess.Popen", side_effect=fake_popen):
            response = self.client.post(
                "/api/v1/jobs",
                json={
                    "experiment_id": experiment["experiment_id"],
                    "provider": "docker_image",
                    "environment_id": environment["environment_id"],
                    "command": ["python", "-c", "print(1)"],
                },
            )

        self.assertEqual(response.status_code, 202, response.get_data(as_text=True))
        self.assertIn("AO_TEST_DEFAULT=enabled", response.get_json()["inputs"]["command"])

        export = self.client.post(
            f"/api/v1/environments/{environment['environment_id']}/export-bundle",
            json={"experiment_id": experiment["experiment_id"]},
        )
        self.assertEqual(export.status_code, 201, export.get_data(as_text=True))
        artifact = export.get_json()["artifact"]
        self.assertEqual(artifact["kind"], "environment_reproducibility_bundle")
        archive_path = Path(artifact["local_path"])
        self.assertTrue(archive_path.exists())
        with tarfile.open(archive_path, "r:gz") as archive:
            names = archive.getnames()
        self.assertTrue(any(name.endswith("/environment.json") for name in names))
        self.assertTrue(any(name.endswith("/runtime_manifest.json") for name in names))

    def test_docker_image_sync_evaluation_runs_in_docker_environment(self) -> None:
        ctx = self.client.application.config["AO_CONTEXT"]
        experiment = ctx.control.create_experiment({"task_id": "toy_eval", "mode": "local"})
        assignment = ctx.control.create_assignment(
            {
                "experiment_id": experiment["experiment_id"],
                "task_id": "toy_eval",
                "agent_id": "agent_001",
            }
        )
        environment = ctx.control.upsert_environment(
            {
                "environment_id": "env_docker_toy",
                "environment_type": "task",
                "status": "ready",
                "fingerprint": "dockerfingerprint",
                "python_path": "/usr/local/bin/python",
                "root_path": str(self.state_root / "envs" / "docker"),
                "task_id": "toy_eval",
                "experiment_id": experiment["experiment_id"],
                "spec": {"kind": "docker_image", "provider": "docker_image"},
                "lock": {
                    "format": "docker-image-lock",
                    "provider": "docker_image",
                    "image_ref": "agentic-opt/task-toy:dockerfingerprint",
                    "image_digest": "sha256:locked",
                    "requirements": [],
                },
                "metadata": {
                    "provider": "docker_image",
                    "container_root": "/opt/agentic-opt",
                    "container_python": "/usr/local/bin/python",
                    "container_src_path": "/opt/agentic-opt/src",
                    "container_tasks_root": "/opt/agentic-opt/tasks",
                    "image_ref": "agentic-opt/task-toy:dockerfingerprint",
                    "image_digest": "sha256:locked",
                },
            }
        )
        entry_path = self.tasks_root / "toy_eval" / "public" / "initial.py"
        docker_commands: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            command = [str(item) for item in cmd]
            if "--evaluation-id" not in command:
                return subprocess.CompletedProcess(cmd, 0, "framework==1.0\n", "")
            docker_commands.append(command)
            evaluation_id = command[command.index("--evaluation-id") + 1]
            ctx.control_service.run_evaluation(evaluation_id)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with patch("agentic_opt.control_plane.evaluation.subprocess.run", side_effect=fake_run):
            response = self.client.post(
                "/api/v1/evaluations",
                json={
                    "experiment_id": experiment["experiment_id"],
                    "assignment_id": assignment["assignment_id"],
                    "task_id": "toy_eval",
                    "kind": "submit",
                    "entry_path": str(entry_path),
                    "environment_id": environment["environment_id"],
                    "async": False,
                },
            )

        self.assertEqual(response.status_code, 201, response.get_data(as_text=True))
        evaluation = response.get_json()
        self.assertEqual(evaluation["status"], "completed")
        self.assertEqual(evaluation["request"]["environment_provider"], "docker_image")
        self.assertEqual(evaluation["request"]["environment_lock"]["image_digest"], "sha256:locked")
        self.assertTrue(docker_commands)
        self.assertIn("sha256:locked", docker_commands[0])
        self.assertEqual(evaluation["request"]["runner"]["image_reference"], "sha256:locked")
        leaderboard = ctx.control.list_leaderboard_entries(experiment_id=experiment["experiment_id"])
        self.assertEqual(leaderboard[0]["metadata"]["environment_provider"], "docker_image")
        self.assertEqual(leaderboard[0]["metadata"]["environment_lock"]["image_digest"], "sha256:locked")

    def test_docker_image_job_resolves_image_from_environment(self) -> None:
        ctx = self.client.application.config["AO_CONTEXT"]
        experiment = ctx.control.create_experiment({"task_id": "toy_eval", "mode": "local"})
        environment = ctx.control.upsert_environment(
            {
                "environment_id": "env_docker_job",
                "environment_type": "task",
                "status": "ready",
                "fingerprint": "dockerjob",
                "python_path": "/usr/local/bin/python",
                "root_path": str(self.state_root / "envs" / "docker-job"),
                "task_id": "toy_eval",
                "experiment_id": experiment["experiment_id"],
                "spec": {"kind": "docker_image", "provider": "docker_image"},
                "lock": {"image_ref": "agentic-opt/task-toy:job", "image_digest": "sha256:job"},
                "metadata": {"provider": "docker_image", "image_ref": "agentic-opt/task-toy:job", "image_digest": "sha256:job"},
            }
        )
        popen_calls: list[list[str]] = []

        class FakeProcess:
            pid = 777

            def poll(self):
                return None

            def wait(self, timeout=None):
                return None

        def fake_popen(cmd, **kwargs):
            popen_calls.append([str(item) for item in cmd])
            return FakeProcess()

        with patch("agentic_opt.control_plane.jobs.subprocess.Popen", side_effect=fake_popen):
            response = self.client.post(
                "/api/v1/jobs",
                json={
                    "experiment_id": experiment["experiment_id"],
                    "provider": "docker_image",
                    "environment_id": environment["environment_id"],
                    "command": ["python", "-c", "print(1)"],
                },
            )

        self.assertEqual(response.status_code, 202, response.get_data(as_text=True))
        job = response.get_json()
        self.assertEqual(job["status"], "running")
        self.assertEqual(job["inputs"]["image"], "sha256:job")
        self.assertIn("sha256:job", job["inputs"]["command"])
        self.assertEqual(job["details"]["environment"]["image_digest"], "sha256:job")
        self.assertEqual(job["details"]["runner"]["image"]["reference"], "sha256:job")

    def test_docker_job_rejects_writable_task_context_mount(self) -> None:
        ctx = self.client.application.config["AO_CONTEXT"]
        experiment = ctx.control.create_experiment({"task_id": "toy_eval", "mode": "local"})

        response = self.client.post(
            "/api/v1/jobs",
            json={
                "experiment_id": experiment["experiment_id"],
                "task_id": "toy_eval",
                "provider": "local-docker",
                "image": "python:3.11",
                "command": ["python", "-c", "print(1)"],
                "inputs": {
                    "mounts": [
                        {
                            "source": str(self.root),
                            "target": "/workspace/task",
                            "read_only": False,
                        }
                    ]
                },
            },
        )

        self.assertEqual(response.status_code, 201, response.get_data(as_text=True))
        job = response.get_json()
        self.assertEqual(job["status"], "blocked")
        self.assertEqual(job["details"]["policy_block"]["reason"], "TaskContextMountConflictError")

    def test_local_job_task_context_digest_guard_marks_mutation_failed(self) -> None:
        created = self.client.post(
            "/api/v1/experiments",
            json={"task_id": "toy_eval", "mode": "local", "budget": {"total_evaluator_runs": 2}, "assignment_count": 1},
        )
        self.assertEqual(created.status_code, 201, created.get_data(as_text=True))
        experiment_id = created.get_json()["experiment"]["experiment_id"]
        assignment_id = created.get_json()["assignments"][0]["assignment_id"]
        session = self.client.post(f"/api/v1/assignments/{assignment_id}/sessions", json={})
        self.assertEqual(session.status_code, 201, session.get_data(as_text=True))
        session_id = session.get_json()["session_id"]
        workspace_root = self.state_root / "workspaces" / assignment_id / session_id
        workspace_root.mkdir(parents=True, exist_ok=True)
        snapshot = ensure_task_context_snapshot(task_id="toy_eval", state_root=self.state_root)
        materialize_task_context_snapshot(snapshot=snapshot, workspace_root=workspace_root)
        ctx = self.client.application.config["AO_CONTEXT"]
        ctx.control.update_session(session_id, {"workspace_path": str(workspace_root)})

        response = self.client.post(
            "/api/v1/jobs",
            json={
                "experiment_id": experiment_id,
                "assignment_id": assignment_id,
                "session_id": session_id,
                "task_id": "toy_eval",
                "provider": "local",
                "workspace_root": str(workspace_root),
                "cwd": str(workspace_root),
                "inputs": {
                    "command": [
                        sys.executable,
                        "-c",
                        "from pathlib import Path; p=Path('task/TASK.md'); p.chmod(0o644); p.write_text('mutated')",
                    ]
                },
            },
        )
        self.assertEqual(response.status_code, 202, response.get_data(as_text=True))
        job = self._wait_for(f"/api/v1/jobs/{response.get_json()['job_id']}")
        self.assertEqual(job["status"], "failed")
        self.assertFalse(job["details"]["task_context_postcheck"]["ok"])
        self.assertEqual(job["details"]["task_context_postcheck"]["workspace"]["status"], "mismatch")

    def test_docker_job_audit_policy_starts_outbound_proxy(self) -> None:
        ctx = self.client.application.config["AO_CONTEXT"]
        experiment = ctx.control.create_experiment(
            {
                "task_id": "toy_eval",
                "mode": "local",
                "policy": {"network": {"external_internet": "audit"}},
            }
        )
        popen_calls: list[list[str]] = []

        class FakeProcess:
            pid = 779

            def poll(self):
                return None

            def wait(self, timeout=None):
                return None

        class FakeProxyProcess(FakeProcess):
            pid = 1779

        def fake_popen(cmd, **kwargs):
            popen_calls.append([str(item) for item in cmd])
            return FakeProcess()

        with patch("agentic_opt.control_plane.jobs._free_tcp_port", return_value=50123):
            with patch("agentic_opt.control_plane.jobs.wait_for_tcp", return_value=None):
                with patch("agentic_opt.control_plane.jobs.start_network_proxy_process", return_value=FakeProxyProcess()):
                    with patch("agentic_opt.control_plane.jobs.subprocess.Popen", side_effect=fake_popen):
                        response = self.client.post(
                            "/api/v1/jobs",
                            json={
                                "experiment_id": experiment["experiment_id"],
                                "provider": "local-docker",
                                "image": "python:3.11",
                                "command": "python -c 'print(1)'",
                            },
                        )

        self.assertEqual(response.status_code, 202, response.get_data(as_text=True))
        job = response.get_json()
        command = job["inputs"]["command"]
        self.assertIn("HTTP_PROXY=http://host.docker.internal:50123", command)
        self.assertIn("--add-host", command)
        self.assertEqual(job["details"]["outbound_audit_proxy"]["container_proxy_url"], "http://host.docker.internal:50123")
        self.assertTrue(job["details"]["network_enforcement"]["outbound_audit_proxy_configured"])

    def test_worker_manager_starts_docker_worker_through_relay(self) -> None:
        ctx = self.client.application.config["AO_CONTEXT"]
        experiment = ctx.control.create_experiment(
            {
                "task_id": "toy_eval",
                "mode": "local",
                "policy": {"network": {"external_internet": "deny"}},
                "config": {"worker_image": "agentic-opt-worker:test"},
            }
        )
        assignment = ctx.control.create_assignment(
            {
                "experiment_id": experiment["experiment_id"],
                "task_id": "toy_eval",
                "agent_id": "agent_001",
                "worker_backend": "local-docker",
            }
        )
        manager = WorkerManager(repo_root=Path.cwd(), state_root=self.state_root, control=ctx.control, reaper_interval_s=None)
        popen_calls: list[list[str]] = []

        class FakeProcess:
            pid = 4242

            def poll(self):
                return None

            def wait(self, timeout=None):
                return None

            def terminate(self):
                return None

            def kill(self):
                return None

        class FakeRelayProcess(FakeProcess):
            pid = 2424

        def fake_start_relay_process(*, socket_path, **kwargs):
            Path(socket_path).parent.mkdir(parents=True, exist_ok=True)
            Path(socket_path).write_text("", encoding="utf-8")
            return FakeRelayProcess()

        def fake_popen(cmd, **kwargs):
            popen_calls.append([str(item) for item in cmd])
            return FakeProcess()

        with patch("agentic_opt.web.workers.start_relay_process", side_effect=fake_start_relay_process):
            with patch("agentic_opt.web.workers.subprocess.Popen", side_effect=fake_popen):
                session = manager.start_control_assignment(
                    assignment_id=assignment["assignment_id"],
                    api_url="http://127.0.0.1:5000",
                    dry_run=True,
                    max_turn_wall_time_s=30,
                )
        try:
            self.assertEqual(session["status"], "running")
            self.assertEqual(len(popen_calls), 1)
            command = popen_calls[0]
            self.assertEqual(command[:3], ["docker", "run", "--rm"])
            self.assertIn("agentic-opt-worker:test", command)
            self.assertIn("--network", command)
            self.assertEqual(command[command.index("--network") + 1], "none")
            self.assertIn("AO_CONTROL_API_URL=unix:///ao-control/control.sock", command)
            self.assertEqual(session["details"]["control_plane_relay"]["transport"], "unix-socket")
            self.assertEqual(session["details"]["control_plane_relay"]["worker_relay_url"], "unix:///ao-control/control.sock")
            self.assertEqual(session["details"]["docker_worker"]["image"], "agentic-opt-worker:test")
            self.assertTrue(session["details"]["network_enforcement"]["external_internet_enforced"])
        finally:
            manager.close()

    def test_worker_manager_starts_docker_worker_through_tcp_relay(self) -> None:
        ctx = self.client.application.config["AO_CONTEXT"]
        experiment = ctx.control.create_experiment(
            {
                "task_id": "toy_eval",
                "mode": "local",
                "policy": {"network": {"external_internet": "deny"}},
                "config": {
                    "worker_image": "agentic-opt-worker:test",
                    "docker": {"control_plane_relay_transport": "tcp"},
                },
            }
        )
        assignment = ctx.control.create_assignment(
            {
                "experiment_id": experiment["experiment_id"],
                "task_id": "toy_eval",
                "agent_id": "agent_001",
                "worker_backend": "local-docker",
            }
        )
        manager = WorkerManager(repo_root=Path.cwd(), state_root=self.state_root, control=ctx.control, reaper_interval_s=None)
        popen_calls: list[list[str]] = []

        class FakeProcess:
            pid = 5252

            def poll(self):
                return None

            def wait(self, timeout=None):
                return None

            def terminate(self):
                return None

            def kill(self):
                return None

        class FakeRelayProcess(FakeProcess):
            pid = 2525

        def fake_start_relay_process(*, socket_path=None, transport="unix-socket", tcp_host=None, tcp_port=None, **kwargs):
            self.assertIsNone(socket_path)
            self.assertEqual(transport, "tcp")
            self.assertEqual(tcp_host, "127.0.0.1")
            self.assertEqual(tcp_port, 54321)
            return FakeRelayProcess()

        def fake_popen(cmd, **kwargs):
            popen_calls.append([str(item) for item in cmd])
            return FakeProcess()

        with patch("agentic_opt.web.workers._free_tcp_port", return_value=54321):
            with patch("agentic_opt.web.workers._wait_for_tcp", return_value=None):
                with patch("agentic_opt.web.workers.start_relay_process", side_effect=fake_start_relay_process):
                    with patch("agentic_opt.web.workers.subprocess.Popen", side_effect=fake_popen):
                        session = manager.start_control_assignment(
                            assignment_id=assignment["assignment_id"],
                            api_url="http://127.0.0.1:5000",
                            dry_run=True,
                        )
        try:
            self.assertEqual(session["status"], "running")
            self.assertEqual(len(popen_calls), 1)
            command = popen_calls[0]
            self.assertEqual(command[command.index("--network") + 1], "bridge")
            self.assertIn("AO_CONTROL_API_URL=http://host.docker.internal:54321", command)
            self.assertIn("host.docker.internal:host-gateway", command)
            self.assertEqual(session["details"]["control_plane_relay"]["transport"], "tcp")
            self.assertEqual(session["details"]["control_plane_relay"]["worker_relay_url"], "http://host.docker.internal:54321")
            self.assertTrue(session["details"]["network_enforcement"]["policy_weakened"])
            self.assertEqual(session["details"]["network_enforcement"]["control_plane_relay_transport"], "tcp")
        finally:
            manager.close()

    def test_worker_manager_starts_docker_worker_with_strict_audit_proxy(self) -> None:
        ctx = self.client.application.config["AO_CONTEXT"]
        experiment = ctx.control.create_experiment(
            {
                "task_id": "toy_eval",
                "mode": "local",
                "policy": {"network": {"external_internet": "audit"}},
                "config": {"worker_image": "agentic-opt-worker:test"},
            }
        )
        assignment = ctx.control.create_assignment(
            {
                "experiment_id": experiment["experiment_id"],
                "task_id": "toy_eval",
                "agent_id": "agent_001",
                "worker_backend": "local-docker",
            }
        )
        manager = WorkerManager(repo_root=Path.cwd(), state_root=self.state_root, control=ctx.control, reaper_interval_s=None)
        popen_calls: list[list[str]] = []

        class FakeProcess:
            pid = 6262

            def poll(self):
                return None

            def wait(self, timeout=None):
                return None

            def terminate(self):
                return None

            def kill(self):
                return None

        class FakeRelayProcess(FakeProcess):
            pid = 2626

        class FakeProxyProcess(FakeProcess):
            pid = 3636

        def fake_start_relay_process(*, socket_path, **kwargs):
            Path(socket_path).parent.mkdir(parents=True, exist_ok=True)
            Path(socket_path).write_text("", encoding="utf-8")
            return FakeRelayProcess()

        def fake_start_network_proxy_process(*, socket_path, **kwargs):
            Path(socket_path).parent.mkdir(parents=True, exist_ok=True)
            Path(socket_path).write_text("", encoding="utf-8")
            return FakeProxyProcess()

        def fake_popen(cmd, **kwargs):
            popen_calls.append([str(item) for item in cmd])
            return FakeProcess()

        with patch("agentic_opt.web.workers.start_relay_process", side_effect=fake_start_relay_process):
            with patch("agentic_opt.web.workers.start_network_proxy_process", side_effect=fake_start_network_proxy_process):
                with patch("agentic_opt.web.workers.wait_for_unix_socket", return_value=None):
                    with patch("agentic_opt.web.workers.subprocess.Popen", side_effect=fake_popen):
                        session = manager.start_control_assignment(
                            assignment_id=assignment["assignment_id"],
                            api_url="http://127.0.0.1:5000",
                            dry_run=True,
                        )
        try:
            command = popen_calls[0]
            self.assertEqual(command[command.index("--network") + 1], "none")
            self.assertIn("AO_CONTROL_API_URL=unix:///ao-control/control.sock", command)
            self.assertIn("AO_OUTBOUND_PROXY_SOCKET=/ao-network/proxy.sock", command)
            self.assertIn("HTTP_PROXY=http://127.0.0.1:8765", command)
            self.assertEqual(session["details"]["control_plane_relay"]["transport"], "unix-socket")
            self.assertEqual(session["details"]["outbound_audit_proxy"]["transport"], "unix-socket-http-proxy")
            self.assertEqual(session["details"]["network_enforcement"]["outbound_audit_mode"], "unix_proxy_bridge")
            self.assertTrue(session["details"]["network_enforcement"]["external_internet_enforced"])
            self.assertFalse(session["details"]["network_enforcement"]["policy_weakened"])
        finally:
            manager.close()

    def test_unix_socket_control_plane_relay(self) -> None:
        target = ThreadingHTTPServer(("127.0.0.1", 0), _RelayTargetHandler)
        target_thread = threading.Thread(target=target.serve_forever, daemon=True)
        target_thread.start()
        relay = ControlPlaneRelayServer(
            self.root / "relay.sock",
            f"http://127.0.0.1:{target.server_address[1]}",
            max_body_bytes=64,
            audit_log_path=self.root / "relay_audit.jsonl",
        )
        relay_thread = threading.Thread(target=relay.serve_forever, daemon=True)
        relay_thread.start()
        try:
            client = ControlPlaneClient(relay_url(self.root / "relay.sock"))
            self.assertEqual(client.get("/api/v1/echo", {"x": "1"})["path"], "/api/v1/echo?x=1")
            posted = client.post("/api/v1/echo", {"hello": "world"})
            self.assertEqual(posted["json"]["hello"], "world")
            with self.assertRaises(ControlPlaneClientError):
                client.get("/not-control-plane")
            with self.assertRaises(ControlPlaneClientError):
                client._request_unix("DELETE", "/api/v1/echo")
            self.assertEqual(
                _validate_relay_path("http://example.test/api/v1/echo"),
                "relay only accepts origin-form control-plane paths",
            )
            with self.assertRaises(ControlPlaneClientError):
                client.post("/api/v1/echo", {"payload": "x" * 128})
        finally:
            relay.shutdown()
            relay.server_close()
            target.shutdown()
            target.server_close()
        audit_lines = (self.root / "relay_audit.jsonl").read_text(encoding="utf-8").strip().splitlines()
        self.assertTrue(any(json.loads(line)["decision"] == "forwarded" for line in audit_lines))
        self.assertTrue(any(json.loads(line)["decision"] == "denied" for line in audit_lines))

    def test_tcp_control_plane_relay(self) -> None:
        target = ThreadingHTTPServer(("127.0.0.1", 0), _RelayTargetHandler)
        target_thread = threading.Thread(target=target.serve_forever, daemon=True)
        target_thread.start()
        relay = ControlPlaneTCPRelayServer(
            ("127.0.0.1", 0),
            f"http://127.0.0.1:{target.server_address[1]}",
            max_body_bytes=64,
            audit_log_path=self.root / "tcp_relay_audit.jsonl",
        )
        relay_thread = threading.Thread(target=relay.serve_forever, daemon=True)
        relay_thread.start()
        try:
            client = ControlPlaneClient(tcp_relay_url("127.0.0.1", int(relay.server_address[1])))
            self.assertEqual(client.get("/api/v1/echo", {"x": "1"})["path"], "/api/v1/echo?x=1")
            posted = client.post("/api/v1/echo", {"hello": "world"})
            self.assertEqual(posted["json"]["hello"], "world")
            with self.assertRaises(ControlPlaneClientError):
                client.get("/not-control-plane")
        finally:
            relay.shutdown()
            relay.server_close()
            target.shutdown()
            target.server_close()
        audit_lines = (self.root / "tcp_relay_audit.jsonl").read_text(encoding="utf-8").strip().splitlines()
        self.assertTrue(any(json.loads(line)["decision"] == "forwarded" for line in audit_lines))
        self.assertTrue(any(json.loads(line)["decision"] == "denied" for line in audit_lines))

    def test_workspace_control_broker_starts_unix_socket_relay(self) -> None:
        target = ThreadingHTTPServer(("127.0.0.1", 0), _RelayTargetHandler)
        target_thread = threading.Thread(target=target.serve_forever, daemon=True)
        target_thread.start()
        broker_process = None
        try:
            broker_url, broker_process, metadata = _start_workspace_control_broker(
                workspace_root=self.root / "worker_workspace",
                target_url=f"http://127.0.0.1:{target.server_address[1]}",
            )
            self.assertTrue(metadata["enabled"])
            self.assertEqual(metadata["transport"], "unix-socket")
            self.assertTrue(broker_url.startswith("unix://"))
            client = ControlPlaneClient(broker_url)
            self.assertEqual(client.get("/api/v1/echo", {"x": "1"})["path"], "/api/v1/echo?x=1")
            self.assertTrue((self.root / "worker_workspace" / ".control" / "control_broker_audit.jsonl").exists())
        finally:
            _terminate_process(broker_process)
            target.shutdown()
            target.server_close()

    def test_local_jsonl_trace_export_writes_redacted_mirror(self) -> None:
        fixture = self._registered_trace_fixture()
        trace_id = fixture["trace"]["trace_id"]

        exported = self.client.post(
            "/api/v1/trace-exports",
            json={"provider": "local-jsonl", "trace_ids": [trace_id]},
        )
        self.assertEqual(exported.status_code, 201, exported.get_data(as_text=True))
        export_payload = exported.get_json()
        self.assertEqual(export_payload["status"], "completed")
        self.assertEqual(export_payload["provider"], "local-jsonl")
        self.assertEqual(export_payload["source_trace_ids"], [trace_id])
        self.assertTrue(export_payload["digest"].startswith("sha256:"))

        export_dir = Path(export_payload["local_path"])
        self.assertTrue(export_dir.exists())
        manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], "agentic_opt.trace_export.local_jsonl.v1")
        self.assertEqual(manifest["record_counts"]["traces"], 1)
        self.assertGreater(manifest["record_counts"]["commands"], 0)
        self.assertGreater(manifest["redaction_summary"]["counts"]["secret_assignment"], 0)
        self.assertGreater(manifest["redaction_summary"]["counts"]["private_path"], 0)
        self.assertGreater(manifest["redaction_summary"]["counts"]["denied_destination"], 0)
        self.assertGreater(manifest["redaction_summary"]["counts"]["sensitive_header"], 0)
        self.assertGreater(manifest["redaction_summary"]["counts"]["sensitive_url_query"], 0)
        self.assertGreater(manifest["redaction_summary"]["counts"]["hidden_grader"], 0)
        self.assertGreater(manifest["redaction_summary"]["counts"]["sensitive_output_truncated"], 0)

        commands_text = (export_dir / "commands.jsonl").read_text(encoding="utf-8")
        raw_events_text = (export_dir / "raw_events.jsonl").read_text(encoding="utf-8")
        self.assertIn("[REDACTED]", commands_text)
        self.assertNotIn("supersecret", commands_text)
        self.assertNotIn("/private/", commands_text)
        self.assertIn("[REDACTED_DESTINATION]", raw_events_text)
        self.assertIn("[REDACTED_HEADER]", raw_events_text)
        self.assertIn("[REDACTED_HIDDEN_GRADER]", raw_events_text)
        self.assertIn("[TRUNCATED_OUTPUT", raw_events_text)
        self.assertNotIn("private-answer.example", raw_events_text)
        self.assertNotIn("header-secret", raw_events_text)
        self.assertNotIn("secret-token", raw_events_text)
        self.assertNotIn("hidden_grader/private.json", raw_events_text)

        artifact = self.client.get(f"/api/v1/artifacts/{export_payload['artifact_id']}")
        self.assertEqual(artifact.status_code, 200, artifact.get_data(as_text=True))
        self.assertEqual(artifact.get_json()["kind"], "trace_export")

        listed = self.client.get("/api/v1/trace-exports", query_string={"experiment_id": fixture["experiment_id"]})
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.get_json()["trace_exports"][0]["trace_export_id"], export_payload["trace_export_id"])

        fetched = self.client.get(f"/api/v1/trace-exports/{export_payload['trace_export_id']}")
        self.assertEqual(fetched.status_code, 200)
        self.assertEqual(fetched.get_json()["digest"], export_payload["digest"])

        exported_again = self.client.post(
            "/api/v1/trace-exports",
            json={"provider": "local-jsonl", "trace_ids": [trace_id]},
        )
        self.assertEqual(exported_again.status_code, 201, exported_again.get_data(as_text=True))
        self.assertEqual(exported_again.get_json()["digest"], export_payload["digest"])

    def test_otlp_trace_export_posts_redacted_payload(self) -> None:
        fixture = self._registered_trace_fixture()
        trace_id = fixture["trace"]["trace_id"]
        _OtlpCollectorHandler.requests = []
        _OtlpCollectorHandler.response_status = 200
        _OtlpCollectorHandler.response_body = {"accepted": True}
        collector = ThreadingHTTPServer(("127.0.0.1", 0), _OtlpCollectorHandler)
        thread = threading.Thread(target=collector.serve_forever, daemon=True)
        thread.start()
        try:
            endpoint = f"http://127.0.0.1:{collector.server_address[1]}/v1/traces"
            exported = self.client.post(
                "/api/v1/trace-exports",
                json={
                    "provider": "otlp",
                    "trace_ids": [trace_id],
                    "endpoint": endpoint,
                    "headers": {
                        "Authorization": "Bearer collector-secret",
                        "x-otlp-test": "yes",
                    },
                },
            )
        finally:
            collector.shutdown()
            collector.server_close()

        self.assertEqual(exported.status_code, 201, exported.get_data(as_text=True))
        export_payload = exported.get_json()
        self.assertEqual(export_payload["status"], "completed")
        self.assertEqual(export_payload["provider"], "otlp")
        self.assertEqual(export_payload["destination_uri"], endpoint)
        self.assertTrue(export_payload["digest"].startswith("sha256:"))
        self.assertNotIn("collector-secret", json.dumps(export_payload, sort_keys=True))

        self.assertEqual(len(_OtlpCollectorHandler.requests), 1)
        captured = _OtlpCollectorHandler.requests[0]
        request_headers = {key.lower(): value for key, value in captured["headers"].items()}
        self.assertEqual(captured["path"], "/v1/traces")
        self.assertEqual(request_headers.get("authorization"), "Bearer collector-secret")
        self.assertEqual(request_headers.get("x-otlp-test"), "yes")

        otlp_payload = captured["json"]
        otlp_text = json.dumps(otlp_payload, sort_keys=True)
        self.assertIn("resourceSpans", otlp_payload)
        self.assertIn("[REDACTED]", otlp_text)
        self.assertNotIn("supersecret", otlp_text)
        self.assertNotIn("/private/", otlp_text)
        self.assertNotIn("private-answer.example", otlp_text)
        self.assertNotIn("header-secret", otlp_text)
        self.assertNotIn("hidden_grader/private.json", otlp_text)
        spans = [
            span
            for resource_spans in otlp_payload["resourceSpans"]
            for scope_spans in resource_spans["scopeSpans"]
            for span in scope_spans["spans"]
        ]
        self.assertGreaterEqual(len(spans), 3)
        self.assertIn("command", {span["name"] for span in spans})
        self.assertIn("agent message", {span["name"] for span in spans})

        export_dir = Path(export_payload["local_path"])
        manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], "agentic_opt.trace_export.otlp.v1")
        self.assertEqual(manifest["destination_uri"], endpoint)
        self.assertEqual(manifest["record_counts"]["traces"], 1)
        self.assertGreaterEqual(manifest["record_counts"]["otlp_spans"], 3)
        self.assertEqual(manifest["http_result"]["http_status"], 200)
        self.assertEqual(manifest["provider_config"]["headers"]["Authorization"], "[REDACTED]")
        self.assertEqual(manifest["provider_config"]["headers"]["x-otlp-test"], "yes")
        self.assertNotIn("collector-secret", json.dumps(manifest, sort_keys=True))

    def test_otlp_trace_export_records_http_failure(self) -> None:
        fixture = self._registered_trace_fixture()
        _OtlpCollectorHandler.requests = []
        _OtlpCollectorHandler.response_status = 500
        _OtlpCollectorHandler.response_body = {"error": "collector rejected payload"}
        collector = ThreadingHTTPServer(("127.0.0.1", 0), _OtlpCollectorHandler)
        thread = threading.Thread(target=collector.serve_forever, daemon=True)
        thread.start()
        try:
            endpoint = f"http://127.0.0.1:{collector.server_address[1]}/v1/traces"
            exported = self.client.post(
                "/api/v1/trace-exports",
                json={"provider": "otlp", "trace_ids": [fixture["trace"]["trace_id"]], "endpoint": endpoint},
            )
        finally:
            collector.shutdown()
            collector.server_close()

        self.assertEqual(exported.status_code, 201, exported.get_data(as_text=True))
        payload = exported.get_json()
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["provider"], "otlp")
        self.assertIn("otlp export failed: HTTP 500", payload["error"]["message"])
        self.assertEqual(len(_OtlpCollectorHandler.requests), 1)

    def test_trace_export_records_provider_failure(self) -> None:
        fixture = self._registered_trace_fixture()

        exported = self.client.post(
            "/api/v1/trace-exports",
            json={"provider": "phoenix", "trace_ids": [fixture["trace"]["trace_id"]]},
        )

        self.assertEqual(exported.status_code, 201, exported.get_data(as_text=True))
        payload = exported.get_json()
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["provider"], "phoenix")
        self.assertIn("unsupported trace export provider", payload["error"]["message"])

    def _completed_budgeted_assignment(
        self,
        *,
        total_budget: int,
        assignment_budget: int,
        used_evals: int,
        metadata: dict | None = None,
        session_status: str = "completed",
        stop_reason: str = "turn_completed",
    ) -> tuple[object, dict, dict, dict]:
        ctx = self.client.application.config["AO_CONTEXT"]
        experiment = ctx.control.create_experiment(
            {
                "task_id": "toy_eval",
                "mode": "local",
                "budget": {"total_evaluator_runs": total_budget},
            }
        )
        assignment = ctx.control.create_assignment(
            {
                "experiment_id": experiment["experiment_id"],
                "task_id": "toy_eval",
                "agent_id": "agent_001",
                "budget": {"evaluator_runs": assignment_budget},
                "metadata": metadata or {},
            }
        )
        session = ctx.control.create_session(
            {
                "assignment_id": assignment["assignment_id"],
                "status": "running",
                "details": {"api_url": "http://127.0.0.1:5010"},
            }
        )
        for index in range(used_evals):
            evaluation = ctx.control.create_evaluation(
                {
                    "evaluation_id": f"eval_budget_test_{index}",
                    "experiment_id": experiment["experiment_id"],
                    "assignment_id": assignment["assignment_id"],
                    "task_id": "toy_eval",
                    "kind": "submit",
                    "status": "completed",
                    "valid": True,
                    "score": float(index),
                }
            )
            ctx.control.create_leaderboard_entry(
                {
                    "experiment_id": experiment["experiment_id"],
                    "task_id": "toy_eval",
                    "assignment_id": assignment["assignment_id"],
                    "evaluation_id": evaluation["evaluation_id"],
                    "score": float(index),
                }
            )
        session = ctx.control.update_session(
            session["session_id"],
            {"status": session_status, "details": {"stop_reason": stop_reason}},
        )
        return ctx, experiment, assignment, session

    def _registered_trace_fixture(self) -> dict:
        created = self.client.post(
            "/api/v1/experiments",
            json={"task_id": "toy_eval", "mode": "local", "assignment_count": 1},
        )
        self.assertEqual(created.status_code, 201, created.get_data(as_text=True))
        experiment_id = created.get_json()["experiment"]["experiment_id"]
        assignment = created.get_json()["assignments"][0]
        assignment_id = assignment["assignment_id"]
        session = self.client.post(f"/api/v1/assignments/{assignment_id}/sessions", json={})
        self.assertEqual(session.status_code, 201, session.get_data(as_text=True))
        session_id = session.get_json()["session_id"]
        trace_dir = self.root / "trace_export_source" / session_id / "turn_trace_export"
        trace_dir.mkdir(parents=True)
        events = [
            {
                "method": "item/started",
                "params": {
                    "startedAtMs": 100,
                    "item": {
                        "id": "cmd-secret",
                        "command": "OPENAI_API_KEY=supersecret python tasks/toy_eval/private/secret.py",
                        "cwd": str(self.root / "tasks" / "toy_eval" / "private"),
                        "source": "exec",
                    },
                },
            },
            {
                "method": "item/completed",
                "params": {
                    "completedAtMs": 200,
                    "item": {
                        "id": "cmd-secret",
                        "command": "OPENAI_API_KEY=supersecret python tasks/toy_eval/private/secret.py",
                        "cwd": str(self.root / "tasks" / "toy_eval" / "private"),
                        "source": "exec",
                        "durationMs": 100,
                        "exitCode": 0,
                        "status": "completed",
                        "request": {
                            "headers": {
                                "Authorization": "Bearer header-secret",
                                "X-Api-Key": "header-secret",
                                "Content-Type": "application/json",
                            },
                            "url": "https://api.example.test/run?token=secret-token&public=1",
                        },
                        "hidden_grader_path": "/tmp/tasks/toy_eval/hidden_grader/private.json",
                        "aggregatedOutput": "wrote /tmp/tasks/toy_eval/private/hidden.txt with OPENAI_API_KEY=supersecret\n"
                        + ("x" * 13_000),
                    },
                },
            },
            {
                "method": "item/agentMessage/delta",
                "params": {"delta": "Export this trace."},
            },
            {
                "method": "network/access",
                "params": {
                    "decision": "denied",
                    "destination": "https://private-answer.example/solution",
                },
            },
        ]
        (trace_dir / "events.jsonl").write_text(
            "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
            encoding="utf-8",
        )
        registered = self.client.post(
            "/api/v1/agent-traces",
            json={
                "experiment_id": experiment_id,
                "assignment_id": assignment_id,
                "session_id": session_id,
                "task_id": "toy_eval",
                "agent_id": assignment["agent_id"],
                "run_id": "run_trace_export",
                "turn_id": "turn_trace_export",
                "trace_dir": str(trace_dir),
                "outcome": "completed",
                "status": "completed",
            },
        )
        self.assertEqual(registered.status_code, 201, registered.get_data(as_text=True))
        return {
            "experiment_id": experiment_id,
            "assignment": assignment,
            "session_id": session_id,
            "trace": registered.get_json(),
        }

    def _wait_for(self, path: str, *, timeout_s: float = 10.0) -> dict:
        deadline = time.time() + timeout_s
        last = None
        while time.time() < deadline:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
            last = response.get_json()
            if last["status"] in {"completed", "failed", "cancelled"}:
                return last
            time.sleep(0.1)
        self.fail(f"timed out waiting for {path}; last={last}")

    def _write_toy_task(self) -> None:
        task_dir = self.tasks_root / "toy_eval"
        public = task_dir / "public"
        public.mkdir(parents=True, exist_ok=True)
        (public / "TASK.md").write_text("# Toy Eval\n\nReturn 42.\n", encoding="utf-8")
        (public / "public_contract.md").write_text("Expose `solve()` in `initial.py`.\n", encoding="utf-8")
        (public / "initial.py").write_text("def solve():\n    return 42\n", encoding="utf-8")
        knowledge = public / "knowledge"
        knowledge.mkdir(parents=True, exist_ok=True)
        (knowledge / "note.md").write_text("Use the toy note.\n", encoding="utf-8")
        (knowledge / "manifest.json").write_text(
            '''
{
  "items": [
    {
      "knowledge_id": "toy_note",
      "title": "Toy note",
      "kind": "note",
      "path": "note.md",
      "media_type": "text/markdown",
      "summary": "Small task-packaged note",
      "tags": ["toy"]
    }
  ]
}
'''.lstrip(),
            encoding="utf-8",
        )
        (task_dir / "task.py").write_text(
            '''
from pathlib import Path

from agentic_opt.common.runtime_env import TaskRuntimeSpec
from agentic_opt.task_api import TaskMetadata


class ToyTask:
    metadata = TaskMetadata(task_id="toy_eval", title="Toy Eval")
    runtime_spec = TaskRuntimeSpec(verify_public_seed=False)

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def public_dir(self) -> Path:
        return self._root / "public"

    def verify_entry(self, entry_path: Path) -> dict:
        text = entry_path.read_text(encoding="utf-8")
        valid = "def solve" in text
        return {"valid": valid, "feedback": {} if valid else {"error": "missing solve"}}

    def probe_entry(self, entry_path: Path, *, kind: str) -> dict:
        return {"valid": True, "kind": kind, "score": 0.5}

    def evaluate_entry(self, entry_path: Path) -> dict:
        text = entry_path.read_text(encoding="utf-8")
        score = 1.0 if "return 42" in text else 0.0
        return {
            "score": score,
            "correct": {"correct": score == 1.0},
            "evaluator": {"public_details": {"matched": score == 1.0}},
        }


def create_task() -> ToyTask:
    return ToyTask(Path(__file__).parent)
'''.lstrip(),
            encoding="utf-8",
        )

class _RelayTargetHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self._send({"method": "GET", "path": self.path})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        self._send({"method": "POST", "path": self.path, "json": json.loads(raw.decode("utf-8"))})

    def do_PATCH(self) -> None:
        self.do_POST()

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send(self, payload: dict) -> None:
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class _OtlpCollectorHandler(BaseHTTPRequestHandler):
    requests: list[dict] = []
    response_status = 200
    response_body: dict = {"accepted": True}

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        type(self).requests.append(
            {
                "path": self.path,
                "headers": dict(self.headers),
                "json": json.loads(raw.decode("utf-8")),
            }
        )
        body = json.dumps(type(self).response_body, sort_keys=True).encode("utf-8")
        self.send_response(type(self).response_status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


if __name__ == "__main__":
    unittest.main()
