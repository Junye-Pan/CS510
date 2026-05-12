from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agentic_opt.control_plane.client import ControlPlaneClient, ControlPlaneClientError
from agentic_opt.control_plane.jobs import DockerNetworkPolicyError, JobService, build_local_docker_command
from agentic_opt.control_plane.relay import ControlPlaneRelayServer, relay_url
from agentic_opt.web.app import create_app
from agentic_opt.web.workers import WorkerManager, WorkerProcess


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
        self.client = app.test_client()

    def tearDown(self) -> None:
        if self._old_tasks_roots is None:
            os.environ.pop("AO_TASKS_ROOTS", None)
        else:
            os.environ["AO_TASKS_ROOTS"] = self._old_tasks_roots
        self.tempdir.cleanup()

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
        self.assertIn("Environment", model.get_json()["resources"])
        self.assertIn("EnvironmentOverlay", model.get_json()["resources"])
        self.assertIn("LeaderboardEntry", model.get_json()["resources"])

        created = self.client.post(
            "/api/v1/experiments",
            json={
                "task_id": "toy_eval",
                "mode": "local",
                "budget": {"total_evaluator_runs": 4},
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
        self.assertEqual(task.get_json()["knowledge_items"][0]["local_id"], "toy_note")

        context = self.client.get("/api/v1/context", query_string={"assignment_id": assignment_id})
        self.assertEqual(context.status_code, 200)
        self.assertEqual(context.get_json()["assignment"]["assignment_id"], assignment_id)
        self.assertTrue(context.get_json()["environments"])
        self.assertEqual(context.get_json()["network_policy"]["policy"]["external_internet"], "deny")
        self.assertTrue(context.get_json()["network_policy"]["enforcement"]["policy_weakened"])
        self.assertTrue(context.get_json()["knowledge_items"])
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

        finding = self.client.post(
            "/api/v1/findings",
            json={
                "experiment_id": experiment_id,
                "assignment_id": assignment_id,
                "task_id": "toy_eval",
                "finding_type": "pattern",
                "title": "Reusable pattern",
                "body": "Finding and pattern are one durable knowledge resource.",
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
                "kind": "text",
                "path": str(artifact_source),
            },
        )
        self.assertEqual(artifact.status_code, 201)
        artifact_payload = artifact.get_json()
        self.assertTrue(artifact_payload["digest"].startswith("sha256:"))
        manifest_path = Path(artifact_payload["metadata"]["manifest_path"])
        self.assertTrue(manifest_path.exists())

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

        knowledge = self.client.get("/api/v1/knowledge", query_string={"task_id": "toy_eval", "query": "note"})
        self.assertEqual(knowledge.status_code, 200)
        knowledge_id = knowledge.get_json()["knowledge_items"][0]["knowledge_id"]
        materialized = self.client.post(
            f"/api/v1/knowledge/{knowledge_id}/materialize",
            json={"destination_path": str(self.root / "knowledge_note.md")},
        )
        self.assertEqual(materialized.status_code, 200, materialized.get_data(as_text=True))
        self.assertEqual((self.root / "knowledge_note.md").read_text(encoding="utf-8"), "Use the toy note.\n")

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

        entry_path = self.tasks_root / "toy_eval" / "public" / "initial.py"
        candidate_artifact = self.client.post(
            "/api/v1/artifacts",
            json={
                "experiment_id": experiment_id,
                "assignment_id": assignment_id,
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
                "task_id": "toy_eval",
                "kind": "verify",
                "artifact_id": candidate_artifact.get_json()["artifact_id"],
                "async": False,
            },
        )
        self.assertEqual(artifact_eval.status_code, 201)
        self.assertEqual(artifact_eval.get_json()["request"]["input_kind"], "artifact")
        self.assertTrue(artifact_eval.get_json()["valid"])

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
        job_id = job.get_json()["job_id"]
        completed_job = self._wait_for(f"/api/v1/jobs/{job_id}")
        self.assertEqual(completed_job["status"], "completed")
        logs = self.client.get(f"/api/v1/jobs/{job_id}/logs")
        self.assertEqual(logs.status_code, 200)
        self.assertIn("ok", logs.get_json()["stdout"])

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

    def test_unix_socket_control_plane_relay(self) -> None:
        target = ThreadingHTTPServer(("127.0.0.1", 0), _RelayTargetHandler)
        target_thread = threading.Thread(target=target.serve_forever, daemon=True)
        target_thread.start()
        relay = ControlPlaneRelayServer(
            self.root / "relay.sock",
            f"http://127.0.0.1:{target.server_address[1]}",
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
        finally:
            relay.shutdown()
            relay.server_close()
            target.shutdown()
            target.server_close()

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


if __name__ == "__main__":
    unittest.main()
