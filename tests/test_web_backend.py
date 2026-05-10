from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

from agentic_opt.web.app import create_app


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

        context = self.client.get("/api/v1/context", query_string={"assignment_id": assignment_id})
        self.assertEqual(context.status_code, 200)
        self.assertEqual(context.get_json()["assignment"]["assignment_id"], assignment_id)
        self.assertTrue(context.get_json()["environments"])

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
        self.assertEqual(async_eval.status_code, 202)
        async_eval_id = async_eval.get_json()["evaluation_id"]
        completed_eval = self._wait_for(f"/api/v1/evaluations/{async_eval_id}")
        self.assertEqual(completed_eval["status"], "completed")
        self.assertEqual(completed_eval["score"], 1.0)

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


if __name__ == "__main__":
    unittest.main()
