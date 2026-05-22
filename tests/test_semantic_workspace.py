from __future__ import annotations

import os
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from agentic_opt.adapter.app_server_client import AppServerClient, AppServerClientError
from agentic_opt.adapter.semantic_worker import private_codex_home_for_workspace
from agentic_opt.adapter.semantic_workspace import (
    SEMANTIC_SKILL_BODIES,
    _prepend_path,
    build_semantic_startup_prompt,
    prepare_semantic_workspace,
)
from agentic_opt.common.runtime_env import PreparedRuntimeEnv, TaskRuntimeSpec
from agentic_opt.control_plane.task_context import ensure_task_context_snapshot
from agentic_opt.worker_tools.semantic_cli import _evaluation_location


class SemanticWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.tasks_root = self.root / "task_bundle"
        self._old_tasks_roots = os.environ.get("AO_TASKS_ROOTS")
        os.environ["AO_TASKS_ROOTS"] = str(self.tasks_root)
        self._write_toy_task()

    def tearDown(self) -> None:
        if self._old_tasks_roots is None:
            os.environ.pop("AO_TASKS_ROOTS", None)
        else:
            os.environ["AO_TASKS_ROOTS"] = self._old_tasks_roots
        self.tempdir.cleanup()

    def test_semantic_workspace_has_only_server_tools(self) -> None:
        runtime_root = self.root / "runtime"
        runtime_root.mkdir(parents=True, exist_ok=True)
        runtime_env = PreparedRuntimeEnv(
            task_id="toy_semantic",
            fingerprint="test",
            root=runtime_root,
            venv_dir=runtime_root,
            python_path=Path(sys.executable),
            manifest_path=runtime_root / "manifest.json",
            spec=TaskRuntimeSpec(verify_public_seed=False),
        )
        assignment = {
            "assignment_id": "assign_test",
            "experiment_id": "exp_test",
            "task_id": "toy_semantic",
            "agent_id": "agent_001",
            "budget": {"evaluator_runs": 1},
        }
        workspace = prepare_semantic_workspace(
            workspace_root=self.root / "workspace",
            api_url="http://127.0.0.1:5000",
            assignment=assignment,
            session_id="session_test",
            runtime_env=runtime_env,
        )

        commands = ("ctx", "attempt", "artifact", "eval", "finding", "notebook", "job", "env", "telemetry", "tool", "network", "trace")
        for command in commands:
            self.assertTrue((workspace.bin_dir / command).exists())
        self.assertTrue((workspace.root / ".agents" / "shell" / ".zprofile").exists())
        self.assertEqual(workspace.env["ZDOTDIR"], str(workspace.root / ".agents" / "shell"))
        self.assertEqual(workspace.env["BASH_ENV"], str(workspace.root / ".agents" / "shell" / "bash_env"))
        self.assertFalse((workspace.bin_dir / "knowledge").exists())
        self.assertFalse((workspace.bin_dir / "fs").exists())
        self.assertFalse((workspace.bin_dir / "ve").exists())
        self.assertEqual(set(SEMANTIC_SKILL_BODIES), {item.name for item in workspace.skills_root.iterdir()})
        self.assertEqual(workspace.env["AO_ENVIRONMENT_ID"], "env_task_toy_semantic_test")

        agents_md = workspace.agents_md_path.read_text(encoding="utf-8")
        self.assertIn("workspace directory tree", agents_md)
        self.assertIn("eval submit", agents_md)
        self.assertIn("context/", agents_md)
        self.assertIn("history/", agents_md)
        self.assertNotIn("fs/ve", agents_md)

        prompt = build_semantic_startup_prompt(assignment=assignment, workspace=workspace)
        self.assertIn("context/current_state.json", prompt)
        self.assertIn("history/findings", prompt)
        self.assertIn("attempt create", prompt)
        self.assertIn("env status", prompt)
        self.assertIn("job create --provider local", prompt)
        self.assertIn("telemetry start", prompt)
        self.assertIn("task/knowledge", prompt)
        self.assertNotIn("knowledge list", prompt)
        self.assertIn("tool publish", prompt)
        self.assertIn("network status", prompt)
        self.assertIn("trace list", prompt)
        self.assertIn("trace commands", prompt)
        self.assertNotIn("fs context", prompt)
        self.assertTrue((workspace.root / "task" / "TASK.md").exists())
        self.assertEqual(
            (workspace.root / "task" / "knowledge" / "domain_refs" / "note.txt").read_text(encoding="utf-8"),
            "Task-defined knowledge layout.\n",
        )
        self.assertTrue((workspace.root / "task" / "knowledge" / "manifest.json").exists())
        self.assertTrue((workspace.root / "task" / "knowledge_inventory.json").exists())
        inventory = json.loads((workspace.root / "task" / "knowledge_inventory.json").read_text(encoding="utf-8"))
        self.assertTrue(inventory["digest"].startswith("sha256:"))
        self.assertTrue(all(item["read_only"] for item in inventory["files"]))
        current_state = json.loads((workspace.root / "context" / "current_state.json").read_text(encoding="utf-8"))
        self.assertEqual(current_state["task_knowledge"]["digest"], inventory["digest"])
        self.assertTrue(current_state["task_context"]["digest"].startswith("sha256:"))
        self.assertTrue(current_state["task_context"]["enforcement"]["policy_weakened"])
        self.assertTrue((workspace.root / "context" / "current_state.json").exists())

        if shutil.which("git"):
            proc = subprocess.run(
                ["git", "status", "--short"],
                cwd=workspace.root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue((workspace.root / ".git").exists())

        if shutil.which("zsh"):
            fake_workspace = Path(tempfile.mkdtemp(dir=str(self.root)))
            self.addCleanup(shutil.rmtree, fake_workspace, ignore_errors=True)
            fake_bin = fake_workspace / "bin"
            fake_bin.mkdir()
            fake_eval = fake_bin / "eval"
            fake_eval.write_text("#!/bin/sh\necho semantic-eval \"$@\"\n", encoding="utf-8")
            fake_eval.chmod(0o755)
            proc = subprocess.run(
                ["zsh", "-lc", "eval verify --entry initial.py"],
                capture_output=True,
                text=True,
                check=False,
                env={
                    **os.environ,
                    "AO_WORKSPACE_ROOT": str(fake_workspace),
                    "ZDOTDIR": workspace.env["ZDOTDIR"],
                },
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stdout.strip(), "semantic-eval verify --entry initial.py")
        self.assertTrue((workspace.root / "history" / "leaderboard.jsonl").exists())
        self.assertFalse((workspace.root / "history" / "knowledge").exists())

    def test_semantic_workspace_can_use_canonical_task_context_snapshot(self) -> None:
        runtime_root = self.root / "runtime_snapshot"
        runtime_root.mkdir(parents=True, exist_ok=True)
        runtime_env = PreparedRuntimeEnv(
            task_id="toy_semantic",
            fingerprint="test",
            root=runtime_root,
            venv_dir=runtime_root,
            python_path=Path(sys.executable),
            manifest_path=runtime_root / "manifest.json",
            spec=TaskRuntimeSpec(verify_public_seed=False),
        )
        assignment = {
            "assignment_id": "assign_test",
            "experiment_id": "exp_test",
            "task_id": "toy_semantic",
            "agent_id": "agent_001",
        }
        snapshot = ensure_task_context_snapshot(task_id="toy_semantic", state_root=self.root / "state")

        workspace = prepare_semantic_workspace(
            workspace_root=self.root / "workspace_snapshot",
            api_url="http://127.0.0.1:5000",
            assignment=assignment,
            session_id="session_test",
            runtime_env=runtime_env,
            bootstrap={"task_context": snapshot},
        )

        current_state = json.loads((workspace.root / "context" / "current_state.json").read_text(encoding="utf-8"))
        self.assertEqual(current_state["task_context"]["digest"], snapshot["digest"])
        self.assertEqual(workspace.task_context["digest"], snapshot["digest"])
        self.assertEqual(
            (workspace.root / "task" / "knowledge" / "domain_refs" / "note.txt").read_text(encoding="utf-8"),
            "Task-defined knowledge layout.\n",
        )

    def test_eval_location_materializes_workspace_history_file(self) -> None:
        old_workspace_root = os.environ.get("AO_WORKSPACE_ROOT")
        os.environ["AO_WORKSPACE_ROOT"] = str(self.root / "workspace_eval")
        self.addCleanup(self._restore_env, "AO_WORKSPACE_ROOT", old_workspace_root)

        payload = {
            "evaluation_id": "eval_test",
            "status": "completed",
            "kind": "submit",
            "valid": True,
            "score": 1.25,
            "request": {"entry_path": "initial.py"},
            "result": {"score": 1.25},
            "public_feedback": {"min_slack": 0.0},
            "feedback": {"ok": True},
        }

        location = _evaluation_location(payload)

        evaluation_root = (self.root / "workspace_eval" / "history" / "evaluations" / "eval_test").resolve()
        self.assertEqual(location["files"]["evaluation"], str(evaluation_root / "evaluation.json"))
        self.assertEqual(json.loads((evaluation_root / "evaluation.json").read_text(encoding="utf-8"))["score"], 1.25)
        self.assertEqual(json.loads((evaluation_root / "request.json").read_text(encoding="utf-8"))["entry_path"], "initial.py")
        self.assertEqual(json.loads((evaluation_root / "public_feedback.json").read_text(encoding="utf-8"))["min_slack"], 0.0)
        self.assertEqual(json.loads((evaluation_root / "feedback.json").read_text(encoding="utf-8"))["ok"], True)
        self.assertIn(
            "eval_test",
            (self.root / "workspace_eval" / "history" / "evaluations" / "index.jsonl").read_text(encoding="utf-8"),
        )

        no_private_feedback = {
            "evaluation_id": "eval_public_only",
            "status": "completed",
            "kind": "submit",
            "public_feedback": {"score": 2.0},
        }
        _evaluation_location(no_private_feedback)
        public_only_root = self.root / "workspace_eval" / "history" / "evaluations" / "eval_public_only"
        self.assertEqual(json.loads((public_only_root / "feedback.json").read_text(encoding="utf-8"))["score"], 2.0)
        self.assertIsNone(json.loads((public_only_root / "result.json").read_text(encoding="utf-8")))

    def test_private_codex_home_is_outside_agent_workspace(self) -> None:
        workspace_root = self.root / "run" / "workspaces" / "assign_test" / "session_test"

        codex_home = private_codex_home_for_workspace(
            workspace_root=workspace_root,
            session_id="../session/test",
        )

        self.assertEqual(
            codex_home,
            self.root.resolve() / "run" / "provider_state" / "codex_home" / "session_test",
        )
        self.assertFalse(codex_home.is_relative_to(workspace_root))
        self.assertFalse((workspace_root / ".codex-home").exists())

    def test_prepend_path_preserves_existing_tail(self) -> None:
        self.assertEqual(_prepend_path("/usr/local/bin", "/usr/local/bin:/usr/bin:/bin"), "/usr/local/bin:/usr/bin:/bin")
        self.assertEqual(_prepend_path("/workspace/bin", "/usr/local/bin:/usr/bin"), "/workspace/bin:/usr/local/bin:/usr/bin")

    def test_bootstrap_context_is_materialized_as_files(self) -> None:
        runtime_root = self.root / "runtime"
        runtime_root.mkdir(parents=True, exist_ok=True)
        runtime_env = PreparedRuntimeEnv(
            task_id="toy_semantic",
            fingerprint="test",
            root=runtime_root,
            venv_dir=runtime_root,
            python_path=Path(sys.executable),
            manifest_path=runtime_root / "manifest.json",
            spec=TaskRuntimeSpec(verify_public_seed=False),
        )
        assignment = {
            "assignment_id": "assign_test",
            "experiment_id": "exp_test",
            "task_id": "toy_semantic",
            "agent_id": "agent_001",
            "budget": {"evaluator_runs": 1},
        }
        bootstrap = {
            "context_snapshot": {
                "assignment": assignment,
                "experiment": {"experiment_id": "exp_test", "task_id": "toy_semantic"},
                "recent_findings": [
                    {
                        "finding_id": "finding_1",
                        "title": "Use pair swaps",
                        "body": "Pair swaps improved the toy candidate.",
                    }
                ],
                "evaluations": [
                    {
                        "evaluation_id": "eval_1",
                        "status": "completed",
                        "request": {"kind": "probe"},
                        "public_feedback": {"score": 1.0},
                    }
                ],
                "jobs": [{"job_id": "job_1", "status": "completed", "outputs": {"stdout_path": "/tmp/stdout.log"}}],
                "agent_traces": [
                    {
                        "trace_id": "trace_1",
                        "trace_root": "/tmp/trace_1",
                        "metadata": {"files": {"commands": "commands.jsonl", "events": "events.jsonl"}},
                    }
                ],
                "leaderboard": [{"evaluation_id": "eval_1", "score": 1.0}],
                "network_policy": {"policy": {"external_internet": "deny"}},
                "network_access_events": [{"event_id": "net_1", "decision": "denied"}],
                "notebook_checkpoints": [{"checkpoint_id": "note_1", "content": "remember this"}],
            },
            "task_contract": {
                "task_id": "toy_semantic",
                "public_context": {
                    "task_markdown": "# Toy Semantic\n",
                    "public_contract_markdown": "Expose `solve()`.\n",
                    "public_files": [{"path": "notes/guide.md", "content": "Guide text\n"}],
                    "research_directions": [{"direction_id": "dir_1", "doc_markdown": "# Direction\n"}],
                },
            },
        }

        workspace = prepare_semantic_workspace(
            workspace_root=self.root / "workspace_files",
            api_url="http://127.0.0.1:5000",
            assignment=assignment,
            session_id="session_test",
            runtime_env=runtime_env,
            bootstrap=bootstrap,
        )

        self.assertIn("Pair swaps", (workspace.root / "history" / "findings" / "finding_1.json").read_text(encoding="utf-8"))
        self.assertEqual((workspace.root / "task" / "public_files" / "notes" / "guide.md").read_text(encoding="utf-8"), "Guide text\n")
        self.assertTrue((workspace.root / "history" / "evaluations" / "eval_1" / "public_feedback.json").exists())
        self.assertIn("commands.jsonl", (workspace.root / "history" / "traces" / "trace_1" / "manifest.json").read_text(encoding="utf-8"))
        self.assertIn("net_1", (workspace.root / "history" / "network" / "events.jsonl").read_text(encoding="utf-8"))

    def test_semantic_workspace_allows_tasks_without_knowledge_tree(self) -> None:
        shutil.rmtree(self.tasks_root / "toy_semantic" / "public" / "knowledge")
        runtime_root = self.root / "runtime_no_knowledge"
        runtime_root.mkdir(parents=True, exist_ok=True)
        runtime_env = PreparedRuntimeEnv(
            task_id="toy_semantic",
            fingerprint="test",
            root=runtime_root,
            venv_dir=runtime_root,
            python_path=Path(sys.executable),
            manifest_path=runtime_root / "manifest.json",
            spec=TaskRuntimeSpec(verify_public_seed=False),
        )
        assignment = {
            "assignment_id": "assign_test",
            "experiment_id": "exp_test",
            "task_id": "toy_semantic",
            "agent_id": "agent_001",
            "budget": {"evaluator_runs": 1},
        }

        workspace = prepare_semantic_workspace(
            workspace_root=self.root / "workspace_no_knowledge",
            api_url="http://127.0.0.1:5000",
            assignment=assignment,
            session_id="session_test",
            runtime_env=runtime_env,
        )

        self.assertTrue((workspace.root / "task" / "knowledge").is_dir())
        self.assertEqual(list((workspace.root / "task" / "knowledge").iterdir()), [])
        inventory = json.loads((workspace.root / "task" / "knowledge_inventory.json").read_text(encoding="utf-8"))
        self.assertFalse(inventory["available"])
        self.assertEqual(inventory["file_count"], 0)

    def test_semantic_workspace_allows_knowledge_without_manifest(self) -> None:
        (self.tasks_root / "toy_semantic" / "public" / "knowledge" / "manifest.json").unlink()
        runtime_root = self.root / "runtime_no_manifest"
        runtime_root.mkdir(parents=True, exist_ok=True)
        runtime_env = PreparedRuntimeEnv(
            task_id="toy_semantic",
            fingerprint="test",
            root=runtime_root,
            venv_dir=runtime_root,
            python_path=Path(sys.executable),
            manifest_path=runtime_root / "manifest.json",
            spec=TaskRuntimeSpec(verify_public_seed=False),
        )
        assignment = {
            "assignment_id": "assign_test",
            "experiment_id": "exp_test",
            "task_id": "toy_semantic",
            "agent_id": "agent_001",
            "budget": {"evaluator_runs": 1},
        }

        workspace = prepare_semantic_workspace(
            workspace_root=self.root / "workspace_no_manifest",
            api_url="http://127.0.0.1:5000",
            assignment=assignment,
            session_id="session_test",
            runtime_env=runtime_env,
        )

        self.assertTrue((workspace.root / "task" / "knowledge" / "domain_refs" / "note.txt").exists())
        inventory = json.loads((workspace.root / "task" / "knowledge_inventory.json").read_text(encoding="utf-8"))
        self.assertIsNone(inventory["manifest"])
        self.assertEqual(inventory["file_count"], 1)

    def test_app_server_rejects_codex_home_inside_workspace(self) -> None:
        workspace_root = self.root / "workspace"
        workspace_root.mkdir(parents=True, exist_ok=True)
        client = AppServerClient(
            root_cwd=str(workspace_root),
            codex_home=str(workspace_root / ".codex-home"),
        )

        with self.assertRaises(AppServerClientError):
            client._build_env()

    @staticmethod
    def _restore_env(name: str, value: str | None) -> None:
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value

    def _write_toy_task(self) -> None:
        task_dir = self.tasks_root / "toy_semantic"
        public = task_dir / "public"
        public.mkdir(parents=True, exist_ok=True)
        (public / "TASK.md").write_text("# Toy Semantic\n", encoding="utf-8")
        (public / "public_contract.md").write_text("Expose `solve()`.\n", encoding="utf-8")
        (public / "initial.py").write_text("def solve():\n    return 1\n", encoding="utf-8")
        (public / "knowledge" / "domain_refs").mkdir(parents=True, exist_ok=True)
        (public / "knowledge" / "domain_refs" / "note.txt").write_text("Task-defined knowledge layout.\n", encoding="utf-8")
        (public / "knowledge" / "manifest.json").write_text('{"items": [{"path": "domain_refs/note.txt"}]}\n', encoding="utf-8")
        (task_dir / "task.py").write_text(
            '''
from pathlib import Path

from agentic_opt.common.runtime_env import TaskRuntimeSpec
from agentic_opt.task_api import TaskMetadata


class ToyTask:
    metadata = TaskMetadata(task_id="toy_semantic", title="Toy Semantic")
    runtime_spec = TaskRuntimeSpec(verify_public_seed=False)

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def public_dir(self) -> Path:
        return self._root / "public"

    def verify_entry(self, entry_path: Path) -> dict:
        return {"valid": entry_path.exists(), "feedback": {}}

    def probe_entry(self, entry_path: Path, *, kind: str) -> dict:
        return {"valid": True, "kind": kind}

    def evaluate_entry(self, entry_path: Path) -> dict:
        return {"score": 1.0, "correct": {"correct": True}, "evaluator": {"public_details": {}}}


def create_task() -> ToyTask:
    return ToyTask(Path(__file__).parent)
'''.lstrip(),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
