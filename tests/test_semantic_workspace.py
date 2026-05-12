from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

from agentic_opt.adapter.app_server_client import AppServerClient, AppServerClientError
from agentic_opt.adapter.semantic_worker import private_codex_home_for_workspace
from agentic_opt.adapter.semantic_workspace import (
    SEMANTIC_SKILL_BODIES,
    build_semantic_startup_prompt,
    prepare_semantic_workspace,
)
from agentic_opt.common.runtime_env import PreparedRuntimeEnv, TaskRuntimeSpec


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

        commands = ("ctx", "artifact", "eval", "finding", "notebook", "job", "env", "telemetry", "tool", "knowledge", "network")
        for command in commands:
            self.assertTrue((workspace.bin_dir / command).exists())
        self.assertFalse((workspace.bin_dir / "fs").exists())
        self.assertFalse((workspace.bin_dir / "ve").exists())
        self.assertEqual(set(SEMANTIC_SKILL_BODIES), {item.name for item in workspace.skills_root.iterdir()})
        self.assertEqual(workspace.env["AO_ENVIRONMENT_ID"], "env_task_toy_semantic_test")

        agents_md = workspace.agents_md_path.read_text(encoding="utf-8")
        self.assertIn("semantic tools", agents_md)
        self.assertIn("eval submit", agents_md)
        self.assertNotIn("fs/ve", agents_md)

        prompt = build_semantic_startup_prompt(assignment=assignment, workspace=workspace)
        self.assertIn("ctx context", prompt)
        self.assertIn("env status", prompt)
        self.assertIn("job create --provider local", prompt)
        self.assertIn("telemetry start", prompt)
        self.assertIn("knowledge list", prompt)
        self.assertIn("tool publish", prompt)
        self.assertIn("network status", prompt)
        self.assertNotIn("fs context", prompt)

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

    def test_app_server_rejects_codex_home_inside_workspace(self) -> None:
        workspace_root = self.root / "workspace"
        workspace_root.mkdir(parents=True, exist_ok=True)
        client = AppServerClient(
            root_cwd=str(workspace_root),
            codex_home=str(workspace_root / ".codex-home"),
        )

        with self.assertRaises(AppServerClientError):
            client._build_env()

    def _write_toy_task(self) -> None:
        task_dir = self.tasks_root / "toy_semantic"
        public = task_dir / "public"
        public.mkdir(parents=True, exist_ok=True)
        (public / "TASK.md").write_text("# Toy Semantic\n", encoding="utf-8")
        (public / "public_contract.md").write_text("Expose `solve()`.\n", encoding="utf-8")
        (public / "initial.py").write_text("def solve():\n    return 1\n", encoding="utf-8")
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
