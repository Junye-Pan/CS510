from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from agentic_opt.adapter.app_server_client import AppServerClient
from agentic_opt.adapter.base import BudgetPolicy, InstructionBundle, WorkspacePolicy
from agentic_opt.adapter.codex_adapter import AppServerAdapterConfig, AppServerCodexAdapter


@unittest.skipUnless(os.environ.get("AO_ENABLE_LIVE_APPSERVER_TESTS") == "1", "set AO_ENABLE_LIVE_APPSERVER_TESTS=1 to run live App Server tests")
class LiveLocalAppServerTests(unittest.TestCase):
    def test_codex_app_server_session_bootstrap_and_turn(self) -> None:
        if shutil.which("codex") is None:
            self.skipTest("codex binary is not available")

        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            workspace_root = root / "workspace"
            workspace_root.mkdir(parents=True, exist_ok=True)
            (workspace_root / "AGENTS.md").write_text(
                "# Agentic Optimization Workspace\n\nReply briefly after reading this file.\n",
                encoding="utf-8",
            )
            client = AppServerClient(
                codex_binary="codex",
                root_cwd=str(workspace_root),
                codex_home=str(root / "codex-home"),
            )
            adapter = AppServerCodexAdapter(
                client=client,
                config=AppServerAdapterConfig(
                    model="gpt-5.4",
                    reasoning_effort="medium",
                    approval_policy="never",
                ),
            )
            session = None
            try:
                session = adapter.start_session(
                    task_id="circle_packing_26",
                    agent_id="agent_live",
                    run_id="run_live",
                    workspace=WorkspacePolicy(
                        workspace_root=str(workspace_root),
                        writable_roots=[str(workspace_root)],
                        readable_roots=[str(workspace_root)],
                        allow_network=False,
                    ),
                    instructions=InstructionBundle(
                        task_id="circle_packing_26",
                        startup_prompt="Read AGENTS.md and reply with a short acknowledgement.",
                        agents_md_path=str(workspace_root / "AGENTS.md"),
                    ),
                    budget=BudgetPolicy(max_turn_wall_time_s=180),
                )
                turn = adapter.start_turn(
                    session_id=session.session_id,
                    kind="smoke",
                    prompt="Read AGENTS.md and respond in one short sentence.",
                    budget=BudgetPolicy(max_turn_wall_time_s=180),
                )
                result = adapter.wait_turn(
                    session_id=session.session_id,
                    turn_id=turn.turn_id,
                    timeout_s=180,
                )
                self.assertIn(result.outcome, {"completed", "success"})
                self.assertTrue(result.final_message)
            finally:
                if session is not None:
                    try:
                        adapter.close_session(session_id=session.session_id, final_status="completed")
                    except Exception:
                        pass
                client.close()


if __name__ == "__main__":
    unittest.main()
