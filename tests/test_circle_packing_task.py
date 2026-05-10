from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentic_opt.control_plane.service import task_contract
from agentic_opt.task_registry import get_task
from agentic_opt.web.app import create_app


class CirclePackingTaskTests(unittest.TestCase):
    def test_task_contract_exposes_semantic_public_context(self) -> None:
        contract = task_contract("circle_packing_26")

        self.assertEqual(contract["candidate_contract"]["workspace_entrypoint"], "initial.py")
        self.assertIn("eval verify", contract["public_context"]["task_markdown"])
        self.assertIn("Official scores come only", contract["public_context"]["public_contract_markdown"])

        public_paths = {item["path"] for item in contract["public_context"]["public_files"]}
        self.assertIn("TASK.md", public_paths)
        self.assertIn("public_contract.md", public_paths)
        self.assertIn("research_directions/manifest.json", public_paths)

        directions = contract["public_context"]["research_directions"]
        self.assertGreaterEqual(len(directions), 3)
        self.assertTrue(all("direction_id" in item for item in directions))
        self.assertTrue(all("doc_markdown" in item for item in directions))

    def test_seed_candidate_verifies_probes_and_evaluates(self) -> None:
        task = get_task("circle_packing_26")
        entry_path = task.public_dir / "initial.py"
        client, experiment_id, assignment_id = self._control_client()

        verified = self._evaluate(
            client,
            experiment_id=experiment_id,
            assignment_id=assignment_id,
            kind="verify",
            entry_path=entry_path,
        )
        self.assertTrue(verified["valid"], verified.get("public_feedback"))
        self.assertEqual(verified["status"], "completed")
        self.assertEqual(verified["result"]["status"], "passed")

        probed = self._evaluate(
            client,
            experiment_id=experiment_id,
            assignment_id=assignment_id,
            kind="probe",
            entry_path=entry_path,
        )
        self.assertTrue(probed["valid"], probed.get("feedback"))
        self.assertIsInstance(probed["score"], float)
        self.assertIn("strict_safe_score", probed["result"]["diagnostics"])

        evaluated = self._evaluate(
            client,
            experiment_id=experiment_id,
            assignment_id=assignment_id,
            kind="submit",
            entry_path=entry_path,
        )
        self.assertTrue(evaluated["valid"])
        self.assertEqual(evaluated["score"], evaluated["result"]["metrics"]["actual_sum"])
        self.assertIn("visualization", evaluated["result"]["extra"])
        self.assertIsNotNone(evaluated["artifact_id"])

        leaderboard = client.get("/api/v1/leaderboard", query_string={"experiment_id": experiment_id})
        self.assertEqual(leaderboard.status_code, 200, leaderboard.get_data(as_text=True))
        entries = leaderboard.get_json()["leaderboard"]
        self.assertEqual(entries[0]["evaluation_id"], evaluated["evaluation_id"])
        self.assertEqual(entries[0]["score"], evaluated["score"])

        incumbent = client.get("/api/v1/incumbent", query_string={"experiment_id": experiment_id})
        self.assertEqual(incumbent.status_code, 200, incumbent.get_data(as_text=True))
        self.assertEqual(incumbent.get_json()["artifact_id"], evaluated["artifact_id"])

        with tempfile.TemporaryDirectory() as tempdir:
            destination = Path(tempdir) / "incumbent.py"
            checkout = client.post(
                "/api/v1/incumbent/checkout",
                json={"experiment_id": experiment_id, "destination_path": str(destination)},
            )
            self.assertEqual(checkout.status_code, 200, checkout.get_data(as_text=True))
            self.assertTrue(destination.exists())
            self.assertIn("run_packing", destination.read_text(encoding="utf-8"))

    def test_non_finite_candidate_is_invalid(self) -> None:
        client, experiment_id, assignment_id = self._control_client()
        with tempfile.TemporaryDirectory() as tempdir:
            entry_path = Path(tempdir) / "initial.py"
            entry_path.write_text(
                """
import numpy as np


def run_packing():
    centers = np.zeros((26, 2), dtype=float)
    centers[0, 0] = np.nan
    radii = np.full(26, 0.01)
    return centers, radii, float(np.sum(radii))
""".lstrip(),
                encoding="utf-8",
            )

            verified = self._evaluate(
                client,
                experiment_id=experiment_id,
                assignment_id=assignment_id,
                kind="verify",
                entry_path=entry_path,
            )
            self.assertFalse(verified["valid"])
            self.assertIn("finite", verified["public_feedback"]["error"])

    def _control_client(self):
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        root = Path(tempdir.name)
        app = create_app(state_root=root / "state", database_path=root / "state" / "control.sqlite3")
        client = app.test_client()
        created = client.post(
            "/api/v1/experiments",
            json={"task_id": "circle_packing_26", "mode": "local", "assignment_count": 1},
        )
        self.assertEqual(created.status_code, 201, created.get_data(as_text=True))
        payload = created.get_json()
        assignment = payload["assignments"][0]
        self.assertIsNotNone(assignment["direction_id"])
        self.assertIn("research_direction", assignment["metadata"])
        return client, payload["experiment"]["experiment_id"], assignment["assignment_id"]

    def _evaluate(self, client, *, experiment_id: str, assignment_id: str, kind: str, entry_path: Path) -> dict:
        response = client.post(
            "/api/v1/evaluations",
            json={
                "experiment_id": experiment_id,
                "assignment_id": assignment_id,
                "task_id": "circle_packing_26",
                "kind": kind,
                "entry_path": str(entry_path),
                "async": False,
            },
        )
        self.assertEqual(response.status_code, 201, response.get_data(as_text=True))
        return response.get_json()


if __name__ == "__main__":
    unittest.main()
