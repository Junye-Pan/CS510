from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from agentic_opt.common.ids import isoformat_z, make_event_id, make_run_id, make_session_id


def _json(value: Any) -> str:
    return json.dumps(value or {}, sort_keys=True)


def _loads(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    return json.loads(raw)


class ControlPlaneRepository:
    """SQLite-backed server-owned state for the new control plane.

    Filesystem paths can still appear as artifact URIs, but resource meaning
    lives in server-owned records.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path.resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS cp_experiments (
                    experiment_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    config_json TEXT NOT NULL DEFAULT '{}',
                    budget_json TEXT NOT NULL DEFAULT '{}',
                    policy_json TEXT NOT NULL DEFAULT '{}',
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS cp_assignments (
                    assignment_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    direction_id TEXT,
                    status TEXT NOT NULL,
                    worker_backend TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    visible_context_policy_json TEXT NOT NULL DEFAULT '{}',
                    budget_json TEXT NOT NULL DEFAULT '{}',
                    workspace_seed_json TEXT NOT NULL DEFAULT '{}',
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_cp_assignments_experiment
                ON cp_assignments(experiment_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS cp_sessions (
                    session_id TEXT PRIMARY KEY,
                    assignment_id TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    worker_backend TEXT NOT NULL,
                    pid INTEGER,
                    started_at TEXT,
                    ended_at TEXT,
                    updated_at TEXT NOT NULL,
                    workspace_path TEXT,
                    details_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_cp_sessions_assignment
                ON cp_sessions(assignment_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS cp_jobs (
                    job_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    assignment_id TEXT,
                    session_id TEXT,
                    provider TEXT NOT NULL,
                    status TEXT NOT NULL,
                    pid INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    inputs_json TEXT NOT NULL DEFAULT '{}',
                    outputs_json TEXT NOT NULL DEFAULT '{}',
                    cost_json TEXT NOT NULL DEFAULT '{}',
                    details_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_cp_jobs_experiment
                ON cp_jobs(experiment_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS cp_environments (
                    environment_id TEXT PRIMARY KEY,
                    environment_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    python_path TEXT NOT NULL,
                    root_path TEXT NOT NULL,
                    task_id TEXT,
                    experiment_id TEXT,
                    parent_environment_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    spec_json TEXT NOT NULL DEFAULT '{}',
                    lock_json TEXT NOT NULL DEFAULT '{}',
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_cp_environments_task
                ON cp_environments(task_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_cp_environments_experiment
                ON cp_environments(experiment_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS cp_environment_overlays (
                    overlay_id TEXT PRIMARY KEY,
                    base_environment_id TEXT NOT NULL,
                    experiment_id TEXT,
                    assignment_id TEXT,
                    session_id TEXT,
                    status TEXT NOT NULL,
                    requested_by_agent_id TEXT,
                    python_path TEXT,
                    root_path TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    requirements_json TEXT NOT NULL DEFAULT '[]',
                    reason TEXT,
                    approved INTEGER NOT NULL DEFAULT 0,
                    lock_json TEXT NOT NULL DEFAULT '{}',
                    policy_decision_json TEXT NOT NULL DEFAULT '{}',
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_cp_environment_overlays_base
                ON cp_environment_overlays(base_environment_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_cp_environment_overlays_assignment
                ON cp_environment_overlays(assignment_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS cp_artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    experiment_id TEXT,
                    assignment_id TEXT,
                    attempt_id TEXT,
                    kind TEXT NOT NULL,
                    uri TEXT NOT NULL,
                    local_path TEXT,
                    digest TEXT,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_cp_artifacts_experiment
                ON cp_artifacts(experiment_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS cp_evaluations (
                    evaluation_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    assignment_id TEXT,
                    attempt_id TEXT,
                    artifact_id TEXT,
                    job_id TEXT,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    valid INTEGER,
                    score REAL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    request_json TEXT NOT NULL DEFAULT '{}',
                    result_json TEXT NOT NULL DEFAULT '{}',
                    public_feedback_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_cp_evaluations_experiment
                ON cp_evaluations(experiment_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS cp_leaderboard_entries (
                    leaderboard_entry_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    assignment_id TEXT,
                    direction_id TEXT,
                    evaluation_id TEXT NOT NULL UNIQUE,
                    artifact_id TEXT,
                    score REAL NOT NULL,
                    status TEXT NOT NULL,
                    environment_id TEXT,
                    environment_overlay_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_cp_leaderboard_experiment
                ON cp_leaderboard_entries(experiment_id, score DESC, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_cp_leaderboard_task
                ON cp_leaderboard_entries(task_id, score DESC, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_cp_leaderboard_direction
                ON cp_leaderboard_entries(experiment_id, direction_id, score DESC, updated_at DESC);

                CREATE TABLE IF NOT EXISTS cp_findings (
                    finding_id TEXT PRIMARY KEY,
                    experiment_id TEXT,
                    assignment_id TEXT,
                    task_id TEXT NOT NULL,
                    finding_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    links_json TEXT NOT NULL DEFAULT '{}',
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_cp_findings_task
                ON cp_findings(task_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS cp_notebook_checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    assignment_id TEXT NOT NULL,
                    session_id TEXT,
                    agent_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    notebook_uri TEXT,
                    content TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_cp_notebooks_assignment
                ON cp_notebook_checkpoints(assignment_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS cp_events (
                    event_id TEXT PRIMARY KEY,
                    experiment_id TEXT,
                    assignment_id TEXT,
                    session_id TEXT,
                    task_id TEXT,
                    agent_id TEXT,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    summary TEXT,
                    payload_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_cp_events_experiment
                ON cp_events(experiment_id, timestamp DESC);

                CREATE TABLE IF NOT EXISTS cp_telemetry_runs (
                    telemetry_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    assignment_id TEXT,
                    session_id TEXT,
                    job_id TEXT,
                    attempt_id TEXT,
                    artifact_id TEXT,
                    provider TEXT NOT NULL,
                    status TEXT NOT NULL,
                    run_name TEXT,
                    dashboard_url TEXT,
                    external_run_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    params_json TEXT NOT NULL DEFAULT '{}',
                    tags_json TEXT NOT NULL DEFAULT '{}',
                    artifacts_json TEXT NOT NULL DEFAULT '{}',
                    details_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_cp_telemetry_experiment
                ON cp_telemetry_runs(experiment_id, updated_at DESC);
                """
            )

    def create_experiment(self, payload: dict[str, Any]) -> dict[str, Any]:
        now = isoformat_z()
        record = {
            "experiment_id": payload.get("experiment_id") or make_run_id("exp"),
            "task_id": payload["task_id"],
            "status": payload.get("status") or "created",
            "mode": payload.get("mode") or "local",
            "created_at": payload.get("created_at") or now,
            "updated_at": payload.get("updated_at") or now,
            "config": payload.get("config") or {},
            "budget": payload.get("budget") or {},
            "policy": payload.get("policy") or {},
            "metadata": payload.get("metadata") or {},
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cp_experiments (
                    experiment_id, task_id, status, mode, created_at, updated_at,
                    config_json, budget_json, policy_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["experiment_id"],
                    record["task_id"],
                    record["status"],
                    record["mode"],
                    record["created_at"],
                    record["updated_at"],
                    _json(record["config"]),
                    _json(record["budget"]),
                    _json(record["policy"]),
                    _json(record["metadata"]),
                ),
            )
        self.record_event(
            {
                "experiment_id": record["experiment_id"],
                "task_id": record["task_id"],
                "event_type": "experiment.created",
                "summary": "experiment created",
                "payload": record,
            }
        )
        return record

    def update_experiment_status(self, experiment_id: str, status: str, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        now = isoformat_z()
        current = self.get_experiment(experiment_id)
        if current is None:
            raise KeyError(experiment_id)
        merged_metadata = {**(current.get("metadata") or {}), **(metadata or {})}
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE cp_experiments
                SET status = ?, updated_at = ?, metadata_json = ?
                WHERE experiment_id = ?
                """,
                (status, now, _json(merged_metadata), experiment_id),
            )
        record = self.get_experiment(experiment_id)
        assert record is not None
        self.record_event(
            {
                "experiment_id": experiment_id,
                "task_id": record["task_id"],
                "event_type": "experiment.updated",
                "summary": f"experiment status={status}",
                "payload": {"status": status, "metadata": metadata or {}},
            }
        )
        return record

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM cp_experiments WHERE experiment_id = ?", (experiment_id,)).fetchone()
        return None if row is None else self._row_experiment(row)

    def list_experiments(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM cp_experiments ORDER BY updated_at DESC, experiment_id DESC"
            ).fetchall()
        return [self._row_experiment(row) for row in rows]

    def create_assignment(self, payload: dict[str, Any]) -> dict[str, Any]:
        experiment = self.get_experiment(payload["experiment_id"])
        if experiment is None:
            raise KeyError(payload["experiment_id"])
        now = isoformat_z()
        record = {
            "assignment_id": payload.get("assignment_id") or make_run_id("assign"),
            "experiment_id": experiment["experiment_id"],
            "task_id": payload.get("task_id") or experiment["task_id"],
            "agent_id": payload.get("agent_id") or "agent_001",
            "direction_id": payload.get("direction_id"),
            "status": payload.get("status") or "queued",
            "worker_backend": payload.get("worker_backend") or "codex-local",
            "created_at": payload.get("created_at") or now,
            "updated_at": payload.get("updated_at") or now,
            "visible_context_policy": payload.get("visible_context_policy") or {},
            "budget": payload.get("budget") or {},
            "workspace_seed": payload.get("workspace_seed") or {},
            "metadata": payload.get("metadata") or {},
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cp_assignments (
                    assignment_id, experiment_id, task_id, agent_id, direction_id,
                    status, worker_backend, created_at, updated_at,
                    visible_context_policy_json, budget_json, workspace_seed_json,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["assignment_id"],
                    record["experiment_id"],
                    record["task_id"],
                    record["agent_id"],
                    record["direction_id"],
                    record["status"],
                    record["worker_backend"],
                    record["created_at"],
                    record["updated_at"],
                    _json(record["visible_context_policy"]),
                    _json(record["budget"]),
                    _json(record["workspace_seed"]),
                    _json(record["metadata"]),
                ),
            )
        self.record_event(
            {
                "experiment_id": record["experiment_id"],
                "assignment_id": record["assignment_id"],
                "task_id": record["task_id"],
                "agent_id": record["agent_id"],
                "event_type": "assignment.created",
                "summary": "worker assignment created",
                "payload": record,
            }
        )
        return record

    def generate_assignments(self, *, experiment_id: str, count: int, worker_backend: str = "codex-local") -> list[dict[str, Any]]:
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            raise KeyError(experiment_id)
        budget = experiment.get("budget") or {}
        total_evaluator_runs = int(budget.get("total_evaluator_runs") or budget.get("evaluator_runs") or 0)
        base_budget, extra = divmod(total_evaluator_runs, max(1, count))
        assignments: list[dict[str, Any]] = []
        for index in range(count):
            agent_id = f"agent_{index + 1:03d}"
            assignments.append(
                self.create_assignment(
                    {
                        "experiment_id": experiment_id,
                        "agent_id": agent_id,
                        "worker_backend": worker_backend,
                        "budget": {
                            "evaluator_runs": base_budget + (1 if index < extra else 0),
                        },
                    }
                )
            )
        return assignments

    def create_assignments(
        self,
        *,
        experiment_id: str,
        count: int,
        worker_backend: str = "codex-local",
        direction_plan: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        experiment = self.get_experiment(experiment_id)
        if experiment is None:
            raise KeyError(experiment_id)
        budget = experiment.get("budget") or {}
        total_evaluator_runs = int(budget.get("total_evaluator_runs") or budget.get("evaluator_runs") or 0)
        base_budget, extra = divmod(total_evaluator_runs, max(1, count))
        directions = direction_plan or []
        assignments: list[dict[str, Any]] = []
        for index in range(count):
            agent_id = f"agent_{index + 1:03d}"
            direction = directions[index % len(directions)] if directions else None
            direction_id = direction.get("direction_id") if direction else None
            metadata: dict[str, Any] = {}
            if direction is not None:
                metadata["research_direction"] = direction
                if direction.get("startup_note"):
                    metadata["startup_note"] = direction["startup_note"]
            assignments.append(
                self.create_assignment(
                    {
                        "experiment_id": experiment_id,
                        "agent_id": agent_id,
                        "worker_backend": worker_backend,
                        "direction_id": direction_id,
                        "budget": {
                            "evaluator_runs": base_budget + (1 if index < extra else 0),
                        },
                        "metadata": metadata,
                    }
                )
            )
        return assignments

    def update_assignment_status(self, assignment_id: str, status: str, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        current = self.get_assignment(assignment_id)
        if current is None:
            raise KeyError(assignment_id)
        merged_metadata = {**(current.get("metadata") or {}), **(metadata or {})}
        now = isoformat_z()
        with self._connect() as conn:
            conn.execute(
                "UPDATE cp_assignments SET status = ?, updated_at = ?, metadata_json = ? WHERE assignment_id = ?",
                (status, now, _json(merged_metadata), assignment_id),
            )
        record = self.get_assignment(assignment_id)
        assert record is not None
        return record

    def get_assignment(self, assignment_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM cp_assignments WHERE assignment_id = ?", (assignment_id,)).fetchone()
        return None if row is None else self._row_assignment(row)

    def list_assignments(self, *, experiment_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM cp_assignments"
        params: tuple[Any, ...] = ()
        if experiment_id:
            query += " WHERE experiment_id = ?"
            params = (experiment_id,)
        query += " ORDER BY created_at DESC, assignment_id DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_assignment(row) for row in rows]

    def create_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        assignment = self.get_assignment(payload["assignment_id"])
        if assignment is None:
            raise KeyError(payload["assignment_id"])
        now = isoformat_z()
        record = {
            "session_id": payload.get("session_id") or make_session_id(),
            "assignment_id": assignment["assignment_id"],
            "experiment_id": assignment["experiment_id"],
            "task_id": assignment["task_id"],
            "agent_id": assignment["agent_id"],
            "status": payload.get("status") or "starting",
            "worker_backend": payload.get("worker_backend") or assignment["worker_backend"],
            "pid": payload.get("pid"),
            "started_at": payload.get("started_at") or now,
            "ended_at": payload.get("ended_at"),
            "updated_at": payload.get("updated_at") or now,
            "workspace_path": payload.get("workspace_path"),
            "details": payload.get("details") or {},
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cp_sessions (
                    session_id, assignment_id, experiment_id, task_id, agent_id,
                    status, worker_backend, pid, started_at, ended_at, updated_at,
                    workspace_path, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["session_id"],
                    record["assignment_id"],
                    record["experiment_id"],
                    record["task_id"],
                    record["agent_id"],
                    record["status"],
                    record["worker_backend"],
                    record["pid"],
                    record["started_at"],
                    record["ended_at"],
                    record["updated_at"],
                    record["workspace_path"],
                    _json(record["details"]),
                ),
            )
        self.update_assignment_status(assignment["assignment_id"], "running")
        return record

    def update_session(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        current = self.get_session(session_id)
        if current is None:
            raise KeyError(session_id)
        details = {**(current.get("details") or {}), **(payload.get("details") or {})}
        status = payload.get("status") or current["status"]
        now = isoformat_z()
        ended_at = payload.get("ended_at")
        if ended_at is None and status in {"completed", "failed", "stopped", "cancelled"}:
            ended_at = now
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE cp_sessions
                SET status = ?, pid = COALESCE(?, pid), ended_at = COALESCE(?, ended_at),
                    updated_at = ?, workspace_path = COALESCE(?, workspace_path),
                    details_json = ?
                WHERE session_id = ?
                """,
                (
                    status,
                    payload.get("pid"),
                    ended_at,
                    now,
                    payload.get("workspace_path"),
                    _json(details),
                    session_id,
                ),
            )
        record = self.get_session(session_id)
        assert record is not None
        if status in {"completed", "failed", "stopped", "cancelled"}:
            self.update_assignment_status(record["assignment_id"], status)
        return record

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM cp_sessions WHERE session_id = ?", (session_id,)).fetchone()
        return None if row is None else self._row_session(row)

    def list_sessions(self, *, assignment_id: str | None = None, experiment_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM cp_sessions"
        clauses: list[str] = []
        params: list[Any] = []
        if assignment_id:
            clauses.append("assignment_id = ?")
            params.append(assignment_id)
        if experiment_id:
            clauses.append("experiment_id = ?")
            params.append(experiment_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC, session_id DESC"
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [self._row_session(row) for row in rows]

    def upsert_environment(self, payload: dict[str, Any]) -> dict[str, Any]:
        now = isoformat_z()
        existing = self.get_environment(payload["environment_id"])
        record = {
            "environment_id": payload["environment_id"],
            "environment_type": payload.get("environment_type") or payload.get("type") or "task",
            "status": payload.get("status") or "ready",
            "fingerprint": payload["fingerprint"],
            "python_path": payload["python_path"],
            "root_path": payload["root_path"],
            "task_id": payload.get("task_id"),
            "experiment_id": payload.get("experiment_id"),
            "parent_environment_id": payload.get("parent_environment_id"),
            "created_at": (existing or {}).get("created_at") or payload.get("created_at") or now,
            "updated_at": payload.get("updated_at") or now,
            "spec": payload.get("spec") or {},
            "lock": payload.get("lock") or {},
            "metadata": payload.get("metadata") or {},
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cp_environments (
                    environment_id, environment_type, status, fingerprint,
                    python_path, root_path, task_id, experiment_id,
                    parent_environment_id, created_at, updated_at,
                    spec_json, lock_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(environment_id) DO UPDATE SET
                    environment_type = excluded.environment_type,
                    status = excluded.status,
                    fingerprint = excluded.fingerprint,
                    python_path = excluded.python_path,
                    root_path = excluded.root_path,
                    task_id = excluded.task_id,
                    experiment_id = excluded.experiment_id,
                    parent_environment_id = excluded.parent_environment_id,
                    updated_at = excluded.updated_at,
                    spec_json = excluded.spec_json,
                    lock_json = excluded.lock_json,
                    metadata_json = excluded.metadata_json
                """,
                (
                    record["environment_id"],
                    record["environment_type"],
                    record["status"],
                    record["fingerprint"],
                    record["python_path"],
                    record["root_path"],
                    record["task_id"],
                    record["experiment_id"],
                    record["parent_environment_id"],
                    record["created_at"],
                    record["updated_at"],
                    _json(record["spec"]),
                    _json(record["lock"]),
                    _json(record["metadata"]),
                ),
            )
        updated = self.get_environment(record["environment_id"])
        assert updated is not None
        return updated

    def get_environment(self, environment_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM cp_environments WHERE environment_id = ?", (environment_id,)).fetchone()
        return None if row is None else self._row_environment(row)

    def list_environments(
        self,
        *,
        task_id: str | None = None,
        experiment_id: str | None = None,
        environment_type: str | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM cp_environments"
        clauses: list[str] = []
        params: list[Any] = []
        if task_id:
            clauses.append("task_id = ?")
            params.append(task_id)
        if experiment_id:
            clauses.append("experiment_id = ?")
            params.append(experiment_id)
        if environment_type:
            clauses.append("environment_type = ?")
            params.append(environment_type)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC, environment_id DESC"
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [self._row_environment(row) for row in rows]

    def create_environment_overlay(self, payload: dict[str, Any]) -> dict[str, Any]:
        now = isoformat_z()
        record = {
            "overlay_id": payload.get("overlay_id") or make_run_id("env_overlay"),
            "base_environment_id": payload["base_environment_id"],
            "experiment_id": payload.get("experiment_id"),
            "assignment_id": payload.get("assignment_id"),
            "session_id": payload.get("session_id"),
            "status": payload.get("status") or "requested",
            "requested_by_agent_id": payload.get("requested_by_agent_id"),
            "python_path": payload.get("python_path"),
            "root_path": payload.get("root_path"),
            "created_at": payload.get("created_at") or now,
            "updated_at": payload.get("updated_at") or now,
            "requirements": payload.get("requirements") or [],
            "reason": payload.get("reason"),
            "approved": bool(payload.get("approved")),
            "lock": payload.get("lock") or {},
            "policy_decision": payload.get("policy_decision") or {},
            "metadata": payload.get("metadata") or {},
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cp_environment_overlays (
                    overlay_id, base_environment_id, experiment_id, assignment_id,
                    session_id, status, requested_by_agent_id, python_path,
                    root_path, created_at, updated_at, requirements_json,
                    reason, approved, lock_json, policy_decision_json,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["overlay_id"],
                    record["base_environment_id"],
                    record["experiment_id"],
                    record["assignment_id"],
                    record["session_id"],
                    record["status"],
                    record["requested_by_agent_id"],
                    record["python_path"],
                    record["root_path"],
                    record["created_at"],
                    record["updated_at"],
                    json.dumps(record["requirements"], sort_keys=True),
                    record["reason"],
                    1 if record["approved"] else 0,
                    _json(record["lock"]),
                    _json(record["policy_decision"]),
                    _json(record["metadata"]),
                ),
            )
        return record

    def update_environment_overlay(self, overlay_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        current = self.get_environment_overlay(overlay_id)
        if current is None:
            raise KeyError(overlay_id)
        now = isoformat_z()
        requirements = current.get("requirements") if "requirements" not in payload else payload.get("requirements")
        lock = current.get("lock") if "lock" not in payload else payload.get("lock")
        policy_decision = current.get("policy_decision") if "policy_decision" not in payload else payload.get("policy_decision")
        metadata = {**(current.get("metadata") or {}), **(payload.get("metadata") or {})}
        approved = current.get("approved") if "approved" not in payload else bool(payload.get("approved"))
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE cp_environment_overlays
                SET status = ?,
                    python_path = COALESCE(?, python_path),
                    root_path = COALESCE(?, root_path),
                    updated_at = ?,
                    requirements_json = ?,
                    reason = COALESCE(?, reason),
                    approved = ?,
                    lock_json = ?,
                    policy_decision_json = ?,
                    metadata_json = ?
                WHERE overlay_id = ?
                """,
                (
                    payload.get("status") or current["status"],
                    payload.get("python_path"),
                    payload.get("root_path"),
                    payload.get("updated_at") or now,
                    json.dumps(requirements or [], sort_keys=True),
                    payload.get("reason"),
                    1 if approved else 0,
                    _json(lock),
                    _json(policy_decision),
                    _json(metadata),
                    overlay_id,
                ),
            )
        record = self.get_environment_overlay(overlay_id)
        assert record is not None
        return record

    def get_environment_overlay(self, overlay_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM cp_environment_overlays WHERE overlay_id = ?", (overlay_id,)).fetchone()
        return None if row is None else self._row_environment_overlay(row)

    def list_environment_overlays(
        self,
        *,
        base_environment_id: str | None = None,
        assignment_id: str | None = None,
        experiment_id: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM cp_environment_overlays"
        clauses: list[str] = []
        params: list[Any] = []
        if base_environment_id:
            clauses.append("base_environment_id = ?")
            params.append(base_environment_id)
        if assignment_id:
            clauses.append("assignment_id = ?")
            params.append(assignment_id)
        if experiment_id:
            clauses.append("experiment_id = ?")
            params.append(experiment_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC, overlay_id DESC"
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [self._row_environment_overlay(row) for row in rows]

    def create_artifact(self, payload: dict[str, Any]) -> dict[str, Any]:
        now = isoformat_z()
        record = {
            "artifact_id": payload.get("artifact_id") or make_run_id("artifact"),
            "experiment_id": payload.get("experiment_id"),
            "assignment_id": payload.get("assignment_id"),
            "attempt_id": payload.get("attempt_id"),
            "kind": payload.get("kind") or "generic",
            "uri": payload["uri"],
            "local_path": payload.get("local_path"),
            "digest": payload.get("digest"),
            "created_at": payload.get("created_at") or now,
            "metadata": payload.get("metadata") or {},
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cp_artifacts (
                    artifact_id, experiment_id, assignment_id, attempt_id, kind,
                    uri, local_path, digest, created_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["artifact_id"],
                    record["experiment_id"],
                    record["assignment_id"],
                    record["attempt_id"],
                    record["kind"],
                    record["uri"],
                    record["local_path"],
                    record["digest"],
                    record["created_at"],
                    _json(record["metadata"]),
                ),
            )
        return record

    def list_artifacts(self, *, experiment_id: str | None = None, assignment_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM cp_artifacts"
        clauses: list[str] = []
        params: list[Any] = []
        if experiment_id:
            clauses.append("experiment_id = ?")
            params.append(experiment_id)
        if assignment_id:
            clauses.append("assignment_id = ?")
            params.append(assignment_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC, artifact_id DESC"
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [self._row_artifact(row) for row in rows]

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM cp_artifacts WHERE artifact_id = ?", (artifact_id,)).fetchone()
        return None if row is None else self._row_artifact(row)

    def create_evaluation(self, payload: dict[str, Any]) -> dict[str, Any]:
        now = isoformat_z()
        record = {
            "evaluation_id": payload.get("evaluation_id") or make_run_id("eval"),
            "experiment_id": payload["experiment_id"],
            "assignment_id": payload.get("assignment_id"),
            "attempt_id": payload.get("attempt_id"),
            "artifact_id": payload.get("artifact_id"),
            "job_id": payload.get("job_id"),
            "kind": payload.get("kind") or "official",
            "status": payload.get("status") or "completed",
            "valid": payload.get("valid"),
            "score": payload.get("score"),
            "created_at": payload.get("created_at") or now,
            "updated_at": payload.get("updated_at") or now,
            "request": payload.get("request") or {},
            "result": payload.get("result") or {},
            "public_feedback": payload.get("public_feedback") or {},
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cp_evaluations (
                    evaluation_id, experiment_id, assignment_id, attempt_id,
                    artifact_id, job_id, kind, status, valid, score,
                    created_at, updated_at, request_json, result_json,
                    public_feedback_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["evaluation_id"],
                    record["experiment_id"],
                    record["assignment_id"],
                    record["attempt_id"],
                    record["artifact_id"],
                    record["job_id"],
                    record["kind"],
                    record["status"],
                    1 if record["valid"] else 0 if record["valid"] is not None else None,
                    record["score"],
                    record["created_at"],
                    record["updated_at"],
                    _json(record["request"]),
                    _json(record["result"]),
                    _json(record["public_feedback"]),
                ),
            )
        return record

    def list_evaluations(self, *, experiment_id: str | None = None, assignment_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM cp_evaluations"
        clauses: list[str] = []
        params: list[Any] = []
        if experiment_id:
            clauses.append("experiment_id = ?")
            params.append(experiment_id)
        if assignment_id:
            clauses.append("assignment_id = ?")
            params.append(assignment_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC, evaluation_id DESC"
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [self._row_evaluation(row) for row in rows]

    def get_evaluation(self, evaluation_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM cp_evaluations WHERE evaluation_id = ?", (evaluation_id,)).fetchone()
        return None if row is None else self._row_evaluation(row)

    def update_evaluation(self, evaluation_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        current = self.get_evaluation(evaluation_id)
        if current is None:
            raise KeyError(evaluation_id)
        now = isoformat_z()
        request_payload = current.get("request") if "request" not in payload else payload.get("request")
        result_payload = current.get("result") if "result" not in payload else payload.get("result")
        public_feedback = current.get("public_feedback") if "public_feedback" not in payload else payload.get("public_feedback")
        valid = current.get("valid") if "valid" not in payload else payload.get("valid")
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE cp_evaluations
                SET artifact_id = COALESCE(?, artifact_id),
                    job_id = COALESCE(?, job_id),
                    status = ?,
                    valid = ?,
                    score = ?,
                    updated_at = ?,
                    request_json = ?,
                    result_json = ?,
                    public_feedback_json = ?
                WHERE evaluation_id = ?
                """,
                (
                    payload.get("artifact_id"),
                    payload.get("job_id"),
                    payload.get("status") or current["status"],
                    1 if valid else 0 if valid is not None else None,
                    current.get("score") if "score" not in payload else payload.get("score"),
                    payload.get("updated_at") or now,
                    _json(request_payload),
                    _json(result_payload),
                    _json(public_feedback),
                    evaluation_id,
                ),
            )
        record = self.get_evaluation(evaluation_id)
        assert record is not None
        return record

    def create_leaderboard_entry(self, payload: dict[str, Any]) -> dict[str, Any]:
        now = isoformat_z()
        assignment_id = payload.get("assignment_id")
        assignment = self.get_assignment(assignment_id) if assignment_id else None
        record = {
            "leaderboard_entry_id": payload.get("leaderboard_entry_id") or make_run_id("leader"),
            "experiment_id": payload["experiment_id"],
            "task_id": payload["task_id"],
            "assignment_id": assignment_id,
            "direction_id": payload.get("direction_id") or (assignment or {}).get("direction_id"),
            "evaluation_id": payload["evaluation_id"],
            "artifact_id": payload.get("artifact_id"),
            "score": float(payload["score"]),
            "status": payload.get("status") or "active",
            "environment_id": payload.get("environment_id"),
            "environment_overlay_id": payload.get("environment_overlay_id"),
            "created_at": payload.get("created_at") or now,
            "updated_at": payload.get("updated_at") or now,
            "metadata": payload.get("metadata") or {},
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cp_leaderboard_entries (
                    leaderboard_entry_id, experiment_id, task_id, assignment_id,
                    direction_id, evaluation_id, artifact_id, score, status,
                    environment_id, environment_overlay_id, created_at,
                    updated_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(evaluation_id) DO UPDATE SET
                    artifact_id = excluded.artifact_id,
                    score = excluded.score,
                    status = excluded.status,
                    environment_id = excluded.environment_id,
                    environment_overlay_id = excluded.environment_overlay_id,
                    updated_at = excluded.updated_at,
                    metadata_json = excluded.metadata_json
                """,
                (
                    record["leaderboard_entry_id"],
                    record["experiment_id"],
                    record["task_id"],
                    record["assignment_id"],
                    record["direction_id"],
                    record["evaluation_id"],
                    record["artifact_id"],
                    record["score"],
                    record["status"],
                    record["environment_id"],
                    record["environment_overlay_id"],
                    record["created_at"],
                    record["updated_at"],
                    _json(record["metadata"]),
                ),
            )
        existing = self.get_leaderboard_entry_for_evaluation(record["evaluation_id"])
        assert existing is not None
        return existing

    def get_leaderboard_entry(self, leaderboard_entry_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM cp_leaderboard_entries WHERE leaderboard_entry_id = ?",
                (leaderboard_entry_id,),
            ).fetchone()
        return None if row is None else self._row_leaderboard_entry(row)

    def get_leaderboard_entry_for_evaluation(self, evaluation_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM cp_leaderboard_entries WHERE evaluation_id = ?",
                (evaluation_id,),
            ).fetchone()
        return None if row is None else self._row_leaderboard_entry(row)

    def list_leaderboard_entries(
        self,
        *,
        experiment_id: str | None = None,
        task_id: str | None = None,
        direction_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM cp_leaderboard_entries WHERE status = 'active'"
        params: list[Any] = []
        if experiment_id:
            query += " AND experiment_id = ?"
            params.append(experiment_id)
        if task_id:
            query += " AND task_id = ?"
            params.append(task_id)
        if direction_id:
            query += " AND direction_id = ?"
            params.append(direction_id)
        query += " ORDER BY score DESC, updated_at ASC, leaderboard_entry_id ASC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [self._row_leaderboard_entry(row) for row in rows]

    def get_incumbent(
        self,
        *,
        experiment_id: str | None = None,
        task_id: str | None = None,
        direction_id: str | None = None,
    ) -> dict[str, Any] | None:
        entries = self.list_leaderboard_entries(
            experiment_id=experiment_id,
            task_id=task_id,
            direction_id=direction_id,
            limit=1,
        )
        return entries[0] if entries else None

    def create_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        now = isoformat_z()
        record = {
            "job_id": payload.get("job_id") or make_run_id("job"),
            "experiment_id": payload["experiment_id"],
            "assignment_id": payload.get("assignment_id"),
            "session_id": payload.get("session_id"),
            "provider": payload.get("provider") or "local",
            "status": payload.get("status") or "queued",
            "pid": payload.get("pid"),
            "created_at": payload.get("created_at") or now,
            "updated_at": payload.get("updated_at") or now,
            "inputs": payload.get("inputs") or {},
            "outputs": payload.get("outputs") or {},
            "cost": payload.get("cost") or {},
            "details": payload.get("details") or {},
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cp_jobs (
                    job_id, experiment_id, assignment_id, session_id, provider,
                    status, pid, created_at, updated_at, inputs_json,
                    outputs_json, cost_json, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["job_id"],
                    record["experiment_id"],
                    record["assignment_id"],
                    record["session_id"],
                    record["provider"],
                    record["status"],
                    record["pid"],
                    record["created_at"],
                    record["updated_at"],
                    _json(record["inputs"]),
                    _json(record["outputs"]),
                    _json(record["cost"]),
                    _json(record["details"]),
                ),
            )
        return record

    def list_jobs(self, *, experiment_id: str | None = None, assignment_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM cp_jobs"
        clauses: list[str] = []
        params: list[Any] = []
        if experiment_id:
            clauses.append("experiment_id = ?")
            params.append(experiment_id)
        if assignment_id:
            clauses.append("assignment_id = ?")
            params.append(assignment_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC, job_id DESC"
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [self._row_job(row) for row in rows]

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM cp_jobs WHERE job_id = ?", (job_id,)).fetchone()
        return None if row is None else self._row_job(row)

    def update_job(self, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        current = self.get_job(job_id)
        if current is None:
            raise KeyError(job_id)
        now = isoformat_z()
        inputs = current.get("inputs") if "inputs" not in payload else payload.get("inputs")
        outputs = current.get("outputs") if "outputs" not in payload else payload.get("outputs")
        cost = current.get("cost") if "cost" not in payload else payload.get("cost")
        details = {**(current.get("details") or {}), **(payload.get("details") or {})}
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE cp_jobs
                SET status = ?,
                    pid = COALESCE(?, pid),
                    updated_at = ?,
                    inputs_json = ?,
                    outputs_json = ?,
                    cost_json = ?,
                    details_json = ?
                WHERE job_id = ?
                """,
                (
                    payload.get("status") or current["status"],
                    payload.get("pid"),
                    payload.get("updated_at") or now,
                    _json(inputs),
                    _json(outputs),
                    _json(cost),
                    _json(details),
                    job_id,
                ),
            )
        record = self.get_job(job_id)
        assert record is not None
        return record

    def share_finding(self, payload: dict[str, Any]) -> dict[str, Any]:
        now = isoformat_z()
        record = {
            "finding_id": payload.get("finding_id") or make_run_id("finding"),
            "experiment_id": payload.get("experiment_id"),
            "assignment_id": payload.get("assignment_id"),
            "task_id": payload["task_id"],
            "finding_type": payload.get("finding_type") or "insight",
            "title": payload["title"],
            "body": payload["body"],
            "created_at": payload.get("created_at") or now,
            "metrics": payload.get("metrics") or {},
            "links": payload.get("links") or {},
            "metadata": payload.get("metadata") or {},
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cp_findings (
                    finding_id, experiment_id, assignment_id, task_id,
                    finding_type, title, body, created_at, metrics_json,
                    links_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["finding_id"],
                    record["experiment_id"],
                    record["assignment_id"],
                    record["task_id"],
                    record["finding_type"],
                    record["title"],
                    record["body"],
                    record["created_at"],
                    _json(record["metrics"]),
                    _json(record["links"]),
                    _json(record["metadata"]),
                ),
            )
        return record

    def list_findings(self, *, task_id: str | None = None, experiment_id: str | None = None, query: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM cp_findings"
        clauses: list[str] = []
        params: list[Any] = []
        if task_id:
            clauses.append("task_id = ?")
            params.append(task_id)
        if experiment_id:
            clauses.append("experiment_id = ?")
            params.append(experiment_id)
        if query:
            clauses.append("(LOWER(title) LIKE ? OR LOWER(body) LIKE ?)")
            like = f"%{query.lower()}%"
            params.extend([like, like])
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC, finding_id DESC"
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [self._row_finding(row) for row in rows]

    def checkpoint_notebook(self, payload: dict[str, Any]) -> dict[str, Any]:
        now = isoformat_z()
        assignment = self.get_assignment(payload["assignment_id"])
        if assignment is None:
            raise KeyError(payload["assignment_id"])
        record = {
            "checkpoint_id": payload.get("checkpoint_id") or make_run_id("notebook"),
            "experiment_id": payload.get("experiment_id") or assignment["experiment_id"],
            "assignment_id": assignment["assignment_id"],
            "session_id": payload.get("session_id"),
            "agent_id": payload.get("agent_id") or assignment["agent_id"],
            "created_at": payload.get("created_at") or now,
            "notebook_uri": payload.get("notebook_uri"),
            "content": payload.get("content"),
            "metadata": payload.get("metadata") or {},
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cp_notebook_checkpoints (
                    checkpoint_id, experiment_id, assignment_id, session_id,
                    agent_id, created_at, notebook_uri, content, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["checkpoint_id"],
                    record["experiment_id"],
                    record["assignment_id"],
                    record["session_id"],
                    record["agent_id"],
                    record["created_at"],
                    record["notebook_uri"],
                    record["content"],
                    _json(record["metadata"]),
                ),
            )
        return record

    def list_notebook_checkpoints(self, *, assignment_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM cp_notebook_checkpoints
                WHERE assignment_id = ?
                ORDER BY created_at DESC, checkpoint_id DESC
                """,
                (assignment_id,),
            ).fetchall()
        return [self._row_notebook(row) for row in rows]

    def record_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        now = isoformat_z()
        record = {
            "event_id": payload.get("event_id") or make_event_id(),
            "experiment_id": payload.get("experiment_id"),
            "assignment_id": payload.get("assignment_id"),
            "session_id": payload.get("session_id"),
            "task_id": payload.get("task_id"),
            "agent_id": payload.get("agent_id"),
            "event_type": payload["event_type"],
            "timestamp": payload.get("timestamp") or now,
            "summary": payload.get("summary"),
            "payload": payload.get("payload") or {},
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cp_events (
                    event_id, experiment_id, assignment_id, session_id, task_id,
                    agent_id, event_type, timestamp, summary, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["event_id"],
                    record["experiment_id"],
                    record["assignment_id"],
                    record["session_id"],
                    record["task_id"],
                    record["agent_id"],
                    record["event_type"],
                    record["timestamp"],
                    record["summary"],
                    _json(record["payload"]),
                ),
            )
        return record

    def create_telemetry_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        now = isoformat_z()
        record = {
            "telemetry_id": payload.get("telemetry_id") or make_run_id("telemetry"),
            "experiment_id": payload["experiment_id"],
            "assignment_id": payload.get("assignment_id"),
            "session_id": payload.get("session_id"),
            "job_id": payload.get("job_id"),
            "attempt_id": payload.get("attempt_id"),
            "artifact_id": payload.get("artifact_id"),
            "provider": payload.get("provider") or "local",
            "status": payload.get("status") or "running",
            "run_name": payload.get("run_name"),
            "dashboard_url": payload.get("dashboard_url"),
            "external_run_id": payload.get("external_run_id"),
            "created_at": payload.get("created_at") or now,
            "updated_at": payload.get("updated_at") or now,
            "metrics": payload.get("metrics") or {},
            "params": payload.get("params") or {},
            "tags": payload.get("tags") or {},
            "artifacts": payload.get("artifacts") or {},
            "details": payload.get("details") or {},
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cp_telemetry_runs (
                    telemetry_id, experiment_id, assignment_id, session_id,
                    job_id, attempt_id, artifact_id, provider, status, run_name,
                    dashboard_url, external_run_id, created_at, updated_at,
                    metrics_json, params_json, tags_json, artifacts_json,
                    details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["telemetry_id"],
                    record["experiment_id"],
                    record["assignment_id"],
                    record["session_id"],
                    record["job_id"],
                    record["attempt_id"],
                    record["artifact_id"],
                    record["provider"],
                    record["status"],
                    record["run_name"],
                    record["dashboard_url"],
                    record["external_run_id"],
                    record["created_at"],
                    record["updated_at"],
                    _json(record["metrics"]),
                    _json(record["params"]),
                    _json(record["tags"]),
                    _json(record["artifacts"]),
                    _json(record["details"]),
                ),
            )
        return record

    def update_telemetry_run(self, telemetry_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        current = self.get_telemetry_run(telemetry_id)
        if current is None:
            raise KeyError(telemetry_id)
        now = isoformat_z()
        metrics = {**(current.get("metrics") or {}), **(payload.get("metrics") or {})}
        params = {**(current.get("params") or {}), **(payload.get("params") or {})}
        tags = {**(current.get("tags") or {}), **(payload.get("tags") or {})}
        artifacts = {**(current.get("artifacts") or {}), **(payload.get("artifacts") or {})}
        details = {**(current.get("details") or {}), **(payload.get("details") or {})}
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE cp_telemetry_runs
                SET status = ?,
                    run_name = COALESCE(?, run_name),
                    dashboard_url = COALESCE(?, dashboard_url),
                    external_run_id = COALESCE(?, external_run_id),
                    updated_at = ?,
                    metrics_json = ?,
                    params_json = ?,
                    tags_json = ?,
                    artifacts_json = ?,
                    details_json = ?
                WHERE telemetry_id = ?
                """,
                (
                    payload.get("status") or current["status"],
                    payload.get("run_name"),
                    payload.get("dashboard_url"),
                    payload.get("external_run_id"),
                    payload.get("updated_at") or now,
                    _json(metrics),
                    _json(params),
                    _json(tags),
                    _json(artifacts),
                    _json(details),
                    telemetry_id,
                ),
            )
        record = self.get_telemetry_run(telemetry_id)
        assert record is not None
        return record

    def get_telemetry_run(self, telemetry_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM cp_telemetry_runs WHERE telemetry_id = ?", (telemetry_id,)).fetchone()
        return None if row is None else self._row_telemetry(row)

    def list_telemetry_runs(
        self,
        *,
        experiment_id: str | None = None,
        assignment_id: str | None = None,
        job_id: str | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM cp_telemetry_runs"
        clauses: list[str] = []
        params: list[Any] = []
        if experiment_id:
            clauses.append("experiment_id = ?")
            params.append(experiment_id)
        if assignment_id:
            clauses.append("assignment_id = ?")
            params.append(assignment_id)
        if job_id:
            clauses.append("job_id = ?")
            params.append(job_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC, telemetry_id DESC"
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [self._row_telemetry(row) for row in rows]

    def list_events(self, *, experiment_id: str | None = None, assignment_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        query = "SELECT * FROM cp_events"
        clauses: list[str] = []
        params: list[Any] = []
        if experiment_id:
            clauses.append("experiment_id = ?")
            params.append(experiment_id)
        if assignment_id:
            clauses.append("assignment_id = ?")
            params.append(assignment_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [self._row_event(row) for row in rows]

    def context_for_assignment(self, assignment_id: str) -> dict[str, Any]:
        assignment = self.get_assignment(assignment_id)
        if assignment is None:
            raise KeyError(assignment_id)
        experiment = self.get_experiment(assignment["experiment_id"])
        return {
            "assignment": assignment,
            "experiment": experiment,
            "sessions": self.list_sessions(assignment_id=assignment_id),
            "recent_findings": self.list_findings(task_id=assignment["task_id"])[:20],
            "artifacts": self.list_artifacts(assignment_id=assignment_id),
            "evaluations": self.list_evaluations(assignment_id=assignment_id),
            "jobs": self.list_jobs(assignment_id=assignment_id),
            "research_direction": (assignment.get("metadata") or {}).get("research_direction"),
            "leaderboard": self.list_leaderboard_entries(experiment_id=assignment["experiment_id"], limit=10),
            "incumbent": self.get_incumbent(experiment_id=assignment["experiment_id"]),
            "direction_incumbent": (
                self.get_incumbent(experiment_id=assignment["experiment_id"], direction_id=assignment["direction_id"])
                if assignment.get("direction_id")
                else None
            ),
            "environments": self.list_environments(task_id=assignment["task_id"]),
            "environment_overlays": self.list_environment_overlays(assignment_id=assignment_id),
            "telemetry_runs": self.list_telemetry_runs(assignment_id=assignment_id),
            "notebook_checkpoints": self.list_notebook_checkpoints(assignment_id=assignment_id),
        }

    def _row_experiment(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "experiment_id": row["experiment_id"],
            "task_id": row["task_id"],
            "status": row["status"],
            "mode": row["mode"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "config": _loads(row["config_json"], {}),
            "budget": _loads(row["budget_json"], {}),
            "policy": _loads(row["policy_json"], {}),
            "metadata": _loads(row["metadata_json"], {}),
        }

    def _row_assignment(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "assignment_id": row["assignment_id"],
            "experiment_id": row["experiment_id"],
            "task_id": row["task_id"],
            "agent_id": row["agent_id"],
            "direction_id": row["direction_id"],
            "status": row["status"],
            "worker_backend": row["worker_backend"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "visible_context_policy": _loads(row["visible_context_policy_json"], {}),
            "budget": _loads(row["budget_json"], {}),
            "workspace_seed": _loads(row["workspace_seed_json"], {}),
            "metadata": _loads(row["metadata_json"], {}),
        }

    def _row_session(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "session_id": row["session_id"],
            "assignment_id": row["assignment_id"],
            "experiment_id": row["experiment_id"],
            "task_id": row["task_id"],
            "agent_id": row["agent_id"],
            "status": row["status"],
            "worker_backend": row["worker_backend"],
            "pid": row["pid"],
            "started_at": row["started_at"],
            "ended_at": row["ended_at"],
            "updated_at": row["updated_at"],
            "workspace_path": row["workspace_path"],
            "details": _loads(row["details_json"], {}),
        }

    def _row_job(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "job_id": row["job_id"],
            "experiment_id": row["experiment_id"],
            "assignment_id": row["assignment_id"],
            "session_id": row["session_id"],
            "provider": row["provider"],
            "status": row["status"],
            "pid": row["pid"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "inputs": _loads(row["inputs_json"], {}),
            "outputs": _loads(row["outputs_json"], {}),
            "cost": _loads(row["cost_json"], {}),
            "details": _loads(row["details_json"], {}),
        }

    def _row_environment(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "environment_id": row["environment_id"],
            "environment_type": row["environment_type"],
            "status": row["status"],
            "fingerprint": row["fingerprint"],
            "python_path": row["python_path"],
            "root_path": row["root_path"],
            "task_id": row["task_id"],
            "experiment_id": row["experiment_id"],
            "parent_environment_id": row["parent_environment_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "spec": _loads(row["spec_json"], {}),
            "lock": _loads(row["lock_json"], {}),
            "metadata": _loads(row["metadata_json"], {}),
        }

    def _row_environment_overlay(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "overlay_id": row["overlay_id"],
            "base_environment_id": row["base_environment_id"],
            "experiment_id": row["experiment_id"],
            "assignment_id": row["assignment_id"],
            "session_id": row["session_id"],
            "status": row["status"],
            "requested_by_agent_id": row["requested_by_agent_id"],
            "python_path": row["python_path"],
            "root_path": row["root_path"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "requirements": _loads(row["requirements_json"], []),
            "reason": row["reason"],
            "approved": bool(row["approved"]),
            "lock": _loads(row["lock_json"], {}),
            "policy_decision": _loads(row["policy_decision_json"], {}),
            "metadata": _loads(row["metadata_json"], {}),
        }

    def _row_artifact(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "artifact_id": row["artifact_id"],
            "experiment_id": row["experiment_id"],
            "assignment_id": row["assignment_id"],
            "attempt_id": row["attempt_id"],
            "kind": row["kind"],
            "uri": row["uri"],
            "local_path": row["local_path"],
            "digest": row["digest"],
            "created_at": row["created_at"],
            "metadata": _loads(row["metadata_json"], {}),
        }

    def _row_evaluation(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "evaluation_id": row["evaluation_id"],
            "experiment_id": row["experiment_id"],
            "assignment_id": row["assignment_id"],
            "attempt_id": row["attempt_id"],
            "artifact_id": row["artifact_id"],
            "job_id": row["job_id"],
            "kind": row["kind"],
            "status": row["status"],
            "valid": bool(row["valid"]) if row["valid"] is not None else None,
            "score": row["score"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "request": _loads(row["request_json"], {}),
            "result": _loads(row["result_json"], {}),
            "public_feedback": _loads(row["public_feedback_json"], {}),
        }

    def _row_leaderboard_entry(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "leaderboard_entry_id": row["leaderboard_entry_id"],
            "experiment_id": row["experiment_id"],
            "task_id": row["task_id"],
            "assignment_id": row["assignment_id"],
            "direction_id": row["direction_id"],
            "evaluation_id": row["evaluation_id"],
            "artifact_id": row["artifact_id"],
            "score": row["score"],
            "status": row["status"],
            "environment_id": row["environment_id"],
            "environment_overlay_id": row["environment_overlay_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "metadata": _loads(row["metadata_json"], {}),
        }

    def _row_finding(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "finding_id": row["finding_id"],
            "experiment_id": row["experiment_id"],
            "assignment_id": row["assignment_id"],
            "task_id": row["task_id"],
            "finding_type": row["finding_type"],
            "title": row["title"],
            "body": row["body"],
            "created_at": row["created_at"],
            "metrics": _loads(row["metrics_json"], {}),
            "links": _loads(row["links_json"], {}),
            "metadata": _loads(row["metadata_json"], {}),
        }

    def _row_notebook(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "checkpoint_id": row["checkpoint_id"],
            "experiment_id": row["experiment_id"],
            "assignment_id": row["assignment_id"],
            "session_id": row["session_id"],
            "agent_id": row["agent_id"],
            "created_at": row["created_at"],
            "notebook_uri": row["notebook_uri"],
            "content": row["content"],
            "metadata": _loads(row["metadata_json"], {}),
        }

    def _row_event(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "event_id": row["event_id"],
            "experiment_id": row["experiment_id"],
            "assignment_id": row["assignment_id"],
            "session_id": row["session_id"],
            "task_id": row["task_id"],
            "agent_id": row["agent_id"],
            "event_type": row["event_type"],
            "timestamp": row["timestamp"],
            "summary": row["summary"],
            "payload": _loads(row["payload_json"], {}),
        }

    def _row_telemetry(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "telemetry_id": row["telemetry_id"],
            "experiment_id": row["experiment_id"],
            "assignment_id": row["assignment_id"],
            "session_id": row["session_id"],
            "job_id": row["job_id"],
            "attempt_id": row["attempt_id"],
            "artifact_id": row["artifact_id"],
            "provider": row["provider"],
            "status": row["status"],
            "run_name": row["run_name"],
            "dashboard_url": row["dashboard_url"],
            "external_run_id": row["external_run_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "metrics": _loads(row["metrics_json"], {}),
            "params": _loads(row["params_json"], {}),
            "tags": _loads(row["tags_json"], {}),
            "artifacts": _loads(row["artifacts_json"], {}),
            "details": _loads(row["details_json"], {}),
        }
