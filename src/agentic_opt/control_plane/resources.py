from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TaskResource:
    task_id: str
    title: str
    public_context: dict[str, Any] = field(default_factory=dict)
    candidate_contract: dict[str, Any] = field(default_factory=dict)
    validation_contract: dict[str, Any] = field(default_factory=dict)
    probe_contract: dict[str, Any] = field(default_factory=dict)
    evaluation_contract: dict[str, Any] = field(default_factory=dict)
    runtime_policy: dict[str, Any] = field(default_factory=dict)
    artifact_policy: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentResource:
    experiment_id: str
    task_id: str
    status: str
    mode: str
    config: dict[str, Any] = field(default_factory=dict)
    budget: dict[str, Any] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkerAssignmentResource:
    assignment_id: str
    experiment_id: str
    task_id: str
    agent_id: str
    status: str
    worker_backend: str
    direction_id: str | None = None
    visible_context_policy: dict[str, Any] = field(default_factory=dict)
    budget: dict[str, Any] = field(default_factory=dict)
    workspace_seed: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkerSessionResource:
    session_id: str
    assignment_id: str
    experiment_id: str
    task_id: str
    agent_id: str
    status: str
    worker_backend: str
    workspace_path: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JobResource:
    job_id: str
    experiment_id: str
    provider: str
    status: str
    assignment_id: str | None = None
    session_id: str | None = None
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EnvironmentResource:
    environment_id: str
    environment_type: str
    status: str
    fingerprint: str
    python_path: str
    root_path: str
    task_id: str | None = None
    experiment_id: str | None = None
    parent_environment_id: str | None = None
    spec: dict[str, Any] = field(default_factory=dict)
    lock: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EnvironmentOverlayResource:
    overlay_id: str
    base_environment_id: str
    status: str
    experiment_id: str | None = None
    assignment_id: str | None = None
    session_id: str | None = None
    requested_by_agent_id: str | None = None
    python_path: str | None = None
    root_path: str | None = None
    requirements: list[str] = field(default_factory=list)
    reason: str | None = None
    approved: bool = False
    lock: dict[str, Any] = field(default_factory=dict)
    policy_decision: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationResource:
    evaluation_id: str
    experiment_id: str
    kind: str
    status: str
    assignment_id: str | None = None
    attempt_id: str | None = None
    artifact_id: str | None = None
    job_id: str | None = None
    valid: bool | None = None
    score: float | None = None
    request: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    public_feedback: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LeaderboardEntryResource:
    leaderboard_entry_id: str
    experiment_id: str
    task_id: str
    evaluation_id: str
    score: float
    status: str
    assignment_id: str | None = None
    direction_id: str | None = None
    artifact_id: str | None = None
    environment_id: str | None = None
    environment_overlay_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TelemetryRunResource:
    telemetry_id: str
    experiment_id: str
    provider: str
    status: str
    assignment_id: str | None = None
    session_id: str | None = None
    job_id: str | None = None
    attempt_id: str | None = None
    artifact_id: str | None = None
    run_name: str | None = None
    dashboard_url: str | None = None
    external_run_id: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactResource:
    artifact_id: str
    kind: str
    uri: str
    experiment_id: str | None = None
    assignment_id: str | None = None
    attempt_id: str | None = None
    local_path: str | None = None
    digest: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FindingResource:
    finding_id: str
    task_id: str
    finding_type: str
    title: str
    body: str
    experiment_id: str | None = None
    assignment_id: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    links: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NotebookCheckpointResource:
    checkpoint_id: str
    experiment_id: str
    assignment_id: str
    agent_id: str
    session_id: str | None = None
    notebook_uri: str | None = None
    content: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CURRENT_RESOURCE_MAPPING: dict[str, str] = {
    "Task": "task package loaded through configured task roots; server API exposes the semantic task contract",
    "Experiment": "control-plane experiments table",
    "WorkerAssignment": "new assignment row generated by server from an Experiment",
    "WorkerSession": "new session row attached to a WorkerAssignment; Codex App Server sessions are one backend detail",
    "Attempt": "candidate artifact snapshot linked to Artifact and Evaluation resources",
    "Job": "new durable compute record owned by server; provider-specific execution is an adapter",
    "Environment": "server-owned Python runtime prepared from framework or task specs; evaluations and workers run through it",
    "EnvironmentOverlay": "worker-requested dependency extension layered on an Environment under experiment policy",
    "Evaluation": "new server-side scoring record; task evaluator behavior is an implementation detail, not the control plane",
    "LeaderboardEntry": "server-owned official score row; highest entry is the current incumbent for a scope",
    "TelemetryRun": "non-official observability record linked to jobs/artifacts without affecting official scoring",
    "Artifact": "new artifact registry row plus local/object-store blob URI",
    "Finding": "server-visible research claim; historical patterns are findings, usually with finding_type='pattern'",
    "NotebookCheckpoint": "server-owned checkpoint for worker notebook or worklog state",
}


def object_model_schema() -> dict[str, Any]:
    return {
        "resources": {
            "Task": list(TaskResource.__dataclass_fields__),
            "Experiment": list(ExperimentResource.__dataclass_fields__),
            "WorkerAssignment": list(WorkerAssignmentResource.__dataclass_fields__),
            "WorkerSession": list(WorkerSessionResource.__dataclass_fields__),
            "Job": list(JobResource.__dataclass_fields__),
            "Environment": list(EnvironmentResource.__dataclass_fields__),
            "EnvironmentOverlay": list(EnvironmentOverlayResource.__dataclass_fields__),
            "Evaluation": list(EvaluationResource.__dataclass_fields__),
            "LeaderboardEntry": list(LeaderboardEntryResource.__dataclass_fields__),
            "TelemetryRun": list(TelemetryRunResource.__dataclass_fields__),
            "Artifact": list(ArtifactResource.__dataclass_fields__),
            "Finding": list(FindingResource.__dataclass_fields__),
            "NotebookCheckpoint": list(NotebookCheckpointResource.__dataclass_fields__),
        },
        "current_resource_mapping": CURRENT_RESOURCE_MAPPING,
    }
