from __future__ import annotations

import os
import re
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path

from agentic_opt.common.atomic import atomic_write_text
from agentic_opt.common.config import get_repo_root
from agentic_opt.common.runtime_env import PreparedRuntimeEnv, prepare_task_runtime
from agentic_opt.task_api import CandidateSpec, candidate_entry_path, candidate_spec_for
from agentic_opt.task_registry import get_task


@dataclass(frozen=True)
class SemanticWorkspace:
    root: Path
    entry_path: Path
    agents_md_path: Path
    skills_root: Path
    bin_dir: Path
    env: dict[str, str]
    readable_roots: list[str]
    writable_roots: list[str]


def prepare_semantic_workspace(
    *,
    workspace_root: Path,
    api_url: str,
    assignment: dict,
    session_id: str,
    runtime_env: PreparedRuntimeEnv | None = None,
) -> SemanticWorkspace:
    workspace_root = workspace_root.resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)
    task_id = assignment["task_id"]
    task = get_task(task_id)
    runtime_env = runtime_env or prepare_task_runtime(task)
    candidate_spec = candidate_spec_for(task)
    entry_path = candidate_entry_path(workspace_root=workspace_root, spec=candidate_spec)
    if not entry_path.exists():
        _materialize_candidate_seed(public_dir=task.public_dir, workspace_root=workspace_root, spec=candidate_spec)

    for relative in ("reference", "artifacts", "findings", "local_tools"):
        (workspace_root / relative).mkdir(parents=True, exist_ok=True)
    worklog = workspace_root / "WORKLOG.md"
    if not worklog.exists():
        atomic_write_text(worklog, "# WORKLOG\n\nUse this file for local scratch notes when useful.\n")

    agents_md_path = workspace_root / "AGENTS.md"
    atomic_write_text(agents_md_path, _agents_md_text())

    skills_root = workspace_root / ".agents" / "skills"
    for skill_name, body in SEMANTIC_SKILL_BODIES.items():
        skill_dir = skills_root / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(skill_dir / "SKILL.md", body.strip() + "\n")

    bin_dir = workspace_root / "bin"
    _write_semantic_tool_wrappers(
        bin_dir=bin_dir,
        runtime_env=runtime_env,
        api_url=api_url,
        assignment=assignment,
        session_id=session_id,
        workspace_root=workspace_root,
    )
    environment_exports = _environment_exports(runtime_env)
    repo_src = str(get_repo_root() / "src")
    venv_bin = str(runtime_env.venv_dir / "bin")
    env = {
        "AO_CONTROL_API_URL": api_url,
        "AO_ASSIGNMENT_ID": assignment["assignment_id"],
        "AO_EXPERIMENT_ID": assignment["experiment_id"],
        "AO_TASK_ID": assignment["task_id"],
        "AO_AGENT_ID": assignment["agent_id"],
        "AO_SESSION_ID": session_id,
        "AO_WORKSPACE_ROOT": str(workspace_root),
        **environment_exports,
        **runtime_env.exports(),
        "PATH": _prepend_paths([str(bin_dir), venv_bin], os.environ.get("PATH")),
        "PYTHONPATH": _prepend_path(repo_src, os.environ.get("PYTHONPATH")),
        "VIRTUAL_ENV": str(runtime_env.venv_dir),
    }
    return SemanticWorkspace(
        root=workspace_root,
        entry_path=entry_path,
        agents_md_path=agents_md_path,
        skills_root=skills_root,
        bin_dir=bin_dir,
        env=env,
        readable_roots=[str(workspace_root), str(runtime_env.root)],
        writable_roots=[str(workspace_root)],
    )


def build_semantic_startup_prompt(*, assignment: dict, workspace: SemanticWorkspace) -> str:
    budget = assignment.get("budget") or {}
    direction = (assignment.get("metadata") or {}).get("research_direction") or {}
    direction_text = ""
    if direction:
        direction_text = (
            f"\nResearch direction: {direction.get('title') or direction.get('direction_id')}\n"
            f"Direction id: {direction.get('direction_id')}\n"
            f"Direction note: {direction.get('startup_note') or ''}\n"
        )
    return f"""You are a Coding Agent worker in an autonomous optimization experiment.

Task: {assignment['task_id']}
Experiment: {assignment['experiment_id']}
Assignment: {assignment['assignment_id']}
Workspace: {workspace.root}
{direction_text}

The server owns experiments, assignments, artifacts, jobs, evaluation, findings,
notebook checkpoints, and policy. You own the autonomous research behavior inside
this workspace.

Use semantic server tools:
- ctx context
- ctx task
- ctx findings
- ctx evaluations
- ctx artifacts
- ctx jobs
- ctx leaderboard
- ctx incumbent
- env status
- env install --pip <requirement> --reason <why>
- artifact upload --path <path> --kind <kind>
- artifact checkout-incumbent --destination <path>
- eval verify --entry <candidate-entrypoint>
- eval probe --entry <candidate-entrypoint>
- eval submit --entry <candidate-entrypoint>  # official evaluations are async by default
- eval submit --artifact-id <artifact-id>
- eval status <evaluation-id> / eval wait <evaluation-id>
- finding share --type <type> --title <title> --body <text>
- notebook checkpoint --file WORKLOG.md
- job create --provider local --command '<command>'
- job create --provider local-docker --image <image> --command '<command>'
- job create --provider runpod --template-id <template> --command '<command>'
- job status <job-id> / job logs <job-id> / job wait <job-id>
- telemetry start --provider local --name <run-name>
- telemetry log-metrics <telemetry-id> --metric loss=0.1 --step 1
- telemetry finish <telemetry-id>

These are capabilities, not a fixed workflow. Read context, implement, validate,
evaluate, checkpoint, share findings, or launch jobs when evidence makes that
action useful. Official scores must come from eval submit/server evaluation.

This assignment budget is: {budget}
"""


def _agents_md_text() -> str:
    return """# Agentic Optimization Semantic Workspace

Use the server-owned semantic tools instead of relying on raw archive layout.
Do not assume a fixed numbered workflow.
Use `ctx task` for task objective and public contract.
Use `ctx context` for assignment state, research direction, incumbent, leaderboard, prior findings, artifacts, jobs, evaluations, and notebook checkpoints.
Use `eval verify`, `eval probe`, and `eval submit` for server-owned feedback and official scoring.
Use `env`, `artifact`, `finding`, `notebook`, `job`, and `telemetry` for durable server-visible state.
Do not access hidden evaluator internals.
"""


SEMANTIC_SKILL_BODIES: dict[str, str] = {
    "context-use": """
# Context Use

Use `ctx context` at the start of meaningful work and whenever server state may
have changed. The context contains the assignment, experiment, recent findings,
artifacts, evaluations, jobs, leaderboard, incumbent, research direction, and
notebook checkpoints visible to this worker.

Use `ctx task` when you need the public task contract. Treat server context as
authoritative durable state; local files are just your current workspace.
""",
    "evaluation-use": """
# Evaluation Use

Use `eval verify --entry <path>` for quick validity checks and `eval probe` for
public diagnostics. Use `eval submit --entry <path>` or
`eval submit --artifact-id <artifact-id>` for official scoring.

Official submit evaluations are asynchronous by default. Read the returned
`evaluation_id`, then use `eval status <evaluation-id>` or
`eval wait <evaluation-id>` before relying on the result.

Do not inspect hidden evaluator data or task internals outside the public
contract.
""",
    "artifact-use": """
# Artifact Use

Use `artifact upload --path <path> --kind <kind>` for durable outputs that should
survive the worker session: candidate snapshots, models, logs, plots, datasets,
or result bundles. The server stores metadata and a manifest in the artifact
registry. Use `artifact checkout-incumbent --destination <path>` when you need
to copy the current incumbent candidate into the workspace for inspection or
repair.
""",
    "finding-use": """
# Finding Use

Use `finding share` for reusable knowledge: results, hypotheses, insights,
failure diagnoses, and reusable patterns. A historical pattern is just a finding
with a useful `--type`, such as `pattern` or `insight`.
""",
    "notebook-use": """
# Notebook Use

Use `WORKLOG.md` for local scratch planning and observations. Use
`notebook checkpoint --file WORKLOG.md` when the current state should become
server-visible memory for this assignment or future workers.
""",
    "job-use": """
# Job Use

Use `job create --provider local --command '<command>'` for host subprocess jobs,
or `job create --provider local-docker --image <image> --command '<command>'`
for Docker-backed local jobs. Use `job create --provider runpod` only when the
assignment policy and budget allow cloud execution. Use `job status`, `job logs`,
and `job wait` to inspect durable compute that can outlive the current
coding-agent turn. Future providers should use the same job resource contract.
""",
    "environment-use": """
# Environment Use

The server owns the task runtime environment. Use `env status` to inspect the
active base environment. If your candidate or exploratory tooling needs an
additional pip package, request it with `env install --pip <requirement>
--reason <why>`. This creates a separate overlay; it does not mutate the base
environment used for official scoring.
""",
    "telemetry-use": """
# Telemetry Use

Use `telemetry start`, `telemetry log-metrics`, and `telemetry finish` for
non-official process or training metrics. Telemetry helps diagnose jobs and
compare runs, but official scores must still come from `eval submit`.
""",
}


def _materialize_candidate_seed(*, public_dir: Path, workspace_root: Path, spec: CandidateSpec) -> None:
    if spec.public_seed_root is None:
        source = public_dir / spec.public_entrypoint
        destination = workspace_root / spec.workspace_entrypoint
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return
    source_root = public_dir / spec.public_seed_root
    destination_root = workspace_root / (spec.workspace_candidate_root or Path(spec.entrypoint_name).parent)
    for source in sorted(source_root.rglob("*")):
        relative = source.relative_to(source_root)
        destination = destination_root / relative
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)


def _write_semantic_tool_wrappers(
    *,
    bin_dir: Path,
    runtime_env: PreparedRuntimeEnv,
    api_url: str,
    assignment: dict,
    session_id: str,
    workspace_root: Path,
) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    exports = {
        "AO_CONTROL_API_URL": api_url,
        "AO_ASSIGNMENT_ID": assignment["assignment_id"],
        "AO_EXPERIMENT_ID": assignment["experiment_id"],
        "AO_TASK_ID": assignment["task_id"],
        "AO_AGENT_ID": assignment["agent_id"],
        "AO_SESSION_ID": session_id,
        "AO_WORKSPACE_ROOT": str(workspace_root),
        **_environment_exports(runtime_env),
        **runtime_env.exports(),
        "PYTHONPATH": _prepend_path(str(get_repo_root() / "src"), os.environ.get("PYTHONPATH")),
        "VIRTUAL_ENV": str(runtime_env.venv_dir),
    }
    for command_name in ("ctx", "artifact", "eval", "finding", "notebook", "job", "env", "telemetry"):
        lines = ["#!/bin/sh", "set -eu"]
        for key, value in exports.items():
            lines.append(f"export {key}={shlex.quote(str(value))}")
        lines.append(
            f"exec {shlex.quote(str(runtime_env.python_path))} -m agentic_opt.worker_tools.semantic_cli {shlex.quote(command_name)} \"$@\""
        )
        target = bin_dir / command_name
        atomic_write_text(target, "\n".join(lines) + "\n")
        target.chmod(0o755)


def _prepend_path(prefix: str, current: str | None) -> str:
    if not current:
        return prefix
    return prefix if current.split(":")[0] == prefix else f"{prefix}:{current}"


def _prepend_paths(prefixes: list[str], current: str | None) -> str:
    result = current or ""
    for prefix in reversed(prefixes):
        result = _prepend_path(prefix, result)
    return result


def _environment_exports(runtime_env: PreparedRuntimeEnv) -> dict[str, str]:
    safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", runtime_env.task_id)
    environment_id = f"env_task_{safe_task_id}_{runtime_env.fingerprint}"
    return {
        "AO_ENVIRONMENT_ID": environment_id,
        "AO_ENVIRONMENT_TYPE": "task",
        "AO_ENVIRONMENT_ROOT": str(runtime_env.root),
        "AO_ENVIRONMENT_PYTHON": str(runtime_env.python_path),
        "AO_ENVIRONMENT_FINGERPRINT": runtime_env.fingerprint,
    }
