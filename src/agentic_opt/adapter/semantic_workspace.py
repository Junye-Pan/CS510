from __future__ import annotations

import os
import re
import json
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentic_opt.common.atomic import atomic_write_text
from agentic_opt.common.config import get_repo_root
from agentic_opt.common.files import digest_directory as _digest_directory, digest_file as _digest_file
from agentic_opt.common.runtime_env import PreparedRuntimeEnv, prepare_task_runtime
from agentic_opt.control_plane.task_context import (
    digest_directory,
    local_task_context_enforcement,
    materialize_task_context_snapshot,
    verify_task_context_path,
)
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
    network_policy: dict[str, object]
    workspace_seed: dict[str, Any] = field(default_factory=dict)
    checked_out_tools: list[dict[str, Any]] = field(default_factory=list)
    task_context: dict[str, Any] = field(default_factory=dict)


def prepare_semantic_workspace(
    *,
    workspace_root: Path,
    api_url: str,
    assignment: dict,
    session_id: str,
    runtime_env: PreparedRuntimeEnv | None = None,
    network_policy: dict[str, object] | None = None,
    bootstrap: dict[str, Any] | None = None,
    extra_env_exports: dict[str, str] | None = None,
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

    network_policy = network_policy or {}
    for relative in (
        "reference",
        "task",
        "task/public_files",
        "task/knowledge",
        "task/research_directions",
        "context",
        "history",
        "history/attempts",
        "history/artifacts",
        "history/evaluations",
        "history/findings",
        "history/jobs",
        "history/network",
        "history/notebooks",
        "history/shared_tools",
        "history/telemetry",
        "history/traces",
        "outbox",
        "outbox/artifacts",
        "outbox/findings",
        "outbox/notebooks",
        "artifacts",
        "findings",
        "local_tools",
        "shared_tools",
    ):
        (workspace_root / relative).mkdir(parents=True, exist_ok=True)
    if bootstrap:
        atomic_write_text(workspace_root / "reference" / "WORKSPACE_BOOTSTRAP.json", _json_pretty(bootstrap))
    task_context_state = _materialize_worker_files(
        workspace_root=workspace_root,
        assignment=assignment,
        task=task,
        network_policy=network_policy,
        bootstrap=bootstrap or {},
    )
    worklog = workspace_root / "WORKLOG.md"
    if not worklog.exists():
        atomic_write_text(worklog, "# WORKLOG\n\nUse this file for local scratch notes when useful.\n")

    agents_md_path = workspace_root / "AGENTS.md"
    atomic_write_text(agents_md_path, _agents_md_text())

    skills_root = workspace_root / ".agents" / "skills"
    for skill_name, body in SEMANTIC_SKILL_BODIES.items():
        skill_dir = skills_root / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(skill_dir / "SKILL.md", _semantic_skill_document(skill_name, body))

    bin_dir = workspace_root / "bin"
    _write_semantic_tool_wrappers(
        bin_dir=bin_dir,
        runtime_env=runtime_env,
        api_url=api_url,
        assignment=assignment,
        session_id=session_id,
        workspace_root=workspace_root,
        network_policy=network_policy,
        extra_env_exports=extra_env_exports or {},
    )
    shell_dir = workspace_root / ".agents" / "shell"
    _write_shell_tool_shims(shell_dir=shell_dir)
    _initialize_workspace_git(workspace_root)
    environment_exports = _environment_exports(runtime_env)
    network_exports = _network_exports(network_policy)
    repo_src = str(get_repo_root() / "src")
    venv_bin = str(runtime_env.venv_dir / "bin")
    env = {
        **{str(key): str(value) for key, value in (extra_env_exports or {}).items()},
        "AO_CONTROL_API_URL": api_url,
        "AO_ASSIGNMENT_ID": assignment["assignment_id"],
        "AO_EXPERIMENT_ID": assignment["experiment_id"],
        "AO_TASK_ID": assignment["task_id"],
        "AO_AGENT_ID": assignment["agent_id"],
        "AO_SESSION_ID": session_id,
        "AO_WORKSPACE_ROOT": str(workspace_root),
        "AO_AGENT_JOBS_ENABLED": "0",
        **environment_exports,
        **network_exports,
        **runtime_env.exports(),
        "PATH": _prepend_paths([str(bin_dir), venv_bin], os.environ.get("PATH")),
        "PYTHONPATH": _prepend_path(repo_src, os.environ.get("PYTHONPATH")),
        "VIRTUAL_ENV": str(runtime_env.venv_dir),
        "ZDOTDIR": str(shell_dir),
        "BASH_ENV": str(shell_dir / "bash_env"),
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
        network_policy=network_policy,
        workspace_seed=(bootstrap or {}).get("workspace_seed") or {},
        checked_out_tools=(bootstrap or {}).get("checked_out_tools") or [],
        task_context=task_context_state,
    )


def build_semantic_startup_prompt(*, assignment: dict, workspace: SemanticWorkspace) -> str:
    direction = (assignment.get("metadata") or {}).get("research_direction") or {}
    direction_text = ""
    if direction:
        direction_text = (
            f"\nResearch direction: {direction.get('title') or direction.get('direction_id')}\n"
            f"Direction id: {direction.get('direction_id')}\n"
            f"Direction note: {direction.get('startup_note') or ''}\n"
        )
    operator_note = _operator_note_text(assignment)
    bootstrap_text = _workspace_bootstrap_text(workspace)
    network_policy = workspace.network_policy.get("policy") if isinstance(workspace.network_policy, dict) else {}
    network_enforcement = workspace.network_policy.get("enforcement") if isinstance(workspace.network_policy, dict) else {}
    return f"""You are a Coding Agent worker in an autonomous optimization experiment.

Task: {assignment['task_id']}
Experiment: {assignment['experiment_id']}
Assignment: {assignment['assignment_id']}
Workspace: {workspace.root}
{direction_text}
{operator_note}
{bootstrap_text}

The server owns experiments, assignments, artifacts, evaluation, findings,
notebook checkpoints, registered traces, and policy. You own the autonomous
research behavior inside this workspace.

Read server-provided context from the workspace files first:
- task/TASK.md and task/public_contract.md
- task/public_files/
- task/knowledge/
- context/current_state.json
- history/attempts/
- history/findings/
- history/notebooks/
- history/evaluations/
- history/artifacts/
- history/traces/
- history/leaderboard.jsonl and history/incumbent.json

Use `rg`, `jq`, `sed`, `head`, and ordinary file tools to inspect only the
parts you need. Semantic commands are for authority-bearing operations or for
returning file locations, not for dumping all history into the context window.
They are available through `./bin/<tool>` and on `PATH`; `./bin/<tool>` is
always the unambiguous spelling when a shell builtin has the same name.

Use semantic server tools when you need server action or fresh status:
- ./bin/env status
- ./bin/env install --pip <requirement> --reason <why>
- ./bin/network status
- ./bin/trace list
- ./bin/trace show <trace-id>
- ./bin/trace commands <trace-id>
- ./bin/trace search <query>
- ./bin/tool publish --path local_tools/<name> --name <name> --description <text>
- ./bin/tool list
- ./bin/tool show <tool-id>
- ./bin/tool checkout <tool-id> --destination shared_tools/<name>
- ./bin/tool install <tool-id>
- ./bin/attempt create
- ./bin/attempt list
- ./bin/attempt show <attempt-id>
- ./bin/attempt update <attempt-id> --status <status>
- ./bin/artifact upload --path <path> --kind <kind>
- ./bin/artifact checkout-incumbent --destination <path>
- ./bin/eval verify --entry <candidate-entrypoint>
- ./bin/eval probe --entry <candidate-entrypoint>
- ./bin/eval submit --entry <candidate-entrypoint>  # official worker submissions run synchronously
- ./bin/eval submit --artifact-id <artifact-id>
- ./bin/finding share --type <type> --title <title> --body <text>
- ./bin/notebook checkpoint --file WORKLOG.md
- ./bin/telemetry start --provider local --name <run-name>
- ./bin/telemetry log-metrics <telemetry-id> --metric loss=0.1 --step 1
- ./bin/telemetry finish <telemetry-id>
- ./bin/ctx stop --reason <why>  # record why this session has no useful next action
- ./bin/ctx global-stop --reason <why> --confirm-global-stop  # record assignment-level convergence or blocker

These are capabilities, not a fixed workflow. Read context, implement, validate,
evaluate, checkpoint, or share findings when evidence makes that action useful.
Durable job launching is disabled for optimization workers; use direct local
commands inside the current turn for scratch checks, and use eval for
server-owned scoring. Official scores must come from eval submit/server
evaluation.
For kernel optimization tasks, keep candidate changes scoped to the candidate
kernel implementation and manifest entries for those implementations. Do not
remove required model components, rewrite model structure, or change evaluator
inputs to improve score. Do not mine prior global run directories such as
`/workspace/agentic-optimization/runs` or unrelated `/workspace/ao_state`
workspaces/artifacts; use only this workspace and semantic server history made
visible inside it. After a material candidate update, run
`./bin/eval verify --entry candidate/manifest.json` promptly so the server
verifier can provide durable feedback.
If the current session has no useful next action, record useful state and run
`./bin/ctx stop --reason <why>`. If the evidence shows the whole assignment is
converged or blocked rather than just the current session, run
`./bin/ctx global-stop --reason <why> --confirm-global-stop`.

Network policy: {network_policy}
Network enforcement: {network_enforcement}
"""


def _operator_note_text(assignment: dict) -> str:
    metadata = assignment.get("metadata") or {}
    notes: list[str] = []
    raw_budget_instruction = metadata.get("operator_budget_instruction")
    if isinstance(raw_budget_instruction, dict):
        message = raw_budget_instruction.get("message")
        if message:
            notes.append(str(message))
    elif raw_budget_instruction:
        notes.append(str(raw_budget_instruction))
    for key in ("operator_instruction", "supervisor_instruction", "supervisor_note"):
        value = metadata.get(key)
        if value:
            notes.append(str(value))
    if not notes:
        return ""
    body = "\n".join(f"- {note}" for note in notes)
    return f"Operator notes:\n{body}\n"


def _agents_md_text() -> str:
    return """# Agentic Optimization Semantic Workspace

Use the workspace directory tree as your primary read interface.
Do not assume a fixed numbered workflow.
Use `task/TASK.md`, `task/public_contract.md`, `task/public_files/`, and `task/knowledge/` for task objective, public contract, and task-provided read-only context.
Use `context/` and `history/` for assignment state, research direction, incumbent, leaderboard, prior findings, attempts, artifacts, evaluations, trace pointers, and notebook checkpoints.
Use local file tools such as `rg`, `jq`, `sed`, and `head` to inspect only the relevant slices of that state.
Use `./bin/eval verify`, `./bin/eval probe`, and `./bin/eval submit` for server-owned feedback and official scoring.
Use `env`, `attempt`, `artifact`, `finding`, `notebook`, `telemetry`, `tool`, `network`, and `trace` for durable server-visible mutations, permissioned actions, fresh status, or file locations.
Use `trace` only to get registered coding-agent trace locations; inspect trace JSONL files yourself.
Use `local_tools/` for draft helper tools and `tool publish` when a helper should become reusable by later workers.
Check `network status` before any action that might need external internet. Control-plane access does not imply permission to search the public internet.
Do not access hidden evaluator internals or mine prior global run directories.
Only read this workspace, task/public files, task/knowledge, checked-out shared
tools, and server-provided history exposed under this workspace. Do not inspect
`/workspace/agentic-optimization/runs`, unrelated `/workspace/ao_state`
workspaces/artifacts, or any other global experiment output unless it has been
materialized into this workspace by a semantic server tool.
For kernel tasks, keep candidate changes scoped to kernel implementation files
and implementation manifest entries. After a material candidate edit, run
`./bin/eval verify --entry candidate/manifest.json` promptly; do not spend the
turn on extended local-only probing.
If the current session has no useful next action, make useful state durable and run `./bin/ctx stop --reason <why>`.
If the assignment itself is converged or blocked, run `./bin/ctx global-stop --reason <why> --confirm-global-stop` so the outer loop does not keep restarting equivalent sessions.
"""


def _workspace_bootstrap_text(workspace: SemanticWorkspace) -> str:
    lines: list[str] = []
    seed = workspace.workspace_seed or {}
    if seed:
        kind = seed.get("kind") or "unknown"
        if seed.get("artifact_id"):
            score = seed.get("score")
            score_text = f", score={score}" if score is not None else ""
            lines.append(
                f"Workspace seed: {kind} artifact {seed.get('artifact_id')} is already materialized at {workspace.entry_path}{score_text}."
            )
        else:
            reason = seed.get("reason")
            reason_text = f" ({reason})" if reason else ""
            lines.append(f"Workspace seed: {kind}{reason_text}; candidate entrypoint is {workspace.entry_path}.")
    if workspace.checked_out_tools:
        lines.append("Auto-checked-out shared tools:")
        for tool in workspace.checked_out_tools:
            lines.append(
                f"- {tool.get('name')} ({tool.get('tool_id')}): {tool.get('destination_path')}"
            )
    if not lines:
        return ""
    return "\n" + "\n".join(lines) + "\n"


def _json_pretty(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _jsonl(records: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _materialize_worker_files(
    *,
    workspace_root: Path,
    assignment: dict[str, Any],
    task: Any,
    network_policy: dict[str, object],
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    context = dict(bootstrap.get("context_snapshot") or {})
    if not context:
        context = {
            "assignment": assignment,
            "network_policy": network_policy,
        }
    context.setdefault("assignment", assignment)
    context.setdefault("network_policy", network_policy)
    task_contract = bootstrap.get("task_contract") or _local_task_contract(task)

    task_knowledge, task_context_state = _materialize_task_files(
        workspace_root=workspace_root,
        task=task,
        task_contract=task_contract,
        task_context=bootstrap.get("task_context") if isinstance(bootstrap.get("task_context"), dict) else None,
    )
    _materialize_context_files(
        workspace_root=workspace_root,
        context=context,
        task_contract=task_contract,
        task_knowledge=task_knowledge,
        task_context=task_context_state,
    )
    _materialize_history_files(workspace_root=workspace_root, context=context)
    return task_context_state


def _materialize_task_files(
    *,
    workspace_root: Path,
    task: Any,
    task_contract: dict[str, Any],
    task_context: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    task_root = workspace_root / "task"
    if task_context and task_context.get("task_path") and task_context.get("digest"):
        enforcement = materialize_task_context_snapshot(
            snapshot=task_context,
            workspace_root=workspace_root,
            provider="local_venv",
        )
        task_knowledge = _read_json(task_root / "knowledge_inventory.json") or {
            "available": False,
            "workspace_path": "task/knowledge",
            "digest": None,
            "file_count": 0,
            "size_bytes": 0,
            "files": [],
            "manifest": None,
        }
        return task_knowledge, {**task_context, "enforcement": enforcement}

    public_context = task_contract.get("public_context") or {}
    task_markdown = public_context.get("task_markdown")
    public_contract = public_context.get("public_contract_markdown")
    if isinstance(task_markdown, str):
        atomic_write_text(task_root / "TASK.md", task_markdown if task_markdown.endswith("\n") else task_markdown + "\n")
    if isinstance(public_contract, str):
        atomic_write_text(
            task_root / "public_contract.md",
            public_contract if public_contract.endswith("\n") else public_contract + "\n",
        )

    public_files = public_context.get("public_files") or []
    if isinstance(public_files, list):
        for item in public_files:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                continue
            content = item.get("content")
            if not isinstance(content, str):
                continue
            destination = _safe_child(task_root / "public_files", item["path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(destination, content if content.endswith("\n") else content + "\n")

    task_knowledge = _materialize_task_knowledge(task_root=task_root, public_dir=task.public_dir)
    atomic_write_text(task_root / "knowledge_inventory.json", _json_pretty(task_knowledge))

    for index, direction in enumerate(public_context.get("research_directions") or []):
        if not isinstance(direction, dict):
            continue
        direction_id = _safe_name(str(direction.get("direction_id") or direction.get("id") or f"direction_{index}"))
        atomic_write_text(task_root / "research_directions" / f"{direction_id}.json", _json_pretty(direction))
        if isinstance(direction.get("doc_markdown"), str):
            atomic_write_text(task_root / "research_directions" / f"{direction_id}.md", direction["doc_markdown"])

    manifest = _without_public_file_content(task_contract)
    atomic_write_text(task_root / "manifest.json", _json_pretty(manifest))
    digest = digest_directory(task_root)
    verification = verify_task_context_path(task_path=task_root, expected_digest=digest)
    return task_knowledge, {
        "task_id": task_contract.get("task_id"),
        "workspace_path": "task",
        "task_path": str(task_root.resolve()),
        "digest": digest,
        "file_count": len([item for item in task_root.rglob("*") if item.is_file()]),
        "size_bytes": sum(item.stat().st_size for item in task_root.rglob("*") if item.is_file()),
        "task_knowledge": task_knowledge,
        "enforcement": local_task_context_enforcement(
            snapshot={"digest": digest, "task_path": str(task_root.resolve())},
            workspace_root=workspace_root,
            verification=verification,
        ),
    }


def _materialize_task_knowledge(*, task_root: Path, public_dir: Path) -> dict[str, Any]:
    source_path = public_dir / "knowledge"
    destination_root = task_root / "knowledge"
    if destination_root.exists():
        _make_tree_writable(destination_root)
        shutil.rmtree(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    if not source_path.exists():
        return _task_knowledge_inventory(source_path=source_path, destination_root=destination_root, manifest=None)
    if source_path.is_symlink() or not source_path.is_dir():
        raise ValueError(f"task knowledge must be a directory: {source_path}")
    source_root = source_path.resolve()
    manifest = _load_task_knowledge_manifest(source_root)
    for source in sorted(source_root.rglob("*")):
        if source.is_symlink():
            raise PermissionError(f"task knowledge may not contain symlinks: {source}")
        relative = source.relative_to(source_root)
        destination = destination_root / relative
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    _make_tree_read_only(destination_root)
    return _task_knowledge_inventory(source_path=source_root, destination_root=destination_root, manifest=manifest)


def _task_knowledge_inventory(*, source_path: Path, destination_root: Path, manifest: dict[str, Any] | None) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in destination_root.rglob("*") if item.is_file()):
        relative = path.relative_to(destination_root).as_posix()
        files.append(
            {
                "relative_path": relative,
                "workspace_path": f"task/knowledge/{relative}",
                "digest": _digest_file(path),
                "size_bytes": path.stat().st_size,
                "read_only": not bool(path.stat().st_mode & 0o222),
            }
        )
    return {
        "available": source_path.exists(),
        "source_path": str(source_path),
        "workspace_path": "task/knowledge",
        "digest": _digest_directory(destination_root) if files else None,
        "file_count": len(files),
        "size_bytes": sum(int(item["size_bytes"]) for item in files),
        "files": files,
        "manifest": manifest,
    }


def _load_task_knowledge_manifest(root: Path) -> dict[str, Any] | None:
    path = root / "manifest.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("task knowledge manifest must be a JSON object")
    items = manifest.get("items")
    if items is None:
        return manifest
    if not isinstance(items, list):
        raise ValueError("task knowledge manifest items must be a list when present")
    for raw in items:
        if not isinstance(raw, dict):
            raise ValueError("task knowledge manifest items must be objects")
        relative = raw.get("path")
        if not isinstance(relative, str) or not relative:
            raise ValueError("task knowledge manifest item path is required")
        target = _safe_child(root, relative).resolve()
        if not target.is_relative_to(root):
            raise PermissionError(f"task knowledge manifest path escapes {root}: {relative}")
        if not target.exists():
            raise FileNotFoundError(target)
    return manifest


def _make_tree_read_only(root: Path) -> None:
    for item in sorted(root.rglob("*"), reverse=True):
        if item.is_file():
            item.chmod(item.stat().st_mode & ~0o222)
        elif item.is_dir():
            item.chmod(item.stat().st_mode & ~0o222)
    root.chmod(root.stat().st_mode & ~0o222)


def _make_tree_writable(root: Path) -> None:
    for item in root.rglob("*"):
        if item.is_dir():
            item.chmod(item.stat().st_mode | 0o700)
        else:
            item.chmod(item.stat().st_mode | 0o600)
    root.chmod(root.stat().st_mode | 0o700)


def _materialize_context_files(
    *,
    workspace_root: Path,
    context: dict[str, Any],
    task_contract: dict[str, Any],
    task_knowledge: dict[str, Any],
    task_context: dict[str, Any],
) -> None:
    context_root = workspace_root / "context"
    assignment = context.get("assignment") or {}
    experiment = context.get("experiment") or {}
    network_policy = context.get("network_policy") or {}
    atomic_write_text(context_root / "assignment.json", _json_pretty(assignment))
    atomic_write_text(context_root / "experiment.json", _json_pretty(experiment))
    atomic_write_text(context_root / "network_policy.json", _json_pretty(network_policy))
    if context.get("research_direction") is not None:
        atomic_write_text(context_root / "research_direction.json", _json_pretty(context.get("research_direction")))

    counts = {
        key: len(context.get(key) or [])
        for key in (
            "attempts",
            "agent_traces",
            "recent_findings",
            "artifacts",
            "evaluations",
            "jobs",
            "leaderboard",
            "environments",
            "environment_overlays",
            "telemetry_runs",
            "notebook_checkpoints",
            "shared_tools",
            "network_access_events",
            "assignment_sessions",
            "assignment_attempts",
            "assignment_agent_traces",
            "assignment_artifacts",
            "assignment_evaluations",
            "assignment_jobs",
            "assignment_telemetry_runs",
            "assignment_notebook_checkpoints",
        )
    }
    current_state = {
        "assignment_id": assignment.get("assignment_id"),
        "experiment_id": assignment.get("experiment_id") or experiment.get("experiment_id"),
        "task_id": assignment.get("task_id") or task_contract.get("task_id"),
        "agent_id": assignment.get("agent_id"),
        "direction_id": assignment.get("direction_id"),
        "counts": counts,
        "paths": _workspace_read_paths(),
        "task_knowledge": {
            "available": task_knowledge.get("available"),
            "workspace_path": task_knowledge.get("workspace_path"),
            "digest": task_knowledge.get("digest"),
            "file_count": task_knowledge.get("file_count"),
            "size_bytes": task_knowledge.get("size_bytes"),
        },
        "task_context": {
            "workspace_path": task_context.get("workspace_path"),
            "digest": task_context.get("digest"),
            "file_count": task_context.get("file_count"),
            "size_bytes": task_context.get("size_bytes"),
            "enforcement": task_context.get("enforcement"),
        },
    }
    atomic_write_text(context_root / "current_state.json", _json_pretty(current_state))
    atomic_write_text(context_root / "README.md", _context_readme_text())


def _materialize_history_files(*, workspace_root: Path, context: dict[str, Any]) -> None:
    history_root = workspace_root / "history"
    _write_record_dir(history_root / "attempts", context.get("attempts") or [], "attempt_id", "attempt.json")
    _write_record_dir(history_root / "artifacts", context.get("artifacts") or [], "artifact_id", "artifact.json")
    _write_evaluations(history_root / "evaluations", context.get("evaluations") or [])
    _write_flat_records(history_root / "findings", context.get("recent_findings") or [], "finding_id")
    _write_record_dir(history_root / "jobs", context.get("jobs") or [], "job_id", "status.json")
    _write_network(history_root / "network", context)
    _write_notebooks(history_root / "notebooks", context.get("notebook_checkpoints") or [])
    _write_record_dir(history_root / "shared_tools", context.get("shared_tools") or [], "tool_id", "tool.json")
    _write_record_dir(history_root / "telemetry", context.get("telemetry_runs") or [], "telemetry_id", "telemetry.json")
    _write_traces(history_root / "traces", context.get("agent_traces") or [])
    atomic_write_text(history_root / "leaderboard.jsonl", _jsonl(context.get("leaderboard") or []))
    atomic_write_text(history_root / "incumbent.json", _json_pretty(context.get("incumbent") or {}))
    atomic_write_text(history_root / "direction_incumbent.json", _json_pretty(context.get("direction_incumbent") or {}))
    atomic_write_text(history_root / "environments.jsonl", _jsonl(context.get("environments") or []))
    atomic_write_text(history_root / "environment_overlays.jsonl", _jsonl(context.get("environment_overlays") or []))

    assignment_root = history_root / "current_assignment"
    _write_record_dir(assignment_root / "attempts", context.get("assignment_attempts") or [], "attempt_id", "attempt.json")
    _write_record_dir(assignment_root / "artifacts", context.get("assignment_artifacts") or [], "artifact_id", "artifact.json")
    _write_evaluations(assignment_root / "evaluations", context.get("assignment_evaluations") or [])
    _write_record_dir(assignment_root / "jobs", context.get("assignment_jobs") or [], "job_id", "status.json")
    _write_notebooks(assignment_root / "notebooks", context.get("assignment_notebook_checkpoints") or [])
    _write_record_dir(assignment_root / "telemetry", context.get("assignment_telemetry_runs") or [], "telemetry_id", "telemetry.json")
    _write_traces(assignment_root / "traces", context.get("assignment_agent_traces") or [])


def _write_record_dir(root: Path, records: list[dict[str, Any]], id_key: str, filename: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    index_records: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        index_records.append(record)
        record_id = _record_id(record, id_key, index)
        target = root / record_id
        target.mkdir(parents=True, exist_ok=True)
        atomic_write_text(target / filename, _json_pretty(record))
    atomic_write_text(root / "index.jsonl", _jsonl(index_records))


def _write_flat_records(root: Path, records: list[dict[str, Any]], id_key: str) -> None:
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        record_id = _record_id(record, id_key, index)
        atomic_write_text(root / f"{record_id}.json", _json_pretty(record))
    atomic_write_text(root / "index.jsonl", _jsonl([item for item in records if isinstance(item, dict)]))


def _write_evaluations(root: Path, records: list[dict[str, Any]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    index_records: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        index_records.append(record)
        evaluation_id = _record_id(record, "evaluation_id", index)
        target = root / evaluation_id
        target.mkdir(parents=True, exist_ok=True)
        atomic_write_text(target / "evaluation.json", _json_pretty(record))
        for field, filename in (("request", "request.json"), ("result", "result.json"), ("public_feedback", "public_feedback.json")):
            if record.get(field) is not None:
                atomic_write_text(target / filename, _json_pretty(record[field]))
    atomic_write_text(root / "index.jsonl", _jsonl(index_records))


def _write_network(root: Path, context: dict[str, Any]) -> None:
    atomic_write_text(root / "policy.json", _json_pretty(context.get("network_policy") or {}))
    atomic_write_text(root / "events.jsonl", _jsonl(context.get("network_access_events") or []))


def _write_notebooks(root: Path, records: list[dict[str, Any]]) -> None:
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        checkpoint_id = _record_id(record, "checkpoint_id", index)
        atomic_write_text(root / f"{checkpoint_id}.json", _json_pretty(record))
        content = record.get("content")
        if isinstance(content, str):
            atomic_write_text(root / f"{checkpoint_id}.md", content if content.endswith("\n") else content + "\n")
    atomic_write_text(root / "index.jsonl", _jsonl([item for item in records if isinstance(item, dict)]))


def _write_traces(root: Path, records: list[dict[str, Any]]) -> None:
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        trace_id = _record_id(record, "trace_id", index)
        target = root / trace_id
        target.mkdir(parents=True, exist_ok=True)
        atomic_write_text(target / "manifest.json", _json_pretty({**record, "files": _trace_files_for_record(record)}))
    atomic_write_text(root / "index.jsonl", _jsonl([item for item in records if isinstance(item, dict)]))


def _trace_files_for_record(trace: dict[str, Any]) -> dict[str, str | None]:
    root = trace.get("trace_root")
    if not root:
        return {"manifest": None, "events": None, "commands": None, "agent_messages": None, "stdout": None}
    metadata = trace.get("metadata") or {}
    files = metadata.get("files") or {}
    trace_root = Path(str(root))
    return {
        "manifest": str(trace_root / "manifest.json"),
        "events": str(trace_root / (files.get("events") or "events.jsonl")),
        "commands": str(trace_root / (files.get("commands") or "commands.jsonl")),
        "agent_messages": str(trace_root / (files.get("agent_messages") or "agent_messages.jsonl")),
        "stdout": str(trace_root / files["stdout"]) if files.get("stdout") else None,
    }


def _record_id(record: dict[str, Any], id_key: str, index: int) -> str:
    return _safe_name(str(record.get(id_key) or record.get("id") or f"record_{index}"))


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe[:120] or "record"


def _safe_child(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ValueError(f"unsafe workspace materialization path: {relative}")
    return root / path


def _without_public_file_content(task_contract: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(task_contract))
    public_context = payload.get("public_context")
    if isinstance(public_context, dict) and isinstance(public_context.get("public_files"), list):
        public_context["public_files"] = [
            {key: value for key, value in item.items() if key != "content"}
            for item in public_context["public_files"]
            if isinstance(item, dict)
        ]
    return payload


def _local_task_contract(task: Any) -> dict[str, Any]:
    public_dir = task.public_dir
    task_md = _read_optional_text(public_dir / "TASK.md")
    public_contract = _read_optional_text(public_dir / "public_contract.md")
    return {
        "task_id": task.metadata.task_id,
        "title": task.metadata.title,
        "public_context": {
            "task_markdown": task_md,
            "public_contract_markdown": public_contract,
            "public_dir": str(public_dir),
            "public_files": [],
            "research_directions": [],
        },
    }


def _read_optional_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _workspace_read_paths() -> dict[str, Any]:
    return {
        "task": ["task/TASK.md", "task/public_contract.md", "task/public_files/", "task/knowledge/"],
        "context": ["context/current_state.json", "context/assignment.json", "context/network_policy.json"],
        "history": [
            "history/attempts/",
            "history/current_assignment/attempts/",
            "history/findings/",
            "history/notebooks/",
            "history/evaluations/",
            "history/artifacts/",
            "history/traces/",
            "history/leaderboard.jsonl",
            "history/incumbent.json",
        ],
        "writable_outbox": ["outbox/findings/", "outbox/artifacts/", "outbox/notebooks/"],
    }


def _context_readme_text() -> str:
    return """# Worker Context Files

This workspace is the default read interface for the worker.

- `task/` contains the task contract and public task files.
- `task/knowledge/` contains task-provided read-only context files when the
  task package provides them.
- `context/` contains current assignment, experiment, and policy metadata.
- `history/` contains experiment-wide attempts, findings, notebooks,
  evaluations, artifacts, telemetry, shared-tool records, network events, and
  trace pointers. Assignment-scoped shortcuts live under
  `history/current_assignment/`.
- `outbox/` is local scratch space for material you may later publish through
  `finding`, `artifact`, or `notebook` commands.

Use local file tools such as `rg`, `jq`, `sed`, and `head` before asking a
semantic command for broad context. Commands remain useful for server-owned
mutations, fresh status, evaluations, uploads, permissions, and trace file
locations.
"""


SEMANTIC_SKILL_BODIES: dict[str, str] = {
    "context-use": """
# Context Use

Use workspace files as the default read path. Start with
`context/current_state.json`, `task/TASK.md`, `task/public_contract.md`, and
the relevant directories under `history/`. Search with `rg`, inspect JSON with
`jq`, and read only the files that matter for the current question.

The `ctx` read commands return file locations when possible; they should not be
used to dump all server history into the context window. Treat the control
plane as authoritative durable state, with this workspace holding the readable
snapshot and local scratch files for the current session.
""",
    "attempt-use": """
# Attempt Use

Use `attempt create` to mark a candidate attempt as a first-class server record
when you are starting a coherent candidate line. Link artifacts, evaluations,
and telemetry with `--attempt-id` so later workers can reconstruct the attempt
from structured resources. Read `history/attempts/index.jsonl` and the linked
`history/evaluations/`, `history/artifacts/`, and `history/findings/` records
before repeating a prior line of work.

Attempts do not store summaries. Share reusable conclusions with `finding
share`, and checkpoint longer local notes with `notebook checkpoint`.
""",
    "evaluation-use": """
# Evaluation Use

Use `./bin/eval verify --entry <path>` for quick validity checks and
`./bin/eval probe` for public diagnostics. Use
`./bin/eval submit --entry <path>` or `./bin/eval submit --artifact-id
<artifact-id>` for official scoring.

Optimization-worker submit evaluations run synchronously. Treat the returned
record as the server-owned scoring result for that submit.

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
non-official process or training metrics. Telemetry helps compare local runs,
but official scores must still come from `./bin/eval submit`.
""",
    "tool-use": """
# Shared Tool Use

Use `local_tools/` for helper scripts that are only useful inside this session.
Use `tool publish` when a tested helper should become reusable by later workers.
Use `tool list`, `tool show`, `tool checkout`, and `tool install` to reuse
server-owned shared tools instead of copying from another worker workspace.
""",
    "network-use": """
# Network Use

Use `network status` to inspect whether external internet is allowed, denied,
or only audit-logged. Semantic control-plane access is separate from public
internet access.
""",
    "trace-use": """
# Trace Use

Use `trace list`, `trace show`, `trace commands`, `trace events`, and
`trace search` to inspect registered coding-agent turn records. Trace is
read-only worker context: commands return trace file locations and small index
metadata, not full trace contents. Use tools such as `rg`, `head`, `sed`, or
`jq` against the returned JSONL paths when you need to inspect only the relevant
slice. Do not use trace as a summary source; share reusable conclusions through
findings and longer notes through notebook checkpoints.
""",
}


def _semantic_skill_document(skill_name: str, body: str) -> str:
    body = body.strip()
    title = skill_name
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip() or skill_name
            break
    return f"---\nname: {skill_name}\ndescription: {title}\n---\n\n{body}\n"


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
    network_policy: dict[str, object],
    extra_env_exports: dict[str, str] | None = None,
) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    exports = {
        **{str(key): str(value) for key, value in (extra_env_exports or {}).items()},
        "AO_CONTROL_API_URL": api_url,
        "AO_ASSIGNMENT_ID": assignment["assignment_id"],
        "AO_EXPERIMENT_ID": assignment["experiment_id"],
        "AO_TASK_ID": assignment["task_id"],
        "AO_AGENT_ID": assignment["agent_id"],
        "AO_SESSION_ID": session_id,
        "AO_WORKSPACE_ROOT": str(workspace_root),
        "AO_AGENT_JOBS_ENABLED": "0",
        **_environment_exports(runtime_env),
        **_network_exports(network_policy),
        **runtime_env.exports(),
        "PYTHONPATH": _prepend_path(str(get_repo_root() / "src"), os.environ.get("PYTHONPATH")),
        "VIRTUAL_ENV": str(runtime_env.venv_dir),
    }
    for command_name in ("ctx", "attempt", "artifact", "eval", "finding", "notebook", "env", "telemetry", "tool", "network", "trace"):
        lines = ["#!/bin/sh", "set -eu"]
        for key, value in exports.items():
            lines.append(f"export {key}={shlex.quote(str(value))}")
        lines.append(
            f"exec {shlex.quote(str(runtime_env.python_path))} -m agentic_opt.worker_tools.semantic_cli {shlex.quote(command_name)} \"$@\""
        )
        target = bin_dir / command_name
        atomic_write_text(target, "\n".join(lines) + "\n")
        target.chmod(0o755)


def _write_shell_tool_shims(*, shell_dir: Path) -> None:
    shell_dir.mkdir(parents=True, exist_ok=True)
    body = """# Generated by agentic-opt.
# zsh/bash builtins can shadow semantic tool wrappers. Keep bare `eval ...`
# usable for workers while preserving `builtin eval ...` for shell evaluation.
if [ -n "${AO_WORKSPACE_ROOT:-}" ] && [ -x "$AO_WORKSPACE_ROOT/bin/eval" ]; then
  eval() {
    command "$AO_WORKSPACE_ROOT/bin/eval" "$@"
  }
fi
"""
    # zsh -lc reads .zprofile after system startup files; defining eval in
    # .zshenv would intercept macOS path_helper's own eval call.
    atomic_write_text(shell_dir / ".zprofile", body)
    atomic_write_text(shell_dir / ".zshrc", body)
    atomic_write_text(shell_dir / "bash_env", body)


def _initialize_workspace_git(workspace_root: Path) -> None:
    git = shutil.which("git")
    if not git or (workspace_root / ".git").exists():
        return
    try:
        subprocess.run(
            [git, "init", "--quiet"],
            cwd=workspace_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
        subprocess.run(
            [git, "config", "user.name", "Agentic Optimization Worker"],
            cwd=workspace_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
        subprocess.run(
            [git, "config", "user.email", "agentic-opt@example.invalid"],
            cwd=workspace_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
    except Exception:
        return


def _prepend_path(prefix: str, current: str | None) -> str:
    if not current:
        return prefix
    return current if current.split(":")[0] == prefix else f"{prefix}:{current}"


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


def _network_exports(network_policy: dict[str, object]) -> dict[str, str]:
    policy = network_policy.get("policy") if isinstance(network_policy, dict) else {}
    enforcement = network_policy.get("enforcement") if isinstance(network_policy, dict) else {}
    if not isinstance(policy, dict):
        policy = {}
    if not isinstance(enforcement, dict):
        enforcement = {}
    return {
        "AO_NETWORK_CONTROL_PLANE": str(policy.get("control_plane") or "allow"),
        "AO_NETWORK_EXTERNAL_INTERNET": str(policy.get("external_internet") or "allow"),
        "AO_NETWORK_POLICY_WEAKENED": "1" if enforcement.get("policy_weakened") else "0",
    }
