# Repository Guidelines

## Scope

This repository is the rewrite target for the agentic optimization framework.

The authoritative implementation references are:
- `docs/implementation_plan_codex.md`
- `docs/plan.md`
- `docs/draft_plan.md` for project meaning and research framing

Do not preserve or reintroduce the old prototype architecture.

## Active Layout

Core code lives in `src/agentic_opt/`:

- `common/`: low-level utilities
- `control_plane/`: server-owned resources, artifact registry, environment service, job service, evaluation service, shared tools, task knowledge, network policy state, telemetry, and leaderboard/incumbent state
- `worker_tools/`: agent-facing semantic CLI tools
- `adapter/`: Codex/App Server integration, workspaces, and thin outer loop
- `web/`: Flask + SQLite backend/routes
- `task_api.py`, `task_registry.py`: task protocol and loading boundary

Task implementations live outside the core package in `tasks/`; task-specific benchmark/data bundles should stay inside the owning task directory.

Tests live in `tests/`.

## Working Rules

- Keep the Coding Agent as the inner-loop search actor.
- Keep the outer loop thin: sessions, workers, budgets, and stopping conditions only.
- Do not encode a fixed numbered workflow in prompts, skills, or controller logic.
- Agents should access history, artifacts, feedback, jobs, environments,
  leaderboard/incumbent state, findings, notebook checkpoints, shared tools,
  and network policy through
  semantic server tools: `ctx`, `artifact`, `eval`, `finding`, `notebook`,
  `job`, `env`, `telemetry`, `tool`, and `network`.
- Task knowledge is provided as read-only files under `task/knowledge/` in the
  worker workspace when the task package includes `public/knowledge/`.
- Important unresolved design questions should be recorded in `docs/plan.md`.

## Commands

- `python3 -m compileall src/agentic_opt`
- `PYTHONPATH=src python3 -m unittest tests.test_web_backend tests.test_semantic_workspace tests.test_circle_packing_task -v`
- `PYTHONPATH=src python3 -m agentic_opt.adapter.semantic_worker --help`
- `PYTHONPATH=src python3 -m agentic_opt.worker_tools.semantic_cli --help`
- `PYTHONPATH=src python3 -m agentic_opt.worker_tools.semantic_cli env --help`
- `PYTHONPATH=src python3 -m agentic_opt.worker_tools.semantic_cli tool --help`
- `PYTHONPATH=src python3 -m agentic_opt.worker_tools.semantic_cli network --help`
- `PYTHONPATH=src python3 -m agentic_opt.web.app --help`

## Notes

- Prefer editing the new `agentic_opt` tree only.
- Keep filesystem artifacts explicit and exportable.
- Treat the control-plane database as the source of semantic server state.
  Filesystem artifacts are durable/exportable blobs referenced by server records.
