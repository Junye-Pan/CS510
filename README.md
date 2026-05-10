# Agentic Optimization

Agentic Optimization is a server-first framework for autonomous optimization
work. The control plane owns durable state and policy; coding agents remain the
inner-loop search actors that inspect context, edit candidates, run evaluations,
and publish findings through semantic tools.

This repository is the active rewrite. Core implementation lives in
`src/agentic_opt/`; task packages live in `tasks/`.

## Current Status

Implemented:

- Control-plane resources for experiments, assignments, sessions, jobs,
  evaluations, artifacts, findings, notebooks, telemetry, events, environments,
  environment overlays, leaderboard entries, and incumbents.
- SQLite-backed semantic server state with filesystem artifacts stored as
  explicit durable blobs.
- Flask API and local worker startup path.
- Semantic worker CLI tools: `ctx`, `artifact`, `eval`, `finding`, `notebook`,
  `job`, `env`, and `telemetry`.
- Local job execution, async evaluation, candidate artifact snapshots on
  official submit, and automatic leaderboard/incumbent updates.
- Environment control through the `local_venv` provider, including worker
  dependency overlays.
- `tasks/circle_packing_26` as the current concrete benchmark task.

Planned or incomplete:

- `docker_image` environment provider.
- A first-class attempt/run model distinct from evaluations and jobs.
- Broader task migration and production UI polish.
- More complete remote execution provider hardening.

## Active Layout

```text
src/agentic_opt/
  common/          Low-level utilities.
  control_plane/   Resource models, repository, services, artifact registry,
                   environment control, jobs, evaluations, leaderboard,
                   telemetry, and provider integrations.
  worker_tools/    Agent-facing semantic CLI tools.
  adapter/         Codex/App Server integration, workspaces, and worker loop.
  web/             Flask + SQLite backend and routes.
  task_api.py      Task protocol, candidate/runtime specs, evaluation reports.
  task_registry.py Task loading boundary.

tasks/
  circle_packing_26/
    task.py
    initial.py
    public/
    research_directions/

tests/
  test_web_backend.py
  test_semantic_workspace.py
  test_circle_packing_task.py
```

The old prototype architecture is not the target of this rewrite and should not
be reintroduced.

## Control Plane

The control plane is the source of truth for semantic state. Workers should use
server tools instead of scraping workspace files or inventing local protocols.

Main resource families:

- Task package metadata and public files.
- Experiments, worker assignments, and worker sessions.
- Environments and worker-requested environment overlays.
- Jobs and evaluations.
- Artifacts, including candidate snapshots.
- Leaderboard entries and incumbents.
- Findings, notebook checkpoints, telemetry runs, and events.

Run the local web backend:

```bash
PYTHONPATH=src python3 -m agentic_opt.web.app --state-root ao_state --db ao_state/control.sqlite --host 127.0.0.1 --port 5000
```

Important API groups are under `/api/v1/`, including `tasks`, `experiments`,
`assignments`, `context`, `environments`, `environment-overlays`, `artifacts`,
`evaluations`, `leaderboard`, `incumbent`, `jobs`, `findings`,
`notebook-checkpoints`, `telemetry-runs`, and `events`.

## Worker Tools

Semantic workspaces expose the control plane through `python -m
agentic_opt.worker_tools.semantic_cli`.

Common commands:

```bash
# Context and task state
ao ctx context
ao ctx task
ao ctx leaderboard
ao ctx incumbent

# Environment management
ao env status
ao env ensure
ao env install scipy==1.12.0
ao env list-overlays

# Evaluation
ao eval verify path/to/candidate.py
ao eval probe path/to/candidate.py
ao eval submit path/to/candidate.py
ao eval wait <evaluation-id>

# Artifacts and shared knowledge
ao artifact upload path/to/file --kind candidate
ao artifact checkout-incumbent --destination incumbent.py
ao finding share --title "Observation" --body "..."
ao finding search

# Jobs, notebooks, and telemetry
ao job create --name experiment --command "python script.py"
ao notebook checkpoint --title "local search notes" --path notes.md
ao telemetry start --name local-run
```

Official scores should go through `ao eval submit`. A valid official submission
is snapshotted as an artifact, recorded on the leaderboard, and may become the
experiment incumbent.

## Environment Control

Tasks declare their runtime needs through `TaskRuntimeSpec`. The control plane
resolves those specs into managed environments so tests, controller code,
workers, and task code execute against one coherent dependency set.

Current provider:

- `local_venv`: creates and reuses local virtual environments for task runtime
  specs. It installs declared requirements, verifies required imports, and
  records lock information.

Worker dependency changes should use environment overlays rather than ad hoc
package installation. A worker can request an overlay with `ao env install ...`;
the control plane records the requested dependencies and can approve or reject
the overlay. Verification/probing can run with an approved overlay, while
official submission defaults to the task base environment unless policy allows
the overlay.

Planned provider:

- `docker_image`: a stronger isolation backend where task execution and worker
  execution can share an image contract. This is the long-term path for fully
  reproducible remote execution.

## Task Packages

A task package lives under `tasks/<task_id>/` and is loaded through
`agentic_opt.task_registry`.

Typical files:

- `task.py`: implements the task protocol and declares `CandidateSpec` and
  `TaskRuntimeSpec`.
- `initial.py`: initial candidate, when the task uses a code-candidate workflow.
- `public/`: worker-visible task statement, contracts, helper notes, and seed
  assets.
- `research_directions/manifest.json`: optional direction set for assignment
  generation.

The public contract should describe what a candidate must expose, what metrics
matter, and what is allowed. Private validation logic should remain in the task
package implementation.

## Current Benchmark: `circle_packing_26`

`tasks/circle_packing_26` is configured as the current concrete optimization
task.

The candidate contract is a single Python file exposing:

```python
def run_packing():
    ...
```

It returns circle centers, radii, and the reported objective value. The task
validates geometry feasibility for 26 circles, computes the score, and records
diagnostics. Its base runtime currently uses the `local_venv` provider with
NumPy and SciPy requirements.

Workers can:

- Read the task contract and current context with `ao ctx context`.
- Check dependency state with `ao env status`.
- Run cheap checks with `ao eval verify`.
- Run exploratory scoring with `ao eval probe`.
- Publish official candidates with `ao eval submit`.
- Compare against the current best with `ao ctx leaderboard`,
  `ao ctx incumbent`, and `ao artifact checkout-incumbent`.

When assignments are generated, the control plane can distribute entries from
`research_directions/manifest.json` so workers explore different approaches.
The selected direction is included in assignment metadata and worker context.

## Validation

Use these commands for the current implementation:

```bash
python3 -m compileall src/agentic_opt
PYTHONPATH=src python3 -m unittest tests.test_web_backend tests.test_semantic_workspace tests.test_circle_packing_task -v
PYTHONPATH=src python3 -m agentic_opt.adapter.semantic_worker --help
PYTHONPATH=src python3 -m agentic_opt.worker_tools.semantic_cli --help
PYTHONPATH=src python3 -m agentic_opt.worker_tools.semantic_cli env --help
PYTHONPATH=src python3 -m agentic_opt.web.app --help
```
