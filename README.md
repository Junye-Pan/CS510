# Agentic Optimization

This repository contains the code implementation for a research project on
agentic optimization: using coding agents as the inner-loop search actor for
optimization tasks, while a thin control plane owns task state, evaluation,
artifacts, budgets, and worker sessions.

The detailed design is in `design.md`. The core implementation is under
`src/agentic_opt/`, and benchmark/task definitions live under `tasks/`.

## Layout

```text
src/agentic_opt/
  control_plane/   SQLite-backed state, artifacts, environments, evaluation.
  worker_tools/    Semantic CLI tools exposed to agents.
  adapter/         Codex/App Server worker integration and workspaces.
  web/             Flask API and lightweight operator UI.
  task_api.py      Task protocol and candidate/runtime specs.
  task_registry.py Task discovery/loading.

tasks/
  circle_packing_26/
  llm_inference_qwen3_1_7b_decode_attention_4090/
  llm_inference_qwen3_1_7b_sglang_4090/
```

## Start An Optimization Run

Start the local control plane:

```bash
PYTHONPATH=src python3 -m agentic_opt.web.app \
  --state-root ao_state \
  --db ao_state/control.sqlite3 \
  --host 127.0.0.1 \
  --port 5000
```

Open the operator UI:

```text
http://127.0.0.1:5000/ui
```

For `codex-local` workers, the `codex` CLI must be available on `PATH`.

Or create an experiment and one worker assignment through the API:

```bash
curl -s -X POST http://127.0.0.1:5000/api/v1/experiments \
  -H 'Content-Type: application/json' \
  -d '{
    "task_id": "circle_packing_26",
    "mode": "local",
    "assignment_count": 1,
    "worker_backend": "codex-local",
    "budget": {"total_evaluator_runs": 5}
  }'
```

The response includes an `assignment_id`. Start that assignment:

```bash
curl -s -X POST http://127.0.0.1:5000/api/v1/assignments/<assignment_id>/start-local \
  -H 'Content-Type: application/json' \
  -d '{"max_turn_wall_time_s": 1800}'
```

Worker workspaces, logs, artifacts, environments, and the SQLite database are
written under `ao_state/`.

## Defining A Task

Tasks are Python packages under `tasks/<task_id>/`. A task package should
provide:

- `task.py` with `create_task()`.
- `public/TASK.md` with worker-facing instructions.
- `public/public_contract.md` with the candidate/evaluation contract.
- A public seed candidate, either `public/initial.py` for single-file tasks or
  `public/initial_candidate/` for directory candidates.
- Optional `public/knowledge/` and `public/research_directions/`.

The object returned by `create_task()` must follow `TaskProtocol` from
`src/agentic_opt/task_api.py`:

- `metadata`: `TaskMetadata`, usually with a `CandidateSpec`.
- `runtime_spec`: `TaskRuntimeSpec` for Python dependencies/runtime.
- `public_dir`: path to the task's `public/` directory.
- `verify_entry(path)`: cheap hard-constraint check.
- `probe_entry(path, kind=...)`: diagnostic, non-official feedback.
- `evaluate_entry(path)`: authoritative scoring.

External task roots can be added with `AO_TASKS_ROOTS` or `AO_TASKS_ROOT`.

## Useful Commands

```bash
python3 -m compileall src/agentic_opt
PYTHONPATH=src python3 -m agentic_opt.web.app --help
PYTHONPATH=src python3 -m agentic_opt.adapter.semantic_worker --help
PYTHONPATH=src python3 -m agentic_opt.worker_tools.semantic_cli --help
```
