# Public Contract

## Entrypoint

The candidate shape is a workspace-root harness package with `initial.py` as
the entrypoint. It may import helper files next to it. The required callable is:

```python
def solve_puzzle(puzzle: dict, budget: dict | None = None) -> dict:
    ...
```

`solve_puzzle` is the harness boundary. The candidate should use the public
puzzle context to construct model prompts, parse responses, repair move traces,
cache reusable genre knowledge, and fail cleanly when it cannot produce a
legal-looking answer.

Optimize the harness's agentic behavior. Do not hard-code answer tables for
known puzzle ids or URLs, and do not turn the task into a complete deterministic
solver that bypasses the fixed LLM harness objective.

This no-new-family-solver variant also rejects newly added complete
deterministic solvers for puzzle families. Candidate code may branch by puzzle
type for prompt construction, parser normalization, repair, validation,
abstention, and model-call policy, but it must not add functions/classes/modules
such as `_solve_<puzzle_type>`, `solve_<puzzle_type>`, `*Solver`,
`*backtrack*`, `*exact_cover*`, or `*constraint_search*` that solve a puzzle
family without the LLM harness.

## Puzzle Schema

The host passes one public-safe puzzle dict at a time:

```python
{
    "puzzle_id": str,
    "puzzle_type": str,
    "puzzlink_url": str,
    "width": int | None,
    "height": int | None,
    "metadata": dict,
    "prompt_context": {
        "official_prompt": str | None,
        "board_text": str,
        "rules_text": str | None,
        "coordinate_notes": str,
        "input_examples": list[dict],
        "move_format": list[str],
        ...
    },
}
```

`prompt_context` is non-answer-bearing. It is the intended input for
harness-engineering work: use it instead of treating `puzzlink_url` as the only
source of puzzle information.

## Result Schema

Return a dict:

```python
{
    "status": "solved" | "failed" | "timeout",
    "moves": list[str],
    "meta": dict,
    "summary": str | None,
}
```

`meta` must include:

```python
{
    "llm_capable": True,
    "model_name": "gpt-5.2",
    "model_calls": int,
}
```

The model is fixed to `gpt-5.2`. Use the configured Codex backend
credentials:

```bash
export OPENAI_API_KEY=$(jq -r '.tokens.access_token' ~/.codex/auth.json)
export OPENAI_BASE_URL=https://chatgpt.com/backend-api/codex
export MODEL_NAME=gpt-5.2
```

The candidate may fail cleanly when credentials are absent, but it must remain
LLM-capable and keep the fixed model metadata.

The task-provided baseline uses the Codex backend's streamed Responses API
path. Do not spend optimization budget rediscovering the backend URL, fixed
model, or streaming requirement; those are part of the task environment.
Keep the `pydantic_ai` streamed `OpenAIResponsesModel` / `AsyncOpenAI`
transport used by `initial.py`. Do not replace it with hand-written `urllib`,
`requests`, `httpx`, direct `openai`, or raw `/responses` HTTP clients.
Changing the API transport is a verifier failure for this task.

## Seed Baseline

The public task filesystem includes `BASELINE.md` and `baseline_results.json`
with measured seed results for this task version. Treat them as the fixed
control result instead of rerunning the seed harness.

The seed `initial.py` baseline scores:

- `probe10`: 2/10 resolved, 10/10 legal, 0 invalid
- `private50`: 6/50 resolved, 50/50 legal, 0 invalid

Use `ve probe` before private submits to check that a candidate has not
regressed below the public `probe10` baseline.

## Budget Semantics

The budget dict includes:

```python
{
    "phase": "verify" | "probe" | "evaluate",
    "wall_clock_s": float,
    "max_model_calls": int,
    "max_tokens": int,
    "request_timeout_s": float,
    "preferred_reasoning_effort": "high",
    "allow_network_search": False,
    "model_name": "gpt-5.2",
    "openai_base_url": "https://chatgpt.com/backend-api/codex",
}
```

`verify` is structural and defaults to zero model calls. Probe and evaluator
runs provide a small multi-call budget by default so agentic harnesses can do
limited retry/repair loops.

`ve submit` evaluates the complete hidden private50 split. It is scheduled by
the task host with ten puzzle instances in flight at a time, but each
`solve_puzzle(...)` call still receives one puzzle record.

## Status Semantics

- `solved`: candidate believes the returned move trace solves the puzzle.
- `failed`: candidate did not solve the puzzle and returned a legal empty or
  partial trace.
- `timeout`: candidate stopped itself before the host-side timeout.

The host evaluator decides whether a trace actually solves a puzzle by replaying
the moves on a fresh PPBench puzzle and checking completion.

## Invalid Outcomes

The task treats these as invalid:

- import error
- missing `solve_puzzle`
- malformed result object
- non-string move entries
- unsupported move syntax
- replay exception / illegal move trace
- host-side timeout
- uncaught per-puzzle crash
- missing `meta.llm_capable=True`
- `meta.model_name` different from `gpt-5.2`
- changing the task-provided `pydantic_ai` streamed Codex transport
- adding complete deterministic puzzle-family solvers

## Privacy Boundary

Candidate code must not import PPBench dataset loaders, inspect answer-bearing
records, read private split manifests, or persist private evaluator inputs.
The task host passes only public-safe puzzle dicts into `solve_puzzle`.
