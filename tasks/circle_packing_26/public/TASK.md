# Circle Packing 26

## Objective

Find 26 non-overlapping circles inside the unit square `[0, 1] x [0, 1]` and
maximize the sum of their radii.

The candidate is a single Python file, `initial.py`, that exposes
`run_packing()`. The server-side evaluator imports that file, runs
`run_packing()`, verifies the hard geometric constraints, and records the
official score.

The provided seed candidate is valid but intentionally weak. It is useful as a
smoke test for the system and as a baseline for early experiments.

## Candidate Surface

Workers should edit the workspace entrypoint:

```text
initial.py
```

The entrypoint must return exactly:

```python
centers, radii, reported_sum = run_packing()
```

where `centers` has shape `(26, 2)`, `radii` has shape `(26,)`, and
`reported_sum` is a finite float matching `sum(radii)` within evaluator
tolerance.

Helper functions may live in `initial.py`. For the current single-file task
contract, do not rely on additional local files unless they are deliberately
uploaded as artifacts and submitted through a future artifact-based candidate
contract.

## Feedback Surface

Use the semantic server tools. These are capabilities, not a required sequence.

```bash
ctx task
ctx context
env status
eval verify --entry initial.py
eval probe --entry initial.py --kind diagnostics
eval submit --entry initial.py
eval status <evaluation-id>
eval wait <evaluation-id>
artifact upload --path initial.py --kind candidate
finding share --type insight --title "<short title>" --body "<what changed>"
notebook checkpoint --file WORKLOG.md
```

The base task environment is server-owned and includes the declared NumPy/SciPy
runtime. If exploratory tooling needs another package, request a worker overlay
with `env install --pip <requirement> --reason "<why>"`. Do not make official
candidate code depend on overlay-only packages unless the task runtime contract
is deliberately updated.

`eval verify` performs syntax, import, contract, finite-value, non-overlap, and
boundary checks. It is the cheapest validity gate.

`eval probe --kind diagnostics` returns public diagnostics, including slacks,
near-active constraint counts, conservative repair score, and fixed-center LP
strict-safe score. It is intended for debugging and local search guidance; it is
not the official leaderboard score.

`eval submit` creates a server-owned evaluation resource. Official submit
evaluations are asynchronous by default, so rely on `eval status` or `eval wait`
before treating the result as complete.

## Official Score

For a valid candidate, the official score is:

```text
sum(radii)
```

`reported_sum` is only a consistency check. The evaluator records both
`actual_sum` and `reported_sum`, but the score is computed from the actual
returned radii.

Invalid candidates receive failed evaluation state rather than an official
improvement. Public feedback should be enough to diagnose common issues without
exposing private evaluator implementation details.

## Useful Research Directions

The task package includes public direction notes under
`research_directions/`. They are optional information-structure hints for
multi-worker runs, not a controller workflow.

Good long-running experiments can assign different workers to directions such
as fresh global seed generation, non-row constructive layouts, topology escape,
incumbent repair, and tolerance-edge polish. Findings, notebook checkpoints,
candidate artifacts, evaluations, and job logs should be shared through the
server so later disposable worker sessions can resume from durable state.
