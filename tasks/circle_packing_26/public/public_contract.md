# Public Contract

## Entrypoint

Candidate programs for `circle_packing_26` must expose:

```python
def run_packing():
    """
    Return:
        centers: array-like numeric data with shape (26, 2)
        radii: array-like numeric data with shape (26,)
        reported_sum: finite float
    """
```

The evaluator coerces `centers` and `radii` to numeric NumPy arrays. All values
must be finite.

## Hard Constraints

- Exactly 26 circles are required.
- Every radius must be non-negative.
- Every circle must lie fully inside `[0, 1] x [0, 1]`.
- Every pair of circles must be non-overlapping.
- `reported_sum` must match the actual `sum(radii)` within tolerance.
- The candidate must not shadow declared runtime dependencies such as `numpy`
  or `scipy`.

## Runtime

The task runtime is a server-owned `Environment` resource with NumPy and SciPy
available from the task's declared runtime spec. `run_packing()` must complete
within the configured circle-packing timeout (`AO_CIRCLE_PACKING_TIMEOUT_S`,
default `180` seconds).

Workers may request dependency overlays for exploratory tooling through
`env install`, but official scoring uses the base task environment. Candidate
code intended for `eval submit` should rely only on the declared task runtime
unless the task contract is deliberately changed.

## Server-Side Operations

The task is evaluated through the current semantic control plane:

- `env status` shows the active task environment and any assignment overlays.
- `eval verify --entry initial.py` checks syntax, imports, contract shape,
  finite values, geometry, and score consistency.
- `eval probe --entry initial.py --kind diagnostics` returns public diagnostics
  for repair and search.
- `eval submit --entry initial.py` records the official server-owned
  evaluation. Submit is asynchronous unless `--sync` is explicitly requested.
- `artifact upload --path initial.py --kind candidate` can preserve a candidate
  snapshot before or after evaluation.
- `finding share` and `notebook checkpoint` preserve reusable knowledge for
  later sessions.

The worker should treat these as available operations rather than as a fixed
workflow. Official scores come only from server-owned evaluation records.
