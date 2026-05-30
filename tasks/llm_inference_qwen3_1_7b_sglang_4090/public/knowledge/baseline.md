# Near-Parity Starter Baseline

The initial candidate is intended to be a valid, near-parity starting point,
not a blank kernel scaffold.

The baseline files are already materialized in the worker workspace under
`candidate/kernels/`:

- `rmsnorm.py` calls the installed `sgl_kernel` RMSNorm operator.
- `fused_add_rmsnorm.py` calls the installed `sgl_kernel` fused add RMSNorm
  operator in place.
- `swiglu.py` calls the installed `sgl_kernel` SiluAndMul operator.
- `attention_backend.py` delegates to the task-owned official fallback.
- `sampling_backend.py` delegates to the task-owned official fallback.

A read-only snapshot of this near-parity public seed is also available under:

```text
task/knowledge/near_parity_public_seed/
```

Use that snapshot as the stable reference if the writable `candidate/` has
already moved to an incumbent or an experimental branch.

Use this baseline to establish correctness and score scale before changing
kernel code. A useful first step is:

```bash
./bin/eval verify --entry candidate/manifest.json
```

Only change one integration surface at a time unless there is strong evidence
that a coupled change is necessary. Preserve the fallback behavior for paths
that are not explicitly supported by the edited kernel.

Manual standard-profile reference check on 2026-05-28:

```text
score: 1.00202445126698
```

Scores around 1 mean parity with the task-owned unmodified SGLang baseline.
Candidates far below 1 are usually dominated by integration overhead or an
unintended slow path, not by a useful kernel optimization tradeoff.
