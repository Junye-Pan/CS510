# Third Party Notices

This task's private verifier code contains small, task-specific adaptations of
verification ideas from `external/flashinfer-bench`, which is distributed under
the Apache License, Version 2.0.

Adapted concepts include:

- deterministic tensor contract checks and tolerance statistics inspired by
  `flashinfer_bench/bench/evaluators/default.py` and
  `flashinfer_bench/bench/utils.py`;
- top-k/top-p sampling-mask validation, repeated sampling, empirical frequency
  estimation, and TVD checks inspired by
  `flashinfer_bench/bench/evaluators/sampling.py`;
- adapter design principles for guard/fallback behavior and paged-attention
  metadata capture inspired by
  `flashinfer_bench/integration/flashinfer/adapters/`.

The SGLang wrappers and task ABI in this directory remain task-specific and do
not directly import FlashInfer-Bench runtime adapters.
