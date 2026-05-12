# Open Design Questions

## LLM Kernel Optimization Task

- Fill the H200 environment manifest with exact versions and immutable
  identifiers before the first agent assignment: model revision/digest,
  tokenizer revision/digest, framework version, Torch, Triton, CUDA, driver,
  Python, and baseline artifact id.
- Confirm whether the first implementation standardizes on vLLM for the MVP or
  keeps a framework adapter boundary that can switch to SGLang later.
- Decide whether dependency overlays are disabled entirely for this task or
  allowed only for non-official diagnostics. Official evaluation should use the
  pinned task base environment.
- Define the acceptable fallback policy for official scoring: ordinary guard
  misses can fallback, but candidate runtime exceptions and excessive fallback
  rate should likely invalidate the submission.
