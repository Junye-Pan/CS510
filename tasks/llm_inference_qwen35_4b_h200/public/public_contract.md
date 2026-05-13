# Public Contract

The model, tokenizer, vLLM version, CUDA stack, prompts, sequence lengths,
sampling parameters, hidden workloads, and baseline metrics are fixed by the
server. Do not download, replace, or mutate them.

Allowed candidate changes:

- kernel source files declared in `candidate/manifest.json`;
- dispatch guards and shape coverage metadata in the manifest;
- implementation priorities and baseline fallback declarations.

Disallowed candidate content:

- model weights or tokenizer files;
- absolute source paths or paths containing `..`;
- hidden/probe workload data;
- runtime dependency changes for official evaluation;
- framework monkey patches outside the official apply runtime.

Manifest schema:

```json
{
  "schema": "agentic_opt.llm_kernel_bundle.v1",
  "target": {
    "model": "qwen-3.5-4b",
    "framework": "vllm",
    "gpu": "H200",
    "dtype": "fp16"
  },
  "implementations": []
}
```

An RMSNorm implementation entry should look like:

```json
{
  "id": "rmsnorm_h2560_triton_v1",
  "definition": "qwen_rmsnorm_h2560_fp16",
  "language": "triton",
  "binding": "torch",
  "entry_point": "kernels/rmsnorm.py::run",
  "sources": ["kernels/rmsnorm.py"],
  "shape_guard": {
    "num_tokens": [1, 32768],
    "hidden": 2560
  },
  "priority": 50,
  "fallback": "baseline"
}
```

For the MVP, `python` and `triton` source files are accepted for static
validation. Live correctness and speed scoring require the prepared H200
environment.

Runtime apply contract:

- candidates do not patch vLLM directly;
- the task-owned apply runtime installs verified RMSNorm implementations through
  vLLM's `vllm.general_plugins` path;
- Qwen 3.5 hidden-size layer norms use vLLM `GemmaRMSNorm`, so the adapter
  supplies the effective `1 + weight` tensor to the candidate;
- the candidate should write into `output` and return `None`;
- unsupported shapes, unsupported dtypes, and guard misses fall back to vLLM's
  original implementation;
- candidate runtime exceptions or missing required candidate calls invalidate
  official submit.

Official integrated checks:

- `eval probe` may use a small integrated smoke workload for cheap feedback;
- live submit uses a pinned logits-distribution baseline artifact under
  `CS510/runs/baselines/` and compares full-logprob KL(P||P'), total variation,
  centered-logit L2/Linf, selected token identity, and argmax identity on
  fixed baseline-selected prefill, decode, low-margin, and long-context
  positions;
- live submit uses a separate pinned prefill/decode/mixed serving-style
  evaluation-suite baseline artifact under `CS510/runs/baselines/`;
- generated token ids and generated text must match the pinned deterministic
  baseline for every evaluation-suite request;
- the default fallback policy requires at least one candidate call and fallback
  rate no higher than `0.50`;
- official score for optimized candidates is the geometric mean of prefill,
  decode, and mixed family p90 request-latency speedups from fixed
  batch/concurrency sweeps;
- model load time and RMSNorm microbenchmark speedup are reported as
  diagnostics and are not the leaderboard score;
- request latency, TTFT proxy, TPOT proxy, p50/p90, and throughput are also
  reported as serving diagnostics;
- `eval probe` reports aggregate dispatch hit rate, fallback counts, latency
  deltas, policy thresholds, and bottleneck hints without exposing hidden
  workload details.
