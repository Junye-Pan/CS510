# Qwen 3.5 4B H200 LLM Kernel Optimization

Optimize allowed inference kernels for a fixed Qwen 3.5 4B serving setup on
H200. The server owns the model, tokenizer, vLLM runtime, workload mix,
sampling parameters, verifier, apply runtime, and official scoring.

Your candidate is a directory rooted at `candidate/`. The entrypoint is
`candidate/manifest.json`.

```text
candidate/
  manifest.json
  kernels/
    rmsnorm.py
```

The MVP public definition is `qwen_rmsnorm_h2560_fp16`. It exposes a
destination-passing interface for RMSNorm over `[num_tokens, 2560]` FP16
activations. The baseline seed candidate declares no optimized implementation
and uses the server-owned framework path.

For Qwen 3.5, vLLM routes the model's hidden-size layer norms through
`GemmaRMSNorm`. The task-owned apply runtime preserves that framework semantic
by passing the effective `1 + weight` scale to your RMSNorm implementation.
Your `run(hidden_states, weight, output)` function should implement ordinary
RMSNorm with the weight tensor it receives.

Useful commands through the control plane:

- `eval verify` validates the manifest and candidate bundle;
- `eval probe` returns public aggregate dispatch, fallback, latency, and
  bottleneck diagnostics;
- official submit reruns verification before scoring.

Full H200/vLLM speed scoring is gated by the server-side live verifier. Without
that live path, only the baseline-only seed candidate receives the baseline
score. Optimized candidates require the live Qwen/vLLM model preflight.

The small integrated smoke workload used by `eval probe` is not the leaderboard
benchmark. Official submit uses a separate prefill/decode/mixed serving-style
sweep suite with fixed batch/concurrency settings.

Official scoring steps for optimized candidates:

1. rerun static verification;
2. run live RMSNorm correctness and microbenchmark diagnostics;
3. load the pinned logits-distribution and evaluation-suite baseline artifacts from
   `CS510/runs/baselines/`;
4. install the candidate through vLLM's `vllm.general_plugins` apply path;
5. run a fixed Qwen/vLLM full-logprob distribution probe over prefill logits,
   decode logits, low-margin positions, and a long-context decode case;
6. reject candidates whose KL(P||P'), total variation, centered-logit L2/Linf,
   selected token identity, or argmax identity drift beyond the pinned baseline;
7. run fixed Qwen/vLLM prefill/decode/mixed batch/concurrency sweeps;
8. compare suite identity, per-request identity, generated token ids, and
   generated text against the pinned baseline;
9. enforce candidate-call and fallback-policy thresholds;
10. score as the geometric mean of prefill, decode, and mixed family p90
   request-latency speedups.

RMSNorm microbenchmark speedup is reported as a diagnostic. The leaderboard
score is the integrated Qwen/vLLM serving-style sweep speedup. Request latency,
TTFT proxy, TPOT proxy, p50, p90, and throughput are reported as diagnostics.
