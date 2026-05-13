# Validate the RMSNorm Contract

Start from the baseline candidate and add a minimal `qwen_rmsnorm_h2560_fp16`
implementation entry. Keep the implementation simple and focus on making the
manifest, source paths, callable name, and shape guard pass static validation.

The official RMSNorm path is destination-passing:

```python
def run(x, weight, output, eps=1e-6):
    ...
    return None
```

For Qwen 3.5, the live vLLM path uses `GemmaRMSNorm`. The apply adapter passes
the effective `1 + weight` tensor to the candidate, so the candidate should use
the supplied `weight` directly and should not apply another Gemma offset.

Minimum acceptance sequence:

1. `eval verify` should pass static manifest and source validation.
2. With live verification enabled, RMSNorm correctness should pass for public
   and hidden `[num_tokens, 2560]` shapes.
3. `eval submit` should show `candidate_rmsnorm_used_in_vllm=true`.
4. The full-logprob distribution probe must stay within the pinned baseline
   bounds for KL(P||P'), total variation, centered-logit L2/Linf, selected
   token identity, and argmax identity.
5. Integrated smoke output must match the pinned baseline artifact for workload
   identity, generated token ids, generated text, and top-logprob token sets.
6. The fallback policy must pass: at least one candidate call and fallback rate
   no higher than the official threshold, currently `0.50` by default.

Use the RMSNorm microbenchmark speedup as a diagnostic while validating this
contract. The small integrated smoke is also probe-only. The official
optimized-candidate score comes from the pinned Qwen/vLLM prefill/decode/mixed
serving-style evaluation suite:

```text
geomean(prefill_speedup, decode_speedup, mixed_speedup)
```

Each family speedup is baseline family p90 request latency divided by candidate
family p90 request latency over fixed batch/concurrency sweeps. That score only
counts after distribution correctness, integrated deterministic correctness,
and fallback policy pass. TTFT proxy, TPOT proxy, throughput, and RMSNorm
microbenchmarks are diagnostics.
