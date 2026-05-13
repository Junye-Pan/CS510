# Plan And Remaining Gaps

This file tracks what is still missing before the Qwen 3.5 4B H200 kernel
optimization task is a complete task design rather than a working RMSNorm MVP.

## Current Baseline

The MVP now has a real end-to-end path:

- Qwen 3.5 4B is downloaded under `CS510/runs/models/`;
- the RMSNorm task package exists under
  `tasks/llm_inference_qwen35_4b_h200/`;
- static verifier and public candidate contract are implemented;
- H200 live RMSNorm correctness and benchmark checks work when
  `AO_LLM_KERNEL_ENABLE_LIVE=1`;
- Qwen/vLLM integrated smoke can require candidate RMSNorm use;
- the candidate RMSNorm is loaded through vLLM's official
  `vllm.general_plugins` mechanism;
- the pinned baseline smoke artifact is recorded at
  `CS510/runs/baselines/qwen35_4b_vllm_smoke_baseline.json`;
- candidate integrated smoke is compared against the pinned baseline token ids
  and top-logprob token sets;
- official optimized-candidate evaluation now also loads the pinned
  logits-distribution baseline at
  `CS510/runs/baselines/qwen35_4b_logits_distribution_baseline.json` plus its
  NPZ tensor archive, then compares full-logprob KL(P||P'), total variation,
  centered-logit L2/Linf/RMSE, selected token identity, and argmax identity on
  fixed baseline-selected positions;
- the logits-distribution probe covers prefill logits, decode logits,
  low-margin positions, and a long-context decode case before the serving-style
  score is computed;
- the fallback policy is enforced with default thresholds of at least one
  candidate call and at most `0.50` fallback rate;
- `eval probe` can still use the small integrated smoke workload, but
  `evaluate_entry()` now uses a distinct prefill/decode/mixed Qwen/vLLM
  serving-style sweep suite for optimized-candidate scoring;
- official optimized-candidate score is the geometric mean of prefill, decode,
  and mixed family p90 request-latency speedups from fixed batch/concurrency
  sweeps;
- the serving-style suite records request latency, TTFT proxy, TPOT proxy,
  throughput, p50, and p90 diagnostics per family;
- the full environment manifest is materialized at
  `CS510/runs/models/qwen35_4b_environment_manifest.json` and validated by
  live preflight for model/tokenizer identity, vLLM, Torch, Triton, CUDA,
  driver, Python, GPU, dtype, and baseline artifact presence;
- RMSNorm hidden-shape geomean speedup is retained as a diagnostic, not the
  leaderboard score;
- `eval probe` now returns aggregate dispatch, fallback, latency-delta, policy,
  and bottleneck-hint diagnostics, with opt-in integrated model probe via
  `AO_LLM_KERNEL_PROBE_MODEL_SMOKE=1`;
- the latest live end-to-end `evaluate_entry()` check completed at
  `CS510/runs/llm_kernel_evaluate_20260512T174307Z/` with official score
  `1.4252560227998312`, baseline generate elapsed
  `13.791041135787964s`, candidate generate elapsed `9.676185131072998s`,
  and RMSNorm microbenchmark diagnostic speedup `6.846887702134944`;
- that run completed with
  `candidate_rmsnorm_used_in_vllm=true`, `780` candidate calls,
  `384` fallback calls, fallback rate `0.32989690721649484`, token match rate
  `1.0`, and min top-logprob overlap `0.8`;
- the previous evaluation-suite baseline run completed at
  `CS510/runs/qwen_vllm_eval_suite_20260512T182326Z/`, wrote
  `CS510/runs/baselines/qwen35_4b_vllm_eval_suite_baseline.json`, and confirmed
  baseline self-score `1.0` for score component
  `prefill_decode_mixed_geomean_speedup`;
- the serving-style suite v2 baseline was regenerated at
  `CS510/runs/qwen_vllm_eval_suite_20260512T185436Z/`; the pinned baseline
  artifact now validates and self-scores `1.0` for score component
  `prefill_decode_mixed_serving_p90_geomean_speedup`;
- the logits-distribution baseline was generated at
  `CS510/runs/qwen_logits_distribution_20260512T193225Z/`; baseline
  self-compare passes with max KL, TV, centered-logit L2, centered-logit Linf,
  and centered-logit RMSE all equal to `0.0`;
- a live example RMSNorm candidate evaluation completed at
  `CS510/runs/llm_kernel_evaluate_20260512T185914Z/`; it exercised the vLLM
  plugin path with candidate RMSNorm calls and trace output, then was correctly
  invalidated by deterministic decode token drift against the pinned baseline;
- after adding the logits-distribution gate, the same example class of RMSNorm
  candidate completed at `CS510/runs/llm_kernel_evaluate_20260512T193744Z/`;
  it invoked candidate RMSNorm in vLLM, then was invalidated before serving
  scoring because centered-logit drift exceeded tolerance
  (`record 0 centered-logit L2 15.7729 > 10`);
- all observed run artifacts are stored under `CS510/runs/`.

## Resolved In Current MVP

These gaps from the previous plan are implemented for the RMSNorm MVP:

- pinned baseline metrics exist for both the probe smoke workload and the
  evaluation-suite workload;
- integrated correctness now compares candidate evaluation-suite output against
  pinned baseline suite identity, request identity, token ids, and generated
  text;
- fallback policy thresholds are codified and enforced during official
  evaluation;
- probe feedback now exposes aggregate dispatch/fallback/latency-policy
  diagnostics and can run an opt-in integrated model probe;
- official score has moved from RMSNorm microbenchmark geomean speedup to an
  integrated Qwen/vLLM prefill/decode/mixed serving-style sweep suite;
- public task documentation describes live apply, `GemmaRMSNorm` semantics,
  fallback policy, baseline comparison, and the official scoring steps.
- official scoring has been broadened from a compact one-shot in-process suite
  to deterministic serving-style batch/concurrency sweeps with p50/p90 request
  latency, TTFT proxy, and TPOT proxy diagnostics;
- token/text matching is no longer the only official integrated correctness
  gate: a Qwen/vLLM full-logprob distribution probe now checks KL(P||P'), total
  variation, centered-logit L2/Linf/RMSE, selected token identity, and argmax
  identity before performance scoring;
- environment manifest handling now accepts the legacy download manifest,
  normalizes it to `agentic_opt.llm_kernel_environment.v1`, materializes a
  complete H200/vLLM runtime manifest, and fails live preflight closed when
  required model/runtime/baseline identifiers are missing or inconsistent.

## Remaining Gaps To Complete The Task Design

1. Official evaluation still uses a local framework environment.
   The latest run used `env_framework_current` with the existing swestar-vllm
   Python. The final task should use a pinned task base environment, and
   eventually the planned `docker_image` or equivalent H200 provider.

2. Candidate apply runtime only covers RMSNorm.
   Add more definitions after RMSNorm is stable: RoPE, attention decode,
   attention prefill, KV cache update, logits postprocess, and sampling. Each
   definition needs shape contracts, verifier rules, apply hooks, fallback
   policy, and public/probe/hidden workloads.

3. Build/cache isolation is minimal.
   Candidate imports work for Python/Triton sources, but there is no digest-keyed
   build cache, no persistent compile artifact record, and no hardened import
   sandbox beyond path/schema validation.

4. Fallback coverage is still aggregate-first.
   Official submit enforces candidate calls and a max fallback rate, but it does
   not yet require a coverage matrix by phase, workload family, layer bucket,
   shape, and fallback reason. That matrix is needed to explain mixed
   fallback/candidate performance and to prevent aggregate rates from hiding
   missing coverage on important paths.

5. Sampling-kernel validation is not implemented yet.
   RMSNorm is deterministic, but future logits postprocess or sampling kernels
   must use repeated fixed-logit statistical tests rather than single-output
   comparisons. Planned checks include empirical TV distance, chi-square or
   G-test statistics, top-p boundary token frequency, multi-seed runs, and
   enough samples to catch distribution drift.

6. Distribution thresholds are not yet calibrated from repeated baseline runs.
   The new logits-distribution gate has fixed conservative thresholds. The next
   hardening step is baseline-vs-baseline calibration across repeated runs and
   threshold selection from p99/p999 self-drift, with separate reporting for
   low-margin and long-context cases.

7. Distribution-gate regression tests are unit-level plus one live baseline
   self-compare.
   Add a live negative-control candidate that perturbs RMSNorm just enough to
   keep some tokens unchanged but exceed KL/TV or centered-logit thresholds.

## Next Implementation Steps

1. Add fallback coverage matrices by phase, family, layer bucket, shape, and
   reason, then make critical-path coverage a hard gate.
2. Add repeated-sampling statistical validation before opening sampling or
   logits-postprocess kernels to candidates.
3. Run repeated baseline-vs-baseline calibration and update distribution
   thresholds from measured self-drift.
4. Add the next kernel definition and apply hook, likely RoPE or attention
   decode.
5. Add digest-keyed build/cache records and stronger import isolation.
