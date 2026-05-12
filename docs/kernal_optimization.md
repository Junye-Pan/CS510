# LLM Kernel Optimization Task Design

This document designs a new `agentic_opt` task for testing whether coding
agents can continuously improve small-LLM inference efficiency through repeated
kernel-level optimization. The task name is intentionally not finalized here;
the likely package name is `tasks/llm_inference_qwen35_4b_h200/`.

The core requirement is that the coding agent remains the inner-loop search
actor, while the system owns the model, runtime environment, verifier, apply
mechanism, workloads, official evaluation, and leaderboard state.

## Decision: Model And Runtime Are Preinstalled

Before any optimization assignment starts, the target model and serving stack
must already be installed, pinned, and verified on the target machine.

For this task, an agent must not download, replace, modify, or reconfigure the
model during optimization. The model, tokenizer, framework, CUDA stack,
benchmark workload, sampling parameters, and baseline are part of the
server-owned task environment.

The task environment must expose a read-only environment manifest to agents,
for example:

```json
{
  "model": {
    "name": "qwen-3.5-4b",
    "local_path": "/models/qwen-3.5-4b",
    "revision": "<exact model revision or digest>",
    "tokenizer_revision": "<exact tokenizer revision or digest>"
  },
  "framework": {
    "name": "vllm",
    "version": "<pinned version>",
    "commit": "<optional git commit>"
  },
  "runtime": {
    "gpu": "H200",
    "dtype": "fp16",
    "python": "<pinned version>",
    "torch": "<pinned version>",
    "triton": "<pinned version>",
    "cuda": "<pinned version>",
    "driver": "<pinned version>"
  },
  "baseline": {
    "artifact_id": "<server artifact>",
    "created_from_environment_fingerprint": "<fingerprint>",
    "metrics_path": "baseline_metrics.json"
  }
}
```

Assignment creation should be gated on an environment preflight:

- model directory exists and is read-only to the worker;
- tokenizer files match the pinned revision;
- framework imports and reports the expected version;
- CUDA device is available and is an H200-class GPU;
- dtype and inference configuration match the task contract;
- a baseline inference run has completed and produced baseline metrics;
- public verifier seed can be evaluated in the same environment.

This repository can still be edited on a no-GPU development machine, but actual
agent sessions and official evaluations for this task should run only on the
prepared H200 environment.

## Objective

The experiment measures whether coding agents can improve end-to-end inference
efficiency for a fixed small LLM by iteratively discovering bottlenecks,
modifying allowed kernels or wrappers, running verifier/probe feedback, and
submitting official evaluations.

The score is not a single microbenchmark result. The official score is an
end-to-end speedup over a fixed baseline serving configuration.

## Fixed Experimental Conditions

The following are fixed before the first agent session:

- model: Qwen 3.5 4B, installed locally and pinned by digest or revision;
- framework: one fixed inference framework, initially vLLM for the MVP;
- hardware: H200 GPU;
- dtype: FP16;
- workload families: prefill-heavy, decode-heavy, and mixed chat;
- baseline: unoptimized framework run in the same pinned environment;
- prompts, sequence lengths, generation lengths, seeds, and sampling parameters.

Agents may not change:

- model weights;
- tokenizer;
- prompt corpus;
- benchmark shape distribution;
- generation length;
- sampling parameters;
- official framework version;
- official environment dependencies;
- hidden/probe workload data.

Agents may optimize only candidate kernels, wrappers, dispatch guards, layout
handling, and allowed replacement logic.

## Candidate Shape

The task should use a directory candidate, not a single Python file:

```python
CandidateSpec(
    candidate_root="candidate",
    public_seed_root="initial_candidate",
    entrypoint_name="manifest.json",
)
```

Workspace layout:

```text
candidate/
  manifest.json
  kernels/
    rmsnorm.py
    rope.py
    attention_decode.py
    sampling.py
```

The manifest declares a bundle of implementations:

```json
{
  "schema": "agentic_opt.llm_kernel_bundle.v1",
  "target": {
    "model": "qwen-3.5-4b",
    "framework": "vllm",
    "gpu": "H200",
    "dtype": "fp16"
  },
  "implementations": [
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
  ]
}
```

The manifest should be close to FlashInfer-Bench `Solution` semantics, but a
top-level bundle is needed because a single agent attempt may submit multiple
candidate kernels for different definitions.

## Kernel Definitions

Task-owned public definitions should describe the candidate interface:

- RMSNorm and fused RMSNorm;
- RoPE;
- MLP activation or epilogue fusion;
- GEMM/Linear where replacement is safe;
- attention prefill;
- attention decode;
- KV cache update;
- logits postprocess;
- top-k/top-p/temperature sampling.

Each definition must specify:

- name and op type;
- input tensors, output tensors, dtype, and shape axes;
- shape constraints and allowed ranges;
- reference behavior;
- verifier class: deterministic, low precision, or stochastic;
- framework hook or adapter target;
- public workload shapes;
- hidden/probe workload membership.

The design should reuse FlashInfer-Bench concepts:

- `Definition` for input/output/axis/reference schema;
- `Solution` for source files, language, binding, entrypoint, and build spec;
- `Workload` for generated or recorded inputs;
- `Evaluation` for correctness/performance status.

## Verifier

The verifier is a required gate. A candidate that fails verifier never reaches
official end-to-end evaluation.

Verifier stages:

1. Static bundle validation:
   - manifest schema is valid;
   - source paths are relative and stay under `candidate/`;
   - no absolute paths or `..`;
   - every definition exists;
   - every implementation targets the pinned model/framework/dtype;
   - forbidden framework/model/tokenizer files are not present in the bundle.

2. Build validation:
   - compile or import the declared implementation;
   - validate callable signature against the definition;
   - cache build output by candidate digest;
   - fail closed on compile/import errors.

3. Per-kernel correctness:
   - deterministic kernels use strict shape/dtype/finite/allclose checks;
   - low precision kernels use matched-ratio checks plus max error caps;
   - stochastic kernels use distribution checks rather than single-output equality.

4. Integrated smoke test:
   - insert verified kernels through the official apply runtime;
   - run a small fixed inference batch;
   - compare selected logits and top-k token sets against baseline;
   - verify that generated behavior stays within the task tolerance.

### Deterministic Kernels

For RMSNorm, RoPE, GEMM, attention, KV update, logits processors, and similar
deterministic kernels:

- output shape must exactly match reference;
- output dtype must exactly match definition;
- output must not contain NaN or Inf;
- elementwise error must satisfy task tolerance across multiple shapes;
- integrated logits must stay close to baseline.

Any violation makes the candidate invalid.

### Low Precision Kernels

For FP8, INT8, INT4, or other lower-precision paths:

- most output elements must satisfy a strict tolerance;
- a small outlier ratio may be allowed;
- max absolute and relative error must still be bounded;
- logits distribution drift must be bounded;
- top-k token set overlap must remain high.

This follows the FlashInfer-Bench low-bit evaluator pattern: tolerate realistic
quantization error without allowing crude approximations that only look fast.

### Stochastic Kernels

For sampling kernels:

- use fixed logits and sampling parameters;
- run the candidate many times;
- compare empirical distribution against the theoretical distribution;
- fail immediately if a sample appears outside the valid mask;
- use total variation distance or a similar distribution metric.

This follows the FlashInfer-Bench sampling evaluator pattern.

## Apply Runtime

Agents must not monkey patch the inference framework directly. The task owns an
official apply runtime.

Flow:

1. agent submits a candidate bundle;
2. verifier compiles and validates each implementation;
3. only verified implementations enter the usable pool;
4. apply runtime receives framework calls through official adapters;
5. dispatch uses definition, dtype, phase, runtime shape, and candidate guard;
6. if no implementation matches, fallback to the original framework path;
7. if a candidate raises at runtime, fallback is used and the event is recorded.

For official scoring, repeated candidate runtime errors should invalidate the
submission rather than silently scoring the baseline path.

The apply table should be inspired by FlashInfer-Bench `ApplyRuntime` and
`ApplyTable`, but adapted for end-to-end LLM inference:

- FlashInfer-Bench table key: definition plus workload axes;
- this task key: definition plus runtime phase, dtype, shape bucket, head count,
  head dim, page size, sequence lengths, batch size, and sampling mode.

## Workloads

The task has three workload visibility levels.

Public workload:

- visible to agents;
- used for local debugging and public verifier shapes;
- should cover representative but not exhaustive request patterns.

Probe workload:

- not fully visible to agents;
- accessible only through `eval probe`;
- returns aggregate diagnostics such as dispatch hit rate, fallback count,
  per-family latency deltas, and kernel-level regressions.

Hidden workload:

- used only by official submit/evaluation;
- prompt text, shape distribution, seeds, and exact request mix are private;
- determines leaderboard score.

Workload families:

- prefill-heavy: score mainly TTFT;
- decode-heavy: score mainly TPOT;
- mixed chat: score request latency or throughput.

## Official Evaluation

Official evaluation must rerun verifier first. If verifier fails, score is
zero and no end-to-end evaluation runs.

If verifier passes:

1. load the pinned baseline environment;
2. load the candidate artifact snapshot;
3. build the verified implementation pool;
4. enable official apply runtime;
5. run hidden workload;
6. record correctness, speed, memory, dispatch, and fallback metrics;
7. compute final speedup score;
8. publish a leaderboard entry only if the result is valid.

Suggested score:

```text
score = geomean(
  prefill_heavy_ttft_speedup,
  decode_heavy_tpot_speedup,
  mixed_chat_latency_or_throughput_speedup
)
```

Invalid conditions:

- verifier failure;
- logits drift beyond tolerance;
- sampling distribution failure;
- OOM;
- framework/model/tokenizer mutation;
- hidden workload parameter mutation;
- excessive candidate exceptions;
- excessive fallback rate when the candidate claimed coverage.

## Probe Feedback

Probe should help agents search without leaking hidden workloads.

Public-safe feedback may include:

- per-definition verifier status;
- build error summaries;
- dispatch hit/miss counts by kernel family;
- fallback counts;
- candidate exception counts;
- median/p95 TTFT, TPOT, and request latency deltas by workload family;
- peak memory delta;
- high-level bottleneck hints such as "attention_decode hit but slower".

Probe should not expose:

- hidden prompt text;
- exact hidden shape distribution;
- exact hidden seeds;
- exact per-request hidden traces.

## Integration With Current `agentic_opt`

The current system already supports the main task boundary:

- `CandidateSpec` supports directory candidates;
- official `submit` snapshots the candidate artifact;
- `EvaluationService` already runs `verify_entry()` before `evaluate_entry()`;
- leaderboard/incumbent records are created only after valid official results;
- semantic tools expose `ctx`, `artifact`, `eval`, `finding`, `notebook`,
  `job`, `env`, `telemetry`, `tool`, `knowledge`, and `network`.

The task should implement:

```text
tasks/llm_inference_qwen35_4b_h200/task.py
tasks/llm_inference_qwen35_4b_h200/public/
tasks/llm_inference_qwen35_4b_h200/private/
```

`task.py` should keep public/private separation:

- public files describe candidate contract and public definitions;
- private code owns verifier, workloads, baseline metrics, and apply runtime;
- hidden/probe files are never materialized into the worker workspace.

## Runtime Environment Requirements

This task should eventually use the planned `docker_image` environment provider
for reproducible H200 execution. For the first implementation, it can run on an
H200 host with a pinned `local_venv` and preinstalled model/framework paths.

Important implementation detail: because development machines may not have a
GPU, task authoring should not require running the full model verifier locally.
The GPU environment readiness check belongs to H200 assignment/evaluation
preflight, not ordinary documentation or source editing.

Recommended environment policy:

- official submit uses only the task base environment;
- dependency overlays are allowed for local worker diagnostics only if policy
  permits them;
- overlays do not affect leaderboard-eligible official evaluation;
- external internet should be denied for official evaluation;
- model and tokenizer mounts are read-only;
- candidate artifact is the only mutable input to official evaluation.

## Initial Implementation Plan

1. Add task skeleton with directory candidate contract.
2. Add static manifest verifier and public seed candidate.
3. Add FlashInfer-style schema conversion for definitions and solutions.
4. Add deterministic verifier for one or two initial kernels.
5. Add probe feedback over synthetic/public workloads.
6. Add H200 environment manifest and baseline preflight.
7. Add official vLLM apply runtime.
8. Add hidden workloads and final speedup scoring.
9. Add low-precision and stochastic verifier classes.
10. Move official evaluation to `docker_image` when the provider is ready.

## Minimal MVP Scope

The first usable version should support:

- pinned H200 vLLM environment;
- installed Qwen 3.5 4B model and tokenizer;
- candidate bundle manifest;
- RMSNorm or RoPE deterministic kernel definition;
- public verifier shapes;
- official apply wrapper for that kernel;
- one public workload and one hidden workload;
- score near 1.0 for the empty/baseline candidate.

After this MVP works, add attention decode and sampling, since those are more
important for real inference speed but much harder to verify safely.
