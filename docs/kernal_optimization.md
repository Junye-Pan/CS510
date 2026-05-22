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

The current concrete model source is the Hugging Face repository
`Qwen/Qwen3.5-4B`. The downloaded local copy and its manifest should live under
`CS510/runs/models/`, for example:

```text
CS510/runs/models/Qwen--Qwen3.5-4B/
CS510/runs/models/qwen35_4b_manifest.json
```

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

The small `qwen35_4b_vllm_integrated_smoke_v1` workload is only a probe/smoke
path. It is used to confirm model loading, vLLM plugin installation,
candidate-call visibility, and basic deterministic output matching. It should
not determine the official leaderboard score.

Current implemented RMSNorm MVP scoring uses a separate Qwen/vLLM serving-style
evaluation suite named
`qwen35_4b_vllm_prefill_decode_mixed_serving_sweeps_v2`. If verifier passes:

1. run static candidate verification;
2. run H200 live RMSNorm correctness and microbenchmark diagnostics;
3. load the pinned logits-distribution and evaluation-suite baseline artifacts from
   `CS510/runs/baselines/`;
4. build the verified implementation pool;
5. install the candidate through the vLLM `vllm.general_plugins` apply path;
6. run the Qwen/vLLM full-logprob distribution probe on fixed baseline-selected
   positions covering prefill logits, decode logits, low-margin positions, and
   a long-context decode case;
7. compare KL(P||P'), total variation, shift-invariant centered-logit L2/Linf,
   selected token identity, and argmax identity against the pinned baseline;
8. run the Qwen/vLLM prefill/decode/mixed batch/concurrency sweeps;
9. compare suite identity, per-family request identity, generated token ids,
   and generated text against the pinned baseline;
10. enforce candidate-call and fallback-policy thresholds;
11. compute one speedup per family as
   `baseline_family_p90_request_latency_s / candidate_family_p90_request_latency_s`;
12. compute final score as the geometric mean of:
   - prefill family speedup;
   - decode family speedup;
   - mixed family speedup;
13. publish a leaderboard entry only if the result is valid.

Model load time, TTFT proxy, TPOT proxy, throughput, and hidden RMSNorm
microbenchmark geomean speedup are diagnostics. They do not directly determine
the official optimized-candidate score.

The implemented evaluation families are:

- `prefill`: long deterministic prompts with short generation, measuring
  prompt ingestion and first-token path pressure;
- `decode`: short prompts with longer generation, measuring sustained decode
  path pressure;
- `mixed`: varied prompt lengths with medium generation, measuring a
  chat-like latency mix.

This mirrors the FlashInfer-Bench separation between a concrete workload
definition and measured evaluation traces: the suite binds fixed prompts,
generation lengths, sampling parameters, sweep concurrency, and correctness
outputs, then records p50/p90 request latency, TTFT proxy, TPOT proxy,
throughput, and per-family speedup.

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
  `job`, `env`, `telemetry`, `tool`, and `network`; task knowledge is exposed
  as files under `task/knowledge/`.

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

All experiment data, including model download manifests, smoke-test outputs,
control-plane smoke databases, candidate snapshots, verifier summaries, and
benchmark summaries, should be written under `CS510/runs/`. Task code should not
write smoke outputs to `/tmp` or ad-hoc locations outside the repository run
root.

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

## Concrete Build Plan From Current Repository

The first implementation should be an MVP task that can be loaded and statically
verified in the normal development environment, while full model execution stays
gated behind the prepared H200 runtime. This keeps ordinary source editing and
unit tests independent of heavyweight `torch`, `triton`, `vllm`, and local model
installations.

Repository integration:

- create `tasks/llm_inference_qwen35_4b_h200/`;
- expose `LLMInferenceQwen35B4H200Task` from `task.py`;
- declare `CandidateSpec(candidate_root="candidate",
  public_seed_root="initial_candidate", entrypoint_name="manifest.json")`;
- keep import-time dependencies limited to the standard library and
  `agentic_opt`;
- place public contract, public definitions, and the seed candidate under
  `public/`;
- place verifier, schema, workload, scoring, preflight, and apply-runtime code
  under `private/`.

MVP file layout:

```text
tasks/llm_inference_qwen35_4b_h200/
  __init__.py
  task.py
  public/
    TASK.md
    public_contract.md
    definitions/qwen_rmsnorm_h2560_fp16.json
    workloads/public_rmsnorm_shapes.json
    initial_candidate/
      manifest.json
      kernels/rmsnorm.py
  private/
    __init__.py
    schema.py
    verifier.py
    definitions.py
    workloads.py
    scoring.py
    preflight.py
    apply_runtime.py
    vllm_plugin_runtime.py
    vllm_rmsnorm_plugin.py
    qwen_vllm_smoke.py
```

MVP behavior:

- `verify_entry()` always runs static bundle validation;
- the empty/baseline seed candidate is valid and reports baseline fallback
  coverage;
- candidate source paths must be relative, must stay under the candidate root,
  and must not contain `..` or absolute paths;
- each implementation must target the pinned model/framework/GPU/dtype and an
  existing task definition;
- RMSNorm is the first public deterministic kernel definition:
  `qwen_rmsnorm_h2560_fp16`;
- GPU correctness/build checks are skipped unless the H200 live verifier is
  explicitly enabled;
- `probe_entry()` returns public aggregate diagnostics only;
- `evaluate_entry()` reruns verifier and, until the live H200 path is enabled,
  returns a baseline score of `1.0` for a valid baseline-only candidate and a
  clear `official_live_enabled=false` metric.

Current vLLM RMSNorm apply path:

- Qwen/vLLM smoke can receive the verified candidate manifest path from
  `evaluate_entry()`;
- the smoke creates a local `vllm.general_plugins` entry point under the active
  run directory in `CS510/runs/`;
- vLLM engine and worker processes load that plugin through vLLM's normal
  general-plugin mechanism;
- the plugin patches vLLM `RMSNorm` and `GemmaRMSNorm` forward paths before
  model layers are instantiated;
- Qwen 3.5 uses `GemmaRMSNorm`, so the adapter passes `1 + weight` to the
  candidate implementation to preserve Qwen's official norm semantics;
- only `[*, 2560]` CUDA FP16/BF16 tensors covered by the candidate shape guard
  dispatch to the candidate kernel; unsupported head-dim norms fall back to
  vLLM;
- smoke output records `candidate_rmsnorm_used_in_vllm`, candidate call counts,
  shapes, process ids, and fallback reasons in
  `vllm_rmsnorm_apply_trace.jsonl` under the run directory;
- when model smoke is required, `evaluate_entry()` treats missing candidate
  calls as a failed integrated smoke.

Current implementation status as of 2026-05-12:

- the concrete task package exists at
  `tasks/llm_inference_qwen35_4b_h200/`;
- the public candidate contract is directory-based and uses
  `candidate/manifest.json`;
- the public seed candidate is an empty baseline bundle;
- the public example candidate provides a Triton RMSNorm implementation for
  `qwen_rmsnorm_h2560_fp16`;
- the verifier supports static manifest checks, source path validation, target
  checks, forbidden model/tokenizer artifact rejection, and optional H200 live
  RMSNorm correctness checks;
- Qwen 3.5 4B has been downloaded from `Qwen/Qwen3.5-4B` into
  `CS510/runs/models/Qwen--Qwen3.5-4B/`;
- the model manifest is recorded at
  `CS510/runs/models/qwen35_4b_manifest.json`;
- the pinned model revision used for this smoke is
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`;
- the current H200 runtime uses the existing
  `/home/junyep2/local/envs/swestar-vllm/bin/python` environment with
  Torch/Triton/vLLM installed;
- the pinned integrated-smoke baseline artifact is recorded at
  `CS510/runs/baselines/qwen35_4b_vllm_smoke_baseline.json`;
- the pinned evaluation-suite baseline artifact is expected at
  `CS510/runs/baselines/qwen35_4b_vllm_eval_suite_baseline.json`;
- integrated correctness compares candidate Qwen/vLLM outputs against the
  pinned evaluation suite, per-family request identities, generated token ids,
  and generated text;
- official fallback policy requires at least one candidate call and a fallback
  rate no higher than `0.50` by default;
- official optimized-candidate score is now the geometric mean of prefill,
  decode, and mixed Qwen/vLLM family speedups;
- RMSNorm hidden-shape geomean speedup is still recorded as a diagnostic;
- `eval probe` exposes aggregate dispatch, fallback, latency-delta, policy, and
  bottleneck-hint diagnostics, with opt-in integrated model probing through
  `AO_LLM_KERNEL_PROBE_MODEL_SMOKE=1`;
- all download, smoke, verifier, benchmark, control-plane, candidate snapshot,
  and trace outputs are written under `CS510/runs/`.

Latest integrated smoke evidence:

- baseline Qwen/vLLM smoke completed at
  `CS510/runs/qwen_vllm_smoke_20260512T103106Z/`;
- official control-plane `submit` path with baseline correctness and fallback
  policy completed at
  `CS510/runs/control_plane_policy_correctness_20260512T103352Z/`;
- that control-plane run produced evaluation
  `eval_20260512_103355_55c23ed7` and candidate artifact
  `artifact_20260512_103355_819b0808`;
- after switching to end-to-end scoring, a live `evaluate_entry()` check
  completed at `CS510/runs/llm_kernel_evaluate_20260512T174307Z/`;
- that check wrote the direct summary at
  `CS510/runs/direct_end2end_score_file_20260512T174258Z/direct_evaluate_summary.json`;
- the official score component was `generate_elapsed_speedup`;
- baseline generate elapsed was `13.791041135787964s`, candidate generate
  elapsed was `9.676185131072998s`, and the resulting score was
  `1.4252560227998312`;
- RMSNorm microbenchmark diagnostic speedup for that same check was
  `6.846887702134944`;
- the smoke required candidate RMSNorm use and reported
  `candidate_rmsnorm_used_in_vllm=true`;
- vLLM trace recorded `780` candidate calls through `GemmaRMSNorm`;
- fallback policy recorded `384` fallback calls and fallback rate
  `0.32989690721649484`, which is below the default `0.50` threshold;
- integrated correctness recorded token match rate `1.0` and minimum
  top-logprob overlap `0.8` against the pinned baseline;
- observed candidate shapes included `[1, 2560]`, `[2, 2560]`, `[9, 2560]`,
  `[12, 2560]`, `[1024, 2560]`, and `[16384, 2560]`;
- before the end-to-end score switch, that run reported RMSNorm microbenchmark
  geomean speedup `6.777929507002357`.

Latest evaluation-suite baseline evidence:

- baseline Qwen/vLLM prefill/decode/mixed suite completed at
  `CS510/runs/qwen_vllm_eval_suite_20260512T182326Z/`;
- pinned evaluation-suite baseline artifact was written to
  `CS510/runs/baselines/qwen35_4b_vllm_eval_suite_baseline.json`;
- baseline self-score summary was written to
  `CS510/runs/qwen_vllm_eval_suite_20260512T182326Z/baseline_self_score.json`;
- baseline family timings were:
  - prefill: `0.8334658145904541s` for `2748` input tokens and `16` output
    tokens;
  - decode: `1.0406031608581543s` for `22` input tokens and `128` output
    tokens;
  - mixed: `0.5589745044708252s` for `891` input tokens and `128` output
    tokens;
- that baseline was for the previous compact suite and must be regenerated for
  serving-sweep suite v2 before the next live optimized-candidate submit;
- a live example RMSNorm candidate evaluation reached the new suite path at
  `CS510/runs/llm_kernel_evaluate_20260512T182550Z/`, invoked the candidate
  through vLLM, and passed fallback policy, but was invalidated by deterministic
  decode/mixed output drift against the pinned suite baseline.

Current scoring caveat: serving-sweep suite v2 is still executed through the
local vLLM Python runtime rather than a containerized H200 provider.

Environment split:

- normal development and CI use CPU-only static tests;
- live GPU verification is enabled only when an operator sets an explicit flag
  such as `AO_LLM_KERNEL_ENABLE_LIVE=1`;
- live preflight checks the H200 device, pinned model path, tokenizer revision,
  `torch`, `triton`, `vllm`, CUDA/driver versions, and baseline metrics;
- full official scoring should not run unless that preflight passes.

FlashInfer-Bench mapping:

- mirror `Definition` for axes, input/output tensor specs, constraints, and
  reference behavior;
- mirror `Solution` for source files, language, binding, entry point, and build
  metadata;
- mirror `Workload` for generated/public/probe/hidden shapes;
- mirror deterministic evaluator semantics for shape, dtype, finite, and
  elementwise tolerance checks;
- mirror sampling evaluator semantics later for valid masks and total variation
  distance;
- adapt `ApplyRuntime`/`ApplyTable` dispatch from definition+axes to
  definition+phase+dtype+shape bucket+LLM runtime features.

Testing plan:

- task registry can load the new task;
- task contract exposes the directory candidate metadata;
- public seed candidate verifies successfully without GPU packages;
- malformed manifests fail closed with useful public feedback;
- path traversal, absolute source paths, duplicate implementation ids, unknown
  definitions, unsupported languages, and target mismatches are rejected;
- baseline score math and geomean scoring are covered by unit tests;
- H200/vLLM smoke tests are isolated behind an opt-in environment variable.

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
