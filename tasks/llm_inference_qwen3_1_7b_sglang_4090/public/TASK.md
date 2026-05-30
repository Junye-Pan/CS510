# Qwen3 1.7B SGLang Kernel Optimization

Optimize a fixed kernel surface used while serving `Qwen/Qwen3-1.7B` with
SGLang on one RTX 4090-class GPU. The goal is to improve inference performance
while preserving the model's externally visible behavior.

## Target Runtime

- Model: `Qwen/Qwen3-1.7B`
- Framework: SGLang
- GPU class: RTX 4090
- Runtime dtype: BF16
- Candidate kernel language: Python/Triton
- Execution mode: server-side SGLang inference with task-owned integration

The serving stack, model weights, tokenizer, launch settings, request set, and
official scoring are controlled by the task. Candidate code should implement
only the declared kernel entrypoints.

## Editable Files

The candidate entrypoint is:

```text
candidate/manifest.json
```

Do not edit `manifest.json`. The writable files are only:

```text
candidate/kernels/rmsnorm.py
candidate/kernels/fused_add_rmsnorm.py
candidate/kernels/swiglu.py
candidate/kernels/attention_backend.py
candidate/kernels/sampling_backend.py
```

The candidate directory must not contain additional files, generated artifacts,
copied framework code, model weights, private adapters, or symlinks.

The fixed public ABI is described in `public_contract.md`. Preserve the
declared function names, call signatures, return shapes, dtypes, and semantic
behavior.

## Starting Baseline

The workspace starts from a valid baseline already materialized under
`candidate/kernels/`. Treat those files as the reference implementation before
making changes:

- `rmsnorm.py`, `fused_add_rmsnorm.py`, and `swiglu.py` delegate to the
  installed `sgl_kernel` operators;
- `attention_backend.py` and `sampling_backend.py` preserve the task-owned
  fallback path.

Read the current baseline code first, then run the verifier before or soon
after the first material edit. Optimize incrementally from this baseline rather
than replacing the full surface at once. Additional public notes are available
under `task/knowledge/`.

## Safety Boundary

Candidate code must stay inside the declared kernel surface. It must not:

- patch, monkeypatch, replace, or reconfigure SGLang;
- import task-private modules;
- import SGLang scheduler, tokenizer, model runner, or model internals;
- mutate `sys.modules` or use dynamic imports;
- start subprocesses;
- read or write files outside the allowed candidate files;
- load model weights or create a separate model instance;
- change request text, token ids, tokenizer behavior, sampling parameters,
  response metadata, or output formatting.

If a runtime path is unsupported by your implementation, return through the
provided fallback path when one is part of the public ABI. Do not guess at
unknown framework internals.

## Validation And Scoring

The task provides a verifier for correctness and safety checks, and an official
evaluator for scoring. Passing local experiments is not enough; submitted
candidates must pass the task verifier and evaluator.

Typical worker commands are:

```bash
eval verify candidate/manifest.json
eval submit candidate/manifest.json
```

Keep temporary experiments outside the candidate directory unless they are one
of the allowed kernel files.
