# Qwen3 1.7B SGLang Decode Attention Kernel Optimization

Optimize SGLang's Triton decode-attention operator for `Qwen/Qwen3-1.7B` on
one RTX 4090-class GPU. This is a tensor-level microkernel task, not a
model-serving task.

## Target Runtime

- Model shape: `Qwen/Qwen3-1.7B`
- Query heads: `16`
- KV heads: `8`
- Head dimension: `128`
- Attention form: GQA with `kv_group_num = 2`
- Runtime dtype: BF16 inputs and BF16 output
- Scratch dtype: FP32 `attn_logits` and `attn_lse`
- Candidate language: Python plus Triton kernels
- Reference: installed SGLang `decode_attention_fwd`

The verifier and evaluator construct synthetic paged-KV decode workloads that
match SGLang's decode kernel ABI. They do not launch a SGLang server, load model
weights, call tokenizer code, or score HTTP generation latency.

## Editable Files

The candidate entrypoint is:

```text
candidate/manifest.json
```

Do not edit `manifest.json`. The only writable kernel file is:

```text
candidate/kernels/decode_attention.py
```

The candidate directory must not contain additional files, generated artifacts,
copied framework code, private adapters, or symlinks.

## Starting Baseline

The initial candidate is a valid fallback-only baseline:

```python
def run(..., fallback):
    return fallback()
```

This delegates to the official SGLang decode-attention implementation. Its
official score should be near `1.0`. Optimize incrementally from this baseline.

## Validation And Scoring

Typical worker commands are:

```bash
eval verify candidate/manifest.json
eval submit candidate/manifest.json
```

The verifier runs:

1. static candidate admission checks;
2. direct tensor-level correctness checks against SGLang;
3. small public CUDA-event timings.

The official evaluator runs a larger direct tensor-level benchmark. The score is
correctness-gated. If all workloads pass, the score is the geometric mean of
per-workload speedups:

```text
reference_decode_attention_ms / candidate_decode_attention_ms
```

The public feedback includes per-workload `ref_ms`, `cand_ms`, `speedup`,
`max_abs_error`, `matched_ratio`, and fallback-call counts.

## Safety Boundary

Candidate code must stay inside the declared kernel surface. It must not:

- import SGLang or task-private modules;
- patch, monkeypatch, replace, or reconfigure runtime state;
- mutate read-only inputs such as `q`, `k_buffer`, `v_buffer`, `kv_indptr`,
  `kv_indices`, or `num_kv_splits`;
- start subprocesses;
- perform network or filesystem access;
- build native extensions;
- use dynamic imports;
- add files outside `candidate/kernels/decode_attention.py`.

Candidate code may mutate only `o`, `attn_logits`, and `attn_lse`. If a shape
or feature is unsupported by your implementation, call the provided
`fallback()` callable.

## Optimization Notes

The main path is Qwen3 GQA decode:

- `q.shape == (batch, 16, 128)`
- `k_buffer.shape == (kv_pool_size, 8, 128)`
- `v_buffer.shape == (kv_pool_size, 8, 128)`
- `o.shape == (batch, 16, 128)`
- page size is `1`
- `kv_indices` maps logical sequence positions to KV-cache slots

Useful directions include specializing small single-split decode, avoiding the
two-stage scratch path where it is unnecessary, tuning split-K behavior for
long contexts, and reducing global-memory traffic through `attn_logits` and
`attn_lse`.
