# Public Contract

The candidate entrypoint is `candidate/manifest.json`. The only writable file
is:

- `candidate/kernels/decode_attention.py`

Kernel entrypoint:

```python
def run(
    q,
    k_buffer,
    v_buffer,
    o,
    kv_indptr,
    kv_indices,
    attn_logits,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    k_scale,
    v_scale,
    logit_cap=0.0,
    sinks=None,
    xai_temperature_len=-1,
    has_mla=False,
    use_pdl=False,
    *,
    fallback,
): ...
```

Expected fixed model facts:

- query heads: `16`
- KV heads: `8`
- head dimension: `128`
- `kv_group_num`: `2`
- input dtype: `torch.bfloat16`
- output dtype: `torch.bfloat16`
- scratch dtype: `torch.float32`
- device: CUDA

The function must write the final decode attention output into `o`. It may
return `None` or `o`; the evaluator compares `o`.

Read-only inputs:

- `q`
- `k_buffer`
- `v_buffer`
- `kv_indptr`
- `kv_indices`
- `num_kv_splits`
- `sinks`, when present

Writable tensors:

- `o`
- `attn_logits`
- `attn_lse`

Unsupported shapes or optional features should delegate to:

```python
return fallback()
```

The fallback accepts keyword overrides for the public ABI names when an
implementation wants to delegate a modified call:

```python
return fallback(q=q, o=o)
```

Candidate code must not patch SGLang, import SGLang internals, start
subprocesses, perform network access, build native extensions, mutate
`sys.modules`, or create files outside the allowed candidate file.
