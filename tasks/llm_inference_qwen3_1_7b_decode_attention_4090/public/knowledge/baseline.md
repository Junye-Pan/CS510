# Baseline

The public seed is fallback-only:

```python
def run(..., fallback):
    return fallback()
```

It calls the task-owned fallback, which invokes SGLang's installed
`decode_attention_fwd` implementation with the same tensor ABI used by the
candidate.

The task intentionally measures only this operator. There is no HTTP server,
tokenizer, sampling path, model runner, or end-to-end request benchmark in the
verification path.

Run the verifier before changing the kernel:

```bash
eval verify candidate/manifest.json
```

The verifier output is intended to show which shapes improved or regressed. A
candidate can call `fallback()` for unsupported workloads and optimize only a
subset of shapes first.
