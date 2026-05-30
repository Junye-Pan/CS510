# Public Contract

The candidate entrypoint is `candidate/manifest.json`. The writable files are:

- `candidate/kernels/rmsnorm.py`
- `candidate/kernels/fused_add_rmsnorm.py`
- `candidate/kernels/swiglu.py`
- `candidate/kernels/attention_backend.py`
- `candidate/kernels/sampling_backend.py`

Kernel entrypoints:

```python
def run(x, weight, eps): ...
def run(x, residual, weight, eps): ...
def run(x): ...
def forward(q, k, v, layer, forward_batch, save_kv_cache=True, *, mode, fallback, **kwargs): ...
def sample(logits_output, sampling_info, return_logprob, top_logprobs_nums, token_ids_logprobs, *, fallback): ...
```

Unsupported shapes should be left to the task-owned fallback path. Candidate
code must not patch SGLang, start subprocesses, perform network access, load
model weights, or mutate files outside `candidate/kernels/`.
