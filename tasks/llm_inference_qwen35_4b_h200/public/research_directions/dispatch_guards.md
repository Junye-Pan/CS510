# Tune Dispatch Guards

Use `shape_guard` narrowly enough that the apply runtime can safely choose the
candidate only for shapes it supports. Overbroad guards may force fallback or
invalidate official scoring when live verification is enabled.

The official apply runtime dispatches only after manifest validation and shape
guard checks. For the MVP target, the useful contract is:

- activation tensors are CUDA FP16 or BF16;
- the final dimension is exactly `2560`;
- the candidate accepts `[num_tokens, 2560]` contiguous and non-pathological
  tensors selected by the task runtime;
- unsupported shapes should miss the guard and use baseline fallback instead of
  raising inside the candidate.

Guard tuning should be driven by `eval probe` before repeated submit attempts.
Probe feedback now includes aggregate dispatch hit rate, fallback counts,
fallback policy thresholds, candidate/fallback shapes, latency deltas by
decode-like and prefill-like shape families, and bottleneck hints. Treat a high
fallback rate as a coverage problem first, then inspect candidate exceptions or
shape misses.

Official submit enforces fallback policy as a gate. The default policy requires
at least one candidate call and fallback rate `<= 0.50`; operators may override
those thresholds for a stricter run. If the candidate claims broad coverage but
falls back on most live Qwen shapes, integrated scoring is invalid even when
the smoke process itself completes.

Do not special-case the smoke prompts or hidden/probe shapes. The only stable
signal available to candidates should be the public definition, public shape
workload, and aggregate probe diagnostics.
