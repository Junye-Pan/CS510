# Implement a Triton RMSNorm Kernel

Replace the placeholder RMSNorm source with a destination-passing Triton-backed
implementation for `[num_tokens, 2560]` FP16 activations. The live verifier will
compare shape, dtype, finite values, and elementwise error before any speed
score can count.

Implementation notes:

- keep the Python entry point stable as `kernels/rmsnorm.py::run`;
- write the result into the provided `output` tensor and return `None`;
- compute using the supplied `weight` tensor directly;
- do not add a second `1 + weight` adjustment for Qwen, because the official
  vLLM adapter already converts `GemmaRMSNorm` weights before calling the
  candidate;
- handle both short decode-like rows and larger prefill-like row counts without
  recompiling on every call;
- let unsupported shapes miss the manifest guard instead of branching into
  partial implementations that raise at runtime.

A useful development loop is:

1. Run static `eval verify` after changing the manifest or entry point.
2. Run live RMSNorm verification to catch numerical or dtype mistakes.
3. Run `eval probe` and inspect latency deltas plus fallback diagnostics.
4. Run integrated submit only after probe shows candidate calls on the intended
   shape family and no candidate exceptions.

Passing a microbenchmark is not enough. Official submit also runs Qwen/vLLM
integrated smoke against the pinned baseline artifact. Generated token ids,
generated text, and top-logprob token sets must remain compatible with the
baseline, and the fallback policy must pass. All trace, smoke, verifier, and
submit artifacts are written under `CS510/runs/`.
