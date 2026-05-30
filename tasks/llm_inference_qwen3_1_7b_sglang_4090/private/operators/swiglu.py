from __future__ import annotations

from types import ModuleType
from typing import Any, Callable

import torch

INPUT_LAST_DIM = 12288
OUTPUT_LAST_DIM = 6144


def install(*, kernel: ModuleType, stats: Any) -> Callable[[], None]:
    from sglang.srt.layers import activation as activation_mod

    target_cls = activation_mod.SiluAndMul
    original = target_cls.forward_cuda
    if getattr(original, "__ao_qwen3_1_7b_wrapper__", False):
        raise RuntimeError("SiluAndMul.forward_cuda is already wrapped by this task")

    def wrapped(self, x: torch.Tensor):
        reason = _guard_reason(x)
        if reason is not None:
            stats.fallback("swiglu", reason)
            return original(self, x)
        try:
            out = kernel.run(x)
            _validate_output(x, out)
            stats.inc("swiglu.kernel_hit")
            return out
        except Exception as exc:
            stats.exception("swiglu", exc)
            raise

    wrapped.__ao_qwen3_1_7b_wrapper__ = True  # type: ignore[attr-defined]
    target_cls.forward_cuda = wrapped
    stats.event("operator_wrapper_installed", target="SiluAndMul.forward_cuda")

    def uninstall() -> None:
        if target_cls.forward_cuda is wrapped:
            target_cls.forward_cuda = original
            stats.event("operator_wrapper_uninstalled", target="SiluAndMul.forward_cuda")

    return uninstall


def _guard_reason(x: torch.Tensor) -> str | None:
    if not isinstance(x, torch.Tensor):
        return "x_not_tensor"
    if not x.is_cuda:
        return "x_not_cuda"
    if x.dtype is not torch.bfloat16:
        return "x_not_bfloat16"
    if x.ndim < 1:
        return "x_rank_lt_1"
    if x.shape[-1] != INPUT_LAST_DIM:
        return f"input_last_dim_{x.shape[-1]}_unsupported"
    if not x.is_contiguous():
        return "x_not_contiguous"
    return None


def _validate_output(x: torch.Tensor, out: Any) -> None:
    if not isinstance(out, torch.Tensor):
        raise TypeError("swiglu kernel must return a tensor")
    expected_shape = x.shape[:-1] + (OUTPUT_LAST_DIM,)
    if out.shape != expected_shape:
        raise ValueError(f"swiglu output shape mismatch: {tuple(out.shape)} != {tuple(expected_shape)}")
    if out.dtype != x.dtype:
        raise TypeError(f"swiglu output dtype mismatch: {out.dtype} != {x.dtype}")
    if out.device != x.device:
        raise ValueError(f"swiglu output device mismatch: {out.device} != {x.device}")
