from __future__ import annotations

from types import ModuleType
from typing import Any, Callable

import torch

RMSNORM_HIDDEN_SIZES = {2048, 128}
FUSED_ADD_HIDDEN_SIZES = {2048}


def install(
    *,
    rmsnorm_kernel: ModuleType,
    fused_add_kernel: ModuleType,
    stats: Any,
) -> Callable[[], None]:
    from sglang.srt.layers import layernorm as layernorm_mod

    target_cls = layernorm_mod.RMSNorm
    original = target_cls.forward_cuda
    if getattr(original, "__ao_qwen3_1_7b_wrapper__", False):
        raise RuntimeError("RMSNorm.forward_cuda is already wrapped by this task")

    def wrapped(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is None:
            reason = _rmsnorm_guard_reason(self, x)
            if reason is not None:
                stats.fallback("rmsnorm", reason)
                return original(self, x, residual)
            try:
                out = rmsnorm_kernel.run(x, self.weight.data, self.variance_epsilon)
                _validate_rmsnorm_output(x, out)
                stats.inc("rmsnorm.kernel_hit")
                return out
            except Exception as exc:
                stats.exception("rmsnorm", exc)
                raise

        reason = _fused_add_guard_reason(self, x, residual)
        if reason is not None:
            stats.fallback("fused_add_rmsnorm", reason)
            return original(self, x, residual)
        try:
            result = fused_add_kernel.run(x, residual, self.weight.data, self.variance_epsilon)
            if result is None:
                result = (x, residual)
            _validate_fused_add_output(x, residual, result)
            stats.inc("fused_add_rmsnorm.kernel_hit")
            return result
        except Exception as exc:
            stats.exception("fused_add_rmsnorm", exc)
            raise

    wrapped.__ao_qwen3_1_7b_wrapper__ = True  # type: ignore[attr-defined]
    target_cls.forward_cuda = wrapped
    stats.event("operator_wrapper_installed", target="RMSNorm.forward_cuda")

    def uninstall() -> None:
        if target_cls.forward_cuda is wrapped:
            target_cls.forward_cuda = original
            stats.event("operator_wrapper_uninstalled", target="RMSNorm.forward_cuda")

    return uninstall


def _base_guard_reason(self, x: torch.Tensor, *, allowed_hidden_sizes: set[int]) -> str | None:
    if not isinstance(x, torch.Tensor):
        return "x_not_tensor"
    if not x.is_cuda:
        return "x_not_cuda"
    if x.dtype is not torch.bfloat16:
        return "x_not_bfloat16"
    if x.ndim < 1:
        return "x_rank_lt_1"
    if x.shape[-1] not in allowed_hidden_sizes:
        return f"hidden_size_{x.shape[-1]}_unsupported"
    if not x.is_contiguous():
        return "x_not_contiguous"
    weight = getattr(self, "weight", None)
    if weight is None or tuple(weight.shape) != (x.shape[-1],):
        return "weight_shape_mismatch"
    return None


def _rmsnorm_guard_reason(self, x: torch.Tensor) -> str | None:
    return _base_guard_reason(self, x, allowed_hidden_sizes=RMSNORM_HIDDEN_SIZES)


def _fused_add_guard_reason(self, x: torch.Tensor, residual: torch.Tensor) -> str | None:
    reason = _base_guard_reason(self, x, allowed_hidden_sizes=FUSED_ADD_HIDDEN_SIZES)
    if reason is not None:
        return reason
    if not isinstance(residual, torch.Tensor):
        return "residual_not_tensor"
    if residual.shape != x.shape:
        return "residual_shape_mismatch"
    if residual.device != x.device:
        return "residual_device_mismatch"
    if residual.dtype != x.dtype:
        return "residual_dtype_mismatch"
    if not residual.is_contiguous():
        return "residual_not_contiguous"
    return None


def _validate_rmsnorm_output(x: torch.Tensor, out: Any) -> None:
    if not isinstance(out, torch.Tensor):
        raise TypeError("rmsnorm kernel must return a tensor")
    if out.shape != x.shape:
        raise ValueError(f"rmsnorm output shape mismatch: {tuple(out.shape)} != {tuple(x.shape)}")
    if out.dtype != x.dtype:
        raise TypeError(f"rmsnorm output dtype mismatch: {out.dtype} != {x.dtype}")
    if out.device != x.device:
        raise ValueError(f"rmsnorm output device mismatch: {out.device} != {x.device}")


def _validate_fused_add_output(x: torch.Tensor, residual: torch.Tensor, result: Any) -> None:
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("fused_add_rmsnorm kernel must return (x, residual) or None")
    out_x, out_residual = result
    if out_x is not x:
        raise ValueError("fused_add_rmsnorm kernel must return the original x tensor")
    if out_residual is not residual:
        raise ValueError("fused_add_rmsnorm kernel must return the original residual tensor")
