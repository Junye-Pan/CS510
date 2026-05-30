from __future__ import annotations

from types import ModuleType
from typing import Any, Callable

import torch

QWEN3_LAYER_IDS = set(range(28))
HEAD_DIM = 128
TP_Q_HEADS = 16
TP_KV_HEADS = 8


class AOCandidateAttentionBackend:
    """Task-owned wrapper around an official SGLang attention backend."""

    def __init__(self, *, fallback_backend: Any, kernel: ModuleType, stats: Any):
        self._ao_fallback_backend = fallback_backend
        self._ao_kernel = kernel
        self._ao_stats = stats

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ao_fallback_backend, name)

    def init_forward_metadata(self, forward_batch):
        return self._ao_fallback_backend.init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int):
        return self._ao_fallback_backend.init_cuda_graph_state(max_bs)

    def init_forward_metadata_capture_cuda_graph(self, *args, **kwargs):
        return self._ao_fallback_backend.init_forward_metadata_capture_cuda_graph(*args, **kwargs)

    def init_forward_metadata_replay_cuda_graph(self, *args, **kwargs):
        return self._ao_fallback_backend.init_forward_metadata_replay_cuda_graph(*args, **kwargs)

    def get_cuda_graph_seq_len_fill_value(self):
        return self._ao_fallback_backend.get_cuda_graph_seq_len_fill_value()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        layer,
        forward_batch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if forward_batch.forward_mode.is_decode():
            return self.forward_decode(q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs)
        return self.forward_extend(q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        layer,
        forward_batch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        return self._run_candidate(
            mode="decode",
            fallback_fn=self._ao_fallback_backend.forward_decode,
            q=q,
            k=k,
            v=v,
            layer=layer,
            forward_batch=forward_batch,
            save_kv_cache=save_kv_cache,
            kwargs=kwargs,
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        layer,
        forward_batch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        return self._run_candidate(
            mode="extend",
            fallback_fn=self._ao_fallback_backend.forward_extend,
            q=q,
            k=k,
            v=v,
            layer=layer,
            forward_batch=forward_batch,
            save_kv_cache=save_kv_cache,
            kwargs=kwargs,
        )

    def _run_candidate(
        self,
        *,
        mode: str,
        fallback_fn: Callable,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        layer,
        forward_batch,
        save_kv_cache: bool,
        kwargs: dict[str, Any],
    ):
        reason = _guard_reason(q=q, k=k, v=v, layer=layer)
        if reason is not None:
            self._ao_stats.fallback("attention", reason)
            return fallback_fn(q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs)

        def fallback(**overrides):
            return fallback_fn(
                overrides.get("q", q),
                overrides.get("k", k),
                overrides.get("v", v),
                overrides.get("layer", layer),
                overrides.get("forward_batch", forward_batch),
                save_kv_cache=overrides.get("save_kv_cache", save_kv_cache),
                **overrides.get("kwargs", kwargs),
            )

        try:
            out = self._ao_kernel.forward(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                mode=mode,
                fallback=fallback,
                **kwargs,
            )
            _validate_output(q, out, layer)
            self._ao_stats.inc(f"attention.{mode}.kernel_hit")
            return out
        except Exception as exc:
            self._ao_stats.exception("attention", exc)
            raise


def install_model_runner(*, model_runner: Any, kernel: ModuleType, stats: Any) -> Callable[[], None]:
    original_backend = getattr(model_runner, "attn_backend", None)
    if original_backend is None:
        raise AttributeError("model_runner has no attn_backend to wrap")
    if isinstance(original_backend, AOCandidateAttentionBackend):
        raise RuntimeError("model_runner.attn_backend is already wrapped by this task")
    wrapped = AOCandidateAttentionBackend(
        fallback_backend=original_backend,
        kernel=kernel,
        stats=stats,
    )
    model_runner.attn_backend = wrapped
    stats.event(
        "backend_wrapper_installed",
        target="model_runner.attn_backend",
        fallback_backend=type(original_backend).__name__,
    )

    def uninstall() -> None:
        if getattr(model_runner, "attn_backend", None) is wrapped:
            model_runner.attn_backend = original_backend
            stats.event("backend_wrapper_uninstalled", target="model_runner.attn_backend")

    return uninstall


def _guard_reason(*, q: torch.Tensor, k: torch.Tensor | None, v: torch.Tensor | None, layer) -> str | None:
    if not isinstance(q, torch.Tensor):
        return "q_not_tensor"
    if not q.is_cuda:
        return "q_not_cuda"
    if q.dtype is not torch.bfloat16:
        return "q_not_bfloat16"
    if getattr(layer, "layer_id", None) not in QWEN3_LAYER_IDS:
        return "layer_id_unsupported"
    if getattr(layer, "qk_head_dim", None) != HEAD_DIM or getattr(layer, "v_head_dim", None) != HEAD_DIM:
        return "head_dim_unsupported"
    if getattr(layer, "tp_q_head_num", None) != TP_Q_HEADS:
        return "q_head_count_unsupported"
    if getattr(layer, "tp_k_head_num", None) != TP_KV_HEADS:
        return "kv_head_count_unsupported"
    for name, tensor in (("k", k), ("v", v)):
        if tensor is None:
            return f"{name}_is_none"
        if not isinstance(tensor, torch.Tensor):
            return f"{name}_not_tensor"
        if not tensor.is_cuda:
            return f"{name}_not_cuda"
        if tensor.dtype != q.dtype:
            return f"{name}_dtype_mismatch"
        if tensor.shape[-2:] != (TP_KV_HEADS, HEAD_DIM):
            return f"{name}_shape_unsupported"
    return None


def _validate_output(q: torch.Tensor, out: Any, layer) -> None:
    if not isinstance(out, torch.Tensor):
        raise TypeError("attention backend kernel must return a tensor")
    expected_last_dim = getattr(layer, "tp_q_head_num") * getattr(layer, "v_head_dim")
    if out.shape[0] != q.reshape(q.shape[0], -1).shape[0] or out.shape[-1] != expected_last_dim:
        raise ValueError(f"attention output shape unsupported: {tuple(out.shape)}")
    if out.dtype != q.dtype:
        raise TypeError(f"attention output dtype mismatch: {out.dtype} != {q.dtype}")
    if out.device != q.device:
        raise ValueError(f"attention output device mismatch: {out.device} != {q.device}")
