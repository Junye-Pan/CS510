from __future__ import annotations

import inspect
from typing import Any, Callable


_DECODE_ATTENTION_FWD: Callable[..., Any] | None = None
_REFERENCE_ACCEPTS_K_SCALE: bool | None = None


def official_decode_attention_fwd(
    q: Any,
    k_buffer: Any,
    v_buffer: Any,
    o: Any,
    kv_indptr: Any,
    kv_indices: Any,
    attn_logits: Any,
    attn_lse: Any,
    num_kv_splits: Any,
    max_kv_splits: int,
    sm_scale: float,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    logit_cap: float = 0.0,
    sinks: Any = None,
    xai_temperature_len: int = -1,
    has_mla: bool = False,
    use_pdl: bool = False,
) -> Any:
    global _DECODE_ATTENTION_FWD, _REFERENCE_ACCEPTS_K_SCALE
    if _DECODE_ATTENTION_FWD is None:
        from sglang.srt.layers.attention.triton_ops.decode_attention import (
            decode_attention_fwd,
        )

        _DECODE_ATTENTION_FWD = decode_attention_fwd
        _REFERENCE_ACCEPTS_K_SCALE = "k_scale" in inspect.signature(decode_attention_fwd).parameters

    if _REFERENCE_ACCEPTS_K_SCALE:
        return _DECODE_ATTENTION_FWD(
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
            logit_cap=logit_cap,
            sinks=sinks,
            xai_temperature_len=xai_temperature_len,
            has_mla=has_mla,
            use_pdl=use_pdl,
        )

    if v_scale != 1.0:
        raise ValueError("installed SGLang decode_attention_fwd does not support v_scale != 1.0")
    if sinks is not None:
        raise ValueError("installed SGLang decode_attention_fwd does not support sinks")
    if xai_temperature_len != -1:
        raise ValueError("installed SGLang decode_attention_fwd does not support xai_temperature_len")
    if has_mla:
        raise ValueError("installed SGLang decode_attention_fwd does not support has_mla")
    if use_pdl:
        raise ValueError("installed SGLang decode_attention_fwd does not support use_pdl")
    return _DECODE_ATTENTION_FWD(
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
        sm_scale * k_scale,
        logit_cap=logit_cap,
    )
