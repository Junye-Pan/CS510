from __future__ import annotations


def forward(
    q,
    k,
    v,
    layer,
    forward_batch,
    save_kv_cache=True,
    *,
    mode: str,
    fallback,
    **kwargs,
):
    return fallback()
