from __future__ import annotations


PUBLIC_WORKLOAD_SHAPES: list[dict[str, int]] = [
    {"num_tokens": 1, "hidden": 2560},
    {"num_tokens": 8, "hidden": 2560},
    {"num_tokens": 128, "hidden": 2560},
]

PROBE_WORKLOAD_SHAPES: list[dict[str, int]] = [
    {"num_tokens": 16, "hidden": 2560},
    {"num_tokens": 512, "hidden": 2560},
]

HIDDEN_RMSNORM_SHAPES: list[dict[str, int]] = [
    {"num_tokens": 32, "hidden": 2560},
    {"num_tokens": 1024, "hidden": 2560},
]

HIDDEN_WORKLOAD_FAMILIES: tuple[str, ...] = (
    "prefill_heavy_ttft",
    "decode_heavy_tpot",
    "mixed_chat_latency",
)
