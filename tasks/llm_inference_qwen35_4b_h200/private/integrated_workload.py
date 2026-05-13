from __future__ import annotations

from typing import Any


DEFAULT_PROMPTS = (
    "In one short sentence, say what an inference kernel does.",
    "Return exactly three words about fast GPUs.",
)
DEFAULT_MAX_TOKENS = 8
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_SEED = 123
DEFAULT_LOGPROBS = 5


def workload_signature(
    *,
    prompts: tuple[str, ...] = DEFAULT_PROMPTS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any]:
    return {
        "name": "qwen35_4b_vllm_integrated_smoke_v1",
        "prompts": list(prompts),
        "max_tokens": int(max_tokens),
        "sampling": {
            "temperature": DEFAULT_TEMPERATURE,
            "top_p": DEFAULT_TOP_P,
            "seed": DEFAULT_SEED,
            "logprobs": DEFAULT_LOGPROBS,
        },
    }
