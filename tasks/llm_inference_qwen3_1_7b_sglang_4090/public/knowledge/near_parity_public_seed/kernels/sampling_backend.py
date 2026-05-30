from __future__ import annotations


def sample(
    logits_output,
    sampling_info,
    return_logprob,
    top_logprobs_nums,
    token_ids_logprobs,
    *,
    fallback,
):
    return fallback()
