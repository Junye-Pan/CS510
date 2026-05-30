from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class SamplingCheckConfig:
    validation_trials: int = 32
    distribution_trials: int = 2048
    tvd_threshold: float = 0.05


DEFAULT_SAMPLING_CONFIG = SamplingCheckConfig()


def expected_probs_from_logits(
    logits,
    *,
    temperatures,
    top_ks,
    top_ps,
    min_ps,
    need_min_p_sampling: bool,
    torch,
):
    probs = torch.softmax(logits / temperatures, dim=-1)
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    vocab_size = probs.shape[-1]
    positions = torch.arange(vocab_size, device=probs.device).view(1, -1)

    probs_sort = probs_sort.clone()
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[positions >= top_ks.view(-1, 1)] = 0.0
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0

    if need_min_p_sampling:
        min_p_thresholds = probs_sort[:, 0] * min_ps
        probs_sort[probs_sort < min_p_thresholds.view(-1, 1)] = 0.0

    denom = probs_sort.sum(dim=-1, keepdim=True)
    if bool((denom <= 0).any().item()):
        raise ValueError("sampling mask removed all probability mass")
    probs_sort = probs_sort / denom
    return torch.zeros_like(probs_sort).scatter_(-1, probs_idx, probs_sort)


def check_sampling_distribution(
    sample_once: Callable[[], Any],
    expected_probs,
    *,
    torch,
    config: SamplingCheckConfig = DEFAULT_SAMPLING_CONFIG,
) -> dict[str, Any]:
    valid_mask = expected_probs > 0
    batch_size, vocab_size = expected_probs.shape

    first_metrics = _run_validation_trials(
        sample_once,
        valid_mask,
        batch_size=batch_size,
        vocab_size=vocab_size,
        torch=torch,
        trials=config.validation_trials,
    )

    counters = torch.zeros_like(expected_probs, dtype=torch.float32)
    for _ in range(config.distribution_trials):
        samples = _normalize_samples(
            sample_once(),
            batch_size=batch_size,
            vocab_size=vocab_size,
            valid_mask=valid_mask,
            torch=torch,
        )
        counters.scatter_add_(
            1,
            samples.view(-1, 1),
            torch.ones((batch_size, 1), dtype=torch.float32, device=expected_probs.device),
        )

    frequencies = counters / float(config.distribution_trials)
    tvds = 0.5 * torch.sum(torch.abs(frequencies - expected_probs.float()), dim=-1)
    max_tvd = float(tvds.max().item())
    valid = max_tvd <= config.tvd_threshold

    return {
        "valid": bool(valid),
        "validation": first_metrics,
        "distribution_trials": config.distribution_trials,
        "tvd_threshold": config.tvd_threshold,
        "max_tvd": max_tvd,
        "tvds_per_batch": [float(x) for x in tvds.detach().cpu().tolist()],
        "expected_active_counts": [int(x) for x in valid_mask.sum(dim=-1).detach().cpu().tolist()],
    }


def _run_validation_trials(
    sample_once: Callable[[], Any],
    valid_mask,
    *,
    batch_size: int,
    vocab_size: int,
    torch,
    trials: int,
) -> dict[str, Any]:
    for trial in range(trials):
        _normalize_samples(
            sample_once(),
            batch_size=batch_size,
            vocab_size=vocab_size,
            valid_mask=valid_mask,
            torch=torch,
        )
    return {"trials": trials, "status": "passed"}


def _normalize_samples(samples, *, batch_size: int, vocab_size: int, valid_mask, torch):
    if not isinstance(samples, torch.Tensor):
        raise TypeError(f"sample output must be a tensor, got {type(samples).__name__}")
    if samples.numel() != batch_size:
        raise ValueError(f"sample output has {samples.numel()} values, expected {batch_size}")
    samples = samples.reshape(batch_size).to(dtype=torch.long, device=valid_mask.device)
    if bool((samples < 0).any().item()) or bool((samples >= vocab_size).any().item()):
        raise ValueError(f"sample output contains token outside [0, {vocab_size})")
    batch_indices = torch.arange(batch_size, device=valid_mask.device)
    if not bool(valid_mask[batch_indices, samples].all().item()):
        invalid = samples[~valid_mask[batch_indices, samples]].detach().cpu().tolist()
        raise ValueError(f"sample output contains token outside active sampling mask: {invalid}")
    return samples
