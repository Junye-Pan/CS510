from __future__ import annotations

from types import ModuleType
from typing import Any, Callable

import torch
from torch import nn


class AOCandidateSampler(nn.Module):
    """Task-owned wrapper around SGLang's sampler module."""

    def __init__(self, *, original_sampler: nn.Module, kernel: ModuleType, stats: Any):
        super().__init__()
        self.original_sampler = original_sampler
        self._ao_kernel = kernel
        self._ao_stats = stats

    def forward(
        self,
        logits_output,
        sampling_info,
        return_logprob,
        top_logprobs_nums,
        token_ids_logprobs,
    ):
        def fallback():
            return self.original_sampler(
                logits_output,
                sampling_info,
                return_logprob,
                top_logprobs_nums,
                token_ids_logprobs,
            )

        try:
            out = self._ao_kernel.sample(
                logits_output,
                sampling_info,
                return_logprob,
                top_logprobs_nums,
                token_ids_logprobs,
                fallback=fallback,
            )
            _validate_output(out)
            self._ao_stats.inc("sampling.kernel_hit")
            return out
        except Exception as exc:
            self._ao_stats.exception("sampling", exc)
            raise


def install_model_runner(*, model_runner: Any, kernel: ModuleType, stats: Any) -> Callable[[], None]:
    original_sampler = getattr(model_runner, "sampler", None)
    if original_sampler is None:
        raise AttributeError("model_runner has no sampler to wrap")
    if isinstance(original_sampler, AOCandidateSampler):
        raise RuntimeError("model_runner.sampler is already wrapped by this task")
    wrapped = AOCandidateSampler(original_sampler=original_sampler, kernel=kernel, stats=stats)
    model_runner.sampler = wrapped
    stats.event(
        "backend_wrapper_installed",
        target="model_runner.sampler",
        fallback_backend=type(original_sampler).__name__,
    )

    def uninstall() -> None:
        if getattr(model_runner, "sampler", None) is wrapped:
            model_runner.sampler = original_sampler
            stats.event("backend_wrapper_uninstalled", target="model_runner.sampler")

    return uninstall


def _validate_output(out: Any) -> None:
    if not isinstance(out, torch.Tensor):
        raise TypeError("sampling backend kernel must return a tensor of token ids")
    if out.ndim != 1:
        raise ValueError(f"sampling output must be rank 1, got shape {tuple(out.shape)}")
    if out.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"sampling output dtype must be int32 or int64, got {out.dtype}")
