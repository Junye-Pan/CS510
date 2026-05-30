from __future__ import annotations

import torch

from sgl_kernel import rmsnorm


def run(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return rmsnorm(x, weight, eps)
