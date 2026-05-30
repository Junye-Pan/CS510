from __future__ import annotations

import torch

from sgl_kernel import fused_add_rmsnorm


def run(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    fused_add_rmsnorm(x, residual, weight, eps)
    return x, residual
