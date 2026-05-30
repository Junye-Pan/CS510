from __future__ import annotations

import torch

from sgl_kernel import silu_and_mul


def run(x: torch.Tensor) -> torch.Tensor:
    output_shape = x.shape[:-1] + (x.shape[-1] // 2,)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    silu_and_mul(x, out)
    return out
