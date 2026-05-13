import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(hidden_states, weight, output, hidden: tl.constexpr, eps: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, block)
    mask = offsets < hidden
    values = tl.load(hidden_states + row * hidden + offsets, mask=mask, other=0.0).to(tl.float32)
    weights = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(values * values, axis=0) / hidden
    normalized = values * tl.rsqrt(variance + eps) * weights
    tl.store(output + row * hidden + offsets, normalized, mask=mask)


def run(hidden_states: torch.Tensor, weight: torch.Tensor, output: torch.Tensor) -> None:
    num_tokens, hidden = hidden_states.shape
    if hidden != 2560:
        raise ValueError(f"expected hidden size 2560, got {hidden}")
    block = triton.next_power_of_2(hidden)
    _rmsnorm_kernel[(num_tokens,)](
        hidden_states,
        weight,
        output,
        hidden,
        1.0e-6,
        block,
        num_warps=8,
    )
