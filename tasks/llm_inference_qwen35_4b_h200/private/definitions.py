from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AxisSpec:
    name: str
    kind: str
    value: int | None = None
    min_value: int | None = None
    max_value: int | None = None

    def contains(self, value: int) -> bool:
        if self.kind == "const":
            return value == self.value
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
        return value >= 0


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[str, ...]
    dtype: str
    description: str


@dataclass(frozen=True)
class DefinitionSpec:
    name: str
    op_type: str
    axes: dict[str, AxisSpec]
    inputs: dict[str, TensorSpec]
    outputs: dict[str, TensorSpec]
    verifier: str
    constraints: tuple[str, ...]
    reference: str

    def to_public_json(self) -> dict[str, Any]:
        axes: dict[str, dict[str, Any]] = {}
        for name, axis in self.axes.items():
            if axis.kind == "const":
                axes[name] = {"type": "const", "value": axis.value}
            else:
                axes[name] = {
                    "type": "var",
                    "min": axis.min_value,
                    "max": axis.max_value,
                }
        return {
            "name": self.name,
            "op_type": self.op_type,
            "axes": axes,
            "inputs": {
                name: {"shape": list(spec.shape), "dtype": spec.dtype, "description": spec.description}
                for name, spec in self.inputs.items()
            },
            "outputs": {
                name: {"shape": list(spec.shape), "dtype": spec.dtype, "description": spec.description}
                for name, spec in self.outputs.items()
            },
            "verifier": self.verifier,
            "constraints": list(self.constraints),
            "reference": self.reference,
        }


RMSNORM_REFERENCE = """
def run(hidden_states, weight, eps=1e-6):
    import torch
    variance = hidden_states.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    output = hidden_states * torch.rsqrt(variance + eps).to(hidden_states.dtype)
    return output * weight
""".strip()


QWEN_RMSNORM_H2560_FP16 = DefinitionSpec(
    name="qwen_rmsnorm_h2560_fp16",
    op_type="rmsnorm",
    axes={
        "num_tokens": AxisSpec("num_tokens", "var", min_value=1, max_value=32768),
        "hidden": AxisSpec("hidden", "const", value=2560),
    },
    inputs={
        "hidden_states": TensorSpec(("num_tokens", "hidden"), "float16", "Input activations."),
        "weight": TensorSpec(("hidden",), "float16", "RMSNorm learned weight."),
    },
    outputs={
        "output": TensorSpec(("num_tokens", "hidden"), "float16", "Normalized activations."),
    },
    verifier="deterministic",
    constraints=("1 <= num_tokens <= 32768", "hidden == 2560"),
    reference=RMSNORM_REFERENCE,
)


DEFINITIONS: dict[str, DefinitionSpec] = {
    QWEN_RMSNORM_H2560_FP16.name: QWEN_RMSNORM_H2560_FP16,
}


def get_definition(name: str) -> DefinitionSpec | None:
    return DEFINITIONS.get(name)
