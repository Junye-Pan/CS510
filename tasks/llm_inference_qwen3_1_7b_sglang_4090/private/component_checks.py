from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .apply_sglang_candidate import load_candidate
from .sampling_verifier import SamplingCheckConfig, check_sampling_distribution, expected_probs_from_logits
from .verifier_primitives import compare_tensor


RMS_EPS = 1.0e-6


@dataclass
class CheckResult:
    name: str
    valid: bool
    metrics: dict[str, Any]
    error: str | None = None

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "valid": self.valid,
            "metrics": self.metrics,
            "error": self.error,
        }


def run_component_checks(entry_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    results: list[CheckResult] = []

    try:
        import torch
    except Exception as exc:
        return _failed_import("torch", exc, started)

    if not torch.cuda.is_available():
        return {
            "valid": False,
            "results": [
                CheckResult(
                    name="cuda_available",
                    valid=False,
                    metrics={},
                    error="CUDA is required for Qwen3 SGLang component checks",
                ).to_jsonable()
            ],
            "elapsed_s": time.perf_counter() - started,
        }

    applied = load_candidate(entry_path)
    modules = applied.modules
    torch.manual_seed(20260528)
    device = torch.device("cuda")

    check_plan: tuple[tuple[str, Callable[[], CheckResult]], ...] = (
        ("rmsnorm", lambda: _check_rmsnorm(modules.rmsnorm, torch, device)),
        ("fused_add_rmsnorm", lambda: _check_fused_add_rmsnorm(modules.fused_add_rmsnorm, torch, device)),
        ("swiglu", lambda: _check_swiglu(modules.swiglu, torch, device)),
        ("attention_backend", lambda: _check_attention_backend(modules.attention_backend, torch, device)),
        ("sampling_backend", lambda: _check_sampling_backend(modules.sampling_backend, torch, device)),
    )

    for name, check in check_plan:
        try:
            results.append(check())
        except Exception as exc:
            results.append(
                CheckResult(
                    name=name,
                    valid=False,
                    metrics={},
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    torch.cuda.synchronize()
    valid = all(result.valid for result in results)
    return {
        "valid": valid,
        "results": [result.to_jsonable() for result in results],
        "elapsed_s": time.perf_counter() - started,
    }


def _failed_import(module: str, exc: BaseException, started: float) -> dict[str, Any]:
    return {
        "valid": False,
        "results": [
            CheckResult(
                name=f"import_{module}",
                valid=False,
                metrics={},
                error=f"{type(exc).__name__}: {exc}",
            ).to_jsonable()
        ],
        "elapsed_s": time.perf_counter() - started,
    }


def _check_rmsnorm(module, torch, device) -> CheckResult:
    from sgl_kernel import rmsnorm as official_rmsnorm

    cases = [
        ("hidden_2048", (3, 2048)),
        ("qk_hidden_128", (7, 128)),
    ]
    metrics: dict[str, Any] = {}
    for label, shape in cases:
        x = torch.randn(shape, dtype=torch.bfloat16, device=device)
        weight = torch.randn((shape[-1],), dtype=torch.bfloat16, device=device)
        candidate = module.run(x.clone(), weight, RMS_EPS)
        reference = official_rmsnorm(x.clone(), weight, RMS_EPS)
        stats = compare_tensor(candidate, reference, name=f"rmsnorm.{label}")
        metrics[label] = stats
        if not stats["allclose"]:
            return CheckResult("rmsnorm", False, metrics, f"{label} output differs from official rmsnorm")
    return CheckResult("rmsnorm", True, metrics)


def _check_fused_add_rmsnorm(module, torch, device) -> CheckResult:
    from sgl_kernel import fused_add_rmsnorm as official_fused_add_rmsnorm

    x0 = torch.randn((4, 2048), dtype=torch.bfloat16, device=device)
    residual0 = torch.randn((4, 2048), dtype=torch.bfloat16, device=device)
    weight = torch.randn((2048,), dtype=torch.bfloat16, device=device)

    cand_x = x0.clone()
    cand_residual = residual0.clone()
    result = module.run(cand_x, cand_residual, weight, RMS_EPS)
    if result is not None:
        if not isinstance(result, tuple) or len(result) != 2:
            return CheckResult(
                "fused_add_rmsnorm",
                False,
                {"returned": type(result).__name__},
                "kernel must return None or (x, residual)",
            )
        if result[0] is not cand_x or result[1] is not cand_residual:
            return CheckResult(
                "fused_add_rmsnorm",
                False,
                {},
                "kernel must preserve in-place x/residual identity",
            )

    ref_x = x0.clone()
    ref_residual = residual0.clone()
    official_fused_add_rmsnorm(ref_x, ref_residual, weight, RMS_EPS)

    x_stats = compare_tensor(cand_x, ref_x, name="fused_add_rmsnorm.x")
    residual_stats = compare_tensor(cand_residual, ref_residual, name="fused_add_rmsnorm.residual")
    metrics = {"x": x_stats, "residual": residual_stats}
    if not x_stats["allclose"]:
        return CheckResult("fused_add_rmsnorm", False, metrics, "mutated x differs from official fused_add_rmsnorm")
    if not residual_stats["allclose"]:
        return CheckResult(
            "fused_add_rmsnorm",
            False,
            metrics,
            "mutated residual differs from official fused_add_rmsnorm",
        )
    return CheckResult("fused_add_rmsnorm", True, metrics)


def _check_swiglu(module, torch, device) -> CheckResult:
    from sgl_kernel import silu_and_mul as official_silu_and_mul

    x = torch.randn((5, 12288), dtype=torch.bfloat16, device=device)
    candidate = module.run(x.clone())
    reference = torch.empty((5, 6144), dtype=torch.bfloat16, device=device)
    official_silu_and_mul(x.clone(), reference)
    stats = compare_tensor(candidate, reference, name="swiglu")
    if not stats["allclose"]:
        return CheckResult("swiglu", False, stats, "output differs from official silu_and_mul")
    return CheckResult("swiglu", True, stats)


def _check_attention_backend(module, torch, device) -> CheckResult:
    class FakeLayer:
        layer_id = 0
        qk_head_dim = 128
        v_head_dim = 128
        tp_q_head_num = 16
        tp_k_head_num = 8

    class FakeForwardBatch:
        pass

    metrics: dict[str, Any] = {}
    for mode, token_count, causal in (
        ("decode", 2, False),
        ("extend", 4, True),
    ):
        q = torch.randn((token_count, 2048), dtype=torch.bfloat16, device=device)
        k = torch.randn((token_count, 8, 128), dtype=torch.bfloat16, device=device)
        v = torch.randn((token_count, 8, 128), dtype=torch.bfloat16, device=device)
        fallback_calls = {"count": 0}
        expected = _reference_gqa_attention(q, k, v, torch=torch, causal=causal)

        def fallback(**overrides):
            fallback_calls["count"] += 1
            return _reference_gqa_attention(
                overrides.get("q", q),
                overrides.get("k", k),
                overrides.get("v", v),
                torch=torch,
                causal=causal,
            )

        out = module.forward(
            q,
            k,
            v,
            FakeLayer(),
            FakeForwardBatch(),
            save_kv_cache=True,
            mode=mode,
            fallback=fallback,
        )
        stats = compare_tensor(out, expected, name=f"attention_backend.{mode}")
        metrics[mode] = {
            "fallback_calls": fallback_calls["count"],
            "tokens": token_count,
            "causal": causal,
            **stats,
        }
        if not stats["allclose"]:
            return CheckResult(
                "attention_backend",
                False,
                metrics,
                f"{mode} attention output differs from GQA reference",
            )
    return CheckResult("attention_backend", True, metrics)


def _check_sampling_backend(module, torch, device) -> CheckResult:
    distribution_metrics = _check_sampling_distribution(module, torch, device)
    metrics = {"distribution": distribution_metrics}
    if not distribution_metrics["valid"]:
        return CheckResult(
            "sampling_backend",
            False,
            metrics,
            f"sampling TVD exceeded threshold: {distribution_metrics['max_tvd']}",
        )
    return CheckResult("sampling_backend", True, metrics)


def _check_sampling_distribution(module, torch, device) -> dict[str, Any]:
    ranks = torch.arange(64, dtype=torch.float32, device=device)
    base_probs = torch.stack(
        [
            torch.softmax(-ranks / 3.5, dim=-1),
            torch.softmax(-ranks / 5.5, dim=-1),
            torch.softmax(-ranks / 7.5, dim=-1),
        ]
    )
    logits = torch.log(base_probs)
    temperatures = torch.ones((3, 1), dtype=torch.float32, device=device)
    top_ks = torch.tensor([8, 16, 32], dtype=torch.int32, device=device)
    top_ps = torch.tensor([0.995, 0.985, 0.975], dtype=torch.float32, device=device)
    min_ps = torch.zeros((3,), dtype=torch.float32, device=device)
    expected_probs = expected_probs_from_logits(
        logits,
        temperatures=temperatures,
        top_ks=top_ks,
        top_ps=top_ps,
        min_ps=min_ps,
        need_min_p_sampling=False,
        torch=torch,
    )

    class FakeLogitsOutput:
        def __init__(self, next_token_logits):
            self.next_token_logits = next_token_logits

    class FakeSamplingInfo:
        def __init__(self):
            self.device = device
            self.is_all_greedy = False
            self.has_custom_logit_processor = False
            self.need_min_p_sampling = False
            self.grammars = None
            self.custom_logit_processor: dict[str, Any] = {}
            self.custom_params: list[Any] = []
            self.temperatures = temperatures.clone()
            self.top_ks = top_ks.clone()
            self.top_ps = top_ps.clone()
            self.min_ps = min_ps.clone()

        def __len__(self) -> int:
            return int(logits.shape[0])

    generator = torch.Generator(device=device)
    generator.manual_seed(20260528)
    fallback_calls = {"count": 0}

    def sample_once():
        logits_output = FakeLogitsOutput(logits.clone())
        sampling_info = FakeSamplingInfo()

        def fallback():
            fallback_calls["count"] += 1
            return torch.multinomial(expected_probs, num_samples=1, generator=generator).squeeze(-1)

        return module.sample(
            logits_output,
            sampling_info,
            False,
            [],
            [],
            fallback=fallback,
        )

    metrics = check_sampling_distribution(
        sample_once,
        expected_probs,
        torch=torch,
        config=SamplingCheckConfig(distribution_trials=2048),
    )
    metrics["fallback_calls"] = fallback_calls["count"]
    metrics["top_ks"] = [int(x) for x in top_ks.detach().cpu().tolist()]
    metrics["top_ps"] = [float(x) for x in top_ps.detach().cpu().tolist()]
    return metrics


def _reference_gqa_attention(q, k, v, *, torch, causal: bool):
    token_count = q.shape[0]
    q_heads = 16
    kv_heads = 8
    head_dim = 128
    group_size = q_heads // kv_heads

    q_view = q.view(token_count, q_heads, head_dim).float()
    k_view = k.float()
    v_view = v.float()
    out = torch.empty((token_count, q_heads, head_dim), dtype=torch.float32, device=q.device)
    scale = 1.0 / math.sqrt(float(head_dim))

    for token_idx in range(token_count):
        key_count = token_idx + 1 if causal else token_count
        for q_head in range(q_heads):
            kv_head = q_head // group_size
            scores = (k_view[:key_count, kv_head, :] * q_view[token_idx, q_head, :]).sum(dim=-1) * scale
            probs = torch.softmax(scores, dim=-1)
            out[token_idx, q_head, :] = (probs[:, None] * v_view[:key_count, kv_head, :]).sum(dim=0)

    return out.reshape(token_count, q_heads * head_dim).to(dtype=q.dtype)
