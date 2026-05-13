from __future__ import annotations

import atexit
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from .rmsnorm_live import RMSNORM_DEFINITION, _load_module
from .schema import BundleManifest, ImplementationSpec, load_manifest, validate_bundle_files
from .vllm_plugin_runtime import EXPECTED_HIDDEN_ENV, MANIFEST_ENV, TRACE_ENV


_ADAPTER: CandidateRMSNormAdapter | None = None
_ORIGINALS: dict[str, Any] = {}
_STATS: dict[str, Any] = {
    "candidate_calls": 0,
    "fallback_calls": 0,
    "classes": {},
    "fallback_reasons": {},
}
_SUMMARY_REGISTERED = False


def register_candidate_rmsnorm() -> None:
    global _ADAPTER, _SUMMARY_REGISTERED
    try:
        _ADAPTER = CandidateRMSNormAdapter.from_env()
        _patch_vllm_layernorm_classes()
        _record_event(
            {
                "event": "plugin_installed",
                "implementation_id": _ADAPTER.implementation.id,
                "manifest_path": str(_ADAPTER.manifest_path),
                "expected_hidden": _ADAPTER.expected_hidden,
            }
        )
        if not _SUMMARY_REGISTERED:
            atexit.register(_flush_summary)
            _SUMMARY_REGISTERED = True
    except Exception as exc:
        _ADAPTER = None
        _record_event({"event": "plugin_install_failed", "error": str(exc)})


class CandidateRMSNormAdapter:
    def __init__(
        self,
        *,
        manifest_path: Path,
        manifest: BundleManifest,
        implementation: ImplementationSpec,
        run: Any,
        expected_hidden: int,
    ) -> None:
        self.manifest_path = manifest_path
        self.manifest = manifest
        self.implementation = implementation
        self.run = run
        self.expected_hidden = expected_hidden

    @classmethod
    def from_env(cls) -> "CandidateRMSNormAdapter":
        manifest_raw = os.environ.get(MANIFEST_ENV)
        if not manifest_raw:
            raise RuntimeError(f"{MANIFEST_ENV} is unset")
        manifest_path = Path(manifest_raw).resolve()
        manifest = load_manifest(manifest_path)
        candidate_root = manifest_path.parent
        validate_bundle_files(manifest, candidate_root=candidate_root)
        implementation = _select_implementation(manifest)
        module = _load_module(candidate_root / implementation.entry_path)
        run = getattr(module, implementation.entry_symbol)
        expected_hidden = int(os.environ.get(EXPECTED_HIDDEN_ENV, "2560"))
        return cls(
            manifest_path=manifest_path,
            manifest=manifest,
            implementation=implementation,
            run=run,
            expected_hidden=expected_hidden,
        )

    def apply_standard(self, layer: Any, x: Any, residual: Any | None) -> Any | None:
        if not getattr(layer, "has_weight", True):
            self._record_fallback("standard_no_weight", layer, x, residual)
            return None
        if getattr(layer, "variance_size_override", None) is not None:
            self._record_fallback("standard_variance_override", layer, x, residual)
            return None
        weight = getattr(layer, "weight", None)
        if weight is None:
            self._record_fallback("standard_missing_weight", layer, x, residual)
            return None
        return self._apply(layer, x, residual, weight.data, class_name="RMSNorm")

    def apply_gemma(self, layer: Any, x: Any, residual: Any | None) -> Any | None:
        weight = getattr(layer, "weight", None)
        if weight is None:
            self._record_fallback("gemma_missing_weight", layer, x, residual)
            return None
        effective_weight = weight.data.float() + 1.0
        return self._apply(layer, x, residual, effective_weight, class_name="GemmaRMSNorm")

    def _apply(self, layer: Any, x: Any, residual: Any | None, weight: Any, *, class_name: str) -> Any | None:
        import torch

        if not _is_supported_tensor(x):
            self._record_fallback("unsupported_input_tensor", layer, x, residual)
            return None
        if not math.isclose(float(getattr(layer, "variance_epsilon", 0.0)), 1.0e-6, rel_tol=0.0, abs_tol=1.0e-12):
            self._record_fallback("unsupported_epsilon", layer, x, residual)
            return None
        hidden = int(x.shape[-1])
        if hidden != self.expected_hidden:
            self._record_fallback("unsupported_hidden", layer, x, residual)
            return None
        if residual is not None and (not _is_supported_tensor(residual) or tuple(residual.shape) != tuple(x.shape)):
            self._record_fallback("unsupported_residual", layer, x, residual)
            return None
        if residual is not None and residual.dtype != x.dtype:
            self._record_fallback("residual_dtype_mismatch", layer, x, residual)
            return None
        if x.dtype not in (torch.float16, torch.bfloat16):
            self._record_fallback("unsupported_dtype", layer, x, residual)
            return None
        if weight.device != x.device:
            self._record_fallback("weight_device_mismatch", layer, x, residual)
            return None

        if residual is None:
            candidate_input = x
            residual_out = None
        elif class_name == "GemmaRMSNorm" and x.dtype == torch.float16:
            candidate_input = x.float() + residual.float()
            residual_out = candidate_input
        else:
            candidate_input = x + residual
            residual_out = candidate_input

        flat_input = _view_as_candidate_matrix(candidate_input, hidden)
        if flat_input is None:
            self._record_fallback("noncontiguous_input", layer, x, residual)
            return None
        output = torch.empty_like(x)
        flat_output = _view_as_candidate_matrix(output, hidden)
        if flat_output is None:
            self._record_fallback("noncontiguous_output", layer, x, residual)
            return None

        num_tokens = int(flat_input.shape[0])
        if not _guard_covers(self.implementation, {"num_tokens": num_tokens, "hidden": hidden}):
            self._record_fallback("shape_guard_miss", layer, x, residual)
            return None

        self.run(flat_input, weight.contiguous(), flat_output)
        _record_candidate_call(
            {
                "class": class_name,
                "implementation_id": self.implementation.id,
                "shape": list(x.shape),
                "num_tokens": num_tokens,
                "hidden": hidden,
                "dtype": str(x.dtype),
                "residual": residual is not None,
            }
        )
        if residual is None:
            return output
        return output, residual_out

    def _record_fallback(self, reason: str, layer: Any, x: Any, residual: Any | None) -> None:
        _STATS["fallback_calls"] = int(_STATS.get("fallback_calls") or 0) + 1
        reasons = _STATS.setdefault("fallback_reasons", {})
        reasons[reason] = int(reasons.get(reason) or 0) + 1
        if int(reasons[reason]) <= 4:
            _record_event(
                {
                    "event": "fallback",
                    "reason": reason,
                    "class": layer.__class__.__name__,
                    "shape": _shape_of(x),
                    "dtype": str(getattr(x, "dtype", "")),
                    "residual": residual is not None,
                }
            )


def _patch_vllm_layernorm_classes() -> None:
    from vllm.model_executor.layers import layernorm as layernorm_module

    if not _ORIGINALS:
        _ORIGINALS["RMSNorm.forward_native"] = layernorm_module.RMSNorm.forward_native
        _ORIGINALS["RMSNorm.forward_cuda"] = layernorm_module.RMSNorm.forward_cuda
        _ORIGINALS["GemmaRMSNorm.forward_native"] = layernorm_module.GemmaRMSNorm.forward_native
        _ORIGINALS["GemmaRMSNorm.forward_cuda"] = layernorm_module.GemmaRMSNorm.forward_cuda

        def rms_forward_native(layer: Any, x: Any, residual: Any | None = None) -> Any:
            if _ADAPTER is not None:
                result = _ADAPTER.apply_standard(layer, x, residual)
                if result is not None:
                    return result
            return _ORIGINALS["RMSNorm.forward_native"](layer, x, residual)

        def rms_forward_cuda(layer: Any, x: Any, residual: Any | None = None) -> Any:
            if _ADAPTER is not None:
                result = _ADAPTER.apply_standard(layer, x, residual)
                if result is not None:
                    return result
            return _ORIGINALS["RMSNorm.forward_cuda"](layer, x, residual)

        def gemma_forward_native(layer: Any, x: Any, residual: Any | None = None) -> Any:
            if _ADAPTER is not None:
                result = _ADAPTER.apply_gemma(layer, x, residual)
                if result is not None:
                    return result
            return _ORIGINALS["GemmaRMSNorm.forward_native"](layer, x, residual)

        def gemma_forward_cuda(layer: Any, x: Any, residual: Any | None = None) -> Any:
            if _ADAPTER is not None:
                result = _ADAPTER.apply_gemma(layer, x, residual)
                if result is not None:
                    return result
            return _ORIGINALS["GemmaRMSNorm.forward_cuda"](layer, x, residual)

        layernorm_module.RMSNorm.forward_native = rms_forward_native
        layernorm_module.RMSNorm.forward_cuda = rms_forward_cuda
        layernorm_module.GemmaRMSNorm.forward_native = gemma_forward_native
        layernorm_module.GemmaRMSNorm.forward_cuda = gemma_forward_cuda


def _select_implementation(manifest: BundleManifest) -> ImplementationSpec:
    candidates = [
        implementation
        for implementation in manifest.implementations
        if implementation.definition == RMSNORM_DEFINITION
    ]
    if not candidates:
        raise RuntimeError(f"manifest does not define {RMSNORM_DEFINITION}")
    return sorted(candidates, key=lambda item: (-item.priority, item.id))[0]


def _guard_covers(implementation: ImplementationSpec, shape: dict[str, int]) -> bool:
    guard = implementation.shape_guard
    token_guard = guard.get("num_tokens")
    hidden_guard = guard.get("hidden")
    if hidden_guard != shape["hidden"]:
        return False
    if not isinstance(token_guard, list) or len(token_guard) != 2:
        return False
    return int(token_guard[0]) <= shape["num_tokens"] <= int(token_guard[1])


def _is_supported_tensor(value: Any) -> bool:
    return bool(
        getattr(value, "is_cuda", False)
        and hasattr(value, "shape")
        and hasattr(value, "dtype")
        and len(value.shape) >= 2
    )


def _view_as_candidate_matrix(value: Any, hidden: int) -> Any | None:
    try:
        matrix = value.view(-1, hidden)
    except Exception:
        return None
    if tuple(matrix.stride()) != (hidden, 1):
        return None
    return matrix


def _shape_of(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(item) for item in shape]


def _record_candidate_call(event: dict[str, Any]) -> None:
    _STATS["candidate_calls"] = int(_STATS.get("candidate_calls") or 0) + 1
    classes = _STATS.setdefault("classes", {})
    class_name = str(event.get("class") or "unknown")
    classes[class_name] = int(classes.get(class_name) or 0) + 1
    event["event"] = "candidate_call"
    _record_event(event)


def _record_event(event: dict[str, Any]) -> None:
    trace_raw = os.environ.get(TRACE_ENV)
    if not trace_raw:
        return
    payload = dict(event)
    payload.setdefault("pid", os.getpid())
    payload.setdefault("time_s", time.time())
    path = Path(trace_raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _flush_summary() -> None:
    _record_event({"event": "plugin_summary", **_STATS})
