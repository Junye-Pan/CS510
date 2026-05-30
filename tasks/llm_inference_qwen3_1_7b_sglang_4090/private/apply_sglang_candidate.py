from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from .backends import attention_backend, sampling_backend
from .operators import rmsnorm, swiglu


KERNEL_ENTRYPOINTS = {
    "rmsnorm": "kernels/rmsnorm.py::run",
    "fused_add_rmsnorm": "kernels/fused_add_rmsnorm.py::run",
    "swiglu": "kernels/swiglu.py::run",
    "attention_backend": "kernels/attention_backend.py::forward",
    "sampling_backend": "kernels/sampling_backend.py::sample",
}
DEFAULT_STATS_FLUSH_EVERY = 2048


@dataclass
class DispatchStats:
    counters: dict[str, int] = field(default_factory=dict)
    fallback_reasons: dict[str, int] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    flush_path: Path | None = None
    flush_every: int = field(default_factory=lambda: _stats_flush_every())

    def inc(self, name: str, amount: int = 1) -> None:
        previous = self.counters.get(name, 0)
        current = previous + amount
        self.counters[name] = current
        if self._should_flush_counter(previous=previous, current=current):
            self.flush()

    def event(self, name: str, **payload: Any) -> None:
        self.events.append({"event": name, **payload})
        self.flush()

    def fallback(self, domain: str, reason: str) -> None:
        self.inc(f"{domain}.fallback")
        key = f"{domain}:{reason}"
        self.fallback_reasons[key] = self.fallback_reasons.get(key, 0) + 1
        self.flush()

    def exception(self, domain: str, exc: BaseException) -> None:
        self.inc(f"{domain}.exception")
        self.event(
            "candidate_exception",
            domain=domain,
            exception_type=type(exc).__name__,
            message=str(exc),
        )

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "counters": dict(sorted(self.counters.items())),
            "fallback_reasons": dict(sorted(self.fallback_reasons.items())),
            "events": list(self.events),
        }

    def flush(self) -> None:
        if self.flush_path is None:
            return
        self.flush_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_jsonable()
        payload["pid"] = os.getpid()
        tmp_path = self.flush_path.with_suffix(self.flush_path.suffix + f".{os.getpid()}.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp_path.replace(self.flush_path)

    def _should_flush_counter(self, *, previous: int, current: int) -> bool:
        if previous <= 0 < current:
            return True
        if self.flush_every <= 0:
            return True
        return current // self.flush_every != previous // self.flush_every


@dataclass
class CandidateModules:
    rmsnorm: ModuleType
    fused_add_rmsnorm: ModuleType
    swiglu: ModuleType
    attention_backend: ModuleType
    sampling_backend: ModuleType


class AppliedCandidateIntegration:
    def __init__(self, *, candidate_root: Path, modules: CandidateModules, stats: DispatchStats | None = None):
        self.candidate_root = candidate_root
        self.modules = modules
        self.stats = stats or DispatchStats()
        self._uninstallers: list[Callable[[], None]] = []
        self._model_runner_wrapped = False

    def __enter__(self) -> "AppliedCandidateIntegration":
        self.install_operators()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.uninstall()

    def install_operators(self) -> None:
        if any(getattr(uninstaller, "__ao_operator__", False) for uninstaller in self._uninstallers):
            return
        rms_uninstall = rmsnorm.install(
            rmsnorm_kernel=self.modules.rmsnorm,
            fused_add_kernel=self.modules.fused_add_rmsnorm,
            stats=self.stats,
        )
        setattr(rms_uninstall, "__ao_operator__", True)
        swiglu_uninstall = swiglu.install(kernel=self.modules.swiglu, stats=self.stats)
        setattr(swiglu_uninstall, "__ao_operator__", True)
        self._uninstallers.extend([swiglu_uninstall, rms_uninstall])

    def wrap_model_runner(self, model_runner: Any) -> None:
        if self._model_runner_wrapped:
            return
        attention_uninstall = attention_backend.install_model_runner(
            model_runner=model_runner,
            kernel=self.modules.attention_backend,
            stats=self.stats,
        )
        sampling_uninstall = sampling_backend.install_model_runner(
            model_runner=model_runner,
            kernel=self.modules.sampling_backend,
            stats=self.stats,
        )
        self._uninstallers.extend([sampling_uninstall, attention_uninstall])
        self._model_runner_wrapped = True

    def uninstall(self) -> None:
        while self._uninstallers:
            uninstall = self._uninstallers.pop()
            uninstall()
        self._model_runner_wrapped = False


def load_candidate(entry_path: Path, *, stats: DispatchStats | None = None) -> AppliedCandidateIntegration:
    entry_path = entry_path.resolve()
    if entry_path.name != "manifest.json":
        raise ValueError(f"entry path must be manifest.json: {entry_path}")
    candidate_root = entry_path.parent
    manifest = json.loads(entry_path.read_text())
    if manifest.get("schema") != "agentic_opt.sglang_fixed_kernel_surface.v1":
        raise ValueError("unsupported candidate manifest schema")
    modules = CandidateModules(
        rmsnorm=_load_kernel(candidate_root, KERNEL_ENTRYPOINTS["rmsnorm"], module_key="rmsnorm"),
        fused_add_rmsnorm=_load_kernel(
            candidate_root,
            KERNEL_ENTRYPOINTS["fused_add_rmsnorm"],
            module_key="fused_add_rmsnorm",
        ),
        swiglu=_load_kernel(candidate_root, KERNEL_ENTRYPOINTS["swiglu"], module_key="swiglu"),
        attention_backend=_load_kernel(
            candidate_root,
            KERNEL_ENTRYPOINTS["attention_backend"],
            module_key="attention_backend",
        ),
        sampling_backend=_load_kernel(
            candidate_root,
            KERNEL_ENTRYPOINTS["sampling_backend"],
            module_key="sampling_backend",
        ),
    )
    applied = AppliedCandidateIntegration(candidate_root=candidate_root, modules=modules, stats=stats)
    applied.stats.event("candidate_loaded", candidate_root=str(candidate_root))
    return applied


def _load_kernel(candidate_root: Path, entrypoint: str, *, module_key: str) -> ModuleType:
    relative, symbol = entrypoint.split("::", 1)
    path = (candidate_root / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if not path.is_relative_to(candidate_root):
        raise PermissionError(f"kernel path escapes candidate root: {path}")
    module_name = f"_ao_qwen3_1_7b_candidate_{module_key}_{abs(hash(path))}"
    previous = sys.modules.pop(module_name, None)
    previous_dont_write_bytecode = sys.dont_write_bytecode
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not import {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
        func = getattr(module, symbol, None)
        if not callable(func):
            raise AttributeError(f"{entrypoint} is not callable")
        return module
    except Exception:
        sys.modules.pop(module_name, None)
        if previous is not None:
            sys.modules[module_name] = previous
        raise
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode


def _stats_flush_every() -> int:
    raw = os.environ.get("AO_QWEN3_1_7B_STATS_FLUSH_EVERY")
    if raw is None or raw == "":
        return DEFAULT_STATS_FLUSH_EVERY
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_STATS_FLUSH_EVERY
