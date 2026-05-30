from __future__ import annotations

import atexit
import importlib.abc
import importlib.machinery
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


ENABLE_ENV = "AO_QWEN3_1_7B_ENABLE_SGLANG_INJECTION"
CANDIDATE_ENTRY_ENV = "AO_QWEN3_1_7B_CANDIDATE_ENTRY"
STATS_DIR_ENV = "AO_QWEN3_1_7B_STATS_DIR"
TARGET_MODULE = "sglang.srt.model_executor.model_runner"


def install() -> None:
    if os.environ.get(ENABLE_ENV) != "1":
        return
    if not os.environ.get(CANDIDATE_ENTRY_ENV):
        _log(f"{CANDIDATE_ENTRY_ENV} is not set; skipping injection")
        return
    existing = sys.modules.get(TARGET_MODULE)
    if existing is not None:
        _patch_model_runner_module(existing)
        return
    if not any(isinstance(finder, _ModelRunnerPatchFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _ModelRunnerPatchFinder())
        _log("installed ModelRunner import hook")


class _ModelRunnerPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any, target: Any = None):
        if fullname != TARGET_MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _ModelRunnerPatchLoader(spec.loader)
        return spec


class _ModelRunnerPatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped_loader: importlib.abc.Loader):
        self._wrapped_loader = wrapped_loader

    def create_module(self, spec):
        create_module = getattr(self._wrapped_loader, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self._wrapped_loader.exec_module(module)
        _patch_model_runner_module(module)


def _patch_model_runner_module(module: ModuleType) -> None:
    model_runner_cls = getattr(module, "ModelRunner", None)
    if model_runner_cls is None:
        _log("ModelRunner class missing; skipping injection")
        return
    if getattr(model_runner_cls, "__ao_qwen3_1_7b_injection_patch__", False):
        return

    from tasks.llm_inference_qwen3_1_7b_sglang_4090.private.apply_sglang_candidate import (
        DispatchStats,
        load_candidate,
    )

    candidate_entry = Path(os.environ[CANDIDATE_ENTRY_ENV]).resolve()
    stats_dir = Path(os.environ.get(STATS_DIR_ENV, "/tmp/ao_qwen3_1_7b_sglang_stats")).resolve()
    stats_path = stats_dir / f"stats_{os.getpid()}.json"
    stats = DispatchStats(flush_path=stats_path)
    applied = load_candidate(candidate_entry, stats=stats)
    applied.install_operators()

    original_init_attention_backend = model_runner_cls.init_attention_backend

    def init_attention_backend_with_ao_candidate(self, *args, **kwargs):
        result = original_init_attention_backend(self, *args, **kwargs)
        if not getattr(self, "_ao_qwen3_1_7b_candidate_wrapped", False):
            applied.wrap_model_runner(self)
            self._ao_qwen3_1_7b_candidate_wrapped = True
            stats.event(
                "model_runner_wrapped",
                tp_rank=getattr(self, "tp_rank", None),
                tp_size=getattr(self, "tp_size", None),
                pp_rank=getattr(self, "pp_rank", None),
                attention_backend=type(getattr(self, "attn_backend", None)).__name__,
                sampler=type(getattr(self, "sampler", None)).__name__,
            )
        return result

    init_attention_backend_with_ao_candidate.__ao_qwen3_1_7b_wrapper__ = True  # type: ignore[attr-defined]
    model_runner_cls.init_attention_backend = init_attention_backend_with_ao_candidate
    model_runner_cls.__ao_qwen3_1_7b_injection_patch__ = True
    stats.event(
        "model_runner_class_patched",
        target=f"{TARGET_MODULE}.ModelRunner.init_attention_backend",
    )
    atexit.register(_safe_uninstall, applied)
    _log(f"patched ModelRunner; stats={stats_path}")


def _safe_uninstall(applied) -> None:
    try:
        applied.uninstall()
    except Exception as exc:  # pragma: no cover - process shutdown best effort
        _log(f"uninstall failed during shutdown: {type(exc).__name__}: {exc}")
    finally:
        try:
            applied.stats.flush()
        except Exception as exc:  # pragma: no cover - process shutdown best effort
            _log(f"stats flush failed during shutdown: {type(exc).__name__}: {exc}")


def _log(message: str) -> None:
    sys.stderr.write(f"[ao-qwen3-inject pid={os.getpid()}] {message}\n")
    sys.stderr.flush()
