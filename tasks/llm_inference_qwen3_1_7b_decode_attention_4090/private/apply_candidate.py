from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


MANIFEST_SCHEMA = "agentic_opt.sglang_decode_attention_kernel.v1"
KERNEL_ENTRYPOINTS = {
    "decode_attention": "kernels/decode_attention.py::run",
}


@dataclass(frozen=True)
class CandidateModules:
    decode_attention: ModuleType


@dataclass(frozen=True)
class LoadedCandidate:
    candidate_root: Path
    modules: CandidateModules


def load_candidate(entry_path: Path) -> LoadedCandidate:
    entry_path = entry_path.resolve()
    if entry_path.name != "manifest.json":
        raise ValueError(f"entry path must be manifest.json: {entry_path}")
    candidate_root = entry_path.parent
    manifest = json.loads(entry_path.read_text())
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported candidate manifest schema")
    modules = CandidateModules(
        decode_attention=_load_kernel(
            candidate_root,
            KERNEL_ENTRYPOINTS["decode_attention"],
            module_key="decode_attention",
        )
    )
    return LoadedCandidate(candidate_root=candidate_root, modules=modules)


def _load_kernel(candidate_root: Path, entrypoint: str, *, module_key: str) -> ModuleType:
    relative, symbol = entrypoint.split("::", 1)
    path = (candidate_root / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if not path.is_relative_to(candidate_root):
        raise PermissionError(f"kernel path escapes candidate root: {path}")

    module_name = f"_ao_qwen3_1_7b_decode_attention_candidate_{module_key}_{abs(hash(path))}"
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
