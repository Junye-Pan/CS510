from __future__ import annotations

import ast
import json
import py_compile
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .definitions import DefinitionSpec, get_definition


EXPECTED_SCHEMA = "agentic_opt.llm_kernel_bundle.v1"
EXPECTED_TARGET = {
    "model": "qwen-3.5-4b",
    "framework": "vllm",
    "gpu": "H200",
    "dtype": "fp16",
}
SUPPORTED_LANGUAGES = {"python", "triton"}
SUPPORTED_BINDINGS = {"torch"}
IMPLEMENTATION_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$")
MAX_SOURCE_BYTES = 2_000_000
MAX_IMPLEMENTATIONS = 32
FORBIDDEN_BUNDLE_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".gguf"}
FORBIDDEN_BUNDLE_NAMES = {"tokenizer.json", "tokenizer.model", "model.safetensors"}


@dataclass(frozen=True)
class ImplementationSpec:
    id: str
    definition: str
    language: str
    binding: str
    entry_point: str
    sources: tuple[str, ...]
    shape_guard: dict[str, Any]
    priority: int
    fallback: str

    @property
    def entry_path(self) -> str:
        return self.entry_point.split("::", 1)[0]

    @property
    def entry_symbol(self) -> str:
        return self.entry_point.split("::", 1)[1]

    def to_solution_like(self, candidate_root: Path) -> dict[str, Any]:
        return {
            "name": self.id,
            "definition": self.definition,
            "author": "agentic_opt_candidate",
            "spec": {
                "language": self.language,
                "target_hardware": ["cuda"],
                "entry_point": self.entry_point,
                "dependencies": [],
                "destination_passing_style": True,
                "binding": self.binding,
            },
            "sources": [
                {"path": source, "content": (candidate_root / source).read_text(encoding="utf-8")}
                for source in self.sources
            ],
        }


@dataclass(frozen=True)
class BundleManifest:
    schema: str
    target: dict[str, str]
    implementations: tuple[ImplementationSpec, ...]

    @property
    def definition_names(self) -> tuple[str, ...]:
        return tuple(sorted({implementation.definition for implementation in self.implementations}))


class ManifestValidationError(ValueError):
    pass


def load_manifest(path: Path) -> BundleManifest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestValidationError(f"manifest is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ManifestValidationError("manifest must be a JSON object")
    schema = raw.get("schema")
    if schema != EXPECTED_SCHEMA:
        raise ManifestValidationError(f"manifest schema must be {EXPECTED_SCHEMA!r}")
    target = raw.get("target")
    if not isinstance(target, dict):
        raise ManifestValidationError("manifest target must be an object")
    for key, expected in EXPECTED_TARGET.items():
        if target.get(key) != expected:
            raise ManifestValidationError(f"manifest target.{key} must be {expected!r}")
    raw_implementations = raw.get("implementations")
    if raw_implementations is None:
        raw_implementations = []
    if not isinstance(raw_implementations, list):
        raise ManifestValidationError("manifest implementations must be a list")
    if len(raw_implementations) > MAX_IMPLEMENTATIONS:
        raise ManifestValidationError(f"manifest may declare at most {MAX_IMPLEMENTATIONS} implementations")
    implementations = tuple(_parse_implementation(item, index) for index, item in enumerate(raw_implementations))
    seen: set[str] = set()
    for implementation in implementations:
        if implementation.id in seen:
            raise ManifestValidationError(f"duplicate implementation id: {implementation.id}")
        seen.add(implementation.id)
    return BundleManifest(
        schema=schema,
        target={key: str(target[key]) for key in EXPECTED_TARGET},
        implementations=implementations,
    )


def validate_bundle_files(manifest: BundleManifest, *, candidate_root: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    _reject_forbidden_files(candidate_root)
    for implementation in manifest.implementations:
        definition = get_definition(implementation.definition)
        if definition is None:
            raise ManifestValidationError(f"unknown definition: {implementation.definition}")
        _validate_shape_guard(implementation, definition)
        _validate_source_list(implementation, candidate_root=candidate_root)
        _validate_entry_symbol(implementation, candidate_root=candidate_root)
        checks.append(
            {
                "name": f"implementation:{implementation.id}",
                "status": "passed",
                "message": f"{implementation.language}/{implementation.binding} targets {implementation.definition}",
            }
        )
    return checks


def _parse_implementation(raw: Any, index: int) -> ImplementationSpec:
    if not isinstance(raw, dict):
        raise ManifestValidationError(f"implementation {index} must be an object")
    raw_id = _required_str(raw, "id", index)
    if not IMPLEMENTATION_ID_RE.fullmatch(raw_id):
        raise ManifestValidationError(f"implementation {index} has invalid id: {raw_id!r}")
    definition = _required_str(raw, "definition", index)
    if get_definition(definition) is None:
        raise ManifestValidationError(f"implementation {raw_id} references unknown definition: {definition}")
    language = _required_str(raw, "language", index)
    if language not in SUPPORTED_LANGUAGES:
        raise ManifestValidationError(f"implementation {raw_id} uses unsupported language: {language}")
    binding = _required_str(raw, "binding", index)
    if binding not in SUPPORTED_BINDINGS:
        raise ManifestValidationError(f"implementation {raw_id} uses unsupported binding: {binding}")
    entry_point = _required_str(raw, "entry_point", index)
    if entry_point.count("::") != 1:
        raise ManifestValidationError(f"implementation {raw_id} entry_point must be '<path>::<function>'")
    sources_raw = raw.get("sources")
    if not isinstance(sources_raw, list) or not sources_raw or not all(isinstance(item, str) for item in sources_raw):
        raise ManifestValidationError(f"implementation {raw_id} sources must be a non-empty string list")
    sources = tuple(sources_raw)
    if len(set(sources)) != len(sources):
        raise ManifestValidationError(f"implementation {raw_id} has duplicate source paths")
    entry_path = entry_point.split("::", 1)[0]
    if entry_path not in sources:
        raise ManifestValidationError(f"implementation {raw_id} entry source {entry_path!r} is not listed in sources")
    shape_guard = raw.get("shape_guard")
    if not isinstance(shape_guard, dict):
        raise ManifestValidationError(f"implementation {raw_id} shape_guard must be an object")
    priority = raw.get("priority", 0)
    if not isinstance(priority, int):
        raise ManifestValidationError(f"implementation {raw_id} priority must be an integer")
    fallback = raw.get("fallback", "baseline")
    if fallback != "baseline":
        raise ManifestValidationError(f"implementation {raw_id} fallback must be 'baseline'")
    return ImplementationSpec(
        id=raw_id,
        definition=definition,
        language=language,
        binding=binding,
        entry_point=entry_point,
        sources=sources,
        shape_guard=shape_guard,
        priority=priority,
        fallback=fallback,
    )


def _required_str(raw: dict[str, Any], key: str, index: int) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ManifestValidationError(f"implementation {index} field {key!r} must be a non-empty string")
    return value


def _validate_source_list(implementation: ImplementationSpec, *, candidate_root: Path) -> None:
    for source in implementation.sources:
        path = _safe_relative_path(source, label=f"implementation {implementation.id} source")
        source_path = (candidate_root / path).resolve()
        root = candidate_root.resolve()
        if not source_path.is_relative_to(root):
            raise ManifestValidationError(f"implementation {implementation.id} source escapes candidate root: {source}")
        if not source_path.exists() or not source_path.is_file():
            raise ManifestValidationError(f"implementation {implementation.id} source does not exist: {source}")
        if source_path.stat().st_size > MAX_SOURCE_BYTES:
            raise ManifestValidationError(f"implementation {implementation.id} source is too large: {source}")
        if source_path.suffix == ".py":
            _syntax_check(source_path)


def _validate_entry_symbol(implementation: ImplementationSpec, *, candidate_root: Path) -> None:
    entry_path = (candidate_root / _safe_relative_path(implementation.entry_path, label="entry path")).resolve()
    if entry_path.suffix != ".py":
        raise ManifestValidationError(f"implementation {implementation.id} MVP entry source must be a Python file")
    try:
        tree = ast.parse(entry_path.read_text(encoding="utf-8"), filename=str(entry_path))
    except SyntaxError as exc:
        raise ManifestValidationError(f"implementation {implementation.id} entry source has invalid syntax: {exc}") from exc
    has_symbol = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == implementation.entry_symbol
        for node in tree.body
    )
    if not has_symbol:
        raise ManifestValidationError(
            f"implementation {implementation.id} entry source does not define {implementation.entry_symbol!r}"
        )


def _validate_shape_guard(implementation: ImplementationSpec, definition: DefinitionSpec) -> None:
    for axis_name, axis in definition.axes.items():
        if axis_name not in implementation.shape_guard:
            raise ManifestValidationError(f"implementation {implementation.id} shape_guard missing axis {axis_name!r}")
        guard = implementation.shape_guard[axis_name]
        if axis.kind == "const":
            if guard != axis.value:
                raise ManifestValidationError(
                    f"implementation {implementation.id} shape_guard.{axis_name} must be {axis.value}"
                )
            continue
        if not (
            isinstance(guard, list)
            and len(guard) == 2
            and all(isinstance(item, int) for item in guard)
            and guard[0] <= guard[1]
        ):
            raise ManifestValidationError(
                f"implementation {implementation.id} shape_guard.{axis_name} must be [min, max]"
            )
        if not axis.contains(guard[0]) or not axis.contains(guard[1]):
            raise ManifestValidationError(
                f"implementation {implementation.id} shape_guard.{axis_name} is outside task range"
            )
    unknown_axes = set(implementation.shape_guard) - set(definition.axes)
    if unknown_axes:
        raise ManifestValidationError(
            f"implementation {implementation.id} shape_guard has unknown axes: {sorted(unknown_axes)}"
        )


def _safe_relative_path(raw_path: str, *, label: str) -> Path:
    if "\\" in raw_path:
        raise ManifestValidationError(f"{label} may not contain backslashes: {raw_path}")
    path = Path(raw_path)
    if path.is_absolute():
        raise ManifestValidationError(f"{label} may not be absolute: {raw_path}")
    if ".." in path.parts:
        raise ManifestValidationError(f"{label} may not contain '..': {raw_path}")
    if not raw_path or raw_path.endswith("/"):
        raise ManifestValidationError(f"{label} must point to a file: {raw_path}")
    return path


def _syntax_check(path: Path) -> None:
    with tempfile.NamedTemporaryFile(suffix=".pyc", delete=False) as handle:
        bytecode_path = handle.name
    try:
        py_compile.compile(str(path), cfile=bytecode_path, doraise=True)
    except py_compile.PyCompileError as exc:
        raise ManifestValidationError(f"source {path.name} failed py_compile: {exc}") from exc
    finally:
        Path(bytecode_path).unlink(missing_ok=True)


def _reject_forbidden_files(candidate_root: Path) -> None:
    for path in candidate_root.rglob("*"):
        if not path.is_file():
            continue
        lowered = path.name.lower()
        if path.suffix.lower() in FORBIDDEN_BUNDLE_SUFFIXES or lowered in FORBIDDEN_BUNDLE_NAMES:
            relative = path.relative_to(candidate_root).as_posix()
            raise ManifestValidationError(f"candidate bundle contains forbidden model/tokenizer artifact: {relative}")
