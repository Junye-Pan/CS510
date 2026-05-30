from __future__ import annotations

import ast
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_opt.common.runtime_env import TaskRuntimeSpec
from agentic_opt.task_api import CandidateSpec, TaskMetadata


TASK_ID = "llm_inference_qwen3_1_7b_sglang_4090"
EXPECTED_KERNELS = {
    "kernels/rmsnorm.py": "run",
    "kernels/fused_add_rmsnorm.py": "run",
    "kernels/swiglu.py": "run",
    "kernels/attention_backend.py": "forward",
    "kernels/sampling_backend.py": "sample",
}
TASK_PRIVATE_IMPORT_PREFIXES = (
    "tasks.llm_inference_qwen3_1_7b_sglang_4090.private",
    "llm_inference_qwen3_1_7b_sglang_4090.private",
)
RUNTIME_OBJECT_NAMES = {
    "forward_batch",
    "layer",
    "logits_output",
    "sampling_info",
}
SYS_MODULES_MUTATORS = {
    "__delitem__",
    "__setitem__",
    "clear",
    "pop",
    "popitem",
    "setdefault",
    "update",
}

QWEN3_RUNTIME = TaskRuntimeSpec(
    python=">=3.11,<3.12",
    required_imports=("torch", "triton", "sglang"),
    forbidden_shadow_modules=("torch", "triton", "sglang", "sgl_kernel"),
    system_site_packages=True,
    verify_public_seed=True,
)


@dataclass(frozen=True)
class Qwen3SGLangKernelTask:
    metadata: TaskMetadata = TaskMetadata(
        task_id=TASK_ID,
        title="Qwen3 1.7B SGLang RTX 4090 BF16 Kernel Optimization",
        candidate_spec=CandidateSpec(
            candidate_root="candidate",
            public_seed_root="initial_candidate",
            entrypoint_name="manifest.json",
            description=(
                "Directory candidate with a read-only manifest and writable "
                "Triton kernel files under candidate/kernels/."
            ),
        ),
    )
    runtime_spec: TaskRuntimeSpec = QWEN3_RUNTIME

    @property
    def public_dir(self) -> Path:
        return Path(__file__).resolve().parent / "public"

    def verify_entry(self, entry_path: Path) -> dict[str, Any]:
        started = time.perf_counter()
        checks: list[dict[str, Any]] = []
        candidate_root = entry_path.resolve().parent

        def fail(name: str, message: str) -> dict[str, Any]:
            checks.append({"name": name, "status": "failed", "message": message})
            return {
                "status": "failed",
                "valid": False,
                "checks": checks,
                "feedback": {"summary": message, "errors": [message], "warnings": []},
                "elapsed_s": time.perf_counter() - started,
            }

        if entry_path.name != "manifest.json":
            return fail("entrypoint_name", f"expected manifest.json, got {entry_path.name}")
        if not entry_path.is_file():
            return fail("entrypoint_exists", f"{entry_path} missing")
        checks.append({"name": "entrypoint_exists", "status": "passed", "message": None})

        reference_manifest = self.public_dir / "initial_candidate" / "manifest.json"
        if entry_path.read_text() != reference_manifest.read_text():
            return fail("manifest_digest", "manifest.json must match the task-owned reference")
        checks.append({"name": "manifest_digest", "status": "passed", "message": None})

        try:
            manifest = json.loads(entry_path.read_text())
        except json.JSONDecodeError as exc:
            return fail("manifest_json", str(exc))
        if manifest.get("schema") != "agentic_opt.sglang_fixed_kernel_surface.v1":
            return fail("manifest_schema", "unsupported manifest schema")
        checks.append({"name": "manifest_schema", "status": "passed", "message": None})

        for forbidden in ("operators", "backends", "private"):
            if (candidate_root / forbidden).exists():
                return fail("candidate_boundary", f"candidate may not contain {forbidden}/")
        checks.append({"name": "candidate_boundary", "status": "passed", "message": None})

        try:
            _check_candidate_tree(candidate_root)
        except Exception as exc:
            return fail("candidate_tree", str(exc))
        checks.append({"name": "candidate_tree", "status": "passed", "message": None})

        for relative, symbol in EXPECTED_KERNELS.items():
            path = candidate_root / relative
            if not path.is_file():
                return fail("kernel_exists", f"{relative} missing")
            try:
                module_ast = ast.parse(path.read_text(), filename=str(path))
            except SyntaxError as exc:
                return fail("kernel_py_compile", f"{relative}: {exc}")
            try:
                _check_kernel_static_safety(relative, module_ast)
            except Exception as exc:
                return fail("kernel_static_safety", f"{relative}: {exc}")
            try:
                _check_symbol(module_ast, symbol)
            except Exception as exc:
                return fail("kernel_abi", f"{relative}: {exc}")
        checks.append({"name": "kernel_files", "status": "passed", "message": None})

        if os.environ.get("AO_LLM_KERNEL_STATIC_VERIFY_ONLY") == "1":
            return {
                "status": "passed",
                "valid": True,
                "checks": checks,
                "feedback": {
                    "summary": "static candidate admission checks passed",
                    "errors": [],
                    "warnings": [],
                },
                "elapsed_s": time.perf_counter() - started,
            }

        try:
            from .private.component_checks import run_component_checks

            component_results = run_component_checks(entry_path)
        except Exception as exc:
            message = f"component correctness checks crashed: {type(exc).__name__}: {exc}"
            checks.append({"name": "component_correctness", "status": "failed", "message": message})
            return {
                "status": "failed",
                "valid": False,
                "checks": checks,
                "feedback": {"summary": message, "errors": [message], "warnings": []},
                "elapsed_s": time.perf_counter() - started,
            }
        if not component_results.get("valid"):
            errors = [
                str(result.get("error"))
                for result in component_results.get("results", [])
                if not result.get("valid") and result.get("error")
            ]
            message = "component correctness checks failed"
            checks.append({"name": "component_correctness", "status": "failed", "message": "; ".join(errors)})
            return {
                "status": "failed",
                "valid": False,
                "checks": checks,
                "component_results": component_results,
                "feedback": {
                    "summary": message,
                    "errors": errors or [message],
                    "warnings": [],
                },
                "elapsed_s": time.perf_counter() - started,
            }
        checks.append({"name": "component_correctness", "status": "passed", "message": None})

        try:
            from .private.sglang_smoke import run_sglang_server_smoke

            smoke_results = run_sglang_server_smoke(entry_path)
        except Exception as exc:
            message = f"SGLang server-level smoke crashed: {type(exc).__name__}: {exc}"
            checks.append({"name": "sglang_server_smoke", "status": "failed", "message": message})
            return {
                "status": "failed",
                "valid": False,
                "checks": checks,
                "component_results": component_results,
                "feedback": {"summary": message, "errors": [message], "warnings": []},
                "elapsed_s": time.perf_counter() - started,
            }
        if not smoke_results.get("valid"):
            errors = [str(error) for error in smoke_results.get("errors", []) if error]
            message = "SGLang server-level smoke failed"
            checks.append({"name": "sglang_server_smoke", "status": "failed", "message": "; ".join(errors)})
            return {
                "status": "failed",
                "valid": False,
                "checks": checks,
                "component_results": component_results,
                "smoke_results": smoke_results,
                "feedback": {
                    "summary": message,
                    "errors": errors or [message],
                    "warnings": [],
                },
                "elapsed_s": time.perf_counter() - started,
            }
        checks.append({"name": "sglang_server_smoke", "status": "passed", "message": None})

        return {
            "status": "passed",
            "valid": True,
            "checks": checks,
            "component_results": component_results,
            "smoke_results": smoke_results,
            "feedback": {
                "summary": "static, component correctness, and SGLang server smoke checks passed",
                "errors": [],
                "warnings": [],
            },
            "elapsed_s": time.perf_counter() - started,
        }

    def probe_entry(self, entry_path: Path, *, kind: str) -> dict[str, Any]:
        return {
            "ok": False,
            "valid": False,
            "kind": kind,
            "status": "probe_disabled",
            "feedback": {
                "summary": "probe is disabled for this task; use eval verify or submit",
                "errors": ["probe_disabled"],
                "warnings": [],
            },
        }

    def evaluate_entry(self, entry_path: Path) -> dict[str, Any]:
        verifier = self.verify_entry(entry_path)
        if not verifier.get("valid"):
            raise ValueError("candidate failed verifier")
        from .private.official_evaluator import run_official_evaluation

        return run_official_evaluation(entry_path, verifier=verifier)


def _check_symbol(module_ast: ast.Module, symbol: str) -> None:
    for node in module_ast.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol:
            if isinstance(node, ast.AsyncFunctionDef):
                raise TypeError(f"{symbol} must not be async")
            return
    raise AttributeError(f"{symbol} function is missing")


def _check_kernel_static_safety(relative: str, module_ast: ast.Module) -> None:
    importlib_aliases = {"importlib"}
    import_module_aliases = {"import_module"}
    sys_aliases = {"sys"}

    for node in ast.walk(module_ast):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                _raise_for_forbidden_import(relative, name)
                if name == "importlib":
                    importlib_aliases.add(alias.asname or "importlib")
                if name == "sys":
                    sys_aliases.add(alias.asname or "sys")
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise PermissionError("relative imports are not allowed")
            module = node.module or ""
            if module == "__future__":
                continue
            _raise_for_forbidden_import(relative, module)
            if module == "importlib":
                for alias in node.names:
                    if alias.name == "import_module":
                        import_module_aliases.add(alias.asname or alias.name)
            if module == "sys":
                for alias in node.names:
                    if alias.name == "modules":
                        raise PermissionError("importing sys.modules is not allowed")

    for node in ast.walk(module_ast):
        if isinstance(node, ast.Call):
            if _is_dynamic_import_call(node, importlib_aliases=importlib_aliases, import_module_aliases=import_module_aliases):
                raise PermissionError("dynamic imports are not allowed in candidate kernels")
            if _is_runtime_mutator_call(node):
                raise PermissionError("mutating SGLang runtime objects is not allowed")
            if _is_sys_modules_mutator_call(node, sys_aliases=sys_aliases):
                raise PermissionError("mutating sys.modules is not allowed")
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if _is_forbidden_assignment_target(target, sys_aliases=sys_aliases):
                    raise PermissionError("candidate kernels may not monkeypatch SGLang runtime state")
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                if _is_forbidden_assignment_target(target, sys_aliases=sys_aliases):
                    raise PermissionError("candidate kernels may not delete SGLang runtime state")


def _raise_for_forbidden_import(relative: str, module: str) -> None:
    if module == "sglang" or module.startswith("sglang."):
        raise PermissionError(
            "candidate kernels may not import sglang; use task-owned wrappers and fallback only"
        )
    if any(module == prefix or module.startswith(prefix + ".") for prefix in TASK_PRIVATE_IMPORT_PREFIXES):
        raise PermissionError("candidate kernels may not import task-private modules")
    if module.endswith(".private") or ".private." in module:
        if "llm_inference_qwen3_1_7b_sglang_4090" in module:
            raise PermissionError("candidate kernels may not import task-private modules")


def _is_dynamic_import_call(
    node: ast.Call,
    *,
    importlib_aliases: set[str],
    import_module_aliases: set[str],
) -> bool:
    if isinstance(node.func, ast.Name):
        return node.func.id == "__import__" or node.func.id in import_module_aliases
    if isinstance(node.func, ast.Attribute):
        return (
            node.func.attr == "import_module"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in importlib_aliases
        )
    return False


def _is_runtime_mutator_call(node: ast.Call) -> bool:
    if isinstance(node.func, ast.Name) and node.func.id in {"setattr", "delattr"}:
        return bool(node.args) and _contains_runtime_object(node.args[0])
    if isinstance(node.func, ast.Attribute) and node.func.attr in {"__setattr__", "__delattr__"}:
        if _contains_runtime_object(node.func.value):
            return True
        return bool(node.args) and _contains_runtime_object(node.args[0])
    return False


def _is_sys_modules_mutator_call(node: ast.Call, *, sys_aliases: set[str]) -> bool:
    if not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr not in SYS_MODULES_MUTATORS:
        return False
    return _is_sys_modules_expr(node.func.value, sys_aliases=sys_aliases)


def _is_forbidden_assignment_target(target: ast.AST, *, sys_aliases: set[str]) -> bool:
    if isinstance(target, ast.Attribute):
        return _contains_runtime_object(target.value)
    if isinstance(target, ast.Subscript):
        return _is_sys_modules_expr(target.value, sys_aliases=sys_aliases)
    if isinstance(target, (ast.Tuple, ast.List)):
        return any(_is_forbidden_assignment_target(item, sys_aliases=sys_aliases) for item in target.elts)
    return False


def _contains_runtime_object(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id in RUNTIME_OBJECT_NAMES or node.id == "sglang"
    if isinstance(node, ast.Attribute):
        return _contains_runtime_object(node.value)
    if isinstance(node, ast.Subscript):
        return _contains_runtime_object(node.value)
    if isinstance(node, ast.Call):
        return any(_contains_runtime_object(arg) for arg in node.args)
    return False


def _is_sys_modules_expr(node: ast.AST, *, sys_aliases: set[str]) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "modules"
        and isinstance(node.value, ast.Name)
        and node.value.id in sys_aliases
    )


def _check_candidate_tree(candidate_root: Path) -> None:
    _prune_runtime_cache_files(candidate_root)

    allowed_files = {Path("manifest.json"), *(Path(relative) for relative in EXPECTED_KERNELS)}
    allowed_dirs = {Path("."), Path("kernels")}

    for path in candidate_root.rglob("*"):
        relative = path.relative_to(candidate_root)
        if path.is_symlink():
            raise PermissionError(f"candidate may not contain symlink: {relative}")
        if path.is_dir():
            if relative not in allowed_dirs:
                raise PermissionError(f"candidate may not contain directory outside fixed surface: {relative}/")
            continue
        if not path.is_file():
            raise PermissionError(f"candidate may not contain special file: {relative}")
        if relative not in allowed_files:
            raise PermissionError(f"candidate may not contain file outside fixed kernel surface: {relative}")


def _prune_runtime_cache_files(candidate_root: Path) -> None:
    for path in sorted(candidate_root.rglob("__pycache__"), key=lambda item: len(item.parts), reverse=True):
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
    for path in candidate_root.rglob("*"):
        if path.is_file() and path.suffix in {".pyc", ".pyo"}:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def create_task() -> Qwen3SGLangKernelTask:
    return Qwen3SGLangKernelTask()
