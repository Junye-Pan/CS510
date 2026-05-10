from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_opt.common.atomic import atomic_write_text
from agentic_opt.common.config import get_repo_root


class RuntimeEnvironmentError(RuntimeError):
    """Raised when a task runtime environment cannot be prepared or validated."""


@dataclass(frozen=True)
class TaskRuntimeSpec:
    kind: str = "local_venv"
    python: str = ">=3.11"
    requirements: tuple[str, ...] = ()
    required_imports: tuple[str, ...] = ()
    forbidden_shadow_modules: tuple[str, ...] = ()
    system_site_packages: bool = False
    verify_public_seed: bool = True

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "python": self.python,
            "requirements": list(self.requirements),
            "required_imports": list(self.required_imports),
            "forbidden_shadow_modules": list(self.forbidden_shadow_modules),
            "system_site_packages": self.system_site_packages,
            "verify_public_seed": self.verify_public_seed,
        }


@dataclass(frozen=True)
class PreparedRuntimeEnv:
    task_id: str
    fingerprint: str
    root: Path
    venv_dir: Path
    python_path: Path
    manifest_path: Path
    spec: TaskRuntimeSpec

    def exports(self) -> dict[str, str]:
        return {
            "AO_TASK_RUNTIME_ENV": str(self.manifest_path),
            "AO_TASK_RUNTIME_ROOT": str(self.root),
            "AO_TASK_RUNTIME_PYTHON": str(self.python_path),
            "AO_TASK_RUNTIME_FINGERPRINT": self.fingerprint,
        }

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "fingerprint": self.fingerprint,
            "root": str(self.root),
            "venv_dir": str(self.venv_dir),
            "python_path": str(self.python_path),
            "manifest_path": str(self.manifest_path),
            "spec": self.spec.to_jsonable(),
        }


def prepare_task_runtime(
    task: Any,
    *,
    envs_root: Path | None = None,
) -> PreparedRuntimeEnv:
    spec = getattr(task, "runtime_spec", None)
    if not isinstance(spec, TaskRuntimeSpec):
        raise RuntimeEnvironmentError(f"task {task.metadata.task_id} does not declare a TaskRuntimeSpec")
    if spec.kind != "local_venv":
        raise RuntimeEnvironmentError(f"unsupported task runtime kind for {task.metadata.task_id}: {spec.kind}")

    repo_root = get_repo_root()
    base_root = envs_root or Path(os.environ.get("AO_TASK_RUNTIME_ENVS_ROOT", repo_root / ".ao_envs"))
    fingerprint = _runtime_fingerprint(task_id=task.metadata.task_id, spec=spec)
    root = (base_root / task.metadata.task_id / fingerprint).resolve()
    venv_dir = root / "venv"
    python_path = _venv_python(venv_dir)
    manifest_path = root / "manifest.json"
    prepared = PreparedRuntimeEnv(
        task_id=task.metadata.task_id,
        fingerprint=fingerprint,
        root=root,
        venv_dir=venv_dir,
        python_path=python_path,
        manifest_path=manifest_path,
        spec=spec,
    )

    root.mkdir(parents=True, exist_ok=True)
    if not python_path.exists():
        _create_venv_with_fallback(venv_dir=venv_dir, spec=spec)
    _install_requirements(prepared)
    _run_import_preflight(prepared)
    if spec.verify_public_seed:
        _run_public_seed_preflight(prepared=prepared, task=task, repo_root=repo_root)
    _write_manifest(prepared)
    return prepared


def check_declared_dependency_shadowing(*, program_dir: Path, runtime_spec: TaskRuntimeSpec) -> None:
    for module in runtime_spec.forbidden_shadow_modules:
        if (program_dir / f"{module}.py").exists() or (program_dir / module).is_dir():
            raise RuntimeEnvironmentError(
                f"candidate workspace may not shadow declared runtime dependency: {module}"
            )


def _runtime_fingerprint(*, task_id: str, spec: TaskRuntimeSpec) -> str:
    payload = {
        "schema": 1,
        "task_id": task_id,
        "spec": spec.to_jsonable(),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:16]


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _create_venv(*, base_python: Path, venv_dir: Path, spec: TaskRuntimeSpec) -> None:
    args = [str(base_python), "-m", "venv"]
    if spec.system_site_packages:
        args.append("--system-site-packages")
    args.append(str(venv_dir))
    proc = subprocess.run(args, capture_output=True, text=True, check=False, timeout=120)
    if proc.returncode != 0:
        raise RuntimeEnvironmentError(
            f"failed to create task runtime venv with {base_python}: {proc.stderr.strip() or proc.stdout.strip()}"
        )


def _create_venv_with_fallback(*, venv_dir: Path, spec: TaskRuntimeSpec) -> None:
    override = os.environ.get("AO_TASK_RUNTIME_PYTHON")
    if override:
        base_python = Path(override).expanduser().resolve()
        _validate_base_python(base_python, spec, source="AO_TASK_RUNTIME_PYTHON")
        _create_venv(base_python=base_python, venv_dir=venv_dir, spec=spec)
        return

    errors: list[str] = []
    for candidate in _iter_python_candidates():
        if not _base_python_is_usable(candidate, spec):
            continue
        try:
            _create_venv(base_python=candidate, venv_dir=venv_dir, spec=spec)
            return
        except RuntimeEnvironmentError as exc:
            errors.append(str(exc))
            if venv_dir.exists():
                shutil.rmtree(venv_dir, ignore_errors=True)
    suffix = "" if not errors else " Venv creation failures: " + " | ".join(errors)
    raise RuntimeEnvironmentError(f"no Python executable can create a task runtime venv for python {spec.python}.{suffix}")


def _install_requirements(prepared: PreparedRuntimeEnv) -> None:
    marker_path = prepared.root / "requirements_installed.json"
    expected = {
        "requirements": list(prepared.spec.requirements),
        "python": str(prepared.python_path),
    }
    if marker_path.exists():
        try:
            if json.loads(marker_path.read_text(encoding="utf-8")) == expected:
                return
        except json.JSONDecodeError:
            pass
    if not prepared.spec.requirements:
        atomic_write_text(marker_path, json.dumps(expected, indent=2, sort_keys=True) + "\n")
        return
    proc = subprocess.run(
        [str(prepared.python_path), "-m", "pip", "install", *prepared.spec.requirements],
        cwd=str(prepared.root),
        env=_clean_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeEnvironmentError(
            f"failed to install task runtime requirements for {prepared.task_id}: "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    atomic_write_text(marker_path, json.dumps(expected, indent=2, sort_keys=True) + "\n")


def _run_import_preflight(prepared: PreparedRuntimeEnv) -> dict[str, Any]:
    modules = prepared.spec.required_imports
    code = (
        "import importlib, json, sys\n"
        "modules = {}\n"
        f"for name in {list(modules)!r}:\n"
        "    mod = importlib.import_module(name)\n"
        "    modules[name] = {'version': getattr(mod, '__version__', None), 'file': getattr(mod, '__file__', None)}\n"
        "print(json.dumps({'python': sys.executable, 'version_info': list(sys.version_info[:3]), 'modules': modules}, sort_keys=True))\n"
    )
    proc = _run_python(prepared.python_path, code, cwd=tempfile.gettempdir(), env=_clean_env())
    if proc.returncode != 0:
        raise RuntimeEnvironmentError(
            f"task runtime import preflight failed for {prepared.task_id}: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    payload = json.loads(proc.stdout)
    _write_preflight(prepared.root / "import_preflight.json", payload)
    return payload


def _run_public_seed_preflight(*, prepared: PreparedRuntimeEnv, task: Any, repo_root: Path) -> dict[str, Any]:
    candidate_spec = getattr(task.metadata, "candidate_spec", None)
    if candidate_spec is None:
        entry_path = task.public_dir / task.metadata.entrypoint_name
    else:
        entry_path = task.public_dir / candidate_spec.public_entrypoint
    script_path = prepared.root / "public_seed_preflight.py"
    atomic_write_text(
        script_path,
        """
from __future__ import annotations

import json
import os
from pathlib import Path

from agentic_opt.task_registry import get_task


def main() -> int:
    task = get_task(os.environ["AO_TASK_ID"])
    result = task.verify_entry(Path(os.environ["AO_TASK_ENTRY"]))
    payload = {
        "valid": bool(result.get("valid")),
        "status": result.get("status"),
        "feedback": result.get("feedback"),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
""".lstrip(),
    )
    env = _clean_env()
    env.update(prepared.exports())
    env["AO_TASK_ID"] = task.metadata.task_id
    env["AO_TASK_ENTRY"] = str(entry_path)
    env.setdefault("AO_TASKS_ROOTS", str(repo_root / "tasks"))
    env["PYTHONPATH"] = str(repo_root / "src")
    proc = _run_python_script(prepared.python_path, script_path, cwd=tempfile.gettempdir(), env=env, timeout_s=240)
    if proc.returncode != 0:
        raise RuntimeEnvironmentError(
            f"task runtime public-seed preflight failed for {prepared.task_id}: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    payload = json.loads(proc.stdout)
    _write_preflight(prepared.root / "public_seed_preflight.json", payload)
    return payload


def _write_manifest(prepared: PreparedRuntimeEnv) -> None:
    payload = prepared.to_jsonable()
    payload["created_at_unix"] = time.time()
    atomic_write_text(prepared.manifest_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_preflight(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _select_base_python(spec: TaskRuntimeSpec) -> Path:
    override = os.environ.get("AO_TASK_RUNTIME_PYTHON")
    if override:
        candidate = Path(override).expanduser().resolve()
        _validate_base_python(candidate, spec, source="AO_TASK_RUNTIME_PYTHON")
        return candidate
    for candidate in _iter_python_candidates():
        if _base_python_is_usable(candidate, spec):
            return candidate
    raise RuntimeEnvironmentError(
        f"no Python executable satisfies task runtime: python {spec.python}"
    )


def _validate_base_python(candidate: Path, spec: TaskRuntimeSpec, *, source: str) -> None:
    if not _base_python_is_usable(candidate, spec):
        raise RuntimeEnvironmentError(
            f"Python selected by {source} is not usable for this task runtime: {candidate}. "
            f"Required python={spec.python}"
        )


def _base_python_is_usable(candidate: Path, spec: TaskRuntimeSpec) -> bool:
    if not candidate.exists():
        return False
    code = (
        "import sys\n"
        f"constraint = {spec.python!r}\n"
        "version = sys.version_info[:3]\n"
        "def check(v, raw):\n"
        "    parts = [p.strip() for p in raw.split(',') if p.strip()]\n"
        "    for part in parts:\n"
        "        if part.startswith('>='):\n"
        "            if not (v >= tuple(map(int, part[2:].split('.')))): return False\n"
        "        elif part.startswith('>'):\n"
        "            if not (v > tuple(map(int, part[1:].split('.')))): return False\n"
        "        elif part.startswith('<='):\n"
        "            if not (v <= tuple(map(int, part[2:].split('.')))): return False\n"
        "        elif part.startswith('<'):\n"
        "            if not (v < tuple(map(int, part[1:].split('.')))): return False\n"
        "        elif part.startswith('=='):\n"
        "            if not (v == tuple(map(int, part[2:].split('.')))): return False\n"
        "    return True\n"
        "if not check(version, constraint): raise SystemExit(3)\n"
    )
    proc = _run_python(candidate, code, cwd=tempfile.gettempdir(), env=_clean_env(), timeout_s=10)
    return proc.returncode == 0


def _iter_python_candidates() -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    def add(path_like: str | Path | None) -> None:
        if not path_like:
            return
        candidate = Path(path_like).expanduser().resolve()
        if candidate in seen:
            return
        seen.add(candidate)
        candidates.append(candidate)

    add(sys.executable)
    add(shutil.which("python3"))
    add(shutil.which("python"))
    add("/usr/bin/python3")
    add("/opt/homebrew/bin/python3")

    framework_root = Path("/Library/Frameworks/Python.framework/Versions")
    if framework_root.exists():
        version_dirs = sorted(
            [item for item in framework_root.iterdir() if item.is_dir()],
            key=lambda item: tuple(int(part) if part.isdigit() else -1 for part in item.name.split(".")),
            reverse=True,
        )
        for version_dir in version_dirs:
            add(version_dir / "bin" / "python3")
            add(version_dir / "bin" / "python")

    return candidates


def _clean_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    return env


def _run_python(
    python_path: Path,
    code: str,
    *,
    cwd: str | Path,
    env: dict[str, str],
    timeout_s: float = 30,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [str(python_path), "-c", code],
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except OSError as exc:
        raise RuntimeEnvironmentError(f"failed to execute Python runtime {python_path}: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeEnvironmentError(f"timed out probing Python runtime {python_path}") from exc


def _run_python_script(
    python_path: Path,
    script_path: Path,
    *,
    cwd: str | Path,
    env: dict[str, str],
    timeout_s: float = 30,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [str(python_path), str(script_path)],
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except OSError as exc:
        raise RuntimeEnvironmentError(f"failed to execute Python runtime {python_path}: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeEnvironmentError(f"timed out probing Python runtime {python_path}") from exc
