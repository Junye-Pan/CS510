from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_opt.common.runtime_env import TaskRuntimeSpec


DEFAULT_ENTRYPOINT_NAME = "initial.py"


def _validate_relative_path(raw: str, *, field_name: str) -> None:
    path = Path(raw)
    if path.is_absolute():
        raise ValueError(f"{field_name} must be relative: {raw!r}")
    if not raw or any(part in ("", ".", "..") for part in path.parts):
        if raw != ".":
            raise ValueError(f"{field_name} must be a clean relative path: {raw!r}")


@dataclass(frozen=True)
class CandidateSpec:
    """Task-defined candidate package shape.

    `candidate_root=None` means the candidate root is the parent directory of
    the submitted entrypoint. This preserves the default single-file task shape.
    When `candidate_root` is set, it is interpreted relative to the workspace
    root, and snapshots archive that directory with `entrypoint_name` inside it.
    """

    entrypoint_name: str = DEFAULT_ENTRYPOINT_NAME
    candidate_root: str | None = None
    public_seed_root: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        _validate_relative_path(self.entrypoint_name, field_name="entrypoint_name")
        if self.candidate_root is not None:
            _validate_relative_path(self.candidate_root, field_name="candidate_root")
        if self.public_seed_root is not None:
            _validate_relative_path(self.public_seed_root, field_name="public_seed_root")

    @property
    def workspace_entrypoint(self) -> Path:
        if self.candidate_root is None:
            return Path(self.entrypoint_name)
        return Path(self.candidate_root) / self.entrypoint_name

    @property
    def workspace_candidate_root(self) -> Path | None:
        if self.candidate_root is None:
            return None
        return Path(self.candidate_root)

    @property
    def public_entrypoint(self) -> Path:
        if self.public_seed_root is not None:
            return Path(self.public_seed_root) / self.entrypoint_name
        if self.candidate_root is not None:
            return Path(self.candidate_root) / self.entrypoint_name
        return Path(self.entrypoint_name)


@dataclass(frozen=True)
class TaskMetadata:
    task_id: str
    title: str
    entrypoint_name: str = DEFAULT_ENTRYPOINT_NAME
    candidate_spec: CandidateSpec | None = None


class TaskProtocol:
    metadata: TaskMetadata
    runtime_spec: TaskRuntimeSpec

    @property
    def public_dir(self) -> Path:
        raise NotImplementedError

    def verify_entry(self, entry_path: Path) -> dict[str, Any]:
        raise NotImplementedError

    def probe_entry(self, entry_path: Path, *, kind: str) -> dict[str, Any]:
        raise NotImplementedError

    def evaluate_entry(self, entry_path: Path) -> dict[str, Any]:
        raise NotImplementedError


def candidate_spec_for(task: Any) -> CandidateSpec:
    metadata = getattr(task, "metadata")
    spec = getattr(metadata, "candidate_spec", None)
    if spec is None:
        return CandidateSpec(entrypoint_name=getattr(metadata, "entrypoint_name", DEFAULT_ENTRYPOINT_NAME))
    if not isinstance(spec, CandidateSpec):
        raise TypeError("metadata.candidate_spec must be a CandidateSpec")
    return spec


def candidate_entry_path(*, workspace_root: Path, spec: CandidateSpec) -> Path:
    return workspace_root / spec.workspace_entrypoint


def candidate_snapshot_paths(
    *,
    entry_path: Path,
    workspace_root: Path | None,
    spec: CandidateSpec,
) -> tuple[Path, str, str | None]:
    entry_path = entry_path.resolve()
    if workspace_root is None or spec.workspace_candidate_root is None:
        return entry_path.parent, entry_path.name, None

    root = (workspace_root.resolve() / spec.workspace_candidate_root).resolve()
    if not entry_path.is_relative_to(root):
        raise PermissionError(f"entry path must stay within task candidate root: {root}")
    return root, entry_path.relative_to(root).as_posix(), spec.workspace_candidate_root.as_posix()
