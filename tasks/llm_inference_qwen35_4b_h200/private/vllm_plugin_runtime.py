from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from .run_artifacts import get_repo_root


PLUGIN_NAME = "agentic_opt_candidate_rmsnorm"
MANIFEST_ENV = "AO_LLM_KERNEL_VLLM_RMSNORM_MANIFEST"
TRACE_ENV = "AO_LLM_KERNEL_VLLM_RMSNORM_TRACE"
EXPECTED_HIDDEN_ENV = "AO_LLM_KERNEL_VLLM_RMSNORM_EXPECTED_HIDDEN"


def prepare_candidate_rmsnorm_plugin(*, run_dir: Path, manifest_path: Path) -> dict[str, Any]:
    plugin_root = run_dir / "vllm_plugin"
    dist_info = plugin_root / "agentic_opt_vllm_kernel_plugin-0.0.0.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(
        "\n".join(
            [
                "Metadata-Version: 2.1",
                "Name: agentic-opt-vllm-kernel-plugin",
                "Version: 0.0.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (dist_info / "entry_points.txt").write_text(
        "\n".join(
            [
                "[vllm.general_plugins]",
                (
                    f"{PLUGIN_NAME} = "
                    "tasks.llm_inference_qwen35_4b_h200.private.vllm_rmsnorm_plugin:"
                    "register_candidate_rmsnorm"
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    trace_path = run_dir / "vllm_rmsnorm_apply_trace.jsonl"
    repo_root = get_repo_root()
    src_root = repo_root / "src"
    _prepend_sys_path(plugin_root)
    _prepend_sys_path(repo_root)
    _prepend_sys_path(src_root)
    _prepend_pythonpath(plugin_root, repo_root, src_root)
    _enable_plugin_name()
    os.environ[MANIFEST_ENV] = str(manifest_path.resolve())
    os.environ[TRACE_ENV] = str(trace_path)
    os.environ[EXPECTED_HIDDEN_ENV] = "2560"

    if "vllm" in sys.modules:
        from .vllm_rmsnorm_plugin import register_candidate_rmsnorm

        register_candidate_rmsnorm()

    return {
        "plugin_name": PLUGIN_NAME,
        "plugin_root": str(plugin_root),
        "manifest_path": str(manifest_path.resolve()),
        "trace_path": str(trace_path),
        "entry_points": str(dist_info / "entry_points.txt"),
    }


def summarize_rmsnorm_trace(trace_path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "trace_path": str(trace_path),
        "trace_exists": trace_path.exists(),
        "plugin_installs": 0,
        "candidate_calls": 0,
        "fallback_calls": 0,
        "classes": {},
        "pids": [],
        "candidate_shapes": [],
        "residual_candidate_calls": 0,
        "fallback_reasons": {},
        "summary_events": [],
    }
    if not trace_path.exists():
        return summary
    pids: set[int] = set()
    shapes: set[tuple[int, ...]] = set()
    classes: dict[str, int] = {}
    fallback_reasons: dict[str, int] = {}
    summary_events: list[dict[str, Any]] = []
    summary_candidate_calls = 0
    summary_fallback_calls = 0
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                summary["fallback_calls"] += 1
                continue
            pid = event.get("pid")
            if isinstance(pid, int):
                pids.add(pid)
            kind = event.get("event")
            if kind == "plugin_installed":
                summary["plugin_installs"] += 1
            elif kind == "candidate_call":
                summary["candidate_calls"] += 1
                cls_name = str(event.get("class") or "unknown")
                classes[cls_name] = classes.get(cls_name, 0) + 1
                shape = event.get("shape")
                if isinstance(shape, list) and all(isinstance(item, int) for item in shape):
                    shapes.add(tuple(shape))
                if event.get("residual"):
                    summary["residual_candidate_calls"] += 1
            elif kind == "fallback":
                summary["fallback_calls"] += 1
                reason = str(event.get("reason") or "unknown")
                fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
            elif kind == "plugin_summary":
                summary_events.append(event)
                summary_candidate_calls = max(summary_candidate_calls, int(event.get("candidate_calls") or 0))
                summary_fallback_calls = max(summary_fallback_calls, int(event.get("fallback_calls") or 0))
                for reason, count in (event.get("fallback_reasons") or {}).items():
                    fallback_reasons[str(reason)] = max(fallback_reasons.get(str(reason), 0), int(count))
    summary["candidate_calls"] = max(int(summary["candidate_calls"]), summary_candidate_calls)
    summary["fallback_calls"] = max(int(summary["fallback_calls"]), summary_fallback_calls)
    summary["classes"] = classes
    summary["pids"] = sorted(pids)
    summary["candidate_shapes"] = [list(shape) for shape in sorted(shapes)]
    summary["fallback_reasons"] = fallback_reasons
    summary["summary_events"] = summary_events
    return summary


def _prepend_sys_path(path: Path) -> None:
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


def _prepend_pythonpath(*paths: Path) -> None:
    existing = [item for item in os.environ.get("PYTHONPATH", "").split(os.pathsep) if item]
    prefix = [str(path) for path in paths]
    deduped: list[str] = []
    for item in prefix + existing:
        if item not in deduped:
            deduped.append(item)
    os.environ["PYTHONPATH"] = os.pathsep.join(deduped)


def _enable_plugin_name() -> None:
    current = os.environ.get("VLLM_PLUGINS")
    if current is None or current == "":
        os.environ["VLLM_PLUGINS"] = PLUGIN_NAME
        return
    names = [name for name in current.split(",") if name]
    if PLUGIN_NAME not in names:
        names.append(PLUGIN_NAME)
    os.environ["VLLM_PLUGINS"] = ",".join(names)
