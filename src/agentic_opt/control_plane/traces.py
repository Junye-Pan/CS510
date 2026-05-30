from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
from pathlib import Path
from typing import Any

from agentic_opt.common.atomic import atomic_write_text
from agentic_opt.common.files import (
    count_files as _count_files,
    digest_directory as _digest_directory,
    size_bytes as _size_bytes,
)
from agentic_opt.common.ids import make_run_id

from .repository import ControlPlaneRepository


SEMANTIC_TOOLS = {
    "ctx",
    "attempt",
    "artifact",
    "eval",
    "finding",
    "notebook",
    "job",
    "env",
    "telemetry",
    "tool",
    "network",
    "trace",
}

ID_PATTERNS = {
    "attempt_ids": re.compile(r"\battempt_[A-Za-z0-9_]*\d[A-Za-z0-9_]*\b"),
    "artifact_ids": re.compile(r"\bartifact_[A-Za-z0-9_]*\d[A-Za-z0-9_]*\b"),
    "evaluation_ids": re.compile(r"\beval_[A-Za-z0-9_]*\d[A-Za-z0-9_]*\b"),
    "job_ids": re.compile(r"\bjob_[A-Za-z0-9_]*\d[A-Za-z0-9_]*\b"),
    "telemetry_ids": re.compile(r"\btelemetry_[A-Za-z0-9_]*\d[A-Za-z0-9_]*\b"),
    "trace_ids": re.compile(r"\btrace_[A-Za-z0-9_]*\d[A-Za-z0-9_]*\b"),
}


class AgentTraceService:
    """Registers coding-agent turn traces as immutable artifacts.

    The trace contents remain file-backed. SQLite stores only a lightweight
    catalog row and enough metadata to discover prior agent behavior.
    """

    def __init__(self, *, repository: ControlPlaneRepository, artifact_root: Path) -> None:
        self.repository = repository
        self.artifact_root = artifact_root.resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)

    def register_trace_directory(self, payload: dict[str, Any]) -> dict[str, Any]:
        trace_dir = Path(payload["trace_dir"]).resolve()
        if not trace_dir.exists():
            raise FileNotFoundError(trace_dir)
        events_path = trace_dir / "events.jsonl"
        if not events_path.exists():
            raise FileNotFoundError(events_path)

        run_id = str(payload.get("run_id") or trace_dir.parent.name)
        turn_id = str(payload.get("turn_id") or trace_dir.name)
        existing = self.repository.get_agent_trace_by_turn(
            session_id=str(payload["session_id"]),
            turn_id=turn_id,
        )
        if existing is not None:
            return existing

        trace_id = payload.get("trace_id") or make_run_id("trace")
        artifact_id = payload.get("artifact_id") or make_run_id("artifact")
        staging_dir = (self.artifact_root.parent / "trace_staging" / trace_id).resolve()
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=False)

        extracted = _extract_trace(events_path)
        stdout_path = trace_dir / "stdout.log"
        if stdout_path.exists():
            shutil.copy2(stdout_path, staging_dir / "stdout.log")
        shutil.copy2(events_path, staging_dir / "events.jsonl")
        _write_jsonl(staging_dir / "commands.jsonl", extracted["commands"])
        _write_jsonl(staging_dir / "agent_messages.jsonl", extracted["agent_messages"])

        metadata = {
            **(payload.get("metadata") or {}),
            "schema_version": 1,
            "source_trace_dir": str(trace_dir),
            "outcome": payload.get("outcome"),
            "event_count": extracted["event_count"],
            "command_count": len(extracted["commands"]),
            "failed_command_count": sum(1 for item in extracted["commands"] if item.get("exit_code") not in {None, 0}),
            "semantic_command_count": sum(1 for item in extracted["commands"] if item.get("semantic_tool")),
            "agent_message_count": len(extracted["agent_messages"]),
            "observed_ids": extracted["observed_ids"],
            "files": {
                "events": "events.jsonl",
                "stdout": "stdout.log" if stdout_path.exists() else None,
                "commands": "commands.jsonl",
                "agent_messages": "agent_messages.jsonl",
            },
        }
        manifest = {
            "trace_id": trace_id,
            "run_id": run_id,
            "turn_id": turn_id,
            "status": payload.get("status") or _status_from_outcome(str(payload.get("outcome") or "")),
            "metadata": metadata,
        }
        atomic_write_text(staging_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        artifact = self._create_trace_artifact(
            artifact_id=artifact_id,
            staging_dir=staging_dir,
            source_trace_dir=trace_dir,
            payload=payload,
            trace_id=trace_id,
            metadata=metadata,
        )
        try:
            shutil.rmtree(staging_dir)
        except OSError:
            pass

        return self.repository.create_agent_trace(
            {
                "trace_id": trace_id,
                "experiment_id": payload.get("experiment_id"),
                "assignment_id": payload.get("assignment_id"),
                "session_id": payload["session_id"],
                "task_id": payload.get("task_id"),
                "agent_id": payload.get("agent_id"),
                "run_id": run_id,
                "turn_id": turn_id,
                "worker_backend": payload.get("worker_backend"),
                "status": manifest["status"],
                "artifact_id": artifact["artifact_id"],
                "trace_root": artifact.get("local_path"),
                "metadata": metadata,
            }
        )

    def list_traces(self, **filters: Any) -> list[dict[str, Any]]:
        attempt_id = filters.pop("attempt_id", None)
        traces = self.repository.list_agent_traces(**{key: value for key, value in filters.items() if value})
        if attempt_id:
            traces = [
                trace
                for trace in traces
                if attempt_id in ((trace.get("metadata") or {}).get("observed_ids") or {}).get("attempt_ids", [])
            ]
        return traces

    def trace_with_manifest(self, trace_id: str) -> dict[str, Any]:
        trace = self.repository.get_agent_trace(trace_id)
        if trace is None:
            raise KeyError(trace_id)
        return {"trace": trace, "manifest": self._read_manifest(trace)}

    def trace_commands(
        self,
        trace_id: str,
        *,
        failed_only: bool = False,
        semantic_only: bool = False,
    ) -> dict[str, Any]:
        trace = self.repository.get_agent_trace(trace_id)
        if trace is None:
            raise KeyError(trace_id)
        commands = self._read_jsonl(trace, "commands.jsonl")
        if failed_only:
            commands = [item for item in commands if item.get("exit_code") not in {None, 0}]
        if semantic_only:
            commands = [item for item in commands if item.get("semantic_tool")]
        return {"trace": trace, "commands": commands}

    def trace_events(self, trace_id: str, *, query: str | None = None, limit: int = 200) -> dict[str, Any]:
        trace = self.repository.get_agent_trace(trace_id)
        if trace is None:
            raise KeyError(trace_id)
        events_path = Path(trace["trace_root"]) / "events.jsonl"
        events: list[dict[str, Any]] = []
        needle = (query or "").lower()
        with events_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if needle and needle not in line.lower():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    event = {"raw": line.rstrip("\n")}
                events.append({"line": line_number, "event": event})
                if len(events) >= limit:
                    break
        return {"trace": trace, "events": events}

    def search_traces(self, *, query: str, **filters: Any) -> dict[str, Any]:
        needle = query.lower().strip()
        if not needle:
            raise ValueError("query is required")
        matches: list[dict[str, Any]] = []
        for trace in self.list_traces(**filters):
            command_hits = []
            for command in self._read_jsonl(trace, "commands.jsonl"):
                searchable = json.dumps(command, sort_keys=True).lower()
                if needle in searchable:
                    command_hits.append(command)
            message_hits = []
            for message in self._read_jsonl(trace, "agent_messages.jsonl"):
                searchable = json.dumps(message, sort_keys=True).lower()
                if needle in searchable:
                    message_hits.append(message)
            metadata_hit = needle in json.dumps(trace.get("metadata") or {}, sort_keys=True).lower()
            if metadata_hit or command_hits or message_hits:
                matches.append(
                    {
                        "trace": trace,
                        "command_hits": command_hits[:10],
                        "agent_message_hits": message_hits[:5],
                        "metadata_hit": metadata_hit,
                    }
                )
        return {"query": query, "matches": matches}

    def _create_trace_artifact(
        self,
        *,
        artifact_id: str,
        staging_dir: Path,
        source_trace_dir: Path,
        payload: dict[str, Any],
        trace_id: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        artifact_dir = self.artifact_root / artifact_id
        if artifact_dir.exists():
            raise FileExistsError(artifact_dir)
        content_path = artifact_dir / "content"
        manifest_path = artifact_dir / "manifest.json"
        artifact_dir.mkdir(parents=True, exist_ok=False)
        shutil.copytree(staging_dir, content_path)
        digest = _digest_directory(content_path)
        artifact_manifest = {
            "artifact_id": artifact_id,
            "kind": "agent_trace_bundle",
            "source_path": str(source_trace_dir),
            "content_path": str(content_path),
            "uri": content_path.as_uri(),
            "storage_provider": "local",
            "digest": digest,
            "size_bytes": _size_bytes(content_path),
            "file_count": _count_files(content_path),
            "metadata": {"trace_id": trace_id, **metadata},
        }
        atomic_write_text(manifest_path, json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n")
        return self.repository.create_artifact(
            {
                "artifact_id": artifact_id,
                "experiment_id": payload.get("experiment_id"),
                "assignment_id": payload.get("assignment_id"),
                "kind": "agent_trace_bundle",
                "uri": content_path.as_uri(),
                "local_path": str(content_path),
                "digest": digest,
                "metadata": {
                "trace_id": trace_id,
                "manifest_path": str(manifest_path),
                "source_path": str(source_trace_dir),
                "storage_provider": "local",
                    "size_bytes": artifact_manifest["size_bytes"],
                    "file_count": artifact_manifest["file_count"],
                    **metadata,
                },
            }
        )

    def _read_manifest(self, trace: dict[str, Any]) -> dict[str, Any]:
        path = Path(trace["trace_root"]) / "manifest.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _read_jsonl(self, trace: dict[str, Any], name: str) -> list[dict[str, Any]]:
        path = Path(trace["trace_root"]) / name
        if not path.exists():
            return []
        items: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    items.append({"raw": line.rstrip("\n")})
        return items


def _extract_trace(events_path: Path) -> dict[str, Any]:
    commands_by_id: dict[str, dict[str, Any]] = {}
    agent_messages: list[dict[str, Any]] = []
    streaming_message_chunks: list[str] = []
    observed_text_parts: list[str] = []
    event_count = 0
    with events_path.open(encoding="utf-8") as handle:
        for event_count, line in enumerate(handle, 1):
            observed_text_parts.append(line)
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            method = event.get("method")
            params = event.get("params") or {}
            item = params.get("item") or {}
            if method == "item/started" and item.get("command"):
                command_id = str(item.get("id") or params.get("itemId") or event_count)
                commands_by_id.setdefault(command_id, {})
                commands_by_id[command_id].update(_command_base(item))
                commands_by_id[command_id]["started_at_ms"] = params.get("startedAtMs")
            elif method == "item/completed" and item.get("command"):
                command_id = str(item.get("id") or params.get("itemId") or event_count)
                commands_by_id.setdefault(command_id, {})
                commands_by_id[command_id].update(_command_base(item))
                commands_by_id[command_id]["completed_at_ms"] = params.get("completedAtMs")
                commands_by_id[command_id]["duration_ms"] = item.get("durationMs")
                commands_by_id[command_id]["exit_code"] = item.get("exitCode")
                commands_by_id[command_id]["status"] = item.get("status")
                _attach_output(commands_by_id[command_id], item.get("aggregatedOutput"))
            elif method in {"item/commandExecution/outputDelta", "command/exec/outputDelta"}:
                command_id = str(params.get("itemId") or "")
                if command_id:
                    commands_by_id.setdefault(command_id, {})
                    previous = commands_by_id[command_id].get("_output") or ""
                    commands_by_id[command_id]["_output"] = previous + str(params.get("delta") or "")
            elif method in {"item/agentMessage/delta", "agent_message.delta", "agent_message.chunk"}:
                delta = params.get("delta") or params.get("text")
                if isinstance(delta, str):
                    streaming_message_chunks.append(delta)
            elif method == "item/completed" and (item.get("type") == "agentMessage" or item.get("text")):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    agent_messages.append(
                        {
                            "text": text,
                            "completed_at_ms": params.get("completedAtMs"),
                        }
                    )
    if streaming_message_chunks and not agent_messages:
        text = "".join(streaming_message_chunks).strip()
        if text:
            agent_messages.append({"text": text, "completed_at_ms": None})

    commands: list[dict[str, Any]] = []
    for command_id, command in commands_by_id.items():
        if "_output" in command and "output_digest" not in command:
            _attach_output(command, command.get("_output"))
        command.pop("_output", None)
        command["id"] = command_id
        semantic = _semantic_command(command.get("command") or "")
        command.update(semantic)
        commands.append(command)
    commands.sort(key=lambda item: item.get("started_at_ms") or item.get("completed_at_ms") or 0)

    observed_ids = {
        key: sorted(set(pattern.findall("\n".join(observed_text_parts))))
        for key, pattern in ID_PATTERNS.items()
    }
    return {
        "event_count": event_count,
        "commands": commands,
        "agent_messages": agent_messages,
        "observed_ids": observed_ids,
    }


def _command_base(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "command": item.get("command"),
        "cwd": item.get("cwd"),
        "process_id": item.get("processId"),
        "source": item.get("source"),
    }


def _attach_output(command: dict[str, Any], output: Any) -> None:
    if not isinstance(output, str):
        return
    command["output_excerpt"] = output[:2000]
    command["output_digest"] = _digest_bytes(output.encode("utf-8", errors="replace"))


def _semantic_command(raw_command: str) -> dict[str, Any]:
    script = raw_command
    try:
        parts = shlex.split(raw_command)
    except ValueError:
        parts = raw_command.split()
    if "-lc" in parts:
        index = parts.index("-lc")
        if index + 1 < len(parts):
            script = parts[index + 1]
    try:
        script_parts = shlex.split(script)
    except ValueError:
        script_parts = script.split()
    if not script_parts:
        return {"semantic_tool": None, "semantic_subcommand": None}
    tool = Path(script_parts[0]).name
    if tool not in SEMANTIC_TOOLS:
        return {"semantic_tool": None, "semantic_subcommand": None}
    return {
        "semantic_tool": tool,
        "semantic_subcommand": script_parts[1] if len(script_parts) > 1 else None,
    }


def _status_from_outcome(outcome: str) -> str:
    return "completed" if outcome in {"completed", "success"} else "partial"


def _write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, sort_keys=True) + "\n")


def _digest_bytes(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"
