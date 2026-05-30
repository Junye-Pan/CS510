from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from agentic_opt.common.atomic import atomic_write_text
from agentic_opt.common.files import digest_file as _digest_file

from .repository import ControlPlaneRepository


TRACE_EXPORT_SCHEMA_VERSION = "agentic_opt.trace_export.local_jsonl.v1"
TRACE_EXPORT_RECORD_SCHEMA_VERSION = "agentic_opt.trace_export.record.v1"
OTLP_TRACE_EXPORT_SCHEMA_VERSION = "agentic_opt.trace_export.otlp.v1"


class TraceExportService:
    """Server-side trace export provider boundary.

    Trace exports are observability mirrors. The immutable AgentTraceBundle
    artifact remains the source of truth.
    """

    def __init__(self, *, repository: ControlPlaneRepository, export_root: Path) -> None:
        self.repository = repository
        self.export_root = export_root.resolve()
        self.export_root.mkdir(parents=True, exist_ok=True)

    def create_export(self, payload: dict[str, Any]) -> dict[str, Any]:
        provider = str(payload.get("provider") or "local-jsonl")
        redaction_policy = _normalize_redaction_policy(payload.get("redaction_policy"))
        provider_config = _provider_config(provider, payload)
        traces = self._resolve_traces(payload)
        if not traces:
            raise ValueError("trace export requires at least one source trace")
        inferred = _infer_scope(traces)
        request = {
            "provider": provider,
            "trace_ids": [trace["trace_id"] for trace in traces],
            "filters": _trace_filter_request(payload),
            "redaction_policy": redaction_policy,
            "provider_config": _public_provider_config(provider_config),
        }
        export = self.repository.create_trace_export_run(
            {
                "provider": provider,
                "status": "running",
                "experiment_id": payload.get("experiment_id") or inferred.get("experiment_id"),
                "assignment_id": payload.get("assignment_id") or inferred.get("assignment_id"),
                "session_id": payload.get("session_id") or inferred.get("session_id"),
                "task_id": payload.get("task_id") or inferred.get("task_id"),
                "agent_id": payload.get("agent_id") or inferred.get("agent_id"),
                "source_trace_ids": [trace["trace_id"] for trace in traces],
                "redaction_policy": redaction_policy,
                "request": request,
                "metadata": payload.get("metadata") or {},
            }
        )
        try:
            if provider == "local-jsonl":
                result = self._export_local_jsonl(export=export, traces=traces, redaction_policy=redaction_policy)
            elif provider == "otlp":
                result = self._export_otlp(
                    export=export,
                    traces=traces,
                    redaction_policy=redaction_policy,
                    provider_config=provider_config,
                )
            else:
                raise ValueError(f"unsupported trace export provider: {provider}")
            artifact = self.repository.create_artifact(
                {
                    "experiment_id": export.get("experiment_id"),
                    "assignment_id": export.get("assignment_id"),
                    "kind": "trace_export",
                    "uri": Path(result["export_dir"]).as_uri(),
                    "local_path": result["export_dir"],
                    "digest": result["payload_digest"],
                    "metadata": {
                        "trace_export_id": export["trace_export_id"],
                        "provider": provider,
                        "schema_version": result.get("schema_version") or TRACE_EXPORT_SCHEMA_VERSION,
                        "manifest_path": result["manifest_path"],
                        "source_trace_ids": export["source_trace_ids"],
                        "file_digests": result["file_digests"],
                        "record_counts": result["record_counts"],
                        "redaction_summary": result["redaction_summary"],
                    },
                }
            )
            return self.repository.update_trace_export_run(
                export["trace_export_id"],
                {
                    "status": "completed",
                    "destination_uri": result.get("destination_uri") or Path(result["export_dir"]).as_uri(),
                    "local_path": result["export_dir"],
                    "artifact_id": artifact["artifact_id"],
                    "digest": result["payload_digest"],
                    "result": {**result, "artifact_id": artifact["artifact_id"]},
                    "error": {},
                },
            )
        except Exception as exc:
            return self.repository.update_trace_export_run(
                export["trace_export_id"],
                {
                    "status": "failed",
                    "error": {
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                },
            )

    def get_export(self, trace_export_id: str) -> dict[str, Any]:
        export = self.repository.get_trace_export_run(trace_export_id)
        if export is None:
            raise KeyError(trace_export_id)
        return export

    def list_exports(self, **filters: Any) -> list[dict[str, Any]]:
        return self.repository.list_trace_export_runs(**{key: value for key, value in filters.items() if value})

    def _resolve_traces(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        trace_ids = [str(item) for item in (payload.get("trace_ids") or payload.get("source_trace_ids") or [])]
        if trace_ids:
            traces = []
            for trace_id in trace_ids:
                trace = self.repository.get_agent_trace(trace_id)
                if trace is None:
                    raise KeyError(trace_id)
                traces.append(trace)
            return traces

        filters = _trace_filter_request(payload)
        if not any(filters.values()):
            raise ValueError("trace export requires trace_ids or at least one trace filter")
        attempt_id = filters.pop("attempt_id", None)
        traces = self.repository.list_agent_traces(**{key: value for key, value in filters.items() if value})
        if attempt_id:
            traces = [
                trace
                for trace in traces
                if str(attempt_id) in ((trace.get("metadata") or {}).get("observed_ids") or {}).get("attempt_ids", [])
            ]
        return traces

    def _export_local_jsonl(
        self,
        *,
        export: dict[str, Any],
        traces: list[dict[str, Any]],
        redaction_policy: dict[str, Any],
    ) -> dict[str, Any]:
        export_dir = self.export_root / export["trace_export_id"]
        if export_dir.exists():
            shutil.rmtree(export_dir)
        export_dir.mkdir(parents=True, exist_ok=False)

        normalized = _build_normalized_export_payload(traces=traces, redaction_policy=redaction_policy)
        commands = normalized["commands"]
        agent_messages = normalized["agent_messages"]
        events = normalized["events"]
        raw_events = normalized["raw_events"]
        redactor = normalized["redactor"]

        files = {
            "commands": export_dir / "commands.jsonl",
            "agent_messages": export_dir / "agent_messages.jsonl",
            "events": export_dir / "events.jsonl",
            "raw_events": export_dir / "raw_events.jsonl",
        }
        _write_jsonl(files["commands"], commands)
        _write_jsonl(files["agent_messages"], agent_messages)
        _write_jsonl(files["events"], events)
        _write_jsonl(files["raw_events"], raw_events)
        file_digests = {name: _digest_file(path) for name, path in files.items()}
        payload_digest = _digest_named_files(files)
        record_counts = {
            "traces": len(traces),
            "commands": len(commands),
            "agent_messages": len(agent_messages),
            "events": len(events),
            "raw_events": len(raw_events),
        }
        manifest = {
            "schema_version": TRACE_EXPORT_SCHEMA_VERSION,
            "provider": "local-jsonl",
            "source_trace_ids": [trace["trace_id"] for trace in traces],
            "source_traces": [_trace_manifest_ref(trace) for trace in traces],
            "files": {name: path.name for name, path in files.items()},
            "file_digests": file_digests,
            "payload_digest": payload_digest,
            "record_counts": record_counts,
            "redaction_policy": redaction_policy,
            "redaction_summary": redactor.summary(),
            "source_of_truth": "control-plane cp_agent_traces plus immutable agent_trace_bundle artifacts",
        }
        manifest_path = export_dir / "manifest.json"
        atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return {
            "provider": "local-jsonl",
            "schema_version": TRACE_EXPORT_SCHEMA_VERSION,
            "export_dir": str(export_dir),
            "manifest_path": str(manifest_path),
            "files": {name: str(path) for name, path in files.items()},
            "file_digests": file_digests,
            "payload_digest": payload_digest,
            "record_counts": record_counts,
            "redaction_summary": redactor.summary(),
        }

    def _export_otlp(
        self,
        *,
        export: dict[str, Any],
        traces: list[dict[str, Any]],
        redaction_policy: dict[str, Any],
        provider_config: dict[str, Any],
    ) -> dict[str, Any]:
        endpoint = str(provider_config.get("endpoint") or "")
        if not endpoint and not provider_config.get("dry_run"):
            raise ValueError("otlp trace export requires endpoint, otlp_endpoint, or OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")

        export_dir = self.export_root / export["trace_export_id"]
        if export_dir.exists():
            shutil.rmtree(export_dir)
        export_dir.mkdir(parents=True, exist_ok=False)

        normalized = _build_normalized_export_payload(traces=traces, redaction_policy=redaction_policy)
        otlp_payload = _otlp_payload(
            export=export,
            traces=traces,
            normalized=normalized,
            provider_config=provider_config,
        )
        payload_path = export_dir / "otlp_payload.json"
        atomic_write_text(payload_path, json.dumps(otlp_payload, indent=2, sort_keys=True) + "\n")

        http_result = {"dry_run": True, "status": "skipped"}
        if not provider_config.get("dry_run"):
            http_result = _post_otlp_json(
                endpoint=endpoint,
                payload=otlp_payload,
                headers=provider_config.get("headers") or {},
                timeout_s=float(provider_config.get("timeout_s") or 10.0),
            )

        files = {"otlp_payload": payload_path}
        file_digests = {name: _digest_file(path) for name, path in files.items()}
        payload_digest = _digest_named_files(files)
        record_counts = {
            "traces": len(traces),
            "commands": len(normalized["commands"]),
            "agent_messages": len(normalized["agent_messages"]),
            "events": len(normalized["events"]),
            "raw_events": len(normalized["raw_events"]),
            "otlp_spans": _count_otlp_spans(otlp_payload),
        }
        manifest = {
            "schema_version": OTLP_TRACE_EXPORT_SCHEMA_VERSION,
            "provider": "otlp",
            "destination_uri": endpoint or None,
            "source_trace_ids": [trace["trace_id"] for trace in traces],
            "source_traces": [_trace_manifest_ref(trace) for trace in traces],
            "files": {name: path.name for name, path in files.items()},
            "file_digests": file_digests,
            "payload_digest": payload_digest,
            "record_counts": record_counts,
            "redaction_policy": redaction_policy,
            "redaction_summary": normalized["redactor"].summary(),
            "provider_config": _public_provider_config(provider_config),
            "http_result": http_result,
            "source_of_truth": "control-plane cp_agent_traces plus immutable agent_trace_bundle artifacts",
        }
        manifest_path = export_dir / "manifest.json"
        atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return {
            "provider": "otlp",
            "schema_version": OTLP_TRACE_EXPORT_SCHEMA_VERSION,
            "destination_uri": endpoint or None,
            "export_dir": str(export_dir),
            "manifest_path": str(manifest_path),
            "files": {name: str(path) for name, path in files.items()},
            "file_digests": file_digests,
            "payload_digest": payload_digest,
            "record_counts": record_counts,
            "redaction_summary": normalized["redactor"].summary(),
            "provider_config": _public_provider_config(provider_config),
            "http_result": http_result,
        }


class TraceRedactor:
    SENSITIVE_KEY_TOKENS = (
        "access_token",
        "api_key",
        "apikey",
        "authorization",
        "client_secret",
        "cookie",
        "credential",
        "id_token",
        "password",
        "passwd",
        "private_key",
        "proxy_authorization",
        "refresh_token",
        "secret",
        "set_cookie",
        "ssh_key",
        "token",
    )
    SENSITIVE_HEADER_KEYS = {
        "authorization",
        "cookie",
        "proxy-authorization",
        "set-cookie",
        "x-api-key",
        "x-auth-token",
        "x-csrf-token",
    }
    SENSITIVE_QUERY_KEYS = {
        "access_token",
        "api_key",
        "apikey",
        "auth",
        "client_secret",
        "code",
        "credential",
        "id_token",
        "key",
        "password",
        "refresh_token",
        "secret",
        "signature",
        "sig",
        "token",
        "x-amz-credential",
        "x-amz-signature",
    }
    HIDDEN_GRADER_KEYS = {
        "answer_key",
        "expected_answer",
        "gold",
        "grader_secret",
        "ground_truth",
        "hidden_grader",
        "hidden_grader_path",
        "private_dataset",
        "private_dataset_path",
        "private_feedback",
        "reference_solution",
        "solution_key",
    }
    OUTPUT_KEY_TOKENS = ("aggregatedoutput", "output", "output_excerpt", "stderr", "stdout", "transcript")

    def __init__(self, policy: dict[str, Any]) -> None:
        self.policy = policy
        self.enabled = bool(policy.get("enabled", True))
        self.counts: dict[str, int] = {}
        self.max_output_chars = int(policy.get("max_output_chars") or 12_000)
        self.private_path_patterns = [
            re.compile(str(pattern), re.IGNORECASE)
            for pattern in (policy.get("private_path_patterns") or [])
            if str(pattern)
        ]

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "counts": dict(sorted(self.counts.items())),
        }

    def redact(self, value: Any, *, key_path: str = "", parent: dict[str, Any] | None = None) -> Any:
        if not self.enabled:
            return value
        if isinstance(value, dict):
            return {
                str(key): self._redact_value_for_key(str(key), item, key_path=key_path, parent=value)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self.redact(item, key_path=key_path, parent=parent) for item in value]
        if isinstance(value, str):
            return self._redact_string(value)
        return value

    def _redact_value_for_key(self, key: str, value: Any, *, key_path: str, parent: dict[str, Any]) -> Any:
        lowered = _normalize_secret_key(key)
        child_path = f"{key_path}.{key}" if key_path else key
        if self._is_sensitive_header(key=key, key_path=child_path):
            self._count("sensitive_header")
            return "[REDACTED_HEADER]"
        if self._is_sensitive_key(lowered):
            self._count("sensitive_key")
            return "[REDACTED]"
        if self._is_hidden_grader_key(child_path):
            self._count("hidden_grader")
            return "[REDACTED_HIDDEN_GRADER]"
        if lowered in {"destination", "host", "uri", "url"} and str(parent.get("decision") or "").lower() == "denied":
            self._count("denied_destination")
            return "[REDACTED_DESTINATION]"
        redacted = self.redact(value, key_path=child_path, parent=parent)
        if isinstance(redacted, str) and self._is_sensitive_output_key(child_path):
            return self._truncate_output(redacted)
        return redacted

    def _redact_string(self, value: str) -> str:
        redacted = value
        substitutions = [
            (
                "secret_assignment",
                re.compile(r"\b([A-Z0-9_]*(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD)[A-Z0-9_]*)=([^\s'\";]+)", re.IGNORECASE),
                r"\1=[REDACTED]",
            ),
            (
                "authorization_bearer",
                re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE),
                "Bearer [REDACTED]",
            ),
            (
                "authorization_basic",
                re.compile(r"\bBasic\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE),
                "Basic [REDACTED]",
            ),
            (
                "sensitive_header",
                re.compile(r"\b(Authorization|Cookie|Set-Cookie|Proxy-Authorization|X-Api-Key)\s*:\s*[^\r\n]+", re.IGNORECASE),
                r"\1: [REDACTED_HEADER]",
            ),
            (
                "sensitive_url_query",
                re.compile(
                    r"([?&](?:access_token|api[_-]?key|apikey|auth|client_secret|credential|id_token|key|password|refresh_token|secret|signature|sig|token|x-amz-credential|x-amz-signature)=)([^&\s\"'<>]+)",
                    re.IGNORECASE,
                ),
                r"\1[REDACTED]",
            ),
            ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"), "sk-[REDACTED]"),
            ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{12,}\b"), "gh[REDACTED]"),
            (
                "private_path",
                re.compile(
                    r"((?:[A-Za-z]:)?[\\/][^\s\"'<>]*(?:[\\/](?:private|hidden|hidden_grader|grader|private_datasets?|secrets?|\.ssh|\.aws|\.config|\.codex)(?:[\\/][^\s\"'<>]*)?))",
                    re.IGNORECASE,
                ),
                "[REDACTED_PATH]",
            ),
        ]
        for reason, pattern, replacement in substitutions:
            redacted, count = pattern.subn(replacement, redacted)
            if count:
                self._count(reason, count)
        for pattern in self.private_path_patterns:
            redacted, count = pattern.subn("[REDACTED_PATH]", redacted)
            if count:
                self._count("private_path", count)
        redacted = self._redact_url_query(redacted)
        return redacted

    def _redact_url_query(self, value: str) -> str:
        if "://" not in value:
            return value
        parts = urlsplit(value)
        if not parts.scheme or not parts.netloc or not parts.query:
            return value
        pairs = parse_qsl(parts.query, keep_blank_values=True)
        changed = False
        redacted_pairs = []
        for key, item in pairs:
            if _normalize_secret_key(key) in self.SENSITIVE_QUERY_KEYS:
                redacted_pairs.append((key, "[REDACTED]"))
                changed = True
            else:
                redacted_pairs.append((key, item))
        if not changed:
            return value
        self._count("sensitive_url_query")
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(redacted_pairs), parts.fragment))

    def _is_sensitive_key(self, lowered: str) -> bool:
        return any(token in lowered for token in self.SENSITIVE_KEY_TOKENS)

    def _is_sensitive_header(self, *, key: str, key_path: str) -> bool:
        lowered_key = key.lower().replace("_", "-")
        if lowered_key in self.SENSITIVE_HEADER_KEYS:
            return True
        segments = [_normalize_secret_key(segment) for segment in key_path.split(".")]
        if any(segment in {"headers", "request_headers", "response_headers", "http_headers"} for segment in segments):
            return lowered_key in self.SENSITIVE_HEADER_KEYS or self._is_sensitive_key(_normalize_secret_key(key))
        return False

    def _is_hidden_grader_key(self, key_path: str) -> bool:
        segments = {_normalize_secret_key(segment) for segment in key_path.split(".")}
        return bool(segments & self.HIDDEN_GRADER_KEYS)

    def _is_sensitive_output_key(self, key_path: str) -> bool:
        normalized = _normalize_secret_key(key_path.split(".")[-1])
        return any(token == normalized or token in normalized for token in self.OUTPUT_KEY_TOKENS)

    def _truncate_output(self, value: str) -> str:
        if self.max_output_chars <= 0 or len(value) <= self.max_output_chars:
            return value
        self._count("sensitive_output_truncated")
        omitted = len(value) - self.max_output_chars
        return f"{value[: self.max_output_chars]}\n[TRUNCATED_OUTPUT {omitted} chars omitted]"

    def _count(self, reason: str, count: int = 1) -> None:
        self.counts[reason] = self.counts.get(reason, 0) + count


def _build_normalized_export_payload(*, traces: list[dict[str, Any]], redaction_policy: dict[str, Any]) -> dict[str, Any]:
    redactor = TraceRedactor(redaction_policy)
    commands: list[dict[str, Any]] = []
    agent_messages: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    raw_events: list[dict[str, Any]] = []

    for trace in traces:
        trace_root = _trace_root(trace)
        trace_ref = _trace_ref(trace)
        for index, command in enumerate(_read_jsonl(trace_root / "commands.jsonl"), 1):
            record = redactor.redact(_command_export_record(trace_ref=trace_ref, command=command, index=index))
            commands.append(record)
            events.append(
                _event_record(
                    trace_ref=trace_ref,
                    event_type="command.execution",
                    sequence=len(events) + 1,
                    payload=record,
                )
            )
        for index, message in enumerate(_read_jsonl(trace_root / "agent_messages.jsonl"), 1):
            record = redactor.redact(_agent_message_export_record(trace_ref=trace_ref, message=message, index=index))
            agent_messages.append(record)
            events.append(
                _event_record(
                    trace_ref=trace_ref,
                    event_type="agent.message",
                    sequence=len(events) + 1,
                    payload=record,
                )
            )
        for index, raw_event in enumerate(_read_jsonl(trace_root / "events.jsonl"), 1):
            record = redactor.redact(_raw_event_export_record(trace_ref=trace_ref, event=raw_event, line=index))
            raw_events.append(record)
            events.append(
                _event_record(
                    trace_ref=trace_ref,
                    event_type="app_server.raw_event",
                    sequence=len(events) + 1,
                    payload={
                        "line": record["line"],
                        "method": record.get("method"),
                        "raw_event_digest": _digest_json(record.get("event")),
                    },
                )
            )

    return {
        "commands": commands,
        "agent_messages": agent_messages,
        "events": events,
        "raw_events": raw_events,
        "redactor": redactor,
    }


def _provider_config(provider: str, payload: dict[str, Any]) -> dict[str, Any]:
    config = payload.get("provider_config") if isinstance(payload.get("provider_config"), dict) else {}
    if provider != "otlp":
        return dict(config)

    endpoint = (
        payload.get("endpoint")
        or payload.get("otlp_endpoint")
        or config.get("endpoint")
        or config.get("otlp_endpoint")
        or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        or _append_otlp_traces_path(os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"))
    )
    raw_headers: dict[str, Any] = {}
    raw_headers.update(_parse_otlp_headers(os.environ.get("OTEL_EXPORTER_OTLP_HEADERS")))
    raw_headers.update(_parse_otlp_headers(os.environ.get("OTEL_EXPORTER_OTLP_TRACES_HEADERS")))
    if isinstance(config.get("headers"), dict):
        raw_headers.update(config["headers"])
    if isinstance(payload.get("headers"), dict):
        raw_headers.update(payload["headers"])
    timeout_s = payload.get("timeout_s") or payload.get("timeout") or config.get("timeout_s") or config.get("timeout") or 10.0
    return {
        **config,
        "endpoint": _normalize_otlp_endpoint(str(endpoint)) if endpoint else None,
        "headers": {str(key): str(value) for key, value in raw_headers.items() if key and value is not None},
        "timeout_s": float(timeout_s),
        "dry_run": bool(payload.get("dry_run") or config.get("dry_run")),
        "service_name": str(payload.get("service_name") or config.get("service_name") or "agentic-opt"),
    }


def _public_provider_config(config: dict[str, Any]) -> dict[str, Any]:
    public: dict[str, Any] = {}
    for key, value in config.items():
        normalized = _normalize_secret_key(str(key))
        if normalized in {"headers", "authorization", "api_key", "token", "password", "secret"}:
            continue
        public[str(key)] = value
    headers = config.get("headers")
    if isinstance(headers, dict):
        public["headers"] = {
            str(key): ("[REDACTED]" if _normalize_secret_key(str(key)) in TraceRedactor.SENSITIVE_HEADER_KEYS or TraceRedactor({})._is_sensitive_key(_normalize_secret_key(str(key))) else str(value))
            for key, value in headers.items()
        }
    return public


def _parse_otlp_headers(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    result: dict[str, str] = {}
    for item in raw.split(","):
        if not item.strip() or "=" not in item:
            continue
        key, value = item.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def _append_otlp_traces_path(endpoint: str | None) -> str | None:
    if not endpoint:
        return None
    return _normalize_otlp_endpoint(endpoint)


def _normalize_otlp_endpoint(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    parts = urlsplit(endpoint)
    if not parts.scheme or not parts.netloc:
        raise ValueError(f"otlp endpoint must be an absolute HTTP URL: {endpoint}")
    if parts.scheme not in {"http", "https"}:
        raise ValueError(f"otlp endpoint must use http or https: {endpoint}")
    if parts.path in {"", "/"}:
        return endpoint + "/v1/traces"
    return endpoint


def _post_otlp_json(*, endpoint: str, payload: dict[str, Any], headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    request_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        **headers,
    }
    request = Request(endpoint, data=body, method="POST", headers=request_headers)
    try:
        with urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8", errors="replace")
            return {
                "status": "sent",
                "http_status": response.status,
                "response_body_digest": _digest_bytes(raw.encode("utf-8", errors="replace")) if raw else None,
                "response_body_excerpt": raw[:1000] if raw else "",
            }
    except HTTPError as exc:
        try:
            raw = exc.read().decode("utf-8", errors="replace")
        finally:
            exc.close()
        raise RuntimeError(f"otlp export failed: HTTP {exc.code}: {raw[:1000]}") from exc
    except URLError as exc:
        raise RuntimeError(f"otlp export failed: {exc}") from exc


def _otlp_payload(
    *,
    export: dict[str, Any],
    traces: list[dict[str, Any]],
    normalized: dict[str, Any],
    provider_config: dict[str, Any],
) -> dict[str, Any]:
    base_time = _iso_z_to_unix_nanos(str(export.get("created_at") or ""))
    spans: list[dict[str, Any]] = []
    by_trace: dict[str, dict[str, Any]] = {trace["trace_id"]: trace for trace in traces}
    commands_by_trace = _records_by_trace(normalized["commands"])
    messages_by_trace = _records_by_trace(normalized["agent_messages"])
    events_by_trace = _records_by_trace(normalized["events"])

    for trace_id, trace in by_trace.items():
        otel_trace_id = _stable_hex("trace", trace_id, length=32)
        root_span_id = _stable_hex("span", trace_id, "root", length=16)
        trace_events = events_by_trace.get(trace_id, [])
        trace_commands = commands_by_trace.get(trace_id, [])
        trace_messages = messages_by_trace.get(trace_id, [])
        root_span = {
            "traceId": otel_trace_id,
            "spanId": root_span_id,
            "name": f"agentic_opt.trace {trace.get('turn_id') or trace_id}",
            "kind": 1,
            "startTimeUnixNano": str(base_time),
            "endTimeUnixNano": str(
                base_time
                + _max_trace_offset_ms(events=trace_events, commands=trace_commands, messages=trace_messages) * 1_000_000
            ),
            "attributes": _otlp_attributes(
                {
                    "agentic_opt.trace_id": trace_id,
                    "agentic_opt.experiment_id": trace.get("experiment_id"),
                    "agentic_opt.assignment_id": trace.get("assignment_id"),
                    "agentic_opt.session_id": trace.get("session_id"),
                    "agentic_opt.task_id": trace.get("task_id"),
                    "agentic_opt.agent_id": trace.get("agent_id"),
                    "agentic_opt.run_id": trace.get("run_id"),
                    "agentic_opt.turn_id": trace.get("turn_id"),
                    "agentic_opt.worker_backend": trace.get("worker_backend"),
                    "agentic_opt.source_artifact_id": trace.get("artifact_id"),
                    "agentic_opt.event_count": (trace.get("metadata") or {}).get("event_count"),
                    "agentic_opt.command_count": (trace.get("metadata") or {}).get("command_count"),
                    "agentic_opt.agent_message_count": (trace.get("metadata") or {}).get("agent_message_count"),
                }
            ),
            "events": [
                {
                    "name": str(event.get("event_type") or event.get("record_type") or "agentic_opt.event"),
                    "timeUnixNano": str(base_time + _int_or_default(event.get("sequence"), index) * 1_000_000),
                    "attributes": _otlp_attributes(
                        {
                            "agentic_opt.event.sequence": event.get("sequence") or index,
                            "agentic_opt.event.payload_digest": _digest_json(event.get("payload")),
                        }
                    ),
                }
                for index, event in enumerate(trace_events[:500], 1)
            ],
            "status": {"code": 1},
        }
        spans.append(root_span)
        for command in trace_commands:
            spans.append(_otlp_command_span(command=command, otel_trace_id=otel_trace_id, parent_span_id=root_span_id, base_time=base_time))
        for message in trace_messages:
            spans.append(_otlp_agent_message_span(message=message, otel_trace_id=otel_trace_id, parent_span_id=root_span_id, base_time=base_time))

    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": _otlp_attributes(
                        {
                            "service.name": provider_config.get("service_name") or "agentic-opt",
                            "telemetry.sdk.name": "agentic-opt",
                            "agentic_opt.trace_export_id": export.get("trace_export_id"),
                            "agentic_opt.trace_export_provider": "otlp",
                            "agentic_opt.source_trace_count": len(traces),
                        }
                    )
                },
                "scopeSpans": [
                    {
                        "scope": {
                            "name": "agentic_opt.control_plane.trace_exports",
                            "version": OTLP_TRACE_EXPORT_SCHEMA_VERSION,
                        },
                        "spans": spans,
                    }
                ],
            }
        ]
    }


def _records_by_trace(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        trace = record.get("trace") or {}
        trace_id = trace.get("trace_id")
        if trace_id:
            result.setdefault(str(trace_id), []).append(record)
    return result


def _max_trace_offset_ms(*, events: list[dict[str, Any]], commands: list[dict[str, Any]], messages: list[dict[str, Any]]) -> int:
    offsets = [max(1, len(events))]
    offsets.extend(_int_or_default(event.get("sequence"), index) for index, event in enumerate(events, 1))
    offsets.extend(
        _int_or_default(command.get("completed_at_ms") or command.get("started_at_ms") or command.get("sequence"), 1)
        for command in commands
    )
    offsets.extend(
        _int_or_default(message.get("completed_at_ms") or message.get("sequence"), 1)
        for message in messages
    )
    return max(1, max(offsets or [1]))


def _otlp_command_span(*, command: dict[str, Any], otel_trace_id: str, parent_span_id: str, base_time: int) -> dict[str, Any]:
    sequence = _int_or_default(command.get("sequence"), 0)
    start_offset = _int_or_default(command.get("started_at_ms"), sequence)
    end_offset = _int_or_default(command.get("completed_at_ms"), start_offset + 1)
    start = base_time + start_offset * 1_000_000
    end = base_time + end_offset * 1_000_000
    exit_code = command.get("exit_code")
    failed = exit_code not in {None, 0}
    name = "command"
    if command.get("semantic_tool"):
        subcommand = f" {command.get('semantic_subcommand')}" if command.get("semantic_subcommand") else ""
        name = f"semantic {command.get('semantic_tool')}{subcommand}"
    return {
        "traceId": otel_trace_id,
        "spanId": _stable_hex("span", command.get("trace", {}).get("trace_id"), "command", command.get("command_id"), sequence, length=16),
        "parentSpanId": parent_span_id,
        "name": name,
        "kind": 1,
        "startTimeUnixNano": str(start),
        "endTimeUnixNano": str(max(end, start + 1)),
        "attributes": _otlp_attributes(
            {
                "agentic_opt.record_type": "command",
                "agentic_opt.command_id": command.get("command_id"),
                "agentic_opt.command": command.get("command"),
                "agentic_opt.cwd": command.get("cwd"),
                "agentic_opt.source": command.get("source"),
                "agentic_opt.semantic_tool": command.get("semantic_tool"),
                "agentic_opt.semantic_subcommand": command.get("semantic_subcommand"),
                "agentic_opt.duration_ms": command.get("duration_ms"),
                "agentic_opt.exit_code": exit_code,
                "agentic_opt.status": command.get("status"),
                "agentic_opt.output_digest": command.get("output_digest"),
            }
        ),
        "events": [
            {
                "name": "command.output",
                "timeUnixNano": str(max(end, start + 1)),
                "attributes": _otlp_attributes(
                    {
                        "agentic_opt.output_excerpt": command.get("output_excerpt"),
                        "agentic_opt.output_digest": command.get("output_digest"),
                    }
                ),
            }
        ]
        if command.get("output_excerpt") or command.get("output_digest")
        else [],
        "status": {"code": 2 if failed else 1, **({"message": f"exit_code={exit_code}"} if failed else {})},
    }


def _otlp_agent_message_span(*, message: dict[str, Any], otel_trace_id: str, parent_span_id: str, base_time: int) -> dict[str, Any]:
    sequence = _int_or_default(message.get("sequence"), 0)
    event_time = base_time + _int_or_default(message.get("completed_at_ms"), sequence) * 1_000_000
    text = str(message.get("text") or "")
    return {
        "traceId": otel_trace_id,
        "spanId": _stable_hex("span", message.get("trace", {}).get("trace_id"), "agent_message", sequence, length=16),
        "parentSpanId": parent_span_id,
        "name": "agent message",
        "kind": 1,
        "startTimeUnixNano": str(event_time),
        "endTimeUnixNano": str(event_time + 1),
        "attributes": _otlp_attributes(
            {
                "agentic_opt.record_type": "agent_message",
                "agentic_opt.message_digest": _digest_bytes(text.encode("utf-8", errors="replace")) if text else None,
                "agentic_opt.message_excerpt": text[:4000],
            }
        ),
        "status": {"code": 1},
    }


def _otlp_attributes(values: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"key": str(key), "value": _otlp_any_value(value)}
        for key, value in values.items()
        if value is not None
    ]


def _otlp_any_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, (list, tuple)):
        return {"arrayValue": {"values": [_otlp_any_value(item) for item in value]}}
    if isinstance(value, dict):
        return {"stringValue": json.dumps(value, sort_keys=True)}
    return {"stringValue": str(value)}


def _count_otlp_spans(payload: dict[str, Any]) -> int:
    return sum(
        len(scope_spans.get("spans") or [])
        for resource_spans in payload.get("resourceSpans") or []
        for scope_spans in resource_spans.get("scopeSpans") or []
    )


def _stable_hex(*parts: Any, length: int) -> str:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8", errors="replace")).hexdigest()
    value = digest[:length]
    return value if any(char != "0" for char in value) else ("1" + value[1:])


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _iso_z_to_unix_nanos(raw: str) -> int:
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        parsed = datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1_000_000_000)


def _normalize_redaction_policy(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {
            "enabled": True,
            "profile": "default",
            "max_output_chars": 12_000,
            "private_path_patterns": [],
        }
    if not isinstance(raw, dict):
        raise ValueError("redaction_policy must be a JSON object")
    return {
        "enabled": raw.get("enabled", True),
        "profile": raw.get("profile") or "custom",
        "max_output_chars": raw.get("max_output_chars") or 12_000,
        "private_path_patterns": raw.get("private_path_patterns") or [],
        **raw,
    }


def _normalize_secret_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _trace_filter_request(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": payload.get("experiment_id"),
        "assignment_id": payload.get("assignment_id"),
        "session_id": payload.get("session_id"),
        "task_id": payload.get("task_id"),
        "agent_id": payload.get("agent_id"),
        "status": payload.get("status"),
        "attempt_id": payload.get("attempt_id"),
    }


def _infer_scope(traces: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("experiment_id", "assignment_id", "session_id", "task_id", "agent_id"):
        values = {trace.get(key) for trace in traces if trace.get(key)}
        if len(values) == 1:
            result[key] = next(iter(values))
    return result


def _trace_root(trace: dict[str, Any]) -> Path:
    raw = trace.get("trace_root")
    if not raw:
        raise ValueError(f"trace has no trace_root: {trace.get('trace_id')}")
    root = Path(raw)
    if not root.exists():
        raise FileNotFoundError(root)
    return root


def _trace_ref(trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "trace_id": trace["trace_id"],
        "experiment_id": trace.get("experiment_id"),
        "assignment_id": trace.get("assignment_id"),
        "session_id": trace.get("session_id"),
        "task_id": trace.get("task_id"),
        "agent_id": trace.get("agent_id"),
        "run_id": trace.get("run_id"),
        "turn_id": trace.get("turn_id"),
        "worker_backend": trace.get("worker_backend"),
        "source_artifact_id": trace.get("artifact_id"),
    }


def _trace_manifest_ref(trace: dict[str, Any]) -> dict[str, Any]:
    return {
        **_trace_ref(trace),
        "status": trace.get("status"),
        "trace_root": trace.get("trace_root"),
        "metadata": {
            "schema_version": (trace.get("metadata") or {}).get("schema_version"),
            "event_count": (trace.get("metadata") or {}).get("event_count"),
            "command_count": (trace.get("metadata") or {}).get("command_count"),
            "agent_message_count": (trace.get("metadata") or {}).get("agent_message_count"),
        },
    }


def _command_export_record(*, trace_ref: dict[str, Any], command: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "schema_version": TRACE_EXPORT_RECORD_SCHEMA_VERSION,
        "record_type": "command",
        "trace": trace_ref,
        "sequence": index,
        "command_id": command.get("id"),
        "command": command.get("command"),
        "cwd": command.get("cwd"),
        "source": command.get("source"),
        "semantic_tool": command.get("semantic_tool"),
        "semantic_subcommand": command.get("semantic_subcommand"),
        "started_at_ms": command.get("started_at_ms"),
        "completed_at_ms": command.get("completed_at_ms"),
        "duration_ms": command.get("duration_ms"),
        "exit_code": command.get("exit_code"),
        "status": command.get("status"),
        "output_excerpt": command.get("output_excerpt"),
        "output_digest": command.get("output_digest"),
    }


def _agent_message_export_record(*, trace_ref: dict[str, Any], message: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "schema_version": TRACE_EXPORT_RECORD_SCHEMA_VERSION,
        "record_type": "agent_message",
        "trace": trace_ref,
        "sequence": index,
        "text": message.get("text"),
        "completed_at_ms": message.get("completed_at_ms"),
    }


def _raw_event_export_record(*, trace_ref: dict[str, Any], event: dict[str, Any], line: int) -> dict[str, Any]:
    raw_event = event.get("raw") if set(event) == {"raw"} else event
    return {
        "schema_version": TRACE_EXPORT_RECORD_SCHEMA_VERSION,
        "record_type": "raw_event",
        "trace": trace_ref,
        "line": line,
        "method": raw_event.get("method") if isinstance(raw_event, dict) else None,
        "event": raw_event,
    }


def _event_record(*, trace_ref: dict[str, Any], event_type: str, sequence: int, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": TRACE_EXPORT_RECORD_SCHEMA_VERSION,
        "record_type": "event",
        "event_type": event_type,
        "sequence": sequence,
        "trace": trace_ref,
        "payload": payload,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError:
                decoded = {"raw": line.rstrip("\n")}
            if isinstance(decoded, dict):
                records.append(decoded)
            else:
                records.append({"value": decoded})
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _digest_json(value: Any) -> str:
    return _digest_bytes(json.dumps(value, sort_keys=True).encode("utf-8", errors="replace"))


def _digest_named_files(files: dict[str, Path]) -> str:
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode("utf-8"))
        with files[name].open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _digest_bytes(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"
