from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from agentic_opt.common.atomic import atomic_write_text

from .repository import ControlPlaneRepository


class TelemetryService:
    """Server-owned non-official telemetry layer.

    Telemetry is deliberately separate from Evaluation. It records process and
    training metrics that help debugging, but it never updates official scores.
    """

    def __init__(self, *, repository: ControlPlaneRepository, telemetry_root: Path) -> None:
        self.repository = repository
        self.telemetry_root = telemetry_root.resolve()
        self.telemetry_root.mkdir(parents=True, exist_ok=True)

    def create_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        provider = payload.get("provider") or "local"
        if provider == "mlflow":
            payload = self._create_mlflow_run_payload(payload)
        record = self.repository.create_telemetry_run(payload)
        run_dir = self._run_dir(record["telemetry_id"])
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / "metrics.jsonl"
        params_path = run_dir / "params.json"
        tags_path = run_dir / "tags.json"
        if not metrics_path.exists():
            atomic_write_text(metrics_path, "")
        atomic_write_text(params_path, json.dumps(record.get("params") or {}, indent=2, sort_keys=True) + "\n")
        atomic_write_text(tags_path, json.dumps(record.get("tags") or {}, indent=2, sort_keys=True) + "\n")
        record = self.repository.update_telemetry_run(
            record["telemetry_id"],
            {
                "artifacts": {
                    "metrics_jsonl": str(metrics_path),
                    "params_json": str(params_path),
                    "tags_json": str(tags_path),
                },
                "details": {"telemetry_dir": str(run_dir)},
            },
        )
        self.repository.record_event(
            {
                "experiment_id": record["experiment_id"],
                "assignment_id": record.get("assignment_id"),
                "session_id": record.get("session_id"),
                "event_type": "telemetry.started",
                "summary": f"telemetry run started provider={record['provider']}",
                "payload": {"telemetry_id": record["telemetry_id"], "provider": record["provider"]},
            }
        )
        return record

    def log_metrics(self, telemetry_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        record = self._require_run(telemetry_id)
        step = payload.get("step")
        metrics = {str(key): _as_number(value) for key, value in (payload.get("metrics") or {}).items()}
        timestamp = float(payload.get("timestamp") or time.time())
        metrics_path = Path((record.get("artifacts") or {}).get("metrics_jsonl") or self._run_dir(telemetry_id) / "metrics.jsonl")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"timestamp": timestamp, "step": step, "metrics": metrics}, sort_keys=True) + "\n")
        if record["provider"] == "mlflow":
            self._mlflow_log_metrics(record, metrics=metrics, step=step)
        updated = self.repository.update_telemetry_run(
            telemetry_id,
            {
                "metrics": metrics,
                "artifacts": {"metrics_jsonl": str(metrics_path)},
                "details": {"last_metric_timestamp": timestamp, "last_metric_step": step},
            },
        )
        self.repository.record_event(
            {
                "experiment_id": updated["experiment_id"],
                "assignment_id": updated.get("assignment_id"),
                "session_id": updated.get("session_id"),
                "event_type": "telemetry.metrics",
                "summary": f"telemetry metrics logged: {', '.join(sorted(metrics))}",
                "payload": {"telemetry_id": telemetry_id, "metrics": metrics, "step": step},
            }
        )
        return updated

    def finish_run(self, telemetry_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        record = self._require_run(telemetry_id)
        status = payload.get("status") or "completed"
        if record["provider"] == "mlflow":
            self._mlflow_end_run(record, status=status)
        updated = self.repository.update_telemetry_run(telemetry_id, {"status": status, "details": payload.get("details") or {}})
        self.repository.record_event(
            {
                "experiment_id": updated["experiment_id"],
                "assignment_id": updated.get("assignment_id"),
                "session_id": updated.get("session_id"),
                "event_type": "telemetry.finished",
                "summary": f"telemetry run finished status={status}",
                "payload": {"telemetry_id": telemetry_id, "status": status},
            }
        )
        return updated

    def _create_mlflow_run_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        mlflow = _import_mlflow()
        tracking_uri = payload.get("tracking_uri") or os.environ.get("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        experiment_name = payload.get("experiment_name") or os.environ.get("MLFLOW_EXPERIMENT_NAME")
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        run = mlflow.start_run(run_name=payload.get("run_name"))
        run_id = run.info.run_id
        dashboard_url = _mlflow_dashboard_url(tracking_uri=tracking_uri, run_id=run_id, experiment_id=run.info.experiment_id)
        params = payload.get("params") or {}
        tags = payload.get("tags") or {}
        if params:
            mlflow.log_params(params)
        if tags:
            mlflow.set_tags(tags)
        mlflow.end_run()
        return {
            **payload,
            "provider": "mlflow",
            "external_run_id": run_id,
            "dashboard_url": dashboard_url,
            "details": {
                **(payload.get("details") or {}),
                "mlflow_experiment_id": run.info.experiment_id,
                "mlflow_tracking_uri": tracking_uri,
            },
        }

    def _mlflow_log_metrics(self, record: dict[str, Any], *, metrics: dict[str, float], step: int | None) -> None:
        mlflow = _import_mlflow()
        tracking_uri = (record.get("details") or {}).get("mlflow_tracking_uri")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        with mlflow.start_run(run_id=record.get("external_run_id")):
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)

    def _mlflow_end_run(self, record: dict[str, Any], *, status: str) -> None:
        mlflow = _import_mlflow()
        tracking_uri = (record.get("details") or {}).get("mlflow_tracking_uri")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.start_run(run_id=record.get("external_run_id"))
        mlflow.end_run(status="FINISHED" if status == "completed" else "FAILED")

    def _run_dir(self, telemetry_id: str) -> Path:
        return self.telemetry_root / telemetry_id

    def _require_run(self, telemetry_id: str) -> dict[str, Any]:
        record = self.repository.get_telemetry_run(telemetry_id)
        if record is None:
            raise KeyError(telemetry_id)
        return record


def _as_number(value: Any) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value))


def _import_mlflow():
    try:
        import mlflow  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("MLflow telemetry provider requires the optional 'mlflow' package") from exc
    return mlflow


def _mlflow_dashboard_url(*, tracking_uri: str | None, run_id: str, experiment_id: str) -> str | None:
    if not tracking_uri or not tracking_uri.startswith(("http://", "https://")):
        return None
    return f"{tracking_uri.rstrip('/')}/#/experiments/{experiment_id}/runs/{run_id}"

