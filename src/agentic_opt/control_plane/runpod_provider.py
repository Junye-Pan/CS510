from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


RUNPOD_API_URL = "https://rest.runpod.io/v1/pods"


class RunPodCapacityError(RuntimeError):
    """Transient RunPod capacity/availability error."""


class RunPodPermanentError(RuntimeError):
    """Permanent RunPod launch/configuration error."""


@dataclass(frozen=True)
class RunPodLaunchResult:
    pod_id: str
    status: str
    payload: dict[str, Any]
    response: dict[str, Any]
    dry_run: bool = False


class RunPodProvider:
    """RunPod JobProvider adapter.

    This adapts the proven automated-w2s pattern: construct a pod payload with a
    docker start command, classify capacity errors as retryable, and keep all
    provider details behind the Job resource.
    """

    def __init__(self, *, api_key: str | None = None, api_url: str = RUNPOD_API_URL) -> None:
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        self.api_url = api_url

    def launch(self, payload: dict[str, Any]) -> RunPodLaunchResult:
        inputs = payload.get("inputs") or {}
        provider_config = payload.get("provider_config") or payload.get("runpod") or {}
        command = payload.get("command") or inputs.get("command")
        if not command:
            raise ValueError("runpod job requires command or inputs.command")
        template_id = payload.get("template_id") or inputs.get("template_id") or provider_config.get("template_id") or os.environ.get("RUNPOD_TEMPLATE_ID")
        if not template_id:
            raise ValueError("runpod job requires template_id, inputs.template_id, provider_config.template_id, or RUNPOD_TEMPLATE_ID")
        runpod_payload = {
            "name": payload.get("name") or provider_config.get("pod_name") or f"agentic-opt-{payload.get('experiment_id', 'job')}",
            "templateId": template_id,
            "gpuCount": int(payload.get("gpu_count") or inputs.get("gpu_count") or provider_config.get("gpu_count") or 1),
            "gpuTypeIds": payload.get("gpu_type_ids") or inputs.get("gpu_type_ids") or provider_config.get("gpu_type_ids") or ["NVIDIA H200"],
            "cloudType": payload.get("cloud_type") or provider_config.get("cloud_type") or "SECURE",
            "globalNetworking": bool(provider_config.get("global_networking", True)),
            "computeType": provider_config.get("compute_type") or "GPU",
            "gpuTypePriority": provider_config.get("gpu_type_priority") or "availability",
            "dataCenterIds": provider_config.get("data_center_ids"),
            "dataCenterPriority": provider_config.get("data_center_priority") or "availability",
            "interruptible": bool(provider_config.get("interruptible", False)),
            "locked": bool(provider_config.get("locked", False)),
            "dockerStartCmd": _docker_start_cmd(command=command, env=payload.get("env") or inputs.get("env") or {}),
        }
        runpod_payload = {key: value for key, value in runpod_payload.items() if value is not None}
        if payload.get("dry_run") or inputs.get("dry_run") or os.environ.get("AO_RUNPOD_DRY_RUN") == "1":
            pod_id = f"dryrun-{payload.get('job_id') or payload.get('experiment_id') or 'pod'}"
            return RunPodLaunchResult(
                pod_id=pod_id,
                status="queued",
                payload=runpod_payload,
                response={"id": pod_id, "dry_run": True, "desiredStatus": "RUNNING"},
                dry_run=True,
            )
        if not self.api_key:
            raise ValueError("RUNPOD_API_KEY is required for non-dry-run RunPod jobs")
        response = self._request("POST", self.api_url, runpod_payload)
        pod_id = str(response.get("id") or response.get("podId") or response.get("pod_id") or "")
        if not pod_id:
            raise RunPodPermanentError(f"RunPod response did not include a pod id: {response}")
        return RunPodLaunchResult(pod_id=pod_id, status=str(response.get("desiredStatus") or "running").lower(), payload=runpod_payload, response=response)

    def status(self, pod_id: str) -> dict[str, Any]:
        if pod_id.startswith("dryrun-"):
            return {"id": pod_id, "status": "DRY_RUN", "dry_run": True}
        if not self.api_key:
            raise ValueError("RUNPOD_API_KEY is required to poll RunPod status")
        return self._request("GET", f"{self.api_url}/{pod_id}")

    def stop(self, pod_id: str) -> dict[str, Any]:
        if pod_id.startswith("dryrun-"):
            return {"id": pod_id, "status": "STOPPED", "dry_run": True}
        if not self.api_key:
            raise ValueError("RUNPOD_API_KEY is required to stop RunPod pods")
        return self._request("POST", f"{self.api_url}/{pod_id}/stop", None)

    def delete(self, pod_id: str) -> dict[str, Any]:
        if pod_id.startswith("dryrun-"):
            return {"id": pod_id, "status": "DELETED", "dry_run": True}
        if not self.api_key:
            raise ValueError("RUNPOD_API_KEY is required to delete RunPod pods")
        return self._request("DELETE", f"{self.api_url}/{pod_id}", None)

    def _request(self, method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            url,
            data=data,
            method=method,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=120) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if _is_capacity_error(exc.code, body):
                raise RunPodCapacityError(f"RunPod capacity unavailable (HTTP {exc.code}): {body}") from exc
            raise RunPodPermanentError(f"RunPod request failed (HTTP {exc.code}): {body}") from exc
        return json.loads(raw) if raw else {}


def _docker_start_cmd(*, command: str | list[Any], env: dict[str, Any]) -> list[str]:
    exports = [f"export {key}={shlex.quote(str(value))}" for key, value in env.items()]
    if isinstance(command, str):
        command_text = command
    else:
        command_text = " ".join(shlex.quote(str(item)) for item in command)
    return ["bash", "-lc", " && ".join([*exports, command_text] if exports else [command_text])]


def _is_capacity_error(status_code: int, body: str) -> bool:
    transient_keywords = ("capacity", "unavailable", "no available", "no instances", "insufficient", "try again", "gpu")
    return status_code in (429, 503, 507) or any(keyword in body.lower() for keyword in transient_keywords)

