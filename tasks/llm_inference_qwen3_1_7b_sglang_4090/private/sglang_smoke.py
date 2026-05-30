from __future__ import annotations

import atexit
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path("/workspace")
MODEL_PATH = WORKSPACE_ROOT / "models" / "Qwen" / "Qwen3-1.7B"
RUNS_ROOT = WORKSPACE_ROOT / "runs"
SERVER_START_TIMEOUT_S = 240.0
REQUEST_TIMEOUT_S = 180.0
PORT_RELEASE_TIMEOUT_S = 20.0
PORT_RANGE = range(30000, 30200)
PORT_LOCK_DIR = RUNS_ROOT / ".qwen3_1_7b_sglang_ports"
_ACTIVE_PORT_LOCKS: dict[int, Path] = {}

REQUIRED_COUNTERS = {
    "rmsnorm.kernel_hit",
    "fused_add_rmsnorm.kernel_hit",
    "swiglu.kernel_hit",
    "attention.decode.kernel_hit",
    "attention.extend.kernel_hit",
    "sampling.kernel_hit",
}
BASELINE_CACHE_VERSION = "qwen3_1_7b_verify_smoke_baseline_v2"
LOGPROB_ABS_TOL = 2.0e-1


@dataclass
class SmokePaths:
    run_dir: Path
    stats_dir: Path
    baseline_server_log: Path
    server_log: Path
    requests_path: Path
    baseline_responses_path: Path
    candidate_responses_path: Path
    equivalence_path: Path


@dataclass(frozen=True)
class BaselineCachePaths:
    cache_dir: Path
    metadata_path: Path
    responses_path: Path
    model_info_path: Path
    server_log_path: Path


def run_sglang_server_smoke(entry_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    if not MODEL_PATH.is_dir():
        return _failed(
            "model_missing",
            f"model path does not exist: {MODEL_PATH}",
            started=started,
        )

    port = _pick_free_port()
    paths = _make_paths()
    cache_paths, cache_metadata = _baseline_cache_paths()
    baseline_cache = {
        "enabled": _baseline_cache_enabled(),
        "hit": False,
        "cache_dir": str(cache_paths.cache_dir),
        "key": cache_metadata["cache_key"],
    }
    baseline_proc: subprocess.Popen | None = None
    candidate_proc: subprocess.Popen | None = None
    requests_summary: list[dict[str, Any]] = []
    try:
        cached_baseline = _load_baseline_cache(cache_paths) if baseline_cache["enabled"] else None
        if cached_baseline is not None:
            baseline_cache["hit"] = True
            baseline_model_info = cached_baseline["baseline_model_info"]
            baseline_responses = cached_baseline["baseline_responses"]
            paths.baseline_responses_path.write_text(json.dumps(baseline_responses, indent=2, sort_keys=True))
            paths.baseline_server_log.write_text(
                json.dumps(
                    {
                        "baseline_cache": "hit",
                        "cache_dir": str(cache_paths.cache_dir),
                        "key": cache_metadata["cache_key"],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
        else:
            baseline_proc = _start_server(entry_path=entry_path, port=port, paths=paths, inject_candidate=False)
            baseline_model_info = _wait_for_server(baseline_proc, port=port, log_path=paths.baseline_server_log)
            baseline_responses = _run_requests(port, _equivalence_requests())
            paths.baseline_responses_path.write_text(json.dumps(baseline_responses, indent=2, sort_keys=True))
            if baseline_cache["enabled"]:
                _save_baseline_cache(
                    cache_paths,
                    metadata=cache_metadata,
                    baseline_responses=baseline_responses,
                    baseline_model_info=baseline_model_info,
                    server_log_path=paths.baseline_server_log,
                )
            _terminate_process_group(baseline_proc)
            baseline_proc = None
            _wait_for_port_release(port)

        port = _pick_free_port(exclude={port})
        candidate_proc = _start_server(entry_path=entry_path, port=port, paths=paths, inject_candidate=True)
        candidate_model_info = _wait_for_server(candidate_proc, port=port, log_path=paths.server_log)

        candidate_responses: dict[str, Any] = {}
        for request in _smoke_requests():
            response = _post_json(port, "/generate", request["payload"], timeout=REQUEST_TIMEOUT_S)
            _validate_generate_response(request["name"], response)
            candidate_responses[request["name"]] = response
            response_items = _response_items(response)
            requests_summary.append(
                {
                    "name": request["name"],
                    "status": "passed",
                    "batch_size": len(response_items),
                    "text_len": sum(len(str(item.get("text", ""))) for item in response_items),
                    "meta_keys": sorted(
                        {
                            key
                            for item in response_items
                            for key in ((item.get("meta_info") or {}).keys())
                        }
                    ),
                }
            )

        paths.requests_path.write_text(json.dumps(requests_summary, indent=2, sort_keys=True))
        paths.candidate_responses_path.write_text(json.dumps(candidate_responses, indent=2, sort_keys=True))
        equivalence = _compare_baseline_candidate(baseline_responses, candidate_responses)
        paths.equivalence_path.write_text(json.dumps(equivalence, indent=2, sort_keys=True))
        if not equivalence["valid"]:
            return {
                "valid": False,
                "status": "failed",
                "reason": "baseline_candidate_mismatch",
                "errors": equivalence["errors"],
                "metrics": _metrics(
                    started=started,
                    port=port,
                    paths=paths,
                    baseline_model_info=baseline_model_info,
                    candidate_model_info=candidate_model_info,
                    requests=requests_summary,
                    equivalence=equivalence,
                    baseline_cache=baseline_cache,
                ),
            }

        stats_payloads = _load_stats(paths.stats_dir)
        merged = _merge_stats(stats_payloads)
        missing = {
            counter: merged["counters"].get(counter, 0)
            for counter in sorted(REQUIRED_COUNTERS)
            if merged["counters"].get(counter, 0) <= 0
        }
        exceptions = {
            key: value
            for key, value in merged["counters"].items()
            if key.endswith(".exception") and value > 0
        }
        if missing or exceptions:
            return {
                "valid": False,
                "status": "failed",
                "reason": "dispatch_stats_invalid",
                "errors": _stats_errors(missing=missing, exceptions=exceptions),
                "metrics": _metrics(
                    started=started,
                    port=port,
                    paths=paths,
                    baseline_model_info=baseline_model_info,
                    candidate_model_info=candidate_model_info,
                    requests=requests_summary,
                    merged_stats=merged,
                    equivalence=equivalence,
                    baseline_cache=baseline_cache,
                ),
            }

        return {
            "valid": True,
            "status": "passed",
            "reason": None,
            "errors": [],
            "metrics": _metrics(
                started=started,
                port=port,
                paths=paths,
                baseline_model_info=baseline_model_info,
                candidate_model_info=candidate_model_info,
                requests=requests_summary,
                merged_stats=merged,
                equivalence=equivalence,
                baseline_cache=baseline_cache,
            ),
        }
    except Exception as exc:
        return _failed(
            "smoke_exception",
            f"{type(exc).__name__}: {exc}",
            started=started,
            port=port,
            paths=paths,
            requests=requests_summary,
        )
    finally:
        if baseline_proc is not None:
            _terminate_process_group(baseline_proc)
        if candidate_proc is not None:
            _terminate_process_group(candidate_proc)
        _wait_for_port_release(port)


def _start_server(*, entry_path: Path, port: int, paths: SmokePaths, inject_candidate: bool) -> subprocess.Popen:
    env = os.environ.copy()
    sitecustomize_dir = Path(__file__).resolve().parent / "sitecustomize"
    python_paths = [
        str(WORKSPACE_ROOT),
        str(WORKSPACE_ROOT / "src"),
    ]
    if inject_candidate:
        python_paths.insert(0, str(sitecustomize_dir))
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    if inject_candidate:
        env["AO_QWEN3_1_7B_ENABLE_SGLANG_INJECTION"] = "1"
        env["AO_QWEN3_1_7B_CANDIDATE_ENTRY"] = str(entry_path.resolve())
        env["AO_QWEN3_1_7B_STATS_DIR"] = str(paths.stats_dir)
    else:
        env.pop("AO_QWEN3_1_7B_ENABLE_SGLANG_INJECTION", None)
        env.pop("AO_QWEN3_1_7B_CANDIDATE_ENTRY", None)
        env.pop("AO_QWEN3_1_7B_STATS_DIR", None)

    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(MODEL_PATH),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--attention-backend",
        "triton",
        "--sampling-backend",
        "pytorch",
        "--context-length",
        "1536",
        "--max-running-requests",
        "32",
        "--mem-fraction-static",
        "0.65",
        "--disable-radix-cache",
        "--disable-cuda-graph",
        "--trust-remote-code",
    ]
    log_path = paths.server_log if inject_candidate else paths.baseline_server_log
    log_handle = log_path.open("w")
    try:
        return subprocess.Popen(
            cmd,
            cwd=str(WORKSPACE_ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    finally:
        log_handle.close()


def _wait_for_server(proc: subprocess.Popen, *, port: int, log_path: Path) -> dict[str, Any]:
    deadline = time.monotonic() + SERVER_START_TIMEOUT_S
    last_error = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"SGLang server exited with code {proc.returncode}: {_tail(log_path)}")
        try:
            return _get_json(port, "/get_model_info", timeout=5.0)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(2.0)
    raise TimeoutError(f"SGLang server did not become ready: {last_error}; log tail: {_tail(log_path)}")


def _smoke_requests() -> list[dict[str, Any]]:
    long_prompt = " ".join(["Qwen3 smoke prefill token sequence."] * 48)
    return [
        {
            "name": "greedy_short_decode",
            "payload": {
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 2,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "logprob_start_len": -1,
                "top_logprobs_num": 3,
            },
        },
        {
            "name": "prefill_heavy",
            "payload": {
                "text": long_prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 1,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "logprob_start_len": -1,
                "top_logprobs_num": 3,
            },
        },
        {
            "name": "topk_topp_sampling",
            "payload": {
                "text": "Write three short color names:",
                "sampling_params": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 32,
                    "max_new_tokens": 2,
                    "ignore_eos": True,
                },
            },
        },
        {
            "name": "logprob_top_logprobs",
            "payload": {
                "text": "A compact logprob smoke prompt",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 1,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "logprob_start_len": -1,
                "top_logprobs_num": 3,
            },
        },
        _official_lite_request(
            name="batched_prefill_b4_p256_o1",
            mode="prefill",
            batch_size=4,
            prompt_tokens=256,
            max_new_tokens=1,
        ),
        _official_lite_request(
            name="batched_prefill_b4_p512_o1",
            mode="prefill",
            batch_size=4,
            prompt_tokens=512,
            max_new_tokens=1,
        ),
        _official_lite_request(
            name="batched_decode_b8_p256_o4",
            mode="decode",
            batch_size=8,
            prompt_tokens=256,
            max_new_tokens=4,
        ),
    ]


def _equivalence_requests() -> list[dict[str, Any]]:
    return [request for request in _smoke_requests() if request["name"] != "topk_topp_sampling"]


def _run_requests(port: int, requests: list[dict[str, Any]]) -> dict[str, Any]:
    responses: dict[str, Any] = {}
    for request in requests:
        response = _post_json(port, "/generate", request["payload"], timeout=REQUEST_TIMEOUT_S)
        _validate_generate_response(request["name"], response)
        responses[request["name"]] = response
    return responses


def _validate_generate_response(name: str, response: Any) -> None:
    items = _response_items(response)
    if not items:
        raise ValueError(f"{name}: /generate response is empty")
    for index, item in enumerate(items):
        prefix = f"{name}[{index}]"
        if "text" not in item:
            raise ValueError(f"{prefix}: /generate response missing text")
        meta_info = item.get("meta_info")
        if not isinstance(meta_info, dict):
            raise ValueError(f"{prefix}: /generate response missing meta_info")
        if int(meta_info.get("completion_tokens", 0)) <= 0:
            raise ValueError(f"{prefix}: expected at least one completion token")
        if name != "topk_topp_sampling":
            output_logprobs = meta_info.get("output_token_logprobs")
            output_top_logprobs = meta_info.get("output_top_logprobs")
            if not isinstance(output_logprobs, list) or not output_logprobs:
                raise ValueError(f"{prefix}: missing output_token_logprobs")
            if not isinstance(output_top_logprobs, list) or not output_top_logprobs:
                raise ValueError(f"{prefix}: missing output_top_logprobs")


def _compare_baseline_candidate(
    baseline_responses: dict[str, Any],
    candidate_responses: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    results: list[dict[str, Any]] = []
    for name, baseline in sorted(baseline_responses.items()):
        candidate = candidate_responses.get(name)
        if candidate is None:
            errors.append(f"{name}: candidate response missing")
            continue

        request_errors = _compare_response_items(name, baseline, candidate)
        candidate_items = _response_items(candidate)

        results.append(
            {
                "name": name,
                "valid": not request_errors,
                "errors": request_errors,
                "batch_size": len(candidate_items),
                "text_len": sum(len(str(item.get("text", ""))) for item in candidate_items),
                "completion_tokens": sum(int((item.get("meta_info") or {}).get("completion_tokens", 0)) for item in candidate_items),
                "output_token_ids": [
                    _extract_token_ids((item.get("meta_info") or {}).get("output_token_logprobs"))
                    for item in candidate_items
                ],
            }
        )
        errors.extend(f"{name}: {error}" for error in request_errors)

    return {
        "valid": not errors,
        "errors": errors,
        "results": results,
        "logprob_abs_tol": LOGPROB_ABS_TOL,
    }


def _official_lite_request(
    *,
    name: str,
    mode: str,
    batch_size: int,
    prompt_tokens: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    return {
        "name": name,
        "payload": {
            "input_ids": _make_input_ids(
                batch_size=batch_size,
                prompt_tokens=prompt_tokens,
                salt=_stable_salt(f"{mode}:{name}"),
            ),
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "logprob_start_len": -1,
            "top_logprobs_num": 3,
            "stream": False,
        },
    }


def _make_input_ids(*, batch_size: int, prompt_tokens: int, salt: int) -> list[int] | list[list[int]]:
    vocab_window = 120_000
    rows: list[list[int]] = []
    for row_index in range(batch_size):
        row_salt = salt + row_index * 9973
        rows.append([1000 + ((row_salt + token_index * 37) % vocab_window) for token_index in range(prompt_tokens)])
    if batch_size == 1:
        return rows[0]
    return rows


def _stable_salt(text: str) -> int:
    value = 0
    for index, char in enumerate(text):
        value = (value + (index + 1) * ord(char)) % 100_000
    return value


def _response_items(response: Any) -> list[dict[str, Any]]:
    if isinstance(response, list):
        if not all(isinstance(item, dict) for item in response):
            raise TypeError("/generate batch response must contain objects")
        return response
    if isinstance(response, dict):
        return [response]
    raise TypeError(f"unexpected /generate response type: {type(response).__name__}")


def _compare_response_items(request_name: str, baseline: Any, candidate: Any) -> list[str]:
    errors: list[str] = []
    baseline_items = _response_items(baseline)
    candidate_items = _response_items(candidate)
    if len(baseline_items) != len(candidate_items):
        return [f"response batch size differs: {len(baseline_items)} != {len(candidate_items)}"]

    for batch_index, (baseline_item, candidate_item) in enumerate(zip(baseline_items, candidate_items)):
        prefix = f"batch[{batch_index}]"
        if baseline_item.get("text") != candidate_item.get("text"):
            errors.append(f"{prefix}: text differs")

        baseline_meta = baseline_item.get("meta_info") or {}
        candidate_meta = candidate_item.get("meta_info") or {}
        for field in ("prompt_tokens", "completion_tokens"):
            if baseline_meta.get(field) != candidate_meta.get(field):
                errors.append(f"{prefix}: {field} differs: {baseline_meta.get(field)} != {candidate_meta.get(field)}")

        baseline_token_logprobs = baseline_meta.get("output_token_logprobs")
        candidate_token_logprobs = candidate_meta.get("output_token_logprobs")
        if not isinstance(baseline_token_logprobs, list) or not isinstance(candidate_token_logprobs, list):
            errors.append(f"{prefix}: output_token_logprobs missing")
        else:
            errors.extend(
                f"{prefix}: {error}"
                for error in _compare_token_logprob_pairs(
                    request_name,
                    "output_token_logprobs",
                    baseline_token_logprobs,
                    candidate_token_logprobs,
                )
            )

        baseline_top_logprobs = baseline_meta.get("output_top_logprobs")
        candidate_top_logprobs = candidate_meta.get("output_top_logprobs")
        if not isinstance(baseline_top_logprobs, list) or not isinstance(candidate_top_logprobs, list):
            errors.append(f"{prefix}: output_top_logprobs missing")
        else:
            errors.extend(
                f"{prefix}: {error}"
                for error in _compare_top_logprob_pairs(request_name, baseline_top_logprobs, candidate_top_logprobs)
            )
    return errors


def _compare_token_logprob_pairs(
    request_name: str,
    field_name: str,
    baseline_pairs: list[Any],
    candidate_pairs: list[Any],
) -> list[str]:
    errors: list[str] = []
    if len(baseline_pairs) != len(candidate_pairs):
        return [f"{field_name} length differs: {len(baseline_pairs)} != {len(candidate_pairs)}"]
    for index, (baseline_pair, candidate_pair) in enumerate(zip(baseline_pairs, candidate_pairs)):
        try:
            base_logprob, base_token_id = _parse_logprob_pair(baseline_pair)
            cand_logprob, cand_token_id = _parse_logprob_pair(candidate_pair)
        except (TypeError, ValueError) as exc:
            errors.append(f"{field_name}[{index}] malformed logprob pair: {exc}")
            continue
        if base_token_id != cand_token_id:
            errors.append(f"{field_name}[{index}] token id differs: {base_token_id} != {cand_token_id}")
        if abs(base_logprob - cand_logprob) > LOGPROB_ABS_TOL:
            errors.append(
                f"{field_name}[{index}] logprob differs for {request_name}: "
                f"{base_logprob} != {cand_logprob}"
            )
    return errors


def _compare_top_logprob_pairs(request_name: str, baseline_top: list[Any], candidate_top: list[Any]) -> list[str]:
    errors: list[str] = []
    if len(baseline_top) != len(candidate_top):
        return [f"output_top_logprobs length differs: {len(baseline_top)} != {len(candidate_top)}"]
    for token_index, (baseline_items, candidate_items) in enumerate(zip(baseline_top, candidate_top)):
        if not isinstance(baseline_items, list) or not isinstance(candidate_items, list):
            errors.append(f"output_top_logprobs[{token_index}] is not a list")
            continue
        if not baseline_items or not candidate_items:
            errors.append(f"output_top_logprobs[{token_index}] is empty")
    return errors


def _parse_logprob_pair(pair: Any) -> tuple[float, int]:
    if not isinstance(pair, list) or len(pair) < 2:
        raise ValueError(f"logprob pair must be [logprob, token_id, ...], got {pair!r}")
    return float(pair[0]), int(pair[1])


def _extract_token_ids(pairs: Any) -> list[int]:
    if not isinstance(pairs, list):
        return []
    token_ids: list[int] = []
    for pair in pairs:
        try:
            _, token_id = _parse_logprob_pair(pair)
        except Exception:
            continue
        token_ids.append(token_id)
    return token_ids


def _get_json(port: int, path: str, *, timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_json(port: int, path: str, payload: dict[str, Any], *, timeout: float) -> Any:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {path}: {body}") from exc


def _load_stats(stats_dir: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for path in sorted(stats_dir.glob("stats_*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        payload["_path"] = str(path)
        payloads.append(payload)
    if not payloads:
        raise RuntimeError(f"no stats_*.json files produced under {stats_dir}")
    return payloads


def _merge_stats(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    counters: dict[str, int] = {}
    fallback_reasons: dict[str, int] = {}
    events: list[dict[str, Any]] = []
    paths: list[str] = []
    for payload in payloads:
        paths.append(str(payload.get("_path", "")))
        for key, value in (payload.get("counters") or {}).items():
            counters[key] = counters.get(key, 0) + int(value)
        for key, value in (payload.get("fallback_reasons") or {}).items():
            fallback_reasons[key] = fallback_reasons.get(key, 0) + int(value)
        events.extend(payload.get("events") or [])
    return {
        "stats_files": paths,
        "counters": dict(sorted(counters.items())),
        "fallback_reasons": dict(sorted(fallback_reasons.items())),
        "event_names": sorted({str(event.get("event")) for event in events if event.get("event")}),
        "event_count": len(events),
    }


def _metrics(
    *,
    started: float,
    port: int,
    paths: SmokePaths,
    baseline_model_info: dict[str, Any] | None = None,
    candidate_model_info: dict[str, Any] | None = None,
    requests: list[dict[str, Any]],
    merged_stats: dict[str, Any] | None = None,
    equivalence: dict[str, Any] | None = None,
    baseline_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "elapsed_s": time.perf_counter() - started,
        "port": port,
        "run_dir": str(paths.run_dir),
        "baseline_server_log": str(paths.baseline_server_log),
        "server_log": str(paths.server_log),
        "stats_dir": str(paths.stats_dir),
        "requests_path": str(paths.requests_path),
        "baseline_responses_path": str(paths.baseline_responses_path),
        "candidate_responses_path": str(paths.candidate_responses_path),
        "equivalence_path": str(paths.equivalence_path),
        "baseline_model_info": baseline_model_info or {},
        "model_info": candidate_model_info or {},
        "requests": requests,
        "equivalence": equivalence or {},
        "stats": merged_stats or {},
        "baseline_cache": baseline_cache or {},
    }


def _stats_errors(*, missing: dict[str, int], exceptions: dict[str, int]) -> list[str]:
    errors: list[str] = []
    for counter in sorted(missing):
        errors.append(f"required dispatch counter has no hits: {counter}")
    for counter, value in sorted(exceptions.items()):
        errors.append(f"candidate exception counter is nonzero: {counter}={value}")
    return errors


def _failed(
    reason: str,
    message: str,
    *,
    started: float,
    port: int | None = None,
    paths: SmokePaths | None = None,
    requests: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {"elapsed_s": time.perf_counter() - started}
    if port is not None:
        metrics["port"] = port
    if paths is not None:
        metrics.update(
            {
                "run_dir": str(paths.run_dir),
                "baseline_server_log": str(paths.baseline_server_log),
                "server_log": str(paths.server_log),
                "stats_dir": str(paths.stats_dir),
                "requests_path": str(paths.requests_path),
                "baseline_responses_path": str(paths.baseline_responses_path),
                "candidate_responses_path": str(paths.candidate_responses_path),
                "equivalence_path": str(paths.equivalence_path),
                "baseline_log_tail": _tail(paths.baseline_server_log),
                "log_tail": _tail(paths.server_log),
            }
        )
    if requests is not None:
        metrics["requests"] = requests
    return {
        "valid": False,
        "status": "failed",
        "reason": reason,
        "errors": [message],
        "metrics": metrics,
    }


def _task_runs_root() -> Path:
    return _env_path(
        "AO_QWEN3_1_7B_RUNS_ROOT",
        "AO_TASK_RUNS_ROOT",
        "AO_EVALUATION_RUNS_ROOT",
        default=RUNS_ROOT,
    )


def _baseline_cache_root() -> Path:
    return _env_path(
        "AO_QWEN3_1_7B_CACHE_ROOT",
        "AO_TASK_CACHE_ROOT",
        default=RUNS_ROOT / "_baseline_cache",
    )


def _port_lock_dir() -> Path:
    return _env_path(
        "AO_QWEN3_1_7B_PORT_LOCK_DIR",
        "AO_TASK_PORT_LOCK_DIR",
        default=PORT_LOCK_DIR,
    )


def _env_path(*names: str, default: Path) -> Path:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return Path(value).expanduser().resolve()
    return default


def _make_paths() -> SmokePaths:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = _task_runs_root() / f"qwen3_1_7b_verify_smoke_{stamp}_{os.getpid()}"
    stats_dir = run_dir / "stats"
    run_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    return SmokePaths(
        run_dir=run_dir,
        stats_dir=stats_dir,
        baseline_server_log=run_dir / "baseline_server.log",
        server_log=run_dir / "server.log",
        requests_path=run_dir / "requests.json",
        baseline_responses_path=run_dir / "baseline_responses.json",
        candidate_responses_path=run_dir / "candidate_responses.json",
        equivalence_path=run_dir / "equivalence.json",
    )


def _baseline_cache_enabled() -> bool:
    return os.environ.get("AO_QWEN3_1_7B_SMOKE_BASELINE_CACHE", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }


def _baseline_cache_paths() -> tuple[BaselineCachePaths, dict[str, Any]]:
    metadata: dict[str, Any] = {
        "cache_version": BASELINE_CACHE_VERSION,
        "model_path": str(MODEL_PATH),
        "python": sys.executable,
        "dtype": "bfloat16",
        "attention_backend": "triton",
        "sampling_backend": "pytorch",
        "context_length": 1536,
        "max_running_requests": 32,
        "mem_fraction_static": "0.65",
        "disable_radix_cache": True,
        "disable_cuda_graph": True,
        "requests": _equivalence_requests(),
    }
    key_material = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    metadata["cache_key"] = hashlib.sha256(key_material).hexdigest()[:24]
    cache_dir = _baseline_cache_root() / "qwen3_1_7b_verify_smoke" / metadata["cache_key"]
    paths = BaselineCachePaths(
        cache_dir=cache_dir,
        metadata_path=cache_dir / "metadata.json",
        responses_path=cache_dir / "baseline_responses.json",
        model_info_path=cache_dir / "baseline_model_info.json",
        server_log_path=cache_dir / "baseline_server.log",
    )
    return paths, metadata


def _load_baseline_cache(paths: BaselineCachePaths) -> dict[str, Any] | None:
    if not paths.responses_path.is_file() or not paths.model_info_path.is_file() or not paths.metadata_path.is_file():
        return None
    try:
        return {
            "metadata": json.loads(paths.metadata_path.read_text()),
            "baseline_responses": json.loads(paths.responses_path.read_text()),
            "baseline_model_info": json.loads(paths.model_info_path.read_text()),
        }
    except Exception:
        return None


def _save_baseline_cache(
    paths: BaselineCachePaths,
    *,
    metadata: dict[str, Any],
    baseline_responses: dict[str, Any],
    baseline_model_info: dict[str, Any],
    server_log_path: Path,
) -> None:
    paths.cache_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(metadata)
    payload["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    paths.metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    paths.responses_path.write_text(json.dumps(baseline_responses, indent=2, sort_keys=True))
    paths.model_info_path.write_text(json.dumps(baseline_model_info, indent=2, sort_keys=True))
    try:
        paths.server_log_path.write_text(server_log_path.read_text(errors="replace"))
    except Exception:
        pass


def _pick_free_port(*, exclude: set[int] | None = None) -> int:
    exclude = exclude or set()
    for port in _candidate_ports(exclude=exclude):
        if port in exclude:
            continue
        if not _reserve_port_lock(port):
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                _release_port_lock(port)
                continue
            return port
    raise RuntimeError(f"no free localhost port found in range {PORT_RANGE.start}..{PORT_RANGE.stop - 1}")


def _candidate_ports(*, exclude: set[int]) -> list[int]:
    ports = [port for port in PORT_RANGE if port not in exclude]
    if not ports:
        return []
    seed_material = os.environ.get("AO_EVALUATION_ID") or f"{os.getpid()}:{time.time_ns()}"
    seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:8], 16)
    start = seed % len(ports)
    return ports[start:] + ports[:start]


def _reserve_port_lock(port: int) -> bool:
    if port in _ACTIVE_PORT_LOCKS:
        return True
    port_lock_dir = _port_lock_dir()
    port_lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = port_lock_dir / f"{port}.lock"
    payload = {
        "pid": os.getpid(),
        "evaluation_id": os.environ.get("AO_EVALUATION_ID"),
        "created_at": time.time(),
    }
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            if _port_lock_stale(lock_path):
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            return False
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, sort_keys=True)
        _ACTIVE_PORT_LOCKS[port] = lock_path
        return True


def _port_lock_stale(lock_path: Path) -> bool:
    try:
        payload = json.loads(lock_path.read_text() or "{}")
    except Exception:
        return True
    pid = payload.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return True
    return not Path(f"/proc/{pid}").exists()


def _release_port_lock(port: int) -> None:
    lock_path = _ACTIVE_PORT_LOCKS.pop(port, None)
    if lock_path is None:
        return
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def _release_all_port_locks() -> None:
    for port in list(_ACTIVE_PORT_LOCKS):
        _release_port_lock(port)


def _terminate_process_group(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=20)


def _wait_for_port_release(port: int) -> None:
    deadline = time.monotonic() + PORT_RELEASE_TIMEOUT_S
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                _release_port_lock(port)
                return
            except OSError:
                time.sleep(0.5)


atexit.register(_release_all_port_locks)


def _tail(path: Path, *, max_chars: int = 4000) -> str:
    try:
        data = path.read_text(errors="replace")
    except OSError:
        return ""
    return data[-max_chars:]
