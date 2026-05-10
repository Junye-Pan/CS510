"""
Remote evaluation client for worker pods in AAR mode.

Workers can send predictions to the server to get metrics (including PGR)
without having access to ground truth labels. This enables iterative
improvement even in AAR mode.

Usage:
    from w2s_research.utils.remote_evaluation import evaluate_predictions_remote

    metrics = evaluate_predictions_remote(
        predictions=[0, 1, 0, 1, ...],
        dataset="math-binary",
        weak_model="Qwen/Qwen1.5-0.5B-Chat",
        strong_model="Qwen/Qwen3-4B-Base",
    )
    print(f"PGR: {metrics['pgr']}")
"""
import time
from typing import List, Dict, Any, Optional

import requests

from w2s_research.research_loop.tools.http_utils import get_server_url, DEFAULT_HEADERS

# Retry configuration
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_DELAY = 10  # seconds
DEFAULT_RETRY_BACKOFF = 2  # exponential backoff multiplier


def evaluate_predictions_remote(
    predictions: List[int],
    dataset: str,
    weak_model: Optional[str] = None,
    strong_model: Optional[str] = None,
    split: str = "test",
    server_url: Optional[str] = None,
    timeout: int = 60,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: int = DEFAULT_RETRY_DELAY,
) -> Dict[str, Any]:
    """
    Send predictions to the server for evaluation and get back metrics.
    
    This allows workers in AAR mode to get PGR and other metrics without
    having access to ground truth labels. The server computes metrics
    using its locally stored ground truth.
    
    Includes retry logic with exponential backoff for temporary server downtime.
    
    Args:
        predictions: List of predicted labels (0 or 1)
        dataset: Dataset name (e.g., "math-binary", "chat-0115")
        weak_model: Weak model name (for fixed baseline lookup)
        strong_model: Strong model name (for fixed baseline lookup)
        split: Which split predictions are for ("test" or "train_unlabel")
        server_url: Override server URL (default: from env or DEFAULT_SERVER_URL)
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts (default: 5)
        retry_delay: Initial delay between retries in seconds (default: 10)
        
    Returns:
        Dictionary with metrics:
        - transfer_acc: Transfer accuracy
        - pgr: Performance Gap Recovery (if baselines available)
        - correct: Number correct
        - total: Total predictions
        - fixed_weak_acc: Fixed weak baseline used
        - fixed_strong_acc: Fixed strong baseline used
        
    Raises:
        ConnectionError: If server is unreachable after all retries
        ValueError: If server returns an error (non-retryable)
    """
    if server_url is None:
        server_url = get_server_url()

    endpoint = f"{server_url.rstrip('/')}/api/evaluate-predictions"

    # Prepare request data
    request_data = {
        "predictions": predictions,
        "dataset": dataset,
        "split": split,
    }
    if weak_model:
        request_data["weak_model"] = weak_model
    if strong_model:
        request_data["strong_model"] = strong_model

    headers = {**DEFAULT_HEADERS, "Content-Type": "application/json"}

    # Retry loop with exponential backoff
    last_error = None
    current_delay = retry_delay

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(endpoint, json=request_data, headers=headers, timeout=timeout)
            response.raise_for_status()
            result = response.json()

            if result.get('warnings'):
                for warning in result['warnings']:
                    print(f"[Remote Eval] WARNING: {warning}")
            if result.get('pgr_error'):
                print(f"[Remote Eval] {result['pgr_error']}")

            return result

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else 0
            retryable_codes = [403, 429, 500, 502, 503, 504]
            if status_code in retryable_codes and attempt < max_retries:
                print(f"[Remote Eval] Server error {status_code}, retrying in {current_delay}s...")
                time.sleep(current_delay)
                current_delay = min(current_delay * DEFAULT_RETRY_BACKOFF, 300)
                last_error = e
                continue

            try:
                error_msg = e.response.json().get('error', str(e))
            except Exception:
                error_msg = str(e)
            raise ValueError(f"Server error: {error_msg}")

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt < max_retries:
                print(f"[Remote Eval] Connection failed, retrying in {current_delay}s...")
                time.sleep(current_delay)
                current_delay = min(current_delay * DEFAULT_RETRY_BACKOFF, 300)
                last_error = e
                continue
            raise ConnectionError(f"Cannot connect to {endpoint} after {max_retries + 1} attempts: {e}")

        except Exception as e:
            if attempt < max_retries:
                print(f"[Remote Eval] Error: {e}, retrying in {current_delay}s...")
                time.sleep(current_delay)
                current_delay = min(current_delay * DEFAULT_RETRY_BACKOFF, 300)
                last_error = e
                continue
            raise RuntimeError(f"Failed after {max_retries + 1} attempts: {e}")

    raise RuntimeError(f"Failed after {max_retries + 1} attempts: {last_error}")


def is_server_available(server_url: Optional[str] = None, timeout: int = 5) -> bool:
    """
    Check if the evaluation server is available.

    Args:
        server_url: Override server URL
        timeout: Connection timeout in seconds

    Returns:
        True if server is reachable, False otherwise
    """
    if server_url is None:
        try:
            server_url = get_server_url()
        except ValueError:
            # ORCHESTRATOR_API_URL not set
            return False

    # Try to hit a simple endpoint
    try:
        endpoint = f"{server_url.rstrip('/')}/api/config"
        response = requests.get(endpoint, timeout=timeout)
        return response.status_code == 200
    except:
        return False
