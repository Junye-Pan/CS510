"""
Server API Tools - MCP tools for interacting with the orchestrator server.

Provides tools for:
- Evaluation: Get PGR for predictions (ground truth held server-side)
- Knowledge Sharing: Share and query findings from other runs
- Info: Get leaderboard and ideas list
"""

import json
import os
from typing import Any, Dict, List

from claude_agent_sdk import tool, create_sdk_mcp_server

from .http_utils import get_server_url, async_http_post, async_http_get


@tool(
    "evaluate_predictions",
    "Evaluate predictions and get PGR score. Ground truth is held server-side. "
    "Use this after running experiments to get your transfer accuracy and PGR.",
    {
        "type": "object",
        "properties": {
            "predictions": {"type": "array", "items": {"type": "integer"}},
            "dataset": {"type": "string"},
            "weak_model": {"type": "string"},
            "strong_model": {"type": "string"},
        },
        "required": ["predictions", "dataset", "weak_model", "strong_model"],
    },
)
async def evaluate_predictions(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate predictions against ground truth held server-side.

    Args:
        args: Dict with keys:
            - predictions: List of predictions to evaluate
            - dataset: Dataset name
            - weak_model: Weak model identifier
            - strong_model: Strong model identifier

    Returns:
        MCP-formatted response with metrics:
        {
            "transfer_acc": float,  # Strong model trained on weak labels
            "pgr": float,  # Performance Gap Recovery
            "correct": int,  # Number of correct predictions
            "total": int,  # Total number of predictions
            "fixed_weak_acc": float,  # Weak model baseline accuracy
            "fixed_strong_acc": float,  # Strong model ceiling accuracy
        }
    """
    try:
        # Unpack args
        predictions = args.get("predictions", [])
        dataset = args.get("dataset", "")
        weak_model = args.get("weak_model", "")
        strong_model = args.get("strong_model", "")

        server_url = get_server_url()

        result = await async_http_post(
            f"{server_url}/api/evaluate-predictions",
            {
                "predictions": predictions,
                "dataset": dataset,
                "weak_model": weak_model,
                "strong_model": strong_model,
            },
            timeout=120,
        )

        if not isinstance(result, dict):
            error_response = {"success": False, "error": "Invalid server response format"}
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(error_response, indent=2)
                }]
            }

        if "error" in result:
            error_response = {"success": False, "error": result["error"]}
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(error_response, indent=2)
                }]
            }

        required_fields = ["transfer_acc", "pgr"]
        missing = [f for f in required_fields if result.get(f) is None]
        if missing:
            error_response = {
                "success": False,
                "error": f"Server response missing required fields: {missing}",
            }
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(error_response, indent=2)
                }]
            }

        response_data = {
            "success": True,
            "transfer_acc": result.get("transfer_acc"),
            "pgr": result.get("pgr"),
            "correct": result.get("correct"),
            "total": result.get("total"),
            "fixed_weak_acc": result.get("fixed_weak_acc"),
            "fixed_strong_acc": result.get("fixed_strong_acc"),
        }

        return {
            "content": [{
                "type": "text",
                "text": json.dumps(response_data, indent=2)
            }]
        }

    except Exception as e:
        error_response = {"success": False, "error": str(e)}
        return {
            "content": [{
                "type": "text",
                "text": json.dumps(error_response, indent=2)
            }]
        }


async def _auto_upload_snapshot(
    title: str,
    metrics: dict,
    config: dict,
) -> dict:
    """
    Upload a workspace snapshot to S3.

    Returns a dict with snapshot metadata (commit_id, s3_path, s3_key,
    parent_commit_id, sequence_number, files_snapshot) on success,
    or an empty dict on failure.  All fields are forwarded to
    /api/findings/share by the caller.
    """
    from datetime import datetime, timezone
    from pathlib import Path

    idea_uid = os.environ.get("IDEA_UID")
    run_id = os.environ.get("RUN_ID")
    idea_name = os.environ.get("IDEA_NAME", "")
    workspace_dir = os.environ.get("WORKSPACE_DIR", "/workspace")

    if not idea_uid or not run_id:
        print("[auto-snapshot] Skipping: IDEA_UID or RUN_ID not set")
        return {}

    try:
        server_url = get_server_url()

        # Determine sequence number from existing findings that have snapshots
        try:
            response = await async_http_post(
                f"{server_url}/api/snapshots/search",
                {"query": f"idea_uid:{idea_uid} run_id:{run_id}", "limit": 1000},
                timeout=30,
            )
            existing_commits = response.get("commits", [])
            run_commits = [
                c for c in existing_commits
                if c.get("idea_uid") == idea_uid and c.get("run_id") == run_id
            ]
            sequence_number = len(run_commits)
            parent_commit_id = run_commits[-1]["commit_id"] if run_commits else None
        except Exception:
            sequence_number = 0
            parent_commit_id = None

        timestamp = datetime.now(timezone.utc).isoformat()
        from w2s_research.infrastructure.s3_utils import generate_commit_id, upload_commit_to_s3

        commit_id = generate_commit_id(
            experiment_id=0,
            sequence_number=sequence_number,
            message=title,
            timestamp=timestamp,
        )

        from w2s_research.config import S3_BUCKET as bucket, S3_IDEAS_PREFIX as prefix

        s3_key, archive_size, files_list = upload_commit_to_s3(
            idea_uid=idea_uid,
            run_id=run_id,
            commit_id=commit_id,
            workspace_dir=Path(workspace_dir),
            metadata={
                "title": title,
                "idea_name": idea_name,
                "metrics": metrics or {},
                "config": config or {},
                "parent_commit_id": parent_commit_id,
                "sequence_number": sequence_number,
            },
            bucket_name=bucket,
            prefix=prefix,
        )

        s3_path = f"s3://{bucket}/{s3_key}"
        print(f"[auto-snapshot] Created snapshot {commit_id} at {s3_path}")
        return {
            "commit_id": commit_id,
            "s3_path": s3_path,
            "s3_key": s3_key,
            "parent_commit_id": parent_commit_id,
            "sequence_number": sequence_number,
            "files_snapshot": files_list,
        }

    except Exception as e:
        print(f"[auto-snapshot] Failed: {e}")
        return {}


@tool(
    "share_finding",
    """As you explore your research direction, share your empirical findings with other workers.

Parameters:
- summary: all important details for your finding (in markdown format): method explanations, key implementation code snippets, metric results, analysis, config details.
- title: Short descriptive title for the finding
- idea_name: Name of your current research idea (make it descriptive and concise. Do not use "V2/V3" or the exactly same idea name again.)
- metrics: Dict with results like {"pgr": 0.45, "transfer_acc": 0.78, "weak_acc": 0.65, "pgr_se": 0.03, "transfer_acc_se": 0.02, "num_seeds": 5}
- config: Dict with hyperparameters like {"epochs": 3, "lr": 1e-4, ...}
- worked: Boolean - did the approach improve over baseline?
- finding_type: One of "result" (for your new research idea tested across 5 random seeds),
  "hypothesis" (for your untested ideas), "insight" (for your analysis/takeaways),
  "error" (for critical bugs you have found)""",
    {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "title": {"type": "string"},
            "idea_name": {"type": "string"},
            "metrics": {"type": "object"},
            "config": {"type": "object"},
            "worked": {"type": "boolean"},
            "finding_type": {"type": "string"},
        },
        "required": ["summary"],
    },
)
async def share_finding(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Share a finding with other workers and post to forum.
    Results with PGR automatically appear on the leaderboard (no separate publish step needed).

    For finding_type="result" with metrics, automatically:
    1. Creates a workspace snapshot
    2. Posts finding to forum (results with PGR appear on leaderboard)

    Args:
        args: Dict with keys:
            - summary: Full markdown content for the forum post (required)
            - title: Short descriptive title for the finding
            - idea_name: Name of research idea
            - metrics: Dict with results
            - config: Dict with hyperparameters
            - worked: Boolean - did approach improve over baseline?
            - finding_type: Type of finding - "result", "hypothesis", "insight", "error", "observation" (default: "result")

    Returns:
        MCP-formatted response with finding_id, post_id, snapshot_id, etc.
    """
    try:
        # Unpack args
        summary = args.get("summary", "")
        title = args.get("title")
        idea_name = args.get("idea_name")
        metrics = args.get("metrics")
        config = args.get("config")
        worked = args.get("worked")
        finding_type = args.get("finding_type", "result")

        # Parse metrics/config if they're passed as JSON strings

        if isinstance(metrics, str):
            try:
                metrics = json.loads(metrics) if metrics else None
            except json.JSONDecodeError:
                print(f"[share_finding] Warning: Could not parse metrics JSON: {metrics}")
                metrics = None

        if isinstance(config, str):
            try:
                config = json.loads(config) if config else None
            except json.JSONDecodeError:
                print(f"[share_finding] Warning: Could not parse config JSON: {config}")
                config = None

        # Validate that "result" findings have been tested across 5 random seeds
        if finding_type == "result":
            num_seeds = (metrics or {}).get("num_seeds") if isinstance(metrics, dict) else None
            if num_seeds is None or num_seeds < 5:
                return {
                    "content": [{
                        "type": "text",
                        "text": json.dumps({
                            "success": False,
                            "error": (
                                f"Rejected: finding_type='result' requires metrics tested across 5 random seeds, "
                                f"but num_seeds={num_seeds}. Run your experiment with 5 different seeds and include "
                                f"'num_seeds': 5 in metrics before sharing a result."
                            ),
                        }, indent=2)
                    }]
                }

        server_url = get_server_url()

        # Auto-snapshot: when sharing a result with metrics, upload to S3
        snapshot = {}
        if finding_type == "result" and metrics:
            auto_title = title or f"Result: {idea_name}"
            snapshot = await _auto_upload_snapshot(
                title=auto_title,
                metrics=metrics,
                config=config,
            )

        # Build payload — flatten metrics + snapshot into top-level fields
        resolved_metrics = metrics or {}

        payload = {
            "summary": summary,
            "idea_uid": os.environ.get("IDEA_UID"),
            "run_id": os.environ.get("RUN_ID"),
            "dataset": os.environ.get("DATASET_NAME"),
            "weak_model": os.environ.get("WEAK_MODEL"),
            "strong_model": os.environ.get("STRONG_MODEL"),
            "finding_type": finding_type,
            "pgr": resolved_metrics.get("pgr"),
            "pgr_se": resolved_metrics.get("pgr_se"),
            "transfer_acc": resolved_metrics.get("transfer_acc"),
            "transfer_acc_se": resolved_metrics.get("transfer_acc_se"),
            "weak_acc": resolved_metrics.get("weak_acc"),
            "strong_acc": resolved_metrics.get("strong_acc"),
            "num_seeds": resolved_metrics.get("num_seeds"),
        }

        # Optional fields (only include if non-None)
        optional_fields = {
            "title": title,
            "idea_name": idea_name,
            "config": config,
            "worked": worked,
            # Snapshot fields from _auto_upload_snapshot
            "commit_id": snapshot.get("commit_id"),
            "s3_path": snapshot.get("s3_path"),
            "s3_key": snapshot.get("s3_key"),
            "parent_commit_id": snapshot.get("parent_commit_id"),
            "sequence_number": snapshot.get("sequence_number"),
            "files_snapshot": snapshot.get("files_snapshot"),
        }
        for key, value in optional_fields.items():
            if value is not None:
                payload[key] = value

        result = await async_http_post(
            f"{server_url}/api/findings/share",
            payload,
            timeout=30,
        )

        snapshot_id = snapshot.get("commit_id")
        s3_path = snapshot.get("s3_path")

        message = "Finding shared successfully"
        if snapshot_id:
            message += " (auto-snapshot created)"

        # Save finding locally so this agent can search it immediately
        # (without waiting for the next background sync poll)
        finding_dict = result.get("finding")
        if finding_dict and finding_dict.get("id"):
            try:
                from .findings_sync import save_finding_to_dir
                from w2s_research.config import LOCAL_FINDINGS_DIR
                from pathlib import Path

                saved = save_finding_to_dir(finding_dict, Path(LOCAL_FINDINGS_DIR))
                if saved:
                    message += f" (saved locally: {saved.name})"
            except Exception as e:
                print(f"[share_finding] Warning: local save failed: {e}")

        response_data = {
            "success": True,
            "finding_id": result.get("finding_id"),
            "post_id": result.get("post_id"),
            "snapshot_id": snapshot_id,
            "s3_path": s3_path,
            "message": message,
        }

        return {
            "content": [{
                "type": "text",
                "text": json.dumps(response_data, indent=2)
            }]
        }

    except Exception as e:

        return {
            "content": [{
                "type": "text",
                "text": json.dumps({"success": False, "error": str(e)}, indent=2)
            }]
        }


@tool(
    "get_leaderboard",
    "Get the leaderboard of best results ranked by PGR. See what to beat!",
    {},
)
async def get_leaderboard(args: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get leaderboard of best results.

    Returns:
        MCP-formatted response with leaderboard entries
    """
    try:
        server_url = get_server_url()

        result = await async_http_get(
            f"{server_url}/api/leaderboard",
            timeout=30,
        )

        entries = result.get("experiments", result.get("entries", result.get("leaderboard", [])))
        top_pgr = entries[0].get("pgr") if entries else 0.0


        response_data = {
            "success": True,
            "entries": entries,
            "top_pgr": top_pgr,
            "count": len(entries),
        }

        return {
            "content": [{
                "type": "text",
                "text": json.dumps(response_data, indent=2)
            }]
        }

    except Exception as e:

        error_response = {
            "success": False,
            "error": str(e),
            "entries": [],
            "top_pgr": 0.0,
            "count": 0,
        }
        return {
            "content": [{
                "type": "text",
                "text": json.dumps(error_response, indent=2)
            }]
        }


def create_server_api_tools_server():
    """Create MCP server with all server API tools."""
    return create_sdk_mcp_server(
        name="server-api-tools",
        version="1.0.0",
        tools=[
            evaluate_predictions,
            share_finding,
            get_leaderboard,
        ],
    )
