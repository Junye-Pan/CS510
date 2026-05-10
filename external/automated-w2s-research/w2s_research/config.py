"""
Configuration for the W2S research system.

Single source of truth — all modules import from here.
All values can be overridden via environment variables.
"""
import os
from pathlib import Path

# =============================================================================
# Paths
# =============================================================================

_REPO_ROOT = str(Path(__file__).parent.parent)
WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", _REPO_ROOT)

# =============================================================================
# Dataset & Models
# =============================================================================

DATASET_NAME = os.getenv("DATASET_NAME", "chat")
DATA_DIR = os.getenv("DATA_DIR", f"{WORKSPACE_DIR}/data/{DATASET_NAME}")
WEAK_MODEL = os.getenv("WEAK_MODEL", "Qwen/Qwen1.5-0.5B-Chat")
STRONG_MODEL = os.getenv("STRONG_MODEL", "Qwen/Qwen3-4B-Base")
SEEDS = [42, 43, 44, 45, 46]

# =============================================================================
# S3 / AWS (required for RunPod mode, unused in local mode)
# =============================================================================

S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "")
S3_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_IDEAS_PREFIX = os.getenv("S3_IDEAS_PREFIX", "ideas/")
S3_RESULTS_PREFIX = os.getenv("S3_RESULTS_PREFIX", "results/")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")

# =============================================================================
# Server (orchestrator API for AAR evaluation)
# =============================================================================

SERVER_URL = os.getenv("ORCHESTRATOR_API_URL", "http://localhost:8000")
AAR_MODE = os.getenv("AAR_MODE", "true").lower() in ("1", "true", "yes")

# =============================================================================
# Agent loop
# =============================================================================

FULL_AUTO_MAX_RUNTIME_SECONDS = int(os.getenv("FULL_AUTO_MAX_RUNTIME_SECONDS", str(5 * 24 * 3600)))
LOGS_DIR = os.getenv("LOGS_DIR", f"{WORKSPACE_DIR}/w2s_research/research_loop/logs")
LOCAL_FINDINGS_DIR = os.getenv("LOCAL_FINDINGS_DIR", f"{WORKSPACE_DIR}/w2s_research/research_loop/shared_findings")
FINDINGS_POLL_INTERVAL = int(os.getenv("FINDINGS_POLL_INTERVAL", "60"))
TARGET_IDEA_FILE = f"{WORKSPACE_DIR}/w2s_research/research_loop/target_idea/idea.json"

# =============================================================================
# RunPod deployment (server-side, for spawning worker pods)
# =============================================================================

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_TEMPLATE_ID = os.getenv("RUNPOD_TEMPLATE_ID", "")
RUNPOD_GPU_TYPE = os.getenv("RUNPOD_GPU_TYPE", "NVIDIA H200")
DEPLOY_TO_RUNPOD = os.getenv("DEPLOY_TO_RUNPOD", "false").lower() == "true"

MAX_CONCURRENT_PODS = int(os.getenv("MAX_CONCURRENT_PODS", "1"))
POD_DEPLOY_MAX_RETRIES = int(os.getenv("POD_DEPLOY_MAX_RETRIES", "100000000"))
POD_DEPLOY_RETRY_DELAY_SECONDS = int(os.getenv("POD_DEPLOY_RETRY_DELAY_SECONDS", "300"))
FULL_AUTO_WORKER_MAX_RUNTIME_SECONDS = int(os.getenv("FULL_AUTO_WORKER_MAX_RUNTIME_SECONDS", str(5 * 24 * 3600)))
FULL_AUTO_POD_TIMEOUT_SECONDS = int(os.getenv("FULL_AUTO_POD_TIMEOUT_SECONDS", str(6 * 24 * 3600)))

# =============================================================================
# Docker local mode (isolated container with GPU, no labels inside)
# =============================================================================

DOCKER_LOCAL_MODE = os.getenv("DOCKER_LOCAL_MODE", "false").lower() == "true"
DOCKER_LOCAL_IMAGE = os.getenv("DOCKER_LOCAL_IMAGE", "w2s-research")
DOCKER_EXECUTABLE = os.getenv("DOCKER_EXECUTABLE", "docker")

# =============================================================================
# Training defaults
# =============================================================================

EPOCHS = 5
NUM_GPUS = 5
