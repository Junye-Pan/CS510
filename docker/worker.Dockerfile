FROM python:3.12-slim

ARG CODEX_VERSION=0.130.0

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/agentic-opt/src \
    AO_TASKS_ROOTS=/opt/agentic-opt/tasks \
    AO_WORKER_RUNTIME_PYTHON=/usr/local/bin/python \
    AO_WORKER_RUNTIME_ROOT=/opt/agentic-opt \
    AO_WORKER_RUNTIME_VENV=/usr/local \
    AO_WORKER_RUNTIME_MANIFEST=/opt/agentic-opt/docker_worker_runtime.json

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        jq \
        nodejs \
        npm \
        procps \
        ripgrep \
        zsh \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/agentic-opt
COPY pyproject.toml README.md ./
COPY docs ./docs
COPY src ./src
COPY tasks ./tasks

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir -e . \
    && npm install -g "@openai/codex@${CODEX_VERSION}" \
    && codex --version \
    && python -m compileall src/agentic_opt

RUN printf '%s\n' \
    '{' \
    '  "provider": "docker_worker",' \
    '  "python": "/usr/local/bin/python",' \
    '  "workspace_root": "/opt/agentic-opt",' \
    '  "tasks_root": "/opt/agentic-opt/tasks"' \
    '}' \
    > /opt/agentic-opt/docker_worker_runtime.json

CMD ["python", "-m", "agentic_opt.adapter.semantic_worker", "--help"]
