# Automated Weak-to-Strong Research

This project releases a sandbox for automated weak-to-strong research, together with datasets, baselines, and a baseline automated researcher.

**Weak-to-strong generalization** addresses superhuman AI alignment: how do we align AI systems smarter than us when we can't reliably evaluate their outputs? The setup trains a weak model on labeled data, uses it to pseudo-label unlabeled data, then trains a strong model based on those labels. We measure how much the strong model recovers ground truth performance via **Performance Gap Recovery (PGR)**:

```
PGR = (transfer_acc - weak_acc) / (strong_acc - weak_acc)
```

PGR=0 means the strong model is only as good as the weak model. PGR=1 means full recovery.

## Environment Setup

### 1. Install dependencies

```bash
uv sync
```

This installs all dependencies: ML training (PyTorch, Transformers, Unsloth, vLLM), agent SDK (Anthropic, Claude Agent SDK), server (Flask), and cloud (boto3, RunPod).

### 2. Download datasets

We provide three datasets: **chat**, **math**, and **code**.

Each dataset has three files:
- `test.jsonl` — test set for evaluating both in-distribution and out-of-distribution performance
- `train_label.jsonl` — labeled data for training the weak teacher model
- `train_unlabel.jsonl` — unlabeled data for training the strong student

The datasets are distributed as a tar.gz archive. Unpack and prepare:

```bash
tar xzf labeled_data.tar.gz
python scripts/prepare_data.py
```

`prepare_data.py` generates `data/` from `labeled_data/` by stripping labels and metadata. This is what the automated researcher sees — ground truth is held server-side and accessed via the evaluation API.

### 3. Run baselines

Pre-computed results for all baselines are provided as an archive. Unpack first:

```bash
tar xzf cache_results.tar.gz
```

You can also rerun baselines on customized datasets and models:

```bash
# Run baselines across 5 seeds in parallel (auto-distributes across available GPUs)
python run.py --idea vanilla_w2s --seeds 42,43,44,45,46 --data-dir data/chat
```

**Available baselines:**
| Idea | Description |
|------|-------------|
| `vanilla_w2s` | Train strong model on hard weak labels (standard W2S baseline) |
| `train_only_on_confident_labels` | Filter weak labels by confidence before training |
| `critic` | Use strong model critiques to improve weak labels |
| `ue_zeroshot` | Unsupervised elicitation — zero-shot variant |
| `ue_fewshot` | Unsupervised elicitation — few-shot variant with in-context learning (we are not using this version in main experiments since Qwen3-4B-Base barely can do in-context learning on our three testbeds)|

### 4. Create your own idea

```bash
cp -r w2s_research/ideas/TEMPLATE w2s_research/ideas/my_idea
# Edit w2s_research/ideas/my_idea/run.py — implement your approach
python run.py --idea my_idea --seed 42
```

Each idea's `run.py` receives a `RunConfig` and returns metrics. The template loads pre-cached weak model artifacts so you only implement your novel contribution.

see our idea list in `Idea.md`


## Automated Researcher

The automated researcher is a Claude-powered agent that iteratively proposes ideas, implements them, trains models, evaluates via the server API, and shares findings. 

There are three execution modes, from simplest to most isolated:

### 1. Start the dashboard (required for all modes)

```bash
python run.py server --port 8000
```

This starts a Flask server that provides:
- **Experiment management** — queue, monitor, and manage agent runs
- **Evaluation API** — agents submit predictions and get PGR back (ground truth stays server-side)
- **Leaderboard** — compare results across agents and ideas
- **Findings forum** — agents share and read research findings from other workers

Open `http://localhost:8000` to access the web dashboard. From the dashboard, you can create research directions, select execution mode (local / Docker / RunPod), assign GPUs, and launch experiments.

### 2. Execution modes

All three modes require `ANTHROPIC_API_KEY` to be set before starting the server.

#### Mode A: Local (subprocess)

The simplest mode. A single agent runs as a subprocess on the same machine, with direct access to GPUs and the filesystem. Best for quick debugging. Note that in this mode, AAR would be able to find `labeled_data`, so the result might not be legit.

No Docker, no persistent S3 storage, no parallel AARs that share findings and codebases to each other.

#### Mode B: Local Docker

Runs a singel agent inside a Docker container with GPU passthrough. This provides **isolation**: the container only sees `data/` (no labels) and `cache_results/` as read-only mounts, so the agent cannot cheat by reading ground truth. Uses the same Docker image as RunPod mode.

**Setup:**
```bash
# Build the Docker image
./scripts/docker-build-push.sh  # builds with tag 'latest'

# Start server with Docker mode enabled
export ANTHROPIC_API_KEY=...
export DOCKER_LOCAL_MODE=true
export DOCKER_LOCAL_IMAGE=w2s-research  # default image name
python run.py server --port 8000
```

Then launch experiments from the web dashboard, selecting "Docker (local GPUs)" as the execution mode.

#### Mode C: RunPod (cloud)

Deploys parallel agents to RunPod cloud GPUs. The server orchestrates deployment, monitors pod status, and collects results via S3. Supports multiple concurrent pods with automatic retry on capacity errors.

**Setup:**
```bash
export ANTHROPIC_API_KEY=...
export RUNPOD_API_KEY=...
export RUNPOD_TEMPLATE_ID=...  # Create a template on RunPod using the Docker image
export DEPLOY_TO_RUNPOD=true

# S3 for artifact storage (datasets, results, findings)
export S3_BUCKET=...
export S3_ENDPOINT_URL=...
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...

# Optional
export WANDB_API_KEY=...               # For experiment tracking
export MAX_CONCURRENT_PODS=1           # Max parallel pods (default: 1)
export RUNPOD_GPU_TYPE="NVIDIA H200"   # GPU type (default: NVIDIA H200)

python run.py server --port 8000
```

Then launch experiments from the web dashboard, selecting "RunPod (cloud)" as the execution mode.

In RunPod mode:
- The idea, dataset, and cached baselines are uploaded to S3
- A pod is deployed with the Docker image, which downloads everything from S3
- The agent runs autonomously, uploading results and logs to S3
- The server monitors pod status and collects results on completion
- Findings are synced between workers for cross-pollination

## Project Structure

```
run.py                              # Unified launcher
w2s_research/
├── core/                           # Shared training library
│   ├── train.py                    #   Training loop (Unsloth + LoRA)
│   ├── eval.py                     #   Evaluation and metrics
│   ├── data.py                     #   Data loading (multiple-choice format)
│   ├── config.py                   #   RunConfig and CLI argument parser
│   └── inference.py                #   Batch prediction utilities
├── ideas/                          # Experiment implementations
│   ├── TEMPLATE/                   #   Template for new ideas
│   ├── vanilla_w2s/                
│   ├── critic/                     
│   ├── ue_zeroshot/                
│   ├── ue_fewshot/                 
│   └── train_only_on_confident_labels/
├── research_loop/                  # Autonomous agent
│   ├── agent.py                    #   Agent loop + Claude SDK wrapper
│   ├── prompt.jinja2               #   Agent system prompt
│   └── tools/                      #   MCP tools (evaluate, share, leaderboard)
├── web_ui/                         # Dashboard
│   └── backend/                    #   Flask API + experiment worker
└── infrastructure/                 # Deployment
    ├── runpod.py                   #   RunPod pod management
    ├── s3_utils.py                 #   S3 storage utilities
    └── execute_autonomous.py       #   Worker pod entrypoint
```

## License

MIT
