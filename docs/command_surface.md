# Semantic Command Surface

This document describes the active worker-facing command surface after the
control-plane refactor. The previous filesystem RPC and evaluator socket
interfaces have been removed from the active source tree.

## Worker Tools

Workspace wrappers expose these commands on `PATH`:

| Command | Purpose |
|---|---|
| `ctx` | Read assignment, task contract, findings, artifacts, evaluations, jobs, and notebook checkpoints. |
| `artifact` | Register durable local files or directories in the artifact registry. |
| `eval` | Request server-side verify/probe/submit evaluation. |
| `finding` | Share or search durable knowledge. Historical patterns are findings. |
| `notebook` | Checkpoint local notebook/worklog state to the server. |
| `job` | Launch and inspect durable compute jobs. |

These tools call the web/control-plane API directly. They are not wrappers over
any local socket service.

## Context

```bash
ctx context
ctx task
ctx findings [query]
ctx evaluations
ctx artifacts
ctx jobs
```

The context API is the main server-to-worker hydration path. It returns
server-owned resources rather than requiring the worker to know archive paths.

## Artifacts

```bash
artifact upload --path <path> --kind <kind> [--note <text>]
artifact list
```

Uploaded artifacts are copied into the local object store, assigned an
`Artifact` record, and given a manifest containing digest, source path, content
path, size, file count, and metadata.

## Evaluation

```bash
eval verify (--entry <path>|--artifact-id <id>) [--async|--sync]
eval probe (--entry <path>|--artifact-id <id>) [--kind diagnostics] [--async|--sync]
eval submit (--entry <path>|--artifact-id <id>) [--async|--sync]
eval status <evaluation-id>
eval wait <evaluation-id> [--timeout-s N]
```

`verify` and `probe` default to synchronous execution. `submit` defaults to
asynchronous execution because official or long-running evaluation should be a
durable server resource. All official scoring goes through the server-side
evaluation service.

## Jobs

```bash
job create --provider local --command '<command>' [--cwd <path>]
job create --provider local-docker --image <image> --command '<command>' [--cwd <path>]
job list
job status <job-id>
job logs <job-id> [--max-bytes N]
job wait <job-id> [--timeout-s N]
job cancel <job-id>
```

The initial job providers are local subprocess execution and a local Docker
adapter that wraps the same durable `Job` resource. Job creation also accepts
approval and estimated-cost metadata so the server can block jobs before launch.
Provider-specific implementations such as RunPod should attach to the same
resource contract rather than introducing task-specific worker commands.

## Findings And Notebook

```bash
finding share --type <type> --title <title> (--body <text>|--file <path>)
finding search <query>
notebook checkpoint (--content <text>|--file <path>) [--kind <kind>]
notebook list
```

Findings cover results, hypotheses, insights, errors, and reusable patterns.
Notebook checkpoints make worker-local research state visible to the server and
future workers without making the local workspace the source of truth.
