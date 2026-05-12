# Semantic Command Surface

This document describes the active worker-facing command surface after the
control-plane refactor. The previous filesystem RPC and evaluator socket
interfaces have been removed from the active source tree.

Workspace wrappers expose these commands on `PATH`. They call the
web/control-plane API directly and are not wrappers over any local socket
service.

The worker workspace also prepends the task runtime venv to `PATH` and exports
`VIRTUAL_ENV`, so ordinary `python`/`pip` commands use the same environment as
the semantic wrappers and official task evaluation.

## Worker Tools

| Command | Purpose |
|---|---|
| `ctx` | Read assignment, task contract, findings, artifacts, evaluations, jobs, environments, leaderboard, incumbent, telemetry, and notebook checkpoints. |
| `artifact` | Register durable local files/directories and checkout the current incumbent candidate. |
| `eval` | Request server-owned verify/probe/submit evaluation. |
| `finding` | Share or search durable agent-authored findings. Historical patterns are findings. |
| `notebook` | Checkpoint local notebook/worklog state to the server. |
| `job` | Launch and inspect durable compute jobs. |
| `env` | Inspect task environments and request worker dependency overlays. |
| `telemetry` | Record non-official process/training metrics. |
| `tool` | Publish, search, checkout, and install shared agent-authored tools. |
| `knowledge` | List, inspect, and materialize curated read-only context packaged with the task. |
| `network` | Inspect external internet policy and recorded access events. |

Accepted near-term tool that is not implemented yet:

| Command | Purpose |
|---|---|
| `trace` | Inspect, bundle, and export coding-agent execution traces. |

## Context

```bash
ctx context
ctx assignment
ctx task
ctx findings [query]
ctx evaluations
ctx artifacts
ctx jobs
ctx environments
ctx leaderboard [--limit N]
ctx incumbent [--direction-id ID]
ctx telemetry
```

The context API is the main server-to-worker hydration path. It returns
server-owned resources rather than requiring the worker to know archive paths.
`ctx context` includes assignment state, experiment metadata, research
direction, recent findings, artifacts, evaluations, jobs, leaderboard,
incumbent, direction incumbent, and notebook checkpoints visible to the worker.

## Artifacts

```bash
artifact upload --path <path> --kind <kind> [--note <text>]
artifact list
artifact checkout-incumbent --destination <path> [--direction-id ID] [--force]
```

Uploaded artifacts are copied into the object store, assigned an `Artifact`
record, and given a manifest containing digest, source path, content path, size,
file count, and metadata.

There is not yet a general artifact download/materialization API. Incumbent
checkout is implemented as the current durable candidate reuse path.

## Evaluation

```bash
eval verify (--entry <path>|--artifact-id <id>) [--environment-id ID] [--environment-overlay-id ID] [--async|--sync]
eval probe (--entry <path>|--artifact-id <id>) [--kind diagnostics] [--environment-id ID] [--environment-overlay-id ID] [--async|--sync]
eval submit (--entry <path>|--artifact-id <id>) [--environment-id ID] [--async|--sync]
eval status <evaluation-id>
eval wait <evaluation-id> [--timeout-s N]
```

`verify` and `probe` default to synchronous execution. `submit` defaults to
asynchronous execution because official or long-running evaluation should be a
durable server resource. Submit/official evaluation snapshots path-based
candidates before scoring, records environment metadata, and updates
leaderboard/incumbent state only after a completed valid official result.

Worker overlays are currently allowed for `verify` and `probe`. Official
`submit` defaults to the task base environment; overlay-backed official
submission is a future policy/runner extension.

## Jobs

```bash
job create --provider local --command '<command>' [--cwd <path>] [--env KEY=VALUE]
job create --provider local-docker --image <image> --command '<command>' [--cwd <path>] [--network-mode <mode>] [--requires-control-plane]
job create --provider runpod --template-id <template> --command '<command>' [--gpu-type-id <id>] [--gpu-count N] [--dry-run]
job list
job status <job-id>
job logs <job-id> [--max-bytes N]
job wait <job-id> [--timeout-s N]
job cancel <job-id>
```

The current job providers are local subprocess execution, a local Docker
adapter, and a first RunPod provider path with dry-run support and provider
status refresh. Job creation also accepts approval and estimated-cost metadata:
`--requires-approval`, `--approved`, and `--estimated-cost-usd`.

When experiment network policy sets `external_internet=deny`, Docker-backed
local jobs are launched with `docker run --network none`. In that mode public
internet is blocked by Docker rather than by prompt instruction. A worker cannot
override the deny policy by requesting `bridge` or `host` networking. A Docker
job that explicitly sets `--requires-control-plane` uses the server-owned
Unix-socket control-plane relay. The relay socket is mounted into the container,
`AO_CONTROL_API_URL` points at `unix:///ao-control/control.sock`, and Docker
still runs with `--network none`.

Provider-specific implementations should attach to the same durable `Job`
resource contract rather than introducing task-specific worker commands.
Production cloud workspace bundling, result polling, and cleanup remain future
work.

## Environment

```bash
env status [--environment-id ID]
env ensure
env install --pip '<requirement>' --reason '<why this is needed>' [--approved]
env list-overlays [--environment-id ID] [--status STATUS]
env overlay <overlay-id>
env approve <overlay-id>
```

The server owns task runtime environments. `local_venv` is the current default
provider. A worker that needs extra packages should request an overlay with
`env install`; this creates a separate environment record and does not mutate
the task base environment used for official scoring.

Codex/App Server workers currently require network access so these commands can
reach the local control-plane API. That implementation detail must not be
treated as permission to search the public internet for optimization answers.
The accepted policy split is:

```text
control-plane access: required for semantic tools
external internet: allow, deny, or audit according to experiment policy
```

Turning off external internet must not disable `ctx`, `eval`, `artifact`,
`finding`, `notebook`, `job`, `env`, or `telemetry`. Until the worker backend
can enforce a localhost/control-plane allowlist, local Codex/App Server runs
with external internet denied but general network technically available should
be marked as policy-weakened in session and trace metadata. Docker-backed jobs
are stricter: they use `--network none` for `external_internet=deny`, and
Docker-backed workers use a Unix-socket control-plane relay instead of weakening
the policy.

The planned `docker_image` provider should use the same environment and overlay
resource model plus the same Docker network enforcement rule.

## Findings And Notebook

```bash
finding share --type <type> --title <title> (--body <text>|--file <path>)
finding search <query>
notebook checkpoint (--content <text>|--file <path>) [--kind <kind>]
notebook list
```

Findings cover agent-authored results, hypotheses, insights, errors, and
reusable patterns. They are separate from curated task knowledge. Notebook
checkpoints make worker-local research state visible to the server and future
workers without making the local workspace the source of truth.

## Telemetry

```bash
telemetry start --provider local --name <run-name> [--job-id <job-id>]
telemetry start --provider mlflow --name <run-name> [--tracking-uri <uri>] [--experiment-name <name>]
telemetry log-metrics <telemetry-id> --metric loss=0.1 [--step N]
telemetry log-metrics <telemetry-id> --metrics '{"loss": 0.1}' [--step N]
telemetry status <telemetry-id>
telemetry list
telemetry finish <telemetry-id> [--status completed]
```

Telemetry is server-owned but non-official. It helps inspect jobs and local
experiments, but it does not update official evaluation or leaderboard state.

## Accepted Near-Term Trace Surface

```bash
trace status
trace bundle [--include-workspace-manifest] [--include-codex-home]
trace export --provider local-jsonl|otlp|phoenix|helicone [--trace-id <id>]
```

Trace bundles should register raw Codex/App Server events, command IO, worker
logs, notebook checkpoints, environment metadata, sandbox policy, and network
policy as immutable server-indexed audit artifacts. Export providers such as
OpenTelemetry OTLP, Arize Phoenix, and Helicone are observability mirrors; the
local trace bundle remains the source of truth.

## Shared Tools

```bash
tool publish --path local_tools/<name> --name <name> --description <text>
tool list [query]
tool show <tool-id>
tool checkout <tool-id> --destination shared_tools/<name>
tool install <tool-id>
```

Workers may draft helpers in `local_tools/`. Publishing creates a server-owned
shared tool record backed by an artifact digest and linked to the session trace
that produced it. Later workers can discover and checkout the tool without
guessing another workspace's filesystem layout.

## Knowledge

```bash
knowledge list [query]
knowledge show <knowledge-id>
knowledge materialize <knowledge-id> [--destination knowledge/<name>]
```

Knowledge items are curated, read-only files packaged with the task, such as
papers, notes, references, public benchmark background, or dataset descriptions.
They are different from findings: knowledge is provided by the task author;
findings are produced by workers during the experiment.

## Network

```bash
network status
network policy
network events
```

These commands expose whether external internet is allowed, denied, or only
audited, and whether the current worker backend can actually enforce that
policy. For Docker providers, the status also reports whether
`docker_network_none` enforcement is paired with a configured control-plane
relay. Package downloads should go through environment overlay policy rather
than ad hoc internet use inside the worker shell.
