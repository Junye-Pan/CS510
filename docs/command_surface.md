# Semantic Command Surface

This document describes the active worker-facing command surface after the
control-plane refactor. The previous filesystem RPC and evaluator socket
interfaces have been removed from the active source tree.

Workspace wrappers expose these commands on `PATH`. In `codex-local` runs they
use `AO_CONTROL_API_URL` pointing at a worker-started Unix-socket control-plane
broker/relay with an audit log under workspace `.control/`; the relay forwards
only control-plane API paths.

The worker workspace also prepends the task runtime venv to `PATH` and exports
`VIRTUAL_ENV`, so ordinary `python`/`pip` commands use the same environment as
the semantic wrappers and official task evaluation.

## Worker Tools

| Command | Purpose |
|---|---|
| `ctx` | Return workspace file locations for assignment, task contract, attempts, findings, artifacts, evaluations, jobs, environments, leaderboard, incumbent, telemetry, and notebook checkpoints. |
| `attempt` | Create, inspect, and update first-class candidate attempt records. |
| `artifact` | Register durable local files/directories and checkout the current incumbent candidate. |
| `eval` | Request server-owned verify/probe/submit evaluation. |
| `finding` | Share durable agent-authored findings or return finding file locations. Historical patterns are findings. |
| `notebook` | Checkpoint local notebook/worklog state to the server or return checkpoint file locations. |
| `job` | Launch durable compute jobs, attach later sessions to existing jobs, and return concise status/log file pointers. |
| `env` | Inspect task environments and request worker dependency overlays. |
| `telemetry` | Record non-official process/training metrics. |
| `tool` | Publish, search, checkout, and install shared agent-authored tools. |
| `network` | Inspect external internet policy and recorded access events. |
| `trace` | Return registered coding-agent trace bundle locations and command/event JSONL paths. |

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
ctx attempts
```

The prepared workspace is the main server-to-worker hydration path. On startup
the worker gets `task/`, `context/`, and `history/` as a materialized view of
worker-visible server state. Task-provided knowledge files, when present, are
materialized directly under `task/knowledge/` with whatever directory layout the
task author chose. These `ctx` commands return file and directory locations
when the workspace is available, so the coding agent can choose how much to
inspect with `rg`, `jq`, `sed`, `head`, or `tail` instead of receiving a broad
JSON dump in the context window.

Server state remains authoritative. The materialized files are a readable
snapshot; use server commands for fresh status and all mutations.

## Attempts

```bash
attempt create [--experiment-id ID] [--assignment-id ID] [--session-id ID] [--task-id ID] [--agent-id ID] [--direction-id ID] [--parent-attempt-id ID] [--candidate-artifact-id ID] [--status active] [--metadata JSON]
attempt list [--experiment-id ID] [--assignment-id ID] [--session-id ID] [--task-id ID] [--parent-attempt-id ID] [--status STATUS]
attempt show <attempt-id>
attempt update <attempt-id> [--status STATUS] [--session-id ID] [--candidate-artifact-id ID] [--metadata JSON]
```

An `Attempt` is a server-owned candidate-line record. It links the experiment,
assignment/session, task, agent, research direction, optional parent attempt,
lifecycle status, and optional candidate artifact. It does not store summary
fields; use `finding share` for reusable conclusions and `notebook checkpoint`
for longer local notes. Commands that create or list artifacts, evaluations,
jobs, and telemetry can take `--attempt-id`; workers may also set
`AO_ATTEMPT_ID` in their environment.

## Artifacts

```bash
artifact upload --path <path> --kind <kind> [--note <text>] [--attempt-id ID]
artifact list [--attempt-id ID]
artifact checkout-incumbent --destination <path> [--direction-id ID] [--force]
```

Uploaded artifacts are copied into the object store, assigned an `Artifact`
record, and given a manifest containing digest, source path, content path, size,
file count, and metadata.

There is not yet a general artifact download/materialization API. Incumbent
checkout is implemented as the current durable candidate reuse path.

## Evaluation

```bash
eval verify (--entry <path>|--artifact-id <id>) [--attempt-id ID] [--environment-id ID] [--environment-overlay-id ID] [--async|--sync]
eval probe (--entry <path>|--artifact-id <id>) [--kind diagnostics] [--attempt-id ID] [--environment-id ID] [--environment-overlay-id ID] [--async|--sync]
eval submit (--entry <path>|--artifact-id <id>) [--attempt-id ID] [--environment-id ID] [--async|--sync]
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
job create --provider local --command '<command>' [--cwd <path>] [--env KEY=VALUE] [--environment-id ID] [--environment-overlay-id ID] [--attempt-id ID]
job create --provider local-docker --image <image> --command '<command>' [--cwd <path>] [--network-mode <mode>] [--requires-control-plane] [--attempt-id ID]
job create --provider runpod --template-id <template> --command '<command>' [--gpu-type-id <id>] [--gpu-count N] [--dry-run] [--attempt-id ID]
job list [--attempt-id ID]
job status <job-id>
job logs <job-id> [--max-bytes N]
job attach <job-id> [--attempt-id ID] [--mode observe|continue] [--note TEXT]
job wait <job-id> [--timeout-s N]
job cancel <job-id>
```

The current job providers are local subprocess execution, a local Docker
adapter, and a first RunPod provider path with dry-run support and provider
status refresh. Job creation also accepts approval and estimated-cost metadata:
`--requires-approval`, `--approved`, and `--estimated-cost-usd`.

Task/assignment-scoped local jobs default to the resolved task environment, or
to an explicitly requested approved overlay. The job record stores the resolved
environment ids, and shell commands see the task runtime exports with the venv
`bin` directory prepended to `PATH`.

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
`job logs` returns stdout/stderr file paths and concise job metadata; workers
should use `tail`, `rg`, `sed`, or `less` to inspect only the needed log slices.
`job attach` records that the current session/attempt is now observing or
continuing from a job created by an earlier session. It does not restart the
job, claim exclusive ownership, change the job's original session/attempt, or
publish anything to the leaderboard.
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

Codex/App Server workers now reach the local control-plane API through the
semantic-tool broker, but the App Server backend still does not provide a
provider-grade public-internet allowlist independent of control-plane access.
That implementation detail must not be treated as permission to search the
public internet for optimization answers.
The accepted policy split is:

```text
control-plane access: required for semantic tools
external internet: allow, deny, or audit according to experiment policy
```

Turning off external internet must not disable `ctx`, `attempt`, `eval`,
`artifact`, `finding`, `notebook`, `job`, `env`, or `telemetry`. Until the
worker backend can enforce a localhost/control-plane allowlist, local
Codex/App Server runs with external internet denied but general network
technically available should be marked as policy-weakened in session and trace
metadata. Docker-backed jobs are stricter: they use `--network none` for
`external_internet=deny`, and Docker-backed workers use a Unix-socket
control-plane relay instead of weakening the policy. For Docker-backed workers
with `external_internet=audit`, the strict path also disables direct container
networking and starts a server-owned outbound audit proxy on a Unix socket. The
container runs a local loopback bridge, so ordinary HTTP clients see
`HTTP_PROXY=http://127.0.0.1:<port>` while all external bytes cross the mounted
proxy socket and are recorded as `NetworkAccessEvent` rows. TCP relay/proxy
fallbacks remain explicit compatibility modes and must be recorded as
policy-weakened when they require Docker `bridge` networking.

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
workers without making the local workspace the source of truth. `finding
search` and `notebook list` return materialized file locations when available.

## Telemetry

```bash
telemetry start --provider local --name <run-name> [--job-id <job-id>] [--attempt-id ID]
telemetry start --provider mlflow --name <run-name> [--tracking-uri <uri>] [--experiment-name <name>] [--attempt-id ID]
telemetry log-metrics <telemetry-id> --metric loss=0.1 [--step N]
telemetry log-metrics <telemetry-id> --metrics '{"loss": 0.1}' [--step N]
telemetry status <telemetry-id>
telemetry list [--attempt-id ID]
telemetry finish <telemetry-id> [--status completed]
```

Telemetry is server-owned but non-official. It helps inspect jobs and local
experiments, but it does not update official evaluation or leaderboard state.

## Traces

```bash
trace list [--experiment-id ID] [--assignment-id ID] [--session-id ID] [--task-id ID] [--agent-id ID] [--status STATUS] [--attempt-id ID] [--all-assignments]
trace show <trace-id>
trace commands <trace-id> [--failed-only] [--semantic-only]
trace events <trace-id> [--query TEXT] [--limit N]
trace search <query> [--experiment-id ID] [--assignment-id ID] [--session-id ID] [--task-id ID] [--agent-id ID] [--status STATUS] [--attempt-id ID] [--all-assignments]
```

Trace is read-only from the worker command surface. Coding-agent turn traces are
registered automatically from `.run/traces/<run_id>/<turn_id>/` as immutable
`agent_trace_bundle` artifacts plus lightweight `AgentTraceBundle` database
rows. The indexed metadata includes command count, failed command count,
semantic tool usage, agent-message excerpts, and observed resource ids such as
attempt/artifact/evaluation/job ids. Worker `trace` commands return trace file
locations and small index metadata rather than printing full trace contents.
Workers should inspect the returned JSONL paths with local tools such as `rg`,
`head`, `sed`, or `jq` so they choose how much context to read. They do not
manually bundle, export, or summarize traces.

Trace export providers such as local JSONL mirrors and OpenTelemetry OTLP are
control-plane observability integrations, not part of the active worker command
surface. Phoenix-compatible and Helicone-compatible sinks remain future work;
the local immutable trace artifact remains the source of truth.

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

## Task Knowledge Files

Tasks may include curated, read-only context files under
`tasks/<task_id>/public/knowledge/`. The framework does not prescribe category
names such as papers, code, notes, or specs; the directory layout is task-owned.
During worker startup that tree is copied into `task/knowledge/`, and the coding
agent inspects it with ordinary file tools. Findings remain separate: knowledge
is provided by the task author, while findings are produced by workers during
the experiment.

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
