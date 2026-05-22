# Agentic Optimization Current Architecture

This document describes the current architecture of this repository after the
control-plane refactor and cleanup. It is an implementation design document
with accepted near-term design changes called out explicitly when the code is
not there yet. Planned extensions are called out as gaps.

The current system is a server-first autonomous optimization framework. The
server owns semantic experiment state and official evaluation. Worker sessions
are disposable coding-agent runtimes attached to server-owned assignments.

## 0. Implementation Status Snapshot

This snapshot is the short operational status as of the current codebase.
Later sections give the full architecture and detailed gap inventory.

Implemented production-facing baseline:

- SQLite-backed control-plane resources for experiments, assignments,
  sessions, attempts, jobs, environments, overlays, evaluations, leaderboard
  entries, telemetry, artifacts, findings, notebook checkpoints, shared tools,
  network access events, trace bundles, and trace export runs.
- Flask `/api/v1` routes and a read-only `/ui` operator shell over the same
  resource model.
- Semantic worker workspace materialization with `task/`, `context/`,
  `history/`, semantic tool wrappers, task knowledge files, task-context
  snapshots, candidate seeds, checked-out shared tools, shell shims for
  ambiguous command names such as `eval`, per-evaluation history file refresh,
  and a lightweight local git repository for worker inspection commands.
- Worker-facing semantic tools for `ctx`, `attempt`, `artifact`, `eval`,
  `finding`, `notebook`, `job`, `env`, `telemetry`, `tool`, `network`, and
  read-only `trace`.
- Local and Docker-backed task environment execution through
  `EnvironmentService`, including `local_venv`, `docker_image`, Docker overlay
  images, task-context read-only enforcement, environment bundle export, and
  replay bundles for local and Docker paths.
- Evaluation snapshotting, async evaluation jobs, leaderboard publication,
  incumbent checkout, evaluator-budget accounting, and replay evaluation
  isolation from leaderboard/budget effects by default.
- Local subprocess jobs, Docker jobs, Docker-image jobs, first RunPod dry-run/
  launch adapter path, job attach records, task-scoped local job environment
  enforcement, and Docker network policy enforcement for deny/audit cases.
- Agent trace registration as immutable artifact-backed bundles plus local
  JSONL and OTLP trace export with baseline redaction. OTLP has been validated
  against a live local collector using a real `codex-local` worker run.
- Two concrete task packages: `circle_packing_26` as the current end-to-end
  hardening task, and `llm_inference_qwen35_4b_h200` as an RMSNorm-focused
  Qwen/vLLM/H200 kernel optimization MVP.

Implemented but still partial or policy-weakened:

- `codex-local` worker networking still needs coarse App Server network access
  for model/control-plane connectivity, so denied external internet is a
  recorded weakened policy unless a stricter provider is used.
- Docker worker strict isolation has command construction, relay/proxy support,
  and tests, but still needs Linux-host end-to-end validation and production
  deployment hardening.
- The UI is an inspection shell, not a full operator console. Stop/restart,
  active command/progress, large-run filtering, auth, and polished operational
  workflows remain future work.
- RunPod and S3-compatible storage exist as first adapter paths, not complete
  cloud orchestration with workspace bundling, polling, cleanup, retries, and
  secret handling.
- Trace export has local JSONL and OTLP providers; Phoenix-compatible and
  Helicone-compatible exporters are not implemented and are deferred from the
  current scope unless a concrete provider-specific observability requirement
  appears.
- Official overlay-backed submit is implemented only behind explicit
  experiment policy and is not the default evaluation path.

Not implemented yet:

- Network and sandbox: pure stdio semantic-tool transport; App Server
  localhost/control-plane network allowlists independent of public internet;
  non-Docker production proxy/firewall wrappers; and a full outbound network
  policy engine with robust request/response redaction.
- Operations and UI: first-class stop/restart controls, active worker current
  command/progress display, auth/permissions, large-run pagination/filtering,
  and duplicate auto-continue/stale-recovery event suppression.
- Observability: Phoenix-compatible and Helicone-compatible exporters are
  deferred; OTLP still needs production retry, batching, backoff, timeout, and
  collector-compatibility hardening beyond the current OTLP/HTTP JSON path.
- Reproducibility and cloud: cross-machine replay restore, external
  artifact-store restore, full cloud workspace/data orchestration, provider
  polling/retry/cleanup, and cloud-provider task-context immutability.
- Secrets and provenance: production secret provider integration, rotation,
  retention and snapshot visibility policy, broader non-trace redaction, signed
  images, SBOM/attestation validation, registry credential policy,
  vulnerability scanning, and reproducible worker-image promotion.
- Task coverage: broad task migration beyond the current circle-packing task
  and Qwen/H200 MVP.
- Qwen/H200 kernel-task work beyond RMSNorm: RoPE, attention prefill/decode,
  KV cache update, logits postprocess, sampling kernels, digest-keyed
  build/cache records, stronger candidate import/build sandboxing, fallback
  coverage matrices, sampling statistical validation, repeated baseline
  calibration for distribution thresholds, and live negative-control regression
  candidates.

## 1. Architectural Principles

- The server is the source of truth for experiments, assignments, sessions,
  environments, jobs, evaluations, artifacts, findings, notebook checkpoints,
  and events.
- Workers perform autonomous research behavior, but they do not own official
  experiment state.
- Worker-facing authority is exposed through semantic operations, while
  worker-visible context is materialized as a readable workspace file tree.
  Coding agents should search files directly instead of receiving broad server
  dumps in the context window.
- Runtime environments are controlled resources. Tasks declare dependencies,
  workers may request overlays, and official evaluation records the exact
  environment used.
- Agent execution traces are first-class audit data. Raw Codex/App Server
  events, command IO, worker logs, workspace manifests, and checkpoints must be
  preservable as immutable trace bundles and optionally exported to telemetry
  systems.
- Control-plane connectivity is separate from external internet access.
  Workers must be able to reach semantic server tools even when experiment
  policy forbids web search or outbound answer lookup.
- Agent-authored reusable code is a shared tool resource, not only a loose file
  in one worker workspace.
- Task-provided background material is a read-only task knowledge file tree.
  PDFs, code, text notes, datasets, and reference files packaged with the task
  should be visible to workers under the semantic workspace, not hidden in ad
  hoc paths or gated behind a separate command-line protocol.
- Task packages define domain contracts. The framework should not hard-code
  one benchmark, one ML workflow, or one metric.
- Filesystem paths are the preferred worker read surface for task context,
  history, and trace pointers. The control plane remains the source of truth;
  the workspace tree is a materialized view and scratch area.
- The retired `fsd`, `fs_cli`, and `ve` prototype architecture is not part of
  the active design.

## 2. Active Source Layout

```text
src/agentic_opt/
  common/
    Low-level utilities: atomic writes, IDs, config, runtime spec helpers,
    NumPy/SciPy loading helpers, and snapshot helpers.

  control_plane/
    Server-owned resource model, SQLite repository, artifact registry,
    JobService, EvaluationService, TelemetryService, RunPod provider,
    S3-compatible artifact storage, environment control, asynchronous
    evaluation worker, job worker, and control-plane client.

  worker_tools/
    Agent-facing semantic CLI tools:
    ctx, attempt, artifact, eval, finding, notebook, job, env, telemetry,
      tool, network, trace

  adapter/
    Codex/App Server integration, semantic workspace preparation, and
    server-first semantic worker launcher.

  web/
    Flask app and `/api/v1` routes over the control-plane service.

  task_api.py
    Task protocol, task metadata, candidate specs, and candidate path helpers.

  task_registry.py
    Configured task-root loading boundary.

scripts/
  sync_root_skills.py
    Syncs semantic worker skills into `.agents/skills`.

tests/
  test_web_backend.py
  test_semantic_workspace.py
  test_circle_packing_task.py
  test_live_local_appserver.py
```

Task implementations live outside the core package in `tasks/`.

## 3. High-Level Shape

```text
User / API client
  |
  v
Flask web API
  |
  v
ControlPlaneRepository + ControlPlaneService
  |             |              |
  |             |              +--> Artifact registry under state_root/artifacts
  |             +-----------------> JobService under state_root/jobs
  |             +-----------------> EnvironmentService under state_root/envs
  +-------------------------------> SQLite control database
  |
  v
WorkerManager
  |
  v
semantic_worker process
  |
  v
Codex/App Server session + semantic workspace
  |
  v
ctx / artifact / eval / finding / notebook / job / telemetry / env CLI tools
  |
  v
Flask web API
```

The loop is intentionally capability-based. A worker can inspect context,
modify code, upload artifacts, launch jobs, request evaluations, checkpoint
notes, or share findings in any order. The framework does not encode a fixed
research workflow.

## 4. Control Plane

The control plane has two layers:

- `ControlPlaneRepository`: SQLite-backed persistence and query methods.
- `ControlPlaneService`: higher-level server operations that need filesystem
  artifacts, jobs, or evaluation execution.

### 4.1 Repository Tables

The current SQLite schema creates these resource tables:

```text
cp_experiments
cp_assignments
cp_sessions
cp_attempts
cp_environments
cp_environment_overlays
cp_jobs
cp_artifacts
cp_evaluations
cp_leaderboard_entries
cp_telemetry_runs
cp_findings
cp_notebook_checkpoints
cp_events
cp_shared_tools
cp_agent_traces
```

The repository stores JSON payloads for config, budget, policy, metadata,
inputs, outputs, request bodies, results, public feedback, metrics, links, and
event payloads. This keeps the current schema simple while still preserving
structured resource state.

The implemented shared-tool and network-event tables provide the resource
boundary for agent-authored reusable tools and network policy audit state. Task
knowledge remains task-package input material and is surfaced through workspace
files and task contract inventory, not a separate worker command resource.
Agent traces are indexed in `cp_agent_traces`; the large trace payloads stay in
artifact blobs.

### 4.2 Resource Model

Current first-class resources in the implemented schema:

```text
Task
Experiment
WorkerAssignment
WorkerSession
Attempt
Job
Evaluation
LeaderboardEntry
TelemetryRun
Artifact
Finding
NotebookCheckpoint
Event
Environment
EnvironmentOverlay
SharedTool
TaskKnowledgeFile
NetworkAccessEvent
AgentTraceBundle
```

`Attempt` is a dedicated first-class candidate-line resource backed by
`cp_attempts`. It links experiment, assignment/session, task, agent, research
direction, optional parent attempt, lifecycle status, and optional candidate
artifact. Artifacts, evaluations, jobs, and telemetry runs can attach to an
attempt with `attempt_id`, so later workers can reconstruct the candidate line
from structured resources.

`Attempt` deliberately does not store summary fields. Reusable conclusions,
failure diagnoses, and narrative interpretation belong in `Finding`; longer
local scratch state belongs in `NotebookCheckpoint`.

`LeaderboardEntry` is the official score publication resource. The current
incumbent for an experiment, task, or research direction is the highest active
leaderboard entry in that scope.

`Environment` and `EnvironmentOverlay` are first-class resources. The active
code has both `local_venv` and `docker_image` provider paths behind
`EnvironmentService`. `docker_image` prepares task images, records image
identity in the environment lock, supports Docker-backed evaluation/jobs, and
can build derived overlay images for approved worker dependency additions.

`Finding` is the durable agent-authored finding resource. Historical "patterns"
are represented as findings, usually by choosing a `finding_type` such as
`pattern`, `insight`, `hypothesis`, `result`, or `error`. Curated task-provided
background material belongs in the task knowledge file tree, not `Finding`.

`AgentTraceBundle` is the accepted resource for coding-agent audit trails. The
current Codex adapter writes raw App Server events and summarized output under
each workspace's `.run/traces/` directory. The semantic worker and worker
reaper register those turn traces as artifact-backed records linked to
experiment, assignment, session, worker backend, run id, and turn id. The
database row is intentionally lightweight; large raw events and normalized
command/message JSONL live in the immutable trace artifact.

`SharedTool` is the accepted resource for agent-authored reusable executable
helpers. A worker may draft code in `local_tools/`, but useful tools should be
published into a server-owned registry with digest, version, declared runtime
requirements, owning task or experiment scope, documentation, and provenance
links to the session trace that created them.

`TaskKnowledgeFile` is task-contract inventory for read-only task context
packaged with the task. It covers files such as PDFs, notes, reference
implementations, dataset descriptions, and benchmark background that the task
author intentionally provides to agents. Knowledge differs from findings:
findings are agent-authored during the experiment, while knowledge is curated
task input material defined at task-definition time and materialized under
`task/knowledge/`.

`NetworkAccessEvent` records external network attempts when a provider can
observe or proxy them. The policy distinction is not "network on/off"; it is
"semantic control-plane access allowed" versus "external internet access
allowed, denied, or audited."

### 4.3 ControlPlaneService Responsibilities

`ControlPlaneService` coordinates the active services below. `EnvironmentService`
is now the server-owned environment boundary for task base environments and
worker overlays across both local virtualenv and Docker image providers.

- task contract exposure
- local path artifact registration
- optional S3-compatible artifact upload
- artifact manifest creation with digest, size, file count, and source metadata
- `EvaluationService`
- leaderboard/incumbent publication and incumbent checkout
- `TelemetryService`
- `JobService`
- `EnvironmentService` as the owner of framework, task, and worker-overlay
  runtime environments
- `AgentTraceService` for trace bundle registration and read-only query/search
- shared-tool publishing, listing, checkout, and install operations
- task-knowledge inventory and read-only workspace materialization operations
- network-policy enforcement state for Docker-backed jobs/workers, with future
  proxy/firewall hardening for non-Docker backends

Official evaluation state lives in `cp_evaluations`. Long-running submit
evaluations are queued as jobs that run
`agentic_opt.control_plane.evaluation_worker`. Completed valid submit/official
evaluations create `LeaderboardEntry` rows and may update the incumbent.
Evaluator budget is a leaderboard score budget: `total_evaluator_runs` means
the target number of published submit/official scores for the experiment.
Verify/probe evaluations, failed non-scoring evaluations, and replay evaluations
that do not publish leaderboard entries do not satisfy this budget.

## 5. Web API

The Flask app is created by `agentic_opt.web.app`.

Default command:

```bash
PYTHONPATH=src python3 -m agentic_opt.web.app \
  --state-root ao_state \
  --db ao_state/control.sqlite3
```

Important routes:

```text
GET  /
GET  /healthz
GET  /api/v1/object-model

GET  /api/v1/environments
POST /api/v1/environments
GET  /api/v1/environments/<environment_id>
POST /api/v1/environments/<environment_id>/export-bundle
POST /api/v1/environment-overlays
GET  /api/v1/environment-overlays
GET  /api/v1/environment-overlays/<overlay_id>
POST /api/v1/environment-overlays/<overlay_id>/approve

GET  /api/v1/tasks/<task_id>

GET  /api/v1/experiments
POST /api/v1/experiments
GET  /api/v1/experiments/<experiment_id>
PATCH /api/v1/experiments/<experiment_id>
GET  /api/v1/experiments/<experiment_id>/analysis
POST /api/v1/experiments/<experiment_id>/assignments
POST /api/v1/experiments/<experiment_id>/assignments/generate

GET  /api/v1/assignments/<assignment_id>
PATCH /api/v1/assignments/<assignment_id>
POST /api/v1/assignments/<assignment_id>/sessions
POST /api/v1/assignments/<assignment_id>/start-local

PATCH /api/v1/sessions/<session_id>

GET  /api/v1/context?assignment_id=...

POST /api/v1/attempts
GET  /api/v1/attempts
GET  /api/v1/attempts/<attempt_id>
PATCH /api/v1/attempts/<attempt_id>

POST /api/v1/artifacts
GET  /api/v1/artifacts
GET  /api/v1/artifacts/<artifact_id>

POST /api/v1/evaluations
GET  /api/v1/evaluations
GET  /api/v1/evaluations/<evaluation_id>

GET  /api/v1/leaderboard
GET  /api/v1/incumbent
POST /api/v1/incumbent/checkout

POST /api/v1/telemetry-runs
GET  /api/v1/telemetry-runs
GET  /api/v1/telemetry-runs/<telemetry_id>
POST /api/v1/telemetry-runs/<telemetry_id>/metrics
POST /api/v1/telemetry-runs/<telemetry_id>/finish

POST /api/v1/jobs
GET  /api/v1/jobs
GET  /api/v1/jobs/<job_id>
GET  /api/v1/jobs/<job_id>/logs
POST /api/v1/jobs/<job_id>/attach
POST /api/v1/jobs/<job_id>/cancel

POST /api/v1/findings
GET  /api/v1/findings

POST /api/v1/notebook-checkpoints
GET  /api/v1/notebook-checkpoints

POST /api/v1/events
GET  /api/v1/events
GET  /api/v1/events/stream
GET  /api/v1/sessions/<session_id>/trace

POST /api/v1/shared-tools
GET  /api/v1/shared-tools
GET  /api/v1/shared-tools/<tool_id>
POST /api/v1/shared-tools/<tool_id>/checkout

GET  /api/v1/network-policy
POST /api/v1/network-access-events
GET  /api/v1/network-access-events

POST /api/v1/agent-traces
GET  /api/v1/agent-traces
GET  /api/v1/agent-traces/search
GET  /api/v1/agent-traces/<trace_id>
GET  /api/v1/agent-traces/<trace_id>/commands
GET  /api/v1/agent-traces/<trace_id>/events
```

The web layer is intentionally thin. It validates request shape lightly,
delegates to the repository/service layer, and returns JSON.

The environment routes are implemented for both `local_venv` and
`docker_image`. Both providers use the same `Environment` and
`EnvironmentOverlay` resource records; provider-specific details live in
`spec_json`, `lock_json`, and metadata.

The `agent-traces` routes return artifact-backed trace bundle metadata,
normalized command records, filtered raw events, and basic command/message
search results. They do not implement worker-triggered export.

## 6. Worker Plane

The current worker backend is Codex through the App Server adapter. It can run
as a local process or through the Docker worker wrapper when the assignment
selects a Docker-backed backend and supplies a worker image.

Local worker startup path:

```text
POST /api/v1/assignments/<assignment_id>/start-local
  -> WorkerManager.start_control_assignment
  -> create WorkerSession
  -> spawn `python -m agentic_opt.adapter.semantic_worker`
  -> prepare semantic workspace
  -> start Codex/App Server session
  -> run one autonomous turn
  -> checkpoint WORKLOG.md
  -> update session status
```

The semantic worker startup environment and the prepared workspace tool
environment use these variables. `WorkerManager` starts `semantic_worker` with
assignment/session identifiers; `semantic_worker` then resolves the task
environment and writes the full runtime exports into the workspace/App Server
environment.

The workspace environment prepends both the semantic `bin/` wrappers and the
task runtime venv `bin/` directory to `PATH`, exports `VIRTUAL_ENV`, and exports
repo `src` on `PYTHONPATH`. Worker shell commands such as `python` and `pip`
therefore resolve to the same task runtime used by the semantic tool wrappers,
instead of whichever interpreter happens to be first on the host `PATH`.

Codex/App Server workers currently need network access enabled so semantic
tools can reach the local control-plane HTTP API. This is an implementation
limitation, not the intended optimization policy. The intended policy has two
separate channels:

```text
control_plane_network
  Required for semantic tools. Allows localhost or a configured private control
  endpoint only.

external_internet
  Optional experiment capability. Controls web search, package downloads,
  arbitrary HTTP fetches, and answer lookup against public internet sources.
```

An experiment must be able to run with `external_internet=deny` while
`control_plane_network=allow`. Turning off external internet must not break
`ctx`, `eval`, `artifact`, `finding`, `notebook`, `job`, `env`, `telemetry`,
`trace`, or `tool` commands. It also must not remove the already-materialized
read-only `task/knowledge/` files.

The local `codex-local` implementation can keep the coarse App Server network
flag enabled only to preserve localhost control-plane access, but it should
label the session as "external internet not technically enforced" unless a
provider can enforce a localhost allowlist. Docker-backed execution must not use
that weakening path: when `external_internet=deny`, Docker jobs run with
`--network none`, and requests for broad networking are rejected unless an
explicit TCP control-plane relay fallback has been selected and recorded as
policy-weakened. A Docker worker that needs both semantic tool access and denied
public internet uses the server-owned Unix-socket control-plane relay by
default. The container mounts only that socket, semantic tools use
`AO_CONTROL_API_URL=unix://...`, and the relay forwards only control-plane API
paths to the Flask server. Docker Desktop/Colima-style deployments can opt into
a TCP relay fallback, which keeps the same API allowlist but cannot combine with
Docker `--network none`; those sessions record the weakened enforcement state.
When a Docker worker is allowed audited external egress
(`external_internet=audit`), the production path still disables direct Docker
networking: the host starts an HTTP/CONNECT audit proxy on a Unix socket, the
container mounts that socket, and a container-local loopback bridge exposes it
as `HTTP_PROXY=http://127.0.0.1:<port>` for Codex and ordinary HTTP clients.
This keeps the container on `--network none`; all external bytes go through the
server-owned proxy and become `NetworkAccessEvent` records.

Remaining target implementations are:

- App Server sandbox support for localhost/control-plane allowlists.
- A pure stdio semantic-tool transport. `codex-local` workers now use a
  Unix-socket broker/relay for semantic tools, but the App Server backend still
  lacks provider-grade public-internet allowlist enforcement.
- Production network proxy/firewall wrappers for non-Docker backends and
  stricter allowlist enforcement. Docker-backed workers now have the
  Unix-socket relay path, TCP fallback, `--network none` enforcement, and a
  strict Unix-socket outbound audit proxy bridge that records proxied
  HTTP/CONNECT attempts without granting direct container networking.

Worker instructions remain useful, but they are not sufficient enforcement.
When external internet is denied, any provider that still exposes general
network access must mark the run as policy-weakened and record that fact in the
session, trace bundle, and leaderboard/evaluation metadata.

```text
AO_CONTROL_API_URL
AO_ASSIGNMENT_ID
AO_EXPERIMENT_ID
AO_TASK_ID
AO_AGENT_ID
AO_SESSION_ID
AO_WORKSPACE_ROOT
AO_TASK_RUNTIME_ENV
AO_TASK_RUNTIME_ROOT
AO_TASK_RUNTIME_PYTHON
AO_TASK_RUNTIME_FINGERPRINT
AO_ENVIRONMENT_ID
AO_ENVIRONMENT_TYPE
AO_ENVIRONMENT_ROOT
AO_ENVIRONMENT_PYTHON
AO_ENVIRONMENT_FINGERPRINT
AO_ENVIRONMENT_OVERLAY_ID      # when present
AO_ENVIRONMENT_OVERLAY_ROOT    # when present
AO_ENVIRONMENT_OVERLAY_PYTHON  # when present
```

The worker is disposable. Durable continuity should come from server resources:
findings, artifacts, evaluations, jobs, notebook checkpoints, and events.

## 7. Semantic Workspace

`adapter.semantic_workspace.prepare_semantic_workspace` creates the worker
workspace.

Current workspace contents:

```text
AGENTS.md
WORKLOG.md
bin/
  ctx
  artifact
  eval
  finding
  notebook
  job
  telemetry
  env
  tool
  network
  trace
.agents/skills/
  artifact-use/
  context-use/
  environment-use/
  evaluation-use/
  finding-use/
  job-use/
  notebook-use/
  telemetry-use/
  tool-use/
  network-use/
  trace-use/
reference/
  WORKSPACE_BOOTSTRAP.json
task/
  TASK.md
  public_contract.md
  manifest.json
  public_files/
  knowledge/
  research_directions/
context/
  README.md
  current_state.json
  assignment.json
  experiment.json
  network_policy.json
  research_direction.json
history/
  attempts/<attempt_id>/attempt.json
  artifacts/<artifact_id>/artifact.json
  evaluations/<evaluation_id>/{evaluation,request,result,public_feedback}.json
  findings/*.json
  findings/index.jsonl
  jobs/<job_id>/status.json
  network/{policy.json,events.jsonl}
  notebooks/*.{json,md}
  shared_tools/<tool_id>/tool.json
  telemetry/<telemetry_id>/telemetry.json
  traces/<trace_id>/manifest.json
  traces/index.jsonl
  leaderboard.jsonl
  incumbent.json
  direction_incumbent.json
  environments.jsonl
  environment_overlays.jsonl
outbox/
  artifacts/
  findings/
  notebooks/
artifacts/
findings/
local_tools/
shared_tools/
candidate entrypoint copied from task public seed
```

The tool wrappers call `agentic_opt.worker_tools.semantic_cli` using the
resolved task base Python. Worker overlays are separate environment resources;
the current workspace wrapper environment is not hot-swapped after overlay
creation. The workspace does not materialize `fs` or `ve` wrappers.

The startup prompt tells the agent that the server owns experiments,
assignments, environments, artifacts, jobs, evaluations, leaderboard/incumbent
state, findings, notebook checkpoints, registered traces, and policy. It also
states that `task/`, `context/`, and `history/` are the primary read surface.
The model should use normal coding-agent file tools such as `rg`, `jq`, `sed`,
`head`, and `tail` to decide how much history to inspect.

This follows the pattern from `external/automated-w2s-research`: keep hidden
labels and official evaluation server-side, but sync worker-visible shared
findings, notebooks, logs, and snapshots into ordinary local files so agents can
search them without forcing every record into the prompt. Commands remain the
authority boundary for evaluation, publishing, job creation, environment
requests, network policy, and fresh status.

On-demand workspace materialization targets:

```text
shared_tools/
  checked-out published shared tools selected by the worker or experiment
```

`local_tools/` remains a writable scratch area for worker-authored helper code.
Publishing a tool must copy it into the shared tool registry and produce a
server-owned `SharedTool` record. `task/knowledge/` is materialized at startup
from `tasks/<task_id>/public/knowledge/` and must be treated as read-only task
context; workers can cite or inspect it, but should not mutate it as part of a
candidate.

## 8. Worker Tools

The semantic CLI is implemented in `worker_tools/semantic_cli.py`.

Worker commands are not meant to be a general memory transport. Query-style
commands such as `ctx context`, `ctx task`, `ctx findings`, `finding search`,
`notebook list`, `job logs`, and worker `trace` commands return workspace
paths or concise status/pointer records when possible. Agents then inspect the
files themselves with ordinary shell tools. Mutating or authority-bearing
commands still call the server directly: evaluations, artifact upload,
finding share, notebook checkpoint, job create/cancel, environment overlays,
shared tool publish/checkout, network policy, and stop conditions.

Tool surface:

```text
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

attempt create [--experiment-id ID] [--assignment-id ID] [--session-id ID] [--task-id ID] [--agent-id ID] [--direction-id ID] [--parent-attempt-id ID] [--candidate-artifact-id ID] [--status active] [--metadata JSON]
attempt list [--experiment-id ID] [--assignment-id ID] [--session-id ID] [--task-id ID] [--parent-attempt-id ID] [--status STATUS]
attempt show <attempt-id>
attempt update <attempt-id> [--status STATUS] [--session-id ID] [--candidate-artifact-id ID] [--metadata JSON]

artifact upload --path <path> --kind <kind> [--attempt-id ID]
artifact list [--attempt-id ID]
artifact checkout-incumbent --destination <path> [--direction-id ID] [--force]

eval verify (--entry <path>|--artifact-id <artifact-id>) [--attempt-id ID] [--environment-id ID] [--environment-overlay-id ID] [--sync|--async]
eval probe (--entry <path>|--artifact-id <artifact-id>) [--kind diagnostics] [--attempt-id ID] [--environment-id ID] [--environment-overlay-id ID] [--sync|--async]
eval submit (--entry <path>|--artifact-id <artifact-id>) [--attempt-id ID] [--environment-id ID] [--sync|--async]
eval status <evaluation-id>
eval wait <evaluation-id> [--timeout-s N]

finding share --type <type> --title <title> --body <text>
finding share --type <type> --title <title> --file <path>
finding search <query>

notebook checkpoint (--file WORKLOG.md|--content <text>) [--kind <kind>]
notebook list

job create --provider local --command '<command>' [--cwd <path>] [--env KEY=VALUE] [--environment-id ID] [--environment-overlay-id ID] [--attempt-id ID]
job create --provider local-docker --image <image> --command '<command>' [--cwd <path>] [--network-mode <mode>] [--requires-control-plane] [--attempt-id ID]
job create --provider runpod --template-id <template> --command '<command>' [--gpu-type-id <id>] [--gpu-count N] [--dry-run] [--attempt-id ID]
job list [--attempt-id ID]
job status <job-id>
job logs <job-id> [--max-bytes N]
job attach <job-id> [--attempt-id ID] [--mode observe|continue] [--note TEXT]
job wait <job-id> [--timeout-s N]
job cancel <job-id>

telemetry start --provider local --name <run-name> [--attempt-id ID]
telemetry start --provider mlflow --name <run-name> [--attempt-id ID]
telemetry log-metrics <telemetry-id> --metric loss=0.1 --step 1
telemetry status <telemetry-id>
telemetry list [--attempt-id ID]
telemetry finish <telemetry-id>

env status
env ensure
env install --pip '<requirement>' --reason '<why this is needed>' [--approved]
env list-overlays [--environment-id ID] [--status STATUS]
env overlay <overlay-id>
env approve <overlay-id>
```

Official scoring should go through `eval submit`. Long-running compute should
go through `job create`. Non-official training/process metrics should go through
`telemetry`. Reusable agent-authored observations should go through
`finding share`. A coherent candidate line should be registered with
`attempt create`; related artifacts, evaluations, jobs, and telemetry should use
the same `attempt_id`.
Incumbent discovery should start from `history/leaderboard.jsonl` and
`history/incumbent.json`, or the path-returning `ctx leaderboard` /
`ctx incumbent` helpers. Candidate reuse should go through
`artifact checkout-incumbent` rather than guessing artifact paths.
`job attach` is the worker-facing handoff primitive for durable jobs created by
an earlier session. It records that the current session/attempt is observing or
continuing from an existing job, but it does not restart the job, claim exclusive
ownership, change the job's original `session_id` / `attempt_id`, or affect
leaderboard state.

The `env` command must not install dependencies directly into shared base
environments. It calls the server-owned environment API to inspect base
environments, create worker overlays, and approve blocked overlay requests.

Additional implemented command surface:

```text
tool publish --path local_tools/<name> --name <name> --description <text>
tool list [query]
tool show <tool-id>
tool checkout <tool-id> --destination shared_tools/<name>
tool install <tool-id>

network status
network policy
network events

trace list [--experiment-id ID] [--assignment-id ID] [--session-id ID] [--task-id ID] [--agent-id ID] [--status STATUS] [--attempt-id ID] [--all-assignments]
trace show <trace-id>
trace commands <trace-id> [--failed-only] [--semantic-only]
trace events <trace-id> [--query TEXT] [--limit N]
trace search <query> [--experiment-id ID] [--assignment-id ID] [--session-id ID] [--task-id ID] [--agent-id ID] [--status STATUS] [--attempt-id ID] [--all-assignments]
```

`tool` is for reusable executable helpers; durable claims about what a tool
showed still belong in `finding`. Curated read-only task context is available as
files under `task/knowledge/`, not as a semantic command. `network` lets a
worker inspect whether external internet access is allowed, denied, or only
audit-logged. `trace` is read-only worker context for registered
coding-agent turn records. Worker `trace` commands return trace file locations
and small index metadata; agents should use local file tools such as `rg`,
`head`, `sed`, or `jq` to inspect only the relevant slices of normalized command
history, raw events, and agent messages. Workers do not manually bundle, export,
or summarize traces.

### 8.1 Findings And Failure Reports

Findings are durable agent-authored claims, observations, hypotheses, results,
failure diagnoses, and reusable patterns. The old idea of a separate `Pattern`
resource should remain folded into `Finding`, usually by choosing a
`finding_type` such as `pattern`, `insight`, `result`, `hypothesis`, or
`error`.

Negative findings and failure reports are first-class research assets. Failed
runs often contain useful search information that raw logs alone do not expose:
what was attempted, why it failed, what evidence ruled out a path, and what a
later worker should avoid or revisit. Workers should use `finding share` for
these reusable conclusions and `notebook checkpoint` for larger local scratch
state.

### 8.2 Shared Tools

Shared tools are executable artifacts authored by workers and intentionally
made reusable by later workers. They are different from findings:

```text
Finding
  A durable claim, observation, result, failure diagnosis, or pattern.

SharedTool
  A runnable helper such as an analyzer, comparator, exporter, plotting script,
  search driver, data converter, or repair utility.
```

The lifecycle is:

```text
1. Worker drafts code in local_tools/.
2. Worker tests it locally or through jobs/evaluations as appropriate.
3. Worker publishes it with tool publish.
4. Server snapshots the tool directory/file as an artifact, computes a digest,
   records metadata, and links it to the creating session trace.
5. Later workers discover it with tool list/search and checkout or install it
   into shared_tools/.
```

Shared tool records should include:

```text
tool_id
name
description
task_id
experiment_id
scope                 # task, experiment, direction, global
artifact_id
entrypoint
runtime_requirements
created_by_assignment_id
created_by_session_id
created_by_agent_id
created_from_trace_id
digest
version
status                # active, deprecated, blocked
metadata
```

Publishing a shared tool should not automatically make it trusted for official
evaluation. It only makes the helper available for worker-side research.
Official candidate behavior must still pass through task contracts,
environment policy, and server-owned evaluation.

## 9. Job Service

`control_plane.jobs.JobService` is the server-owned job layer.

Current providers:

```text
local
  Runs a host subprocess through `agentic_opt.control_plane.job_worker`.

local-docker
  Wraps a command in `docker run --rm -v <cwd>:/workspace -w /workspace <image>`
  and then executes it through the same local job path.
  If experiment network policy sets `external_internet=deny`, this adapter adds
  `--network none`, rejects `bridge`/`host` overrides, and uses the
  control-plane relay when a Docker job requires semantic API access.

docker_image
  Runs Docker jobs from a prepared `Environment` image. The job can provide an
  `environment_id` instead of an image tag; the service resolves the image ref,
  records image digest metadata, applies controlled mounts/env/workdir, and uses
  the same Docker network policy enforcement as `local-docker`.

runpod
  Creates a RunPod pod through the RunPod REST API. Dry-run mode records the pod
  launch payload without contacting RunPod.
```

Unknown providers are recorded as queued jobs, but they are not executed by a
provider adapter yet. AWS, Slurm, SkyPilot, Modal, HF Jobs, and similar systems
are future provider adapters.

Current job features:

- durable job record in SQLite
- job directory under `state_root/jobs/<job_id>`
- command manifest
- stdout/stderr log files
- status lookup
- log tailing
- worker-facing attach records that let a later session explicitly observe or
  continue using a durable job without changing the job's origin
- cancellation by PID where possible
- basic approval and cost gates
- max-job budget gate
- experiment-owned auto-approval with hard cost caps
- RunPod capacity errors classified as retryable provider failures

Jobs should run under a declared environment. Docker image jobs can now resolve
their image from an `environment_id` and record provider-specific image
metadata. Local host jobs created in an assignment/task scope now resolve the
task base environment or requested overlay by default, record that environment
on the job, and run shell commands with the resolved runtime exports and venv
`bin` directory on `PATH`.

## 10. Evaluation Service

Evaluation is server-owned.

Request normalization accepts either:

```text
entry_path
artifact_id
```

Evaluation kinds:

```text
verify
  Synchronous by default. Calls `task.verify_entry(entry_path)`.

probe
  Synchronous by default. Calls `task.probe_entry(entry_path, kind=...)`.

submit / official
  Asynchronous by default. First calls `verify_entry`; if valid, calls
  `task.evaluate_entry(entry_path)`.
```

Submit/official evaluation snapshots the candidate into a candidate artifact
before evaluation when the request is path-based. This freezes the candidate
for async execution and gives the leaderboard/incumbent mechanism a durable
checkout target.

Async submit creates an Evaluation record, launches an evaluation job in the
resolved task environment, and updates the Evaluation after
`evaluation_worker` completes. With a `docker_image` environment, the
evaluation job runs inside the prepared Docker image. Public-safe feedback is
stored separately from the full result payload. Completed valid submit/official
evaluations create a `LeaderboardEntry`; the highest active entry is the
incumbent for that experiment/task/direction scope.

Evaluation must also be environment-owned:

```text
verify/probe
  May run against the task base environment or an approved worker overlay.
  The Evaluation request/result records the environment_id and optional
  overlay_id used.

submit / official
  Defaults to the task base environment.
  Current implementation resolves official submit through the task base
  environment even if an overlay id is present in the CLI request. Overlay-backed
  official submit is a future policy/runner extension.
```

Docker-backed official evaluation provides container-level isolation for the
evaluation process and records runner/image metadata. Hidden evaluator
hardening still needs production review for secret handling, redaction,
retention, and provider-specific isolation guarantees.

## 10.1 Leaderboard and Incumbent

Leaderboard entries are generated from completed valid submit/official
evaluations. They record:

```text
experiment_id
task_id
assignment_id
direction_id
evaluation_id
artifact_id
score
environment_id
environment_overlay_id
metadata
```

The current incumbent is derived by sorting active entries by descending score.
Incumbents can be queried globally for an experiment or within a research
direction. `POST /api/v1/incumbent/checkout` copies the incumbent artifact into
a requested destination path so workers can inspect, repair, or compare it
without depending on raw artifact storage layout.

Environment fingerprints, immutable lock summaries, framework environment
metadata, and runner metadata are copied into evaluation requests and
leaderboard metadata for official reproducibility. Environment-level
reproducibility bundles can be exported from environment records. Evaluation
replay bundles can also be exported and replayed through the normal official
evaluation path for both `local_venv` and `docker_image` task environments.
Remaining hardening is cross-machine restore, external artifact backends, and
stricter provider-level context immutability.

## 11. Artifact Registry

Artifacts are server-indexed durable objects.

Path upload flow:

```text
artifact upload --path <path> --kind <kind>
  -> POST /api/v1/artifacts
  -> ControlPlaneService.register_path_artifact
  -> copy file/directory under state_root/artifacts/<artifact_id>/content
  -> compute digest, size, and file count
  -> write manifest.json
  -> create cp_artifacts row
  -> record artifact.registered event
```

The default object store is local filesystem storage under `state_root`.
Artifacts can also be uploaded to an S3-compatible object store with
`storage_provider="s3"` when `boto3` and credentials are available.

## 12. Telemetry

Telemetry is server-owned but non-official.

Current providers:

```text
local
  Stores metrics as JSONL under state_root/telemetry/<telemetry_id>/.

mlflow
  Lazily imports MLflow, creates an MLflow run, logs params/tags/metrics, and
  stores the external run id/dashboard URL in the TelemetryRun record.
```

Telemetry records can link to experiments, assignments, sessions, jobs,
attempts, and artifacts. They help diagnose training/process behavior, but they
do not update official evaluations or leaderboard state.

### 12.1 Agent Trace and Observability Export

Agent traces are audit records for coding-agent behavior. They should capture
enough information to reconstruct what the worker saw and did:

```text
raw App Server event stream
agent visible messages and final message
command invocations, cwd, stdout/stderr, exit code, and duration
worker process stdout/stderr
WORKLOG.md notebook checkpoints
workspace file manifest and selected diffs
semantic tool calls and server responses
environment id / overlay id
sandbox and network policy
model/provider/runtime metadata
```

The implementation writes raw App Server events and summarized output into
`.run/traces/<run_id>/<turn_id>/`, and Codex also writes its rollout JSONL under
`.codex-home/sessions/`. Worker turns automatically register the `.run/traces`
directory as a server-owned `AgentTraceBundle` row backed by an immutable
`agent_trace_bundle` artifact with digest metadata. Every trace bundle is linked
to the worker session; related artifacts, evaluations, findings, shared tools,
and leaderboard entries can reference trace ids in metadata when relevant.

Trace export is provider-backed and optional. The current implementation has a
control-plane `TraceExportRun` resource and a server-side export service. Exports
read immutable `agent_trace_bundle` artifacts and normalized command/message
JSONL; workers do not manually bundle or export traces.

Current providers:

```text
local-jsonl
  Writes normalized commands, agent messages, raw-event mirrors, event records,
  manifest metadata, file digests, payload digest, and redaction summary to
  state_root/trace_exports/<trace_export_id>/.

otlp
  Builds OpenTelemetry-compatible spans/events from the same normalized,
  redacted trace payload, writes a local payload/manifest mirror, and posts
  OTLP/HTTP JSON to the configured endpoint.
```

Planned providers:

```text
phoenix
  Export LLM/tool-call traces to an Arize Phoenix-compatible sink when
  configured by experiment policy.

helicone
  Export LLM request/response metadata to a Helicone-compatible sink when the
  model access path supports it.
```

These exports are observability mirrors, not the source of truth. The
source of truth remains the control-plane database plus immutable trace/artifact
blobs. If an external telemetry provider is unavailable, the local trace bundle
must still be complete.

Trace export must respect privacy and experiment policy. Hidden evaluator
internals, secrets, private datasets, and denied network destinations should be
redacted or omitted before export. The local-jsonl implementation has default
conservative redaction for secret keys/tokens, HTTP auth and cookie headers,
private/hidden/grader paths, hidden grader/private dataset fields, sensitive URL
query parameters, denied destinations, and oversized command-output fields. A
redaction summary is recorded in the export manifest and `TraceExportRun`
result.

### 12.2 Network Access Control

Network control is a first-class experiment policy. It answers whether workers
may use the public internet to search for information, fetch code, download
datasets, or look up benchmark answers during optimization. It must not be
conflated with access to the local semantic control plane.

Policy shape:

```json
{
  "network": {
    "control_plane": "allow",
    "external_internet": "deny",
    "package_indexes": "policy",
    "allowed_hosts": ["127.0.0.1", "localhost"],
    "denied_hosts": [],
    "audit_external_attempts": true,
    "mark_policy_weakened_if_unenforced": true
  }
}
```

Provider behavior:

- `control_plane=allow` permits semantic tools to reach the configured
  `AO_CONTROL_API_URL`.
- `external_internet=deny` blocks or records public web access, including
  browser use, `curl`, `wget`, arbitrary HTTP clients, and search APIs.
- `package_indexes=policy` allows dependency downloads only through
  environment overlay policy, not ad hoc worker commands.
- Docker-backed jobs enforce `external_internet=deny` with `--network none` and
  reject broad Docker networking overrides.
- Docker-backed workers that require semantic tools under `external_internet=deny`
  use a Unix-socket control-plane relay instead of broad Docker networking by
  default. Docker worker startup can opt into an explicit TCP relay fallback for
  Docker Desktop/Colima-style socket-mount constraints; that path records
  `policy_weakened=true` because Docker cannot combine the TCP relay with
  `--network none`.
- If the worker backend cannot enforce the split, it must mark the session and
  trace as policy-weakened instead of pretending the run is internet-isolated.

This distinction is required for meaningful optimization experiments. A task can
ship public papers in its knowledge bundle, and workers may inspect those files
because they are declared task context, while the same run can still be
forbidden from searching the live web for incumbent answers.

## 13. Task Contract

Tasks are loaded through `task_registry.py`.

Search roots:

```text
AO_TASKS_ROOTS
AO_TASKS_ROOT
repo_root/tasks
```

A task package must provide `task.py` with:

```python
def create_task() -> TaskProtocol:
    ...
```

The task object must expose:

```text
metadata: TaskMetadata
runtime_spec: TaskRuntimeSpec
public_dir: Path
verify_entry(entry_path: Path) -> dict
probe_entry(entry_path: Path, *, kind: str) -> dict
evaluate_entry(entry_path: Path) -> dict
```

The task public directory must contain:

```text
TASK.md
public_contract.md
candidate seed entrypoint
```

The default candidate entrypoint is `initial.py`. Tasks can override candidate
shape with `CandidateSpec`.

Task packages may also include a curated knowledge bundle. This bundle is part
of the task definition, not a worker-created memory store and not an
experiment-time web search cache:

```text
public/knowledge/
  manifest.json        # optional
  ... task-defined files and directories ...
```

Knowledge files are intentionally worker-visible task context. Examples include
papers, survey notes, prior-method descriptions, domain references, API
manuals, or public benchmark background that the task author packaged with the
task. The framework does not prescribe directory categories such as `papers/`,
`code/`, or `notes/`; those are task-author choices. During worker startup, the
tree is materialized directly under `task/knowledge/`, and coding agents inspect
it with ordinary file tools such as `find`, `rg`, `sed`, `head`, PDF tooling, or
Python scripts. No worker-facing `knowledge` command is required.

The optional manifest is an index and description file for task authors,
operators, and possible UI/indexing features. It is not the worker access
protocol. A minimal manifest can look like:

```json
{
  "items": [
    {
      "id": "paper_foo_2024",
      "title": "Paper title",
      "kind": "paper",
      "path": "papers/foo_2024.pdf",
      "media_type": "application/pdf",
      "summary": "Why this is relevant",
      "scope": "task",
      "tags": ["method", "baseline"],
      "read_only": true
    }
  ]
}
```

Knowledge is not a loophole around network policy. If a paper or public method
is provided in the task knowledge bundle, it is declared task context for any
experiment using that task. If external internet is denied, workers may still
inspect the provided bundle but may not search the live web for additional
answers unless policy allows it.

For example, `tasks/circle_packing_26/public/knowledge/` could contain several
circle-packing papers and a manifest. A worker reads those files from
`task/knowledge/` during optimization, but they remain read-only task context
and are not confused with agent-authored findings or shared tools.

`TaskRuntimeSpec` is declarative. It says which Python version, Python packages,
imports, and shadowing rules the task requires. The control plane is
responsible for resolving that declaration into an environment before task
code, candidate code, or evaluator code is executed.

### 13.1 Current Converted Task

`tasks/circle_packing_26` is the current concrete converted benchmark. It uses
the default single-file candidate shape: `initial.py` exposes `run_packing()`,
which returns centers, radii, and a reported score. The task declares a
`local_venv` runtime with NumPy/SciPy requirements, validates circle-packing
feasibility for 26 circles, exposes public diagnostics through `probe`, and
publishes official scores through `submit`.

Its public `research_directions/manifest.json` is used by assignment generation:
directions are assigned round-robin and copied into assignment metadata, worker
context, and startup prompts. This task validates the current environment,
evaluation, candidate snapshot, leaderboard, incumbent checkout, and research
direction paths.

## 14. Environment Control

Environment control is part of the core architecture. Dependency management is
not a task workaround or a worker-local side effect. It is a server-owned
resource boundary that must be shared by tests, the control plane, workers,
jobs, and task evaluation.

The current code uses `common.runtime_env` as the low-level local-venv helper
behind `EnvironmentService`: it derives task runtime fingerprints, prepares
local venvs, runs import/public-seed preflights, and exports runtime variables.
The same service also owns the `docker_image` provider: it builds or resolves
task images, runs import/public-seed preflights inside Docker, records image
identity in the environment lock, and prepares provider-specific worker
overlays.

### 14.1 Environment Resource Model

Base environment:

```text
Environment
  environment_id
  environment_type        # framework, task
  parent_environment_id
  task_id
  experiment_id
  fingerprint
  status                  # preparing, ready, failed, retired
  python_path
  root_path
  spec_json               # includes provider-specific config
  lock_json               # pip freeze, image digest, or equivalent lock
  created_at
  updated_at
```

Worker overlay:

```text
EnvironmentOverlay
  overlay_id
  base_environment_id
  experiment_id
  assignment_id
  session_id
  status                  # requested, preparing, ready, blocked, failed
  requested_by_agent_id
  requirements
  reason
  approved
  python_path
  root_path
  lock_json
  policy_decision_json
  created_at
  updated_at
```

The base environment is immutable after it becomes ready. An overlay is
assignment/session-scoped and can be discarded without invalidating official
task state.

### 14.2 Environment Specifications

Tasks continue to declare requirements through `TaskRuntimeSpec`. The current
default provider is `local_venv`:

```python
TaskRuntimeSpec(
    kind="local_venv",
    python=">=3.11,<3.12",
    requirements=("numpy>=1.23", "scipy>=1.10"),
    required_imports=("numpy", "scipy"),
    forbidden_shadow_modules=("numpy", "scipy"),
)
```

The task declares what it needs. It does not decide where installation happens
or which process happens to have those packages installed. The control plane
resolves the spec into a prepared `Environment`.

The provider for reproducible official evaluation is `docker_image`. The same
task-level dependency declaration can be resolved into either a local virtualenv
or a Docker image:

```python
TaskRuntimeSpec(
    kind="docker_image",
    python=">=3.11,<3.12",
    requirements=("numpy>=1.23", "scipy>=1.10"),
    required_imports=("numpy", "scipy"),
    forbidden_shadow_modules=("numpy", "scipy"),
)
```

Docker-specific details live in the environment spec and lock, not in worker
prompts:

```json
{
  "provider": "docker_image",
  "base_image": "python:3.11-slim",
  "dockerfile": "generated-or-task-owned Dockerfile",
  "context_digest": "sha256:...",
  "image_ref": "agentic-opt/task-circle-packing-26:<fingerprint>",
  "image_digest": "sha256:..."
}
```

Framework-level dependencies should be represented by a framework environment
spec. Tests, API servers, semantic CLI wrappers, workers, jobs, and evaluation
workers should not silently rely on whichever `python3` is first on `PATH`.

### 14.3 Environment Service

Accepted module shape:

```text
src/agentic_opt/control_plane/environments.py
```

Primary operations:

```text
ensure_environment(spec) -> Environment
ensure_framework_environment() -> Environment
ensure_task_environment(task_id) -> Environment
create_overlay(payload) -> EnvironmentOverlay
approve_overlay(overlay_id) -> EnvironmentOverlay
get_execution_environment(task_id, environment_id=None, overlay_id=None, allow_overlay=False) -> resolved execution environment
exports_for_environment(environment) -> dict[str, str]
exports_for_overlay(overlay) -> dict[str, str]
```

`EnvironmentService` is responsible for selecting the provider implementation.
The execution adapter boundary is shared by `EvaluationService`, `JobService`,
`environment_providers`, and `docker_runtime`. Docker-backed evaluation and jobs
now run through a common provider run-plan interface so image identity, mounts,
network enforcement, and resource limits are recorded consistently.

All subprocess launch sites should take a resolved environment instead of using
`sys.executable` directly. This includes:

```text
EvaluationService
evaluation_worker
JobService local jobs
semantic_worker
semantic workspace tool wrappers
tests that exercise real task behavior
```

### 14.4 Environment Providers

Current run-plan interface:

```text
EnvironmentProvider.build_run_plan(EnvironmentRunSpec) -> EnvironmentRunPlan
EnvironmentRunPlan.command -> subprocess/docker argv
EnvironmentRunPlan.env -> launcher environment
EnvironmentRunPlan.metadata -> provider runner metadata
EnvironmentRunPlan.network_enforcement -> policy enforcement metadata
```

Environment preparation and exports are still owned by `EnvironmentService`.
The provider run-plan layer is intentionally narrower: it turns a resolved
environment plus command/mount/policy inputs into an executable plan without
mutating environment records.

`local_venv` provider:

- default provider during the current rewrite
- selects a base Python that satisfies the task version constraint
- verifies that the base Python can create a venv with pip
- installs declared Python requirements into an immutable task venv
- runs import and public-seed preflights before marking the environment ready
- captures `pip freeze` into `lock_json`
- exposes `python_path` for `EvaluationService`, `semantic_worker`, and tool
  wrappers. Local worker-created jobs in task/assignment scope use the resolved
  task or overlay environment by default.

`docker_image` provider:

- implemented provider for CI-oriented and cross-machine reproducible
  evaluation paths
- builds task images from the repo/task context or resolves a supplied image ref
- records immutable image identity in `lock_json`, especially `image_digest`
- resolves prepared environments to immutable repo digests or local image IDs at
  execution time; mutable tags are treated as policy-weakened fallback only
- enforces `external_internet=deny` for Docker-backed jobs/workers with
  `--network none` and rejects broad networking overrides
- uses a control-plane relay when a Docker worker/job needs semantic tools while
  public internet is denied. The strict automatic path is a Unix socket mounted
  into the container. Docker worker startup also supports an explicit TCP
  fallback for Docker Desktop/Colima-style environments where socket mounts are
  unreliable; this fallback is recorded as policy-weakened. The lower-level
  Docker runner can consume either a Unix socket relay or a TCP relay URL.
- uses a Unix-socket outbound audit proxy bridge for Docker workers when
  audited external egress is allowed. The worker container remains on
  `--network none`; Codex sees a local loopback HTTP proxy whose upstream is the
  mounted proxy socket.
- runs official evaluations and Docker jobs through provider run plans instead
  of host `sys.executable`
- mounts only explicit workspace, artifact, state, repo/task, and configured
  provider paths
- records runner metadata such as image ref, image digest, working directory,
  network policy, and resource limits
- uses the same `Environment` and `EnvironmentOverlay` resources as the venv
  provider

The Docker runner records image digests for official evaluation and leaderboard
metadata. Tags are acceptable as input convenience, but a prepared Docker
environment is not ready until the image has been inspected and an immutable
identity has been stored in the environment lock.

### 14.5 Worker Dependency Additions

Workers may need extra packages for local exploration. They should request an
overlay instead of mutating the shared task base environment.

Worker-facing command design:

```bash
env status
env install --pip "shapely>=2.0" --reason "geometric diagnostics"
env install --pip "numba" --reason "accelerate local restart search"
env list-overlays
env overlay <overlay-id>
env approve <overlay-id>
```

The server records the request, applies experiment policy, and creates the
overlay when allowed. The current worker process environment is not hot-swapped;
the worker must explicitly request/use the overlay for supported operations.
The overlay id becomes part of evaluation metadata when used and should be
propagated into job, artifact, and finding metadata when relevant.

Overlay implementation is provider-specific:

- with `local_venv`, the overlay is another venv or venv layer prepared from
  the base spec plus requested requirements
- with `docker_image`, the overlay is a derived image layer or an approved
  runtime mount whose image digest is captured

This gives workers flexibility without making the task environment unstable or
unreproducible.

### 14.6 Policy

Environment policy is experiment-owned. Example shape:

```json
{
  "environments": {
    "allow_worker_overlay": true,
    "auto_approve_pip": ["numpy", "scipy", "shapely", "numba"],
    "require_approval_for": ["torch", "tensorflow", "cupy"],
    "max_overlay_size_mb": 2000,
    "allow_private_indexes": false
  }
}
```

Policy gates should consider:

- package allowlists and denylists
- private package indexes and credentials
- native extensions and system package requirements
- expected download/build size
- GPU/CUDA dependencies
- Docker image provenance, registry access, and policy for which immutable
  digests are trusted. The current Docker policy covers registry, repository,
  digest, immutable identity, and local-build gates; signature verification,
  SBOMs, attestations, registry credential policy, and signed provenance remain
  future hardening.
- network access
- reproducibility and lock-file capture

Blocked environment requests should become durable records with public-safe
reasons, just like blocked jobs.

### 14.7 Official Evaluation Rule

Official evaluation defaults to the task base environment.

A worker overlay may be used for `verify` or `probe` when policy allows it.
Official `submit` and `official` evaluations may use an overlay only when the
experiment explicitly opts in with
`policy.environments.allow_official_overlay_submit=true` and the overlay is
both approved and ready. Otherwise `submit` resolves to the task base
environment and an overlay request is rejected.

Leaderboard-eligible evaluations must record:

```text
environment_id
environment_fingerprint
overlay_id, if any
environment_provider
lock_json or equivalent dependency snapshot
```

Current implementation records `environment_id`, `environment_overlay_id`,
`environment_provider`, a lock summary, and runner metadata in the evaluation
request and leaderboard metadata. Environment export bundles and evaluation
replay bundles are implemented; remaining replay hardening is cross-machine
runtime reconstruction, external artifact storage, and richer trace artifact
packing.

If a candidate requires an overlay dependency to run, the result is not fully
reproducible until that overlay is captured and approved. If the dependency is
useful beyond one assignment, the worker should share a finding or proposal;
a maintainer can then promote it into the task base `TaskRuntimeSpec`.

For `docker_image` official evaluation, the record includes the immutable image
digest and runner metadata. Re-running the same evaluation should be possible
from:

```text
candidate artifact
environment image digest
task version
evaluation request
```

### 14.8 Current Status and Gap

The repository now has `EnvironmentService` paths for `local_venv` and
`docker_image`: durable environment and overlay records, task base environment
preparation, environment-aware evaluation, Docker image digest locks,
Docker-backed official evaluation, Docker-backed jobs, and an `env` worker tool
surface.

Remaining environment work is not core Docker execution or baseline replay. It
is broader hardening: add stronger image provenance checks beyond
registry/digest policy, provider-parity policy controls for local job opt-out,
and make replay bundles portable across machines and artifact stores.

Docker implementation checklist in the current codebase:

- task image build/resolve/preflight/lock: implemented in
  `control_plane/environments.py`
- Docker overlay image build and lock capture: implemented in
  `control_plane/environments.py`
- digest-pinned execution reference resolution: implemented in
  `control_plane/environment_providers.py`
- shared Docker command/network enforcement: implemented in
  `control_plane/docker_runtime.py`
- Docker-backed official evaluation, including async job dispatch: implemented
  in `control_plane/evaluation.py` and `control_plane/jobs.py`
- Docker image jobs from `environment_id`: implemented in
  `control_plane/jobs.py`
- Docker worker wrapper and control-plane relay startup: implemented in
  `web/workers.py`
- Unix-socket and TCP allowlisted relay transports: implemented in
  `control_plane/relay.py`
- Docker outbound audit proxy for `external_internet=audit`: implemented in
  `control_plane/network_proxy.py`, `worker_tools/proxy_bridge.py`,
  `control_plane/jobs.py`, and `web/workers.py`
- Docker image registry/repository/digest trust policy: implemented in
  `control_plane/docker_image_policy.py`
- environment default propagation into Docker jobs/evaluation/worker
  workspace/tool environments: implemented in `control_plane/environments.py`,
  `control_plane/environment_providers.py`, `control_plane/jobs.py`, and
  `adapter/semantic_workspace.py`
- environment reproducibility export bundles: implemented in
  `control_plane/service.py` and exposed through
  `POST /api/v1/environments/<environment_id>/export-bundle`
- unit coverage for digest-pinned evaluation/job execution and both relay
  transports: implemented in `tests/test_web_backend.py`

## 15. State and Filesystem Layout

The server state root defaults to:

```text
ao_state/
```

Current local control-plane layout:

```text
ao_state/
  control.sqlite3
  artifacts/
    <artifact_id>/
      content
      manifest.json
  jobs/
    <job_id>/
      command.json
      stdout.log
      stderr.log
  task_contexts/
    <task_id>/
      <task_context_digest>/
        task/
        snapshot.json
  workspaces/
    <assignment_id>/
      <session_id>/
  telemetry/
    <telemetry_id>/
      metrics.jsonl
      params.json
      tags.json
  trace_exports/
    <trace_export_id>/
      manifest.json
      commands.jsonl          # local-jsonl provider
      agent_messages.jsonl    # local-jsonl provider
      events.jsonl            # local-jsonl provider
      raw_events.jsonl        # local-jsonl provider
      otlp_payload.json       # otlp provider
  envs/
    tasks/
      <task_id>/
        <runtime_fingerprint>/  # local_venv provider storage
    overlays/
      <overlay_id>/             # local_venv overlay storage
```

Shared tools are currently stored as `Artifact` blobs with `kind=shared_tool`
and indexed in SQLite through `cp_shared_tools`. Agent traces are stored as
`Artifact` blobs with `kind=agent_trace_bundle` and indexed in SQLite through
`cp_agent_traces`. Knowledge files remain in the task package under
`tasks/<task_id>/public/knowledge/`; workspace startup exposes them through the
server-owned task-context snapshot under `task/knowledge/` and records both the
knowledge digest and whole task-context digest in workspace context.
Network access events are currently SQLite records rather than JSONL
files. Local JSONL trace exports are stored under `trace_exports/` and indexed
through `cp_trace_export_runs`; each export also creates a `trace_export`
artifact record for digest and provenance tracking.

When `EnvironmentService` is not available, the low-level runtime helper can
still default to:

```text
.ao_envs/<task_id>/<runtime_fingerprint>/
```

That path is a fallback implementation detail. In normal control-plane runs,
`ao_state/envs/` plus the SQLite `Environment`/`EnvironmentOverlay` records are
authoritative. Docker image layers may live in the local Docker cache or a
registry, but their immutable image digests must still be recorded in SQLite.

## 16. Current End-to-End Local Flow

```text
1. Start Flask control plane.
2. Create an Experiment with task_id, config, budget, and policy.
3. EnvironmentService ensures the task base environment when the worker or
   evaluation path needs it. Framework environment records are available through
   the same service, but the local server still starts from the current Python
   interpreter.
4. Generate one or more WorkerAssignments; task research directions are assigned
   round-robin when the task exposes `research_directions/manifest.json`.
5. Start a local assignment.
6. WorkerManager creates a WorkerSession and spawns semantic_worker with
   assignment/session identifiers.
7. semantic_worker loads assignment context from the API and resolves the task
   environment.
8. semantic_worker prepares semantic workspace and semantic tool wrappers using
   the resolved task environment, network policy, task knowledge file-tree
   materialization, and selected shared tool materialization.
9. Codex/App Server receives startup prompt and semantic tools.
10. Agent edits candidate code and uses ctx/attempt/eval/job/artifact/finding/
    notebook/telemetry/env/tool/network/trace plus direct file reads from
    task/knowledge/.
11. eval submit snapshots the candidate, then creates a server-owned Evaluation
    with environment metadata.
12. Async submit launches a server-owned Job running evaluation_worker in the
    resolved task environment.
13. Completed valid submit/official evaluations publish LeaderboardEntry rows
    and update the derived incumbent.
14. Job/evaluation/leaderboard/incumbent status can be inspected by any later
    worker session.
15. Attempts, findings, artifacts, notebook checkpoints, events, shared tools,
    network events, and registered trace bundles persist in SQLite and
    state_root artifacts; task knowledge remains task-package input material.
```

### 16.1 Latest Local Validation

The latest validation for `tasks/circle_packing_26` covers both deterministic
control-plane paths and a real model-backed worker run.

Control-plane and reproducibility validation:

- A local control-plane/evaluator E2E created an experiment, assignment,
  session, and attempt; ran `verify`, `probe`, and `submit`; wrote the submit
  candidate snapshot artifact; published a leaderboard entry; derived the
  incumbent; checked out the incumbent; recorded a `job attach` handoff; and
  verified the human-facing
  `/api/v1/experiments/<experiment_id>/analysis` read model.
- `local_venv` replay validation prepared the circle-packing task venv, ran
  `verify`, `probe`, `submit`, exported a replay bundle, and replayed it
  through the normal evaluation path. Submit and replay scores matched, and
  replay did not publish a second leaderboard entry.
- `docker_image` replay validation built the circle-packing task image from
  `python:3.11-slim`, installed NumPy/SciPy, recorded a repo-digest image lock,
  ran Docker-backed `verify`, `probe`, `submit`, exported a replay bundle, and
  replayed it through the Docker-backed path. Submit and replay scores matched,
  and replay did not publish a second leaderboard entry.
- Task-context immutability validation created canonical
  `state_root/task_contexts/<task_id>/<digest>/task/` snapshots. The
  `local_venv` path uses a copied read-only snapshot plus post-run digest guard;
  the Docker path mounts the same snapshot as a read-only bind mount and
  rejects writable task-context mount conflicts.
- The read-only control-plane UI exists under `/ui` and
  `/ui/experiments/<experiment_id>`. It provides Overview, Leaderboard,
  Analysis, Traces, Isolation, and Replay tabs over the existing JSON API.

Real `codex-local` worker validation:

- A real Codex worker ran `tasks/circle_packing_26` with
  `--max-turn-wall-time-s 1800` and completed naturally in about 915 seconds.
  The session finished with `status=completed` and `stop_reason=turn_completed`.
- The worker used semantic tools to create an attempt, run six evaluations,
  upload the final candidate artifact, publish one leaderboard entry, share a
  finding, checkpoint the notebook, and register a completed trace bundle.
- The official submit score from that run was approximately
  `2.63598308291758`, and the leaderboard contained one published score.
- The worker session recorded its live process id in the session row. With the
  web reaper active, no `worker.session.stale` or `assignment.auto_continue`
  event was recorded for the long-running external worker.
- The trace bundle had 69 recorded command executions and was exported through
  the OTLP provider to a local OTLP/HTTP collector. The collector received one
  `/v1/traces` request with 91 spans; the export completed with HTTP 200 and
  wrote both `manifest.json` and `otlp_payload.json`.
- OTLP redaction was checked against the local payload. The test authorization
  secret and `OPENAI_API_KEY` name did not appear in the exported payload.
- The run exposed two worker-experience issues that are now fixed in code:
  fresh submit evaluations did not always create every file advertised under
  `history/evaluations/<evaluation_id>/`, and `git status --short` failed
  because semantic workspaces were not git repositories. The semantic CLI now
  materializes `evaluation.json`, `request.json`, `result.json`,
  `public_feedback.json`, and `feedback.json` for every fresh eval record; new
  semantic workspaces are initialized as lightweight git repositories, and
  directory artifact upload excludes `.git`.

Known validation caveat:

- A previous managed Docker-environment optimization run exposed a remaining
  lifecycle race where normal continuation and stale/unmanaged-session recovery
  can both record `assignment.auto_continue` for the same completed session.
  The observed run did not create duplicate active workers, but duplicate event
  suppression is still listed as remaining lifecycle work.

## 17. What Is Not Implemented Yet

This section is the consolidated inventory of things that appear in the design
or planning documents but are not yet implemented as production-ready code.
Items listed here should not be treated as available capabilities. When an
area has a partial implementation, this section names the implemented baseline
first and then the remaining gap, so future work does not accidentally
reimplement already-working paths.

Current remaining gaps after the task-knowledge file-tree migration, local/
Docker task-context read-only hardening, Docker-image environment provider,
local JSONL and OTLP trace export, semantic workspace shell/history/git
hardening, real `codex-local` worker validation, and read-only UI baseline are
grouped below.

Partial implementations that need production hardening:

- Trace observability: automatic trace bundle registration plus `local-jsonl`
  and OTLP export are implemented. Phoenix-compatible and Helicone-compatible
  exporters, richer retry/batching controls, and an operator surface for export
  runs are not implemented.
- Product/operator UI: `/ui` exists as a read-only inspection shell.
  Product-grade leaderboard publication views, richer run-analysis interactions,
  active-worker current-activity/progress display, trace-export dashboard or
  CLI, first-class stop/restart controls, auth/permissions, and a consolidated
  strict-vs-weakened isolation view are not implemented.
- Worker lifecycle: worker reaping, auto-continue, stale-session recovery, and
  budget exhaustion handling are implemented. Long-running external
  `semantic_worker` processes now record their live PID so the reaper does not
  misclassify them as stale. Duplicate auto-continue event suppression,
  first-class stop/cancel endpoint semantics, active command/watchdog
  visibility, stalled worker detection, and clearer session/worker status
  rollups remain open.
- Docker isolation: Docker worker/job command construction, control-plane relay
  support, `--network none` deny enforcement, and Docker worker audit-proxy
  support are implemented. Linux-host strict e2e validation, Docker job parity
  with the Unix-socket audit-proxy bridge, production worker-image publishing,
  and image provenance/signing/SBOM hardening remain open.
- Replay/reproducibility: replay bundles and replay runner work for
  `local_venv` and `docker_image`; replay is excluded from leaderboard/budget
  effects by default; official overlay submit is policy-gated. Cross-machine
  bundle restore, external artifact-store restore, provider parity for local
  job environment controls, and cloud-provider task-context immutability remain
  open.
- Shared tools and task knowledge: publish/list/show/checkout/install and
  task-context-backed `task/knowledge/` materialization are implemented. Trust
  policy, deprecation workflow, richer dependency metadata, generated indexes
  or previews for binary knowledge, and deeper trace provenance remain open.

Not implemented as usable capabilities:

- Pure stdio semantic-tool transport.
- App Server localhost/control-plane network allowlists independent of public
  internet access.
- Non-Docker production proxy/firewall wrappers.
- Full outbound network policy engine with richer destination categories,
  request/response redaction, denied-destination tests, and response-size
  policy.
- Production secret-provider integration, secret rotation, retention/snapshot
  visibility policy, and redaction coverage for non-trace bundles and external
  providers.
- Full cloud workspace/data orchestration, provider entrypoint packages,
  result polling, retry policy, cleanup policy, and provider failure recovery
  beyond the first RunPod adapter path.
- Broad task migration and richer search/assignment visibility policies for
  islands, explorers, exploiters, cross-pollination, and direction-local reuse.
- Completion of the Qwen 3.5 4B H200 kernel task beyond the RMSNorm MVP:
  pinned/containerized official H200 environment, RoPE/attention/KV/logits/
  sampling definitions, digest-keyed build/cache records, stronger import
  isolation, fallback coverage matrices, sampling statistical validation,
  repeated distribution-threshold calibration, and live negative-control
  regression candidates.

### 17.1 Core Resource Model

- `Attempt`, `AgentTraceBundle`, `SharedTool`, and
  `NetworkAccessEvent` are implemented as first-class control-plane resources.
  `Attempt` intentionally does not contain a summary field; reusable
  conclusions belong in `Finding`, and longer local state belongs in
  `NotebookCheckpoint`.
- `TraceExportRun`, the `local-jsonl` export provider, and the OTLP export
  provider are implemented.
  The implemented trace source of truth remains automatic `AgentTraceBundle`
  registration plus immutable local artifacts; local JSONL exports are
  observability mirrors under `state_root/trace_exports/<trace_export_id>/`.
  The implemented baseline includes:
  - a `TraceExportRun` control-plane record with provider, status, source trace
    ids, destination, artifact id, error, redaction policy, digest, request,
    result, and metadata.
  - a server-side export service that reads immutable `agent_trace_bundle`
    artifacts and normalized command/message JSONL without asking workers to
    bundle or export anything manually.
  - a local JSONL provider that writes manifest, commands, agent messages,
    normalized event records, and raw-event mirrors for offline inspection and
    regression tests.
  - an OTLP provider that maps normalized trace records to OpenTelemetry spans
    and events, posts OTLP/HTTP JSON to a configured endpoint, records provider
    failure on the export run, and writes the redacted OTLP payload plus
    manifest as a local artifact.
  - a first normalized export schema for agent messages, command executions,
    semantic tool calls, raw event references, worker/session/task metadata,
    and trace-artifact provenance.
  - conservative default redaction for obvious secrets/tokens, private/hidden
    paths, HTTP auth/cookie headers, sensitive URL query values, hidden grader
    fields, denied destinations, and oversized command output, with redaction
    counts recorded in export metadata.
  - control-plane API routes for creating, listing, and inspecting export runs.
    These are not worker `trace export` commands.
  - synthetic and real-worker regression coverage for local JSONL export,
    OTLP export, redaction, provider failures, and stable exported-payload
    digests.
  Remaining trace-export work:
  - harden OTLP retry/batching/backoff and live-collector compatibility beyond
    the current OTLP/HTTP JSON endpoint path.
  - implement Phoenix-compatible export for LLM/tool-call observability when an
    experiment explicitly configures it.
  - implement Helicone-compatible export only where the model access path can
    provide useful request/response metadata without leaking hidden evaluator or
    secret data.
  - add an operator/admin CLI or dashboard surface for export runs if the JSON
    API is not enough.
  - add provider-specific tests for external exporter failures, retry behavior,
    timeout behavior, batching, redaction stability, and exported payload
    compatibility.
- Worker-facing `job attach` is implemented as an observation/continuation
  marker on durable jobs. Worker budget-status query and structured
  failure-report protocol are intentionally not part of the worker-facing
  surface; budget remains an operator/control-plane enforcement concern, and
  findings/notebook checkpoints are the failure-analysis mechanism.
- Attempt lineage and run analysis are implemented as a human-facing
  control-plane read model under `GET /api/v1/experiments/<experiment_id>/analysis`.
  It joins attempt graph, score series, candidate lineage, evaluations, traces,
  artifacts, jobs, telemetry, findings, and notebook checkpoints for dashboard
  or post-run inspection. It is not a worker semantic tool.

### 17.2 Network And Sandbox Enforcement

- Docker worker strict isolation now has a concrete implementation for audited
  egress: the worker container runs with `--network none`, semantic control
  plane access goes through a mounted Unix-socket relay, and external HTTP/
  CONNECT traffic goes through a mounted Unix-socket audit proxy exposed inside
  the container by a loopback bridge. This is the production direction for
  model-backed Docker workers that need observable external egress.
- Docker worker `external_internet=deny` also enforces `--network none` and the
  Unix-socket control-plane relay. A model-backed Codex turn cannot reach model
  APIs in this mode unless model access is provided by a future host-side model
  broker or another explicitly controlled non-network transport. In practice,
  autonomous model-backed Docker workers should use `external_internet=audit`
  when model API access is required.
- The TCP control-plane relay remains an explicit compatibility fallback for
  Docker Desktop/Colima-style environments where mounted Unix sockets are
  unreliable. It cannot be combined with `--network none`; sessions using it
  must be marked `policy_weakened=true`. It is not a production strict
  isolation path.
- Strict Docker worker behavior still needs Linux-host end-to-end validation.
  The current macOS/Colima test environment has already shown unreliable
  mounted Unix socket behavior, so passing TCP-fallback e2e does not prove the
  strict path.
- Docker jobs enforce `external_internet=deny` with `--network none`, and
  Docker jobs that require control-plane access can use the Unix-socket relay.
  Docker job `external_internet=audit` still uses the older TCP/env-proxy path
  and should be upgraded to the same Unix-socket proxy bridge used by Docker
  workers.
- `codex-local` workers do not yet enforce the intended
  control-plane-only allowlist. The App Server path still uses coarse network
  availability so semantic tools can reach the local control plane and must mark
  denied-external-internet runs as policy-weakened when enforcement is not
  possible.
- App Server sandbox policy does not yet provide a localhost/control-plane
  network allowlist independent of public internet access.
- `codex-local` semantic tools now use a worker-started control-plane broker
  with a Unix-domain socket relay and workspace-local audit log. On systems with
  short Unix-socket path limits, the socket path may fall back to a short
  temporary path while the audit log remains under workspace `.control/`.
  A pure stdio semantic-tool transport is still future work.
- The outbound audit proxy records destination and decision metadata, but it is
  not yet a full production network policy engine. Remaining work includes
  richer allow/deny matching, provider-specific destination categories,
  proxy-specific request/response redaction, response-size policy, and
  regression tests for denied destinations and sensitive proxied output.
- Non-Docker production proxy/firewall wrappers are not implemented. Host/
  App-Server backends still need provider-grade allowlist enforcement rather
  than prompt-level policy.

### 17.3 Environment And Reproducibility Hardening

- Local worker-created jobs are automatically run under the resolved task or
  requested overlay environment by default for task/assignment-scoped local
  jobs. Remaining hardening is broader provider parity and policy controls for
  when local jobs may opt out.
- Docker image trust policy covers registry/repository/digest gates, immutable
  identity, and local-build policy, but production provenance checks are not
  implemented: signature verification, SBOM validation, attestations,
  registry-auth policy, and signed build provenance remain future work.
- The worker image Dockerfile exists for local Docker e2e, but production image
  publishing is not implemented. Missing pieces include multi-architecture
  builds, pinned base image digest policy, image signing/attestation, registry
  promotion rules, vulnerability scanning, and a reproducible build manifest.
- Evaluation replay bundles are implemented for the local artifact store and
  replay through the normal evaluation path. They include candidate snapshots,
  environment records/bundles, framework environment metadata, evaluation
  request/result, runner metadata, job logs, task context digests, and trace
  references. Replay evaluations are marked as replay records and do not update
  leaderboard or evaluator-budget state by default. Remaining hardening is
  cross-machine restore, external artifact backends, and richer trace artifact
  packing.
- Overlay-backed official `submit` is implemented behind explicit experiment
  policy. By default, overlays remain limited to `verify` and `probe`; official
  scoring accepts an approved ready overlay only when
  `policy.environments.allow_official_overlay_submit` is true.
- Framework environment records are now copied into evaluation requests and
  replay bundles. The lock path first uses `pip freeze` and falls back to
  `importlib.metadata` package enumeration when `pip freeze` is broken on the
  host interpreter. Remaining hardening is making framework-runtime selection
  fully declarative for production deployments rather than relying on the
  current control-plane interpreter.
- Task context snapshots are implemented for local and Docker providers.
  Worker-visible task context is materialized once as a canonical server-owned
  snapshot containing `TASK.md`, `public_contract.md`, `public_files/`,
  `knowledge/`, `knowledge_inventory.json`, `manifest.json`, and
  `research_directions/`, with a whole-tree digest. `local_venv` workspaces use
  a copied read-only snapshot plus post-run digest guards for local jobs and
  evaluations; this is intentionally marked `policy_weakened=true` because
  same-UID host processes can still chmod local files. Docker workers,
  Docker-backed evaluations, and Docker jobs mount the snapshot over the
  container task path with a read-only bind mount and reject writable mounts that
  target `task/` or its protected subdirectories.

### 17.4 Artifacts, Tools, Task Knowledge, And UI

- No general artifact download/materialization API yet. Incumbent checkout is
  implemented for local filesystem destinations.
- Shared tools are implemented as artifact-backed records with publish/list/
  show/checkout/install commands, but trust policy, deprecation workflow,
  richer dependency metadata, and trace provenance remain future hardening.
- Task knowledge is now part of the provider-protected task-context snapshot.
  Workspace startup exposes it under `task/knowledge/`, writes
  `task/knowledge_inventory.json`, records both knowledge and whole-task-context
  digests in `context/current_state.json`, and does not expose a worker-facing
  `knowledge` command. Remaining knowledge work is optional discovery/UI
  hardening: generated text indexes for PDFs or other binary files, content
  search, rich previews, and cloud-provider read-only snapshot integration.
- A read-only control-plane frontend exists under `/ui`. It is intentionally an
  operator inspection shell over existing API resources, not yet a full product
  console. Implemented views include:
  - Overview with score budget, best score, active worker count, and published
    scores.
  - Leaderboard entries with environment/provider and task-context metadata.
  - Run Analysis with score series, candidate lineage, and attempt-graph
    summary; the score series is a dynamic SVG point/line chart.
  - Trace, Isolation, and Replay tabs for existing trace/export, policy, and
    replay records.
- Remaining UI work is product and operations depth rather than basic routing:
  first-class stop/restart controls, active worker current-command/progress
  display, richer leaderboard publication workflows, trace-export operations,
  consolidated strict-vs-weakened isolation summaries, auth/permissions,
  pagination/search/filtering for large runs, and frontend regression tests
  beyond the current shell/static-resource coverage.
- Docker isolation decisions are visible through the read-only Isolation tab,
  but the view is still a raw record surface. It does not yet provide a
  consolidated run-level verdict that explains which sessions/jobs were strict,
  weakened, or blocked and why.

### 17.5 Cloud Providers, Secrets, And Isolation

- RunPod and S3-compatible storage have first provider adapters, but there is no
  production cloud workspace/data bundle orchestration, pod entrypoint package,
  result polling loop, retry policy, cleanup policy, or provider failure
  recovery flow.
- Hidden evaluator isolation still needs production hardening for secrets,
  retention, provider-specific sandbox guarantees, and visibility policy.
  Docker-backed official evaluation provides current container isolation, and
  local-jsonl/OTLP trace export has baseline redaction, but these do not by
  themselves solve full production secret handling.
- Secrets and isolation policy still needs provider secret injection without
  disclosure, safe logging, secret scrubbing, artifact retention policy,
  snapshot visibility policy, and redaction coverage for non-trace bundles and
  external export providers.
- Codex authentication for Docker workers is still copied/mounted from a host
  Codex home into per-session provider state. This works for local e2e, but a
  production deployment needs an explicit secret provider and rotation model
  instead of relying on host-user Codex files.
- Autonomous cloud approval remains a product and policy gap. Approval behavior
  should be clear when no human is actively watching, and auto-approval must
  remain bounded by hard cost, runtime, and resource caps.

### 17.6 Search Policy And Task Coverage

- Assignment visibility policy is still coarse. Future policies should express
  islands, explorers, cross-pollinators, exploiters, direction-local
  incumbents, and cross-direction reuse rules without hard-coding a workflow.
- `tasks/circle_packing_26` is the current converted benchmark task. Broad task
  migration remains future work.
- The Qwen 3.5 4B H200 kernel task exists as an RMSNorm MVP, but the task plan
  still lists unimplemented work: a pinned/containerized official H200
  environment, additional kernel definitions such as RoPE/attention/KV/logits/
  sampling, digest-keyed build/cache records, stronger import isolation,
  fallback coverage matrices, sampling statistical validation, repeated
  baseline calibration for distribution thresholds, and a live negative-control
  candidate for distribution-gate regression testing.

Open design questions that remain actionable:

- How portable should replay bundle restore be across machines and artifact
  stores once the local replay bundle format is no longer enough?
- Which telemetry interface is minimal enough to support MLflow first and later
  Trackio, TensorBoard, W&B, plain logs, and task-specific dashboards?
- What small ML-style task best validates train/predict/evaluate, artifact
  registry, job service, telemetry, and hidden evaluator isolation without
  dominating the framework refactor?

These are future extensions on top of the current semantic control-plane
architecture. They should not reintroduce the retired filesystem-RPC/evaluator
socket design.

## 18. Extension Points

Add a task:

```text
tasks/<task_id>/task.py
tasks/<task_id>/public/TASK.md
tasks/<task_id>/public/public_contract.md
tasks/<task_id>/public/<candidate seed>
```

Add a worker backend:

```text
adapter/
  implement a new WorkerSession backend while keeping Assignment, Session,
  Context, Evaluation, Job, Artifact, Finding, and NotebookCheckpoint resources
  unchanged.
```

Add a job provider:

```text
control_plane/jobs.py
  add provider adapter behavior behind the existing Job resource contract.
```

Add remote artifact storage:

```text
control_plane/service.py
  extend artifact registration/storage while preserving Artifact records and
  manifests.
```

Extend telemetry:

```text
control_plane/
  add additional providers or artifact export behavior without replacing
  official evaluation.
```

Later telemetry providers such as Trackio, TensorBoard, W&B, plain logs, and
task-specific dashboards should remain provider adapters. They must not replace
server-owned official evaluation or leaderboard publication.

Add cloud job providers:

```text
control_plane/jobs.py
  add providers such as HF Jobs, AWS, Slurm, SkyPilot, Modal, or other compute
  systems behind the existing Job resource contract.
```

Provider integrations should not become task logic. A task should be able to use
RunPod, HF Jobs, AWS, Slurm, SkyPilot, or Modal through experiment/job policy
without changing its candidate or evaluation contract.

Add trace observability:

```text
control_plane/
  AgentTraceService registration and read-only query/search are implemented.
  TraceExportRun plus local-jsonl and OTLP export providers are implemented as
  control-plane-only observability mirrors. Remaining work is Phoenix-compatible
  and Helicone-compatible sinks plus OTLP production hardening while preserving
  local immutable trace bundles as the source of truth.
```

Extend shared tools:

```text
control_plane/
  harden SharedTool resource with trust policy, deprecation, richer dependency
  metadata, and richer trace provenance links.
worker_tools/
  tool publish/list/show/checkout/install are implemented.
adapter/semantic_workspace.py
  checked-out tools materialize under shared_tools/.
```

Extend knowledge:

```text
task packages
  optional public/knowledge/ file tree with task-defined layout.
control_plane/
  exposes task-owned knowledge inventory through task contracts and materializes
  canonical task-context snapshots with provider-specific read-only enforcement.
  Future work can add content search/previews for PDFs without changing worker
  access.
worker_tools/
  no dedicated knowledge command; workers read task/knowledge/ directly.
```

Extend network control:

```text
adapter/
  separate control_plane_network and external_internet policy is carried into
  worker startup metadata.
control_plane/
  policy status, access event records, Docker relay enforcement, and Docker
  outbound audit proxy are implemented; future work is non-Docker
  proxy/firewall/App Server allowlist enforcement.
```

Extend environment providers:

```text
control_plane/environments.py
  add or harden providers behind EnvironmentService without changing Task,
  Worker, Evaluation, or Job APIs.

local_venv provider
  current default implementation.

docker_image provider
  implemented: build/resolve image, resolve digest, record lock_json, create
  Docker overlays, and run evaluations/jobs through provider run plans with
  controlled mounts, digest-pinned execution, and policy records.

future provider hardening
  signed provenance checks, SBOM/attestation validation, registry credential
  policy, cross-machine replay restore, and broader local-job environment
  enforcement.
```

## 19. Validation Commands

Current baseline validation:

```bash
python3 -m compileall src/agentic_opt
PYTHONPATH=src python3 -m unittest tests.test_web_backend tests.test_semantic_workspace tests.test_circle_packing_task -v
PYTHONPATH=src python3 -m agentic_opt.adapter.semantic_worker --help
PYTHONPATH=src python3 -m agentic_opt.worker_tools.semantic_cli --help
PYTHONPATH=src python3 -m agentic_opt.worker_tools.semantic_cli attempt --help
PYTHONPATH=src python3 -m agentic_opt.worker_tools.semantic_cli env --help
PYTHONPATH=src python3 -m agentic_opt.worker_tools.semantic_cli tool --help
PYTHONPATH=src python3 -m agentic_opt.worker_tools.semantic_cli network --help
PYTHONPATH=src python3 -m agentic_opt.web.app --help
```
