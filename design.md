# Agentic Optimization Current Architecture

This document describes the current architecture of this repository after the
control-plane refactor and cleanup. It is an implementation design document
with accepted near-term design changes called out explicitly when the code is
not there yet. Planned extensions are called out as gaps.

The current system is a server-first autonomous optimization framework. The
server owns semantic experiment state and official evaluation. Worker sessions
are disposable coding-agent runtimes attached to server-owned assignments.

## 1. Architectural Principles

- The server is the source of truth for experiments, assignments, sessions,
  environments, jobs, evaluations, artifacts, findings, notebook checkpoints,
  and events.
- Workers perform autonomous research behavior, but they do not own official
  experiment state.
- Worker-facing capabilities are semantic operations, not raw filesystem
  archive traversal.
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
- Task-provided background material is a knowledge resource. Papers, notes,
  datasets, and reference files packaged with the task should be read-only
  context available through semantic tools rather than hidden in ad hoc paths.
- Task packages define domain contracts. The framework should not hard-code
  one benchmark, one ML workflow, or one metric.
- Filesystem paths remain useful as artifact URIs and workspace materialization
  details, but they are not the architecture.
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
      ctx, artifact, eval, finding, notebook, job, env, telemetry,
      tool, knowledge, network
    Accepted next tools:
      trace

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
cp_knowledge_items
cp_network_access_events
```

The repository stores JSON payloads for config, budget, policy, metadata,
inputs, outputs, request bodies, results, public feedback, metrics, links, and
event payloads. This keeps the current schema simple while still preserving
structured resource state.

Accepted near-term schema addition:

```text
cp_agent_traces
```

The implemented shared-tool, knowledge, and network-event tables provide the
resource boundary for agent-authored reusable tools, task-packaged read-only
knowledge, and network policy audit state. Agent traces still need a dedicated
server-indexed table/resource.

### 4.2 Resource Model

Current first-class resources in the implemented schema:

```text
Task
Experiment
WorkerAssignment
WorkerSession
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
KnowledgeItem
NetworkAccessEvent
```

Accepted near-term first-class resources:

```text
AgentTraceBundle
```

`Attempt` is not yet a dedicated table. The current implementation represents
attempt-like candidate states through artifact and evaluation records with
optional `attempt_id` fields. A future dedicated Attempt resource can be added
without changing the worker-facing tool model.

`LeaderboardEntry` is the official score publication resource. The current
incumbent for an experiment, task, or research direction is the highest active
leaderboard entry in that scope.

`Environment` and `EnvironmentOverlay` are first-class resources. The active
code has the `local_venv` provider path behind `EnvironmentService`; the missing
provider is `docker_image`.

`Finding` is the durable agent-authored finding resource. Historical "patterns"
are represented as findings, usually by choosing a `finding_type` such as
`pattern`, `insight`, `hypothesis`, `result`, or `error`. Curated task-provided
background material belongs in `KnowledgeItem`, not `Finding`.

`AgentTraceBundle` is the accepted resource for coding-agent audit trails. The
current Codex adapter already writes raw App Server events and summarized output
under each workspace's `.run/traces/` directory, but that trace is not yet a
server-indexed immutable resource. The target design registers every turn trace
as an artifact-backed record linked to experiment, assignment, session, worker
backend, run id, turn id, model, sandbox policy, environment, and network
policy.

`SharedTool` is the accepted resource for agent-authored reusable executable
helpers. A worker may draft code in `local_tools/`, but useful tools should be
published into a server-owned registry with digest, version, declared runtime
requirements, owning task or experiment scope, documentation, and provenance
links to the session trace that created them.

`KnowledgeItem` is the accepted resource for read-only task context packaged
with the task. It covers files such as papers, notes, reference implementations,
dataset descriptions, and benchmark background that the task author
intentionally provides to agents. Knowledge differs from findings: findings are
agent-authored during the experiment, while knowledge is curated task input
material defined at task-definition time.

`NetworkAccessEvent` records external network attempts when a provider can
observe or proxy them. The policy distinction is not "network on/off"; it is
"semantic control-plane access allowed" versus "external internet access
allowed, denied, or audited."

### 4.3 ControlPlaneService Responsibilities

`ControlPlaneService` coordinates the active services below. `EnvironmentService`
is now the server-owned environment boundary for task base environments and
worker overlays. The remaining environment-provider gap is `docker_image`.

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
- future `AgentTraceService` for trace bundle registration and telemetry export
- future `SharedToolService` for publishing, listing, checking out, and
  installing reusable worker-authored tools
- future `KnowledgeService` for indexing task-packaged knowledge and
  materializing read-only knowledge files into worker workspaces
- future network-policy enforcement surface that separates local control-plane
  access from external internet access

Official evaluation state lives in `cp_evaluations`. Long-running submit
evaluations are queued as jobs that run
`agentic_opt.control_plane.evaluation_worker`. Completed valid submit/official
evaluations create `LeaderboardEntry` rows and may update the incumbent.

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
POST /api/v1/environment-overlays
GET  /api/v1/environment-overlays
GET  /api/v1/environment-overlays/<overlay_id>
POST /api/v1/environment-overlays/<overlay_id>/approve

GET  /api/v1/tasks/<task_id>

GET  /api/v1/experiments
POST /api/v1/experiments
GET  /api/v1/experiments/<experiment_id>
PATCH /api/v1/experiments/<experiment_id>
POST /api/v1/experiments/<experiment_id>/assignments
POST /api/v1/experiments/<experiment_id>/assignments/generate

GET  /api/v1/assignments/<assignment_id>
PATCH /api/v1/assignments/<assignment_id>
POST /api/v1/assignments/<assignment_id>/sessions
POST /api/v1/assignments/<assignment_id>/start-local

PATCH /api/v1/sessions/<session_id>

GET  /api/v1/context?assignment_id=...

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

GET  /api/v1/knowledge?task_id=...
GET  /api/v1/knowledge/<knowledge_id>
POST /api/v1/knowledge/<knowledge_id>/materialize

GET  /api/v1/network-policy
POST /api/v1/network-access-events
GET  /api/v1/network-access-events
```

The web layer is intentionally thin. It validates request shape lightly,
delegates to the repository/service layer, and returns JSON.

The environment routes are implemented for the `local_venv` provider. The
remaining provider-level gap is `docker_image`, which should use the same
routes and resource records.

Accepted near-term routes to add:

```text
POST /api/v1/agent-traces
GET  /api/v1/agent-traces
GET  /api/v1/agent-traces/<trace_id>
POST /api/v1/agent-traces/<trace_id>/export
```

The `agent-traces` routes should return artifact-backed trace bundle metadata,
not just control-plane events. The `export` route should support providers such
as OpenTelemetry OTLP, Arize Phoenix, Helicone, local JSONL, or another
experiment-configured sink. The exact provider adapter should be selected by
experiment policy, not hard-coded into worker prompts.

## 6. Worker Plane

The current worker backend is Codex through the App Server adapter.

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
future `trace`, `tool`, or `knowledge` commands.

The local `codex-local` implementation can keep the coarse App Server network
flag enabled only to preserve localhost control-plane access, but it should
label the session as "external internet not technically enforced" unless a
provider can enforce a localhost allowlist. Docker-backed execution must not use
that weakening path: when `external_internet=deny`, Docker jobs run with
`--network none`, and requests for `bridge` or `host` networking are rejected.
A Docker worker that needs both semantic tool access and denied public internet
uses the server-owned Unix-socket control-plane relay. The container mounts only
that socket, semantic tools use `AO_CONTROL_API_URL=unix://...`, and the relay
forwards only control-plane API paths to the Flask server.

The target implementations are:

- App Server sandbox support for localhost/control-plane allowlists.
- A semantic-tool transport that does not require general network access, such
  as stdio, a Unix-domain socket, or a broker process mounted inside the
  workspace.
- A Docker control-plane relay or network proxy/firewall wrapper that allows
  the control-plane endpoint and denies or logs all other outbound
  destinations. The Unix-socket relay path is implemented first.

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
.agents/skills/
  artifact-use/
  context-use/
  environment-use/
  evaluation-use/
  finding-use/
  job-use/
  notebook-use/
  telemetry-use/
reference/
artifacts/
findings/
local_tools/
candidate entrypoint copied from task public seed
```

The tool wrappers call `agentic_opt.worker_tools.semantic_cli` using the
resolved task base Python. Worker overlays are separate environment resources;
the current workspace wrapper environment is not hot-swapped after overlay
creation. The workspace does not materialize `fs` or `ve` wrappers.

The startup prompt tells the agent that the server owns experiments,
assignments, environments, artifacts, jobs, evaluations, leaderboard/incumbent
state, findings, notebook checkpoints, and policy. It presents semantic tools as
capabilities, not as a required sequence.

Accepted target workspace additions:

```text
knowledge/
  read-only materialized knowledge items from the task package
shared_tools/
  checked-out published shared tools selected by the worker or experiment
```

`local_tools/` remains a writable scratch area for worker-authored helper code.
Publishing a tool must copy it into the shared tool registry and produce a
server-owned `SharedTool` record. `knowledge/` is materialized from
`tasks/<task_id>/public/knowledge/` and must be treated as read-only task
context; workers can cite or inspect it, but should not mutate it as part of a
candidate.

## 8. Worker Tools

The semantic CLI is implemented in `worker_tools/semantic_cli.py`.

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

artifact upload --path <path> --kind <kind>
artifact list
artifact checkout-incumbent --destination <path> [--direction-id ID] [--force]

eval verify (--entry <path>|--artifact-id <artifact-id>) [--environment-id ID] [--environment-overlay-id ID] [--sync|--async]
eval probe (--entry <path>|--artifact-id <artifact-id>) [--kind diagnostics] [--environment-id ID] [--environment-overlay-id ID] [--sync|--async]
eval submit (--entry <path>|--artifact-id <artifact-id>) [--environment-id ID] [--sync|--async]
eval status <evaluation-id>
eval wait <evaluation-id> [--timeout-s N]

finding share --type <type> --title <title> --body <text>
finding share --type <type> --title <title> --file <path>
finding search <query>

notebook checkpoint (--file WORKLOG.md|--content <text>) [--kind <kind>]
notebook list

job create --provider local --command '<command>' [--cwd <path>] [--env KEY=VALUE]
job create --provider local-docker --image <image> --command '<command>' [--cwd <path>] [--network-mode <mode>] [--requires-control-plane]
job create --provider runpod --template-id <template> --command '<command>' [--gpu-type-id <id>] [--gpu-count N] [--dry-run]
job list
job status <job-id>
job logs <job-id> [--max-bytes N]
job wait <job-id> [--timeout-s N]
job cancel <job-id>

telemetry start --provider local --name <run-name>
telemetry start --provider mlflow --name <run-name>
telemetry log-metrics <telemetry-id> --metric loss=0.1 --step 1
telemetry status <telemetry-id>
telemetry list
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
`finding share`.
Incumbent discovery should go through `ctx leaderboard` / `ctx incumbent`, and
candidate reuse should go through `artifact checkout-incumbent` rather than
guessing artifact paths.

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

knowledge list [query]
knowledge show <knowledge-id>
knowledge materialize <knowledge-id> [--destination knowledge/<name>]

network status
network policy
network events
```

Accepted next trace command surface:

```text
trace status
trace bundle [--include-workspace-manifest] [--include-codex-home]
trace export --provider local-jsonl|otlp|phoenix|helicone [--trace-id ID]
```

`tool` is for reusable executable helpers; durable claims about what a tool
showed still belong in `finding`. `knowledge` is for curated read-only context
that was packaged as part of the task before the current worker turn. `network`
lets a worker inspect whether external internet access is allowed, denied, or
only audit-logged. `trace` remains the accepted next command for making the
worker's own execution history visible and exportable.
`network` lets a worker inspect whether external internet access is allowed,
denied, or only audit-logged.

### 8.1 Shared Tools

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
  `--network none`, rejects `bridge`/`host` overrides, and blocks commands that
  require direct control-plane access until a relay provider exists.

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
- cancellation by PID where possible
- basic approval and cost gates
- max-job budget gate
- experiment-owned auto-approval with hard cost caps
- RunPod capacity errors classified as retryable provider failures

Jobs should run under a declared environment. Local jobs should default to the
assignment's selected worker environment overlay when launched from a worker,
or to the task base environment for server-owned evaluation jobs. Provider jobs
should record the same environment id plus provider-specific image or bootstrap
details. This is not fully enforced by the current implementation yet.

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

Async submit creates an Evaluation record, launches a local evaluation job, and
updates the Evaluation after `evaluation_worker` completes. Public-safe feedback
is stored separately from the full result payload. Completed valid
submit/official evaluations create a `LeaderboardEntry`; the highest active
entry is the incumbent for that experiment/task/direction scope.

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

The current implementation does not yet provide strong sandboxing for hidden
evaluator internals. The design boundary is present; stronger isolation belongs
in future task/runtime/provider work.

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

Environment fingerprints and dependency locks are currently available through
the linked `Environment`/`EnvironmentOverlay` records rather than copied into
the leaderboard row. Copying an immutable environment lock summary into
leaderboard metadata remains a hardening task for official reproducibility.

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

The current implementation writes raw App Server events and summarized output
into `.run/traces/<run_id>/<turn_id>/`, and Codex also writes its rollout JSONL
under `.codex-home/sessions/`. The accepted target is to register those files
as a server-owned `AgentTraceBundle` artifact with a digest and database row.
Every trace bundle should be linked to the worker session and referenced from
events, evaluations, artifacts, findings, shared tools, and leaderboard entries
when relevant.

Trace export should be optional and provider-backed. The framework should
support at least:

```text
local-jsonl
  Write normalized spans/events to state_root/trace_exports/.

otlp
  Export OpenTelemetry-compatible spans/events through an OTLP endpoint.

phoenix
  Export LLM/tool-call traces to an Arize Phoenix-compatible sink when
  configured by experiment policy.

helicone
  Export LLM request/response metadata to a Helicone-compatible sink when the
  model access path supports it.
```

These exports are observability mirrors, not the source of truth. The source of
truth remains the control-plane database plus immutable trace/artifact blobs.
If an external telemetry provider is unavailable, the local trace bundle must
still be complete.

Trace export must respect privacy and experiment policy. Hidden evaluator
internals, secrets, private datasets, and denied network destinations should be
redacted or omitted before export. The redaction decision should be recorded in
trace metadata.

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
  use a Unix-socket control-plane relay instead of broad Docker networking.
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
  manifest.json
  papers/
  notes/
  references/
  data_descriptions/
```

Knowledge files are intentionally worker-visible task context. Examples include
papers, survey notes, prior-method descriptions, domain references, API
manuals, or public benchmark background that the task author packaged with the
task. They should be indexed as `KnowledgeItem` records so workers can list,
search, show, and materialize them through semantic tools instead of relying on
raw directory traversal.

Knowledge bundle manifest fields:

```json
{
  "items": [
    {
      "knowledge_id": "paper_foo_2024",
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
circle-packing papers and a manifest. A worker can call `knowledge list` or
`knowledge materialize paper_x` to inspect them during optimization, but those
files remain read-only task context and are not confused with agent-authored
findings or shared tools.

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
The control-plane environment module owns durable records and worker overlay
support. The next provider to add behind the same module is `docker_image`.

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

The long-term provider for reproducible official evaluation is `docker_image`.
The same task-level dependency declaration should be resolvable into either a
local virtualenv or a Docker image:

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
The rest of the control plane should not branch on venv-vs-Docker except at the
execution adapter boundary. Some provider-interface methods such as a generic
`run_in_environment` are still design targets rather than implemented methods.

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

Provider interface:

```text
EnvironmentProvider.prepare(spec) -> PreparedEnvironment
EnvironmentProvider.exports(environment) -> dict[str, str]
EnvironmentProvider.run(environment, command, cwd, env, mounts) -> ProcessHandle
EnvironmentProvider.lock(environment) -> dict
```

`local_venv` provider:

- default provider during the current rewrite
- selects a base Python that satisfies the task version constraint
- verifies that the base Python can create a venv with pip
- installs declared Python requirements into an immutable task venv
- runs import and public-seed preflights before marking the environment ready
- captures `pip freeze` into `lock_json`
- exposes `python_path` for `EvaluationService`, `semantic_worker`, and tool
  wrappers. Local worker-created jobs can carry environment metadata, but
  default job execution is not yet fully environment-enforced.

`docker_image` provider:

- long-term provider for CI, official evaluation, and cross-machine
  reproducibility
- builds or pulls an image from the environment spec
- records immutable image identity in `lock_json`, especially `image_digest`
- must enforce `external_internet=deny` with Docker/network isolation below the
  agent process, not just with prompts
- must use a control-plane relay or equivalent allowlist transport when a
  Docker worker needs semantic tools while public internet is denied
- runs evaluations, jobs, and worker tools through a Docker runner instead of
  host `sys.executable`
- mounts only explicit workspace, artifact, state, and task-public paths
- records runner metadata such as image ref, image digest, mount list,
  working directory, user, network policy, and resource limits
- still uses the same `Environment` and `EnvironmentOverlay` resources as the
  venv provider

The Docker runner should use image digests for official evaluation. Tags are
acceptable as input convenience, but the prepared environment is not ready until
the digest has been resolved and stored.

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
- Docker image provenance, digest pinning, and registry access
- network access
- reproducibility and lock-file capture

Blocked environment requests should become durable records with public-safe
reasons, just like blocked jobs.

### 14.7 Official Evaluation Rule

Official evaluation defaults to the task base environment.

A worker overlay may be used for `verify` or `probe` when policy allows it.
Using an overlay for `submit` is not implemented in the current official path.
The CLI flag name reserved for this future path is
`--environment-overlay-id`, but current `EvaluationService` allows overlays
only for `verify` and `probe`; `submit` resolves to the task base environment.

Leaderboard-eligible evaluations must record:

```text
environment_id
environment_fingerprint
overlay_id, if any
environment_provider
lock_json or equivalent dependency snapshot
```

Current implementation records `environment_id` and `environment_overlay_id` in
the evaluation request and leaderboard entry, with fingerprint/lock reachable
through the environment records. Copying the full immutable lock into the
evaluation/leaderboard result is still future hardening.

If a candidate requires an overlay dependency to run, the result is not fully
reproducible until that overlay is captured and approved. If the dependency is
useful beyond one assignment, the worker should share a finding or proposal;
a maintainer can then promote it into the task base `TaskRuntimeSpec`.

For `docker_image` official evaluation, the record must include the immutable
image digest and runner metadata. Re-running the same evaluation should be
possible from:

```text
candidate artifact
environment image digest
task version
evaluation request
```

### 14.8 Current Status and Gap

The repository now has the first `EnvironmentService` path for `local_venv`:
durable environment and overlay records, task base environment preparation,
environment-aware evaluation, and an `env` worker tool surface. This is the
current default implementation.

The remaining environment gap is the `docker_image` provider. It should be
added behind the same EnvironmentService interface without changing task,
worker, or evaluation APIs. Once Docker provider support exists, CI and
leaderboard-eligible official evaluation should prefer Docker image digests,
while local development can continue to use `local_venv` for speed.

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
  workspaces/
    <assignment_id>/
      <session_id>/
  telemetry/
    <telemetry_id>/
      metrics.jsonl
      params.json
      tags.json
  envs/
    tasks/
      <task_id>/
        <runtime_fingerprint>/  # local_venv provider storage
    overlays/
      <overlay_id>/             # local_venv overlay storage
```

Shared tools are currently stored as `Artifact` blobs with `kind=shared_tool`
and indexed in SQLite through `cp_shared_tools`. Knowledge files remain in the
task package under `tasks/<task_id>/public/knowledge/` and are indexed through
`cp_knowledge_items`; materialization copies them into worker workspaces on
demand. Network access events are currently SQLite records rather than JSONL
files. Dedicated `traces/` and `trace_exports/` state-root directories remain
future work for `AgentTraceBundle`.

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
   the resolved task environment, network policy, task knowledge materialization,
   and selected shared tool materialization.
9. Codex/App Server receives startup prompt and semantic tools.
10. Agent edits candidate code and uses ctx/eval/job/artifact/finding/notebook/telemetry/env.
    Accepted next command adds trace.
11. eval submit snapshots the candidate, then creates a server-owned Evaluation
    with environment metadata.
12. Async submit launches a server-owned Job running evaluation_worker in the
    resolved task environment.
13. Completed valid submit/official evaluations publish LeaderboardEntry rows
    and update the derived incumbent.
14. Job/evaluation/leaderboard/incumbent status can be inspected by any later
    worker session.
15. Findings, artifacts, notebook checkpoints, events, shared tools, network
    events, and indexed task knowledge records persist in SQLite and state_root
    artifacts. Codex trace files are still workspace-local until the accepted
    AgentTraceBundle resource is implemented.
```

## 17. What Is Not Implemented Yet

Current gaps:

- No dedicated Attempt table yet.
- No first-class `AgentTraceBundle` table/service yet. Codex traces are written
  under workspace `.run/traces/`, but they are not registered as immutable
  server-indexed trace artifacts or exported through telemetry providers.
- Network policy is enforceable for Docker-backed jobs through `--network none`
  plus the Unix-socket control-plane relay when semantic tools are required, but
  not yet for `codex-local` workers at the intended granularity. The current
  Codex/App Server path uses a coarse network switch to keep localhost semantic
  tools working. Full `docker_image` worker runner integration still remains.
- Environment control has a `local_venv` implementation, durable records, and
  worker-facing `env` tools. The missing provider is `docker_image`.
- Local worker budget expiry is represented as `stopped` with
  `stop_reason=turn_timeout`, not as a worker exception. The worker still writes
  partial traces and checkpoints `WORKLOG.md` when possible.
- `tasks/circle_packing_26` is the current converted benchmark task; broad task
  migration remains future work.
- No product-grade leaderboard publication UI yet; the JSON API and worker
  tools are implemented.
- No general artifact download API yet; incumbent checkout is implemented for
  local filesystem destinations.
- RunPod and S3-compatible storage have first provider adapters, but no
  production cloud workspace bundle orchestration/result polling flow yet.
- No Docker-backed official evaluation runner yet; hidden evaluator isolation is
  still process/environment based rather than container-image based.
- No product frontend after the cleanup; current web surface is JSON API.
- No broad task migration is part of the current baseline.

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

Add trace observability:

```text
control_plane/
  add AgentTraceService and provider exporters for local JSONL, OTLP,
  Phoenix-compatible sinks, and Helicone-compatible sinks while preserving
  local immutable trace bundles as the source of truth.
```

Extend shared tools:

```text
control_plane/
  harden SharedTool resource with trust policy, deprecation, richer dependency
  metadata, and trace provenance once AgentTraceBundle exists.
worker_tools/
  tool publish/list/show/checkout/install are implemented.
adapter/semantic_workspace.py
  checked-out tools materialize under shared_tools/.
```

Extend knowledge:

```text
task packages
  public/knowledge/manifest.json and files are indexed when present.
control_plane/
  task-owned KnowledgeItem records and read-only materialization are
  implemented; future work can add content search/previews for PDFs.
worker_tools/
  knowledge list/show/materialize are implemented.
```

Extend network control:

```text
adapter/
  separate control_plane_network and external_internet policy is carried into
  worker startup metadata.
control_plane/
  policy status and access event records are implemented; future work is actual
  proxy/firewall/App Server allowlist enforcement.
```

Add an environment provider:

```text
control_plane/environments.py
  add a provider behind EnvironmentService without changing Task, Worker,
  Evaluation, or Job APIs.

local_venv provider
  current default implementation.

docker_image provider
  build/pull image, resolve digest, record lock_json, and run evaluations/jobs
  through an explicit Docker runner with controlled mounts and policy.
```

## 19. Validation Commands

Current baseline validation:

```bash
python3 -m compileall src/agentic_opt
PYTHONPATH=src python3 -m unittest tests.test_web_backend tests.test_semantic_workspace tests.test_circle_packing_task -v
PYTHONPATH=src python3 -m agentic_opt.adapter.semantic_worker --help
PYTHONPATH=src python3 -m agentic_opt.worker_tools.semantic_cli --help
PYTHONPATH=src python3 -m agentic_opt.worker_tools.semantic_cli env --help
PYTHONPATH=src python3 -m agentic_opt.web.app --help
```
