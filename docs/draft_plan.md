# Agentic Optimization with CLI Coding Agents

## Abstract

Recent progress in frontier language models has made CLI coding agents
increasingly capable of long-horizon program modification, debugging, and
iterative improvement. Yet in many existing optimization systems, the model
remains confined to candidate generation, while a hand-designed outer loop
determines how prior solutions are sampled, selected, mutated, and retained.

We propose an agentic optimization framework that shifts substantially more of
this search behavior back to the coding agent itself. The framework consists of
a CLI coding agent, a task API, a server-owned control plane, semantic worker
tools, managed runtime environments, and an artifact store. The control plane
owns experiments, assignments, sessions, evaluations, artifacts, jobs,
environments, findings, notebook checkpoints, telemetry, shared tools, task
knowledge, network policy, leaderboard entries, and incumbents. Filesystem
storage remains important for artifacts, logs, workspaces, environment
directories, relay sockets, export bundles, and task packaging; semantic state
is represented by server records.

The outer loop stays thin. It defines the optimization objective, starts worker
sessions, allocates budgets, enforces policy, records events, and owns
evaluation. The coding agent is given tool-mediated freedom over what prior
information to inspect, what solution to inherit or modify, when to run
verification or probes, when to submit for official scoring, what findings to
share, and what reusable tools to publish. The score interface separates
hard-constraint verification, cheap diagnostic probing, and authoritative
evaluation. Rather than requiring a fixed reflection phase inside the same
conversation, the runtime favors fresh autonomous sessions whose continuity
comes from server-visible context, artifacts, findings, notebook checkpoints,
leaderboards, incumbents, task knowledge, and shared tools.

The central question is whether a minimally scaffolded outer loop can better
expose and amplify the optimization capabilities of modern coding agents.

## 1. Introduction

As frontier language models improve at coding, the role of the model is
shifting from passive code generation to active software iteration. Modern
coding agents can inspect repositories, edit files, run shell commands, analyze
outputs, and revise implementations over extended interactions. This makes them
natural candidates for optimization settings in which progress depends on
repeated cycles of inspection, modification, validation, and reflection.

Many recent systems already use language models to improve programs, but the
model is often embedded inside a fixed outer-loop search procedure. The outer
loop determines how parents are selected, how mutations are formed, what
history is exposed, and when evaluation is triggered. In such systems, the model
acts as a powerful generator inside a largely hand-designed search scaffold.

Our goal is to push further in the agentic direction. The coding agent should
not be limited to producing a candidate from a prescribed prompt. Instead, it
should receive semantic access to a structured control plane containing prior
evaluations, artifacts, diagnostics, findings, tools, notebook checkpoints, and
task-provided knowledge. The agent decides what to inspect, what to compare,
what to modify, when to run checks, when to submit, and what lessons or tools to
preserve.

The central hypothesis is that modern coding agents already possess more
optimization ability than current outer-loop designs typically allow them to
express. If the outer loop is reduced to objective definition, policy,
budgeting, verification, evaluation, and durable state, then more of the actual
search behavior can emerge from the agent itself.

## 2. Framework Overview

The current framework is server-first. Its core components are:

- **Coding agent and adapter.** A CLI coding agent, initially Codex through the
  App Server integration, acts as the inner-loop search actor. The adapter is
  intentionally thin: it starts sessions and turns, supplies workspace policy
  and instructions, and captures raw execution traces.
- **Task API.** A task package defines public instructions, candidate contract,
  runtime requirements, verification, diagnostic probes, and authoritative
  evaluation. Private grader code and hidden assets remain outside the worker's
  readable workspace.
- **Control plane.** A Flask + SQLite service owns semantic state. Workers do
  not scrape archive directories for meaning; they call server APIs through
  semantic CLI tools.
- **Semantic worker tools.** Worker-facing commands include `ctx`, `artifact`,
  `eval`, `finding`, `notebook`, `job`, `env`, `telemetry`, `tool`,
  `knowledge`, and `network`. The accepted next command surface is `trace`.
- **Environment providers.** Runtime environments are controlled resources.
  The current default provider is `local_venv`; `docker_image` remains the
  target provider for reproducible containerized workers and evaluation.
- **Artifact and object storage.** Filesystem storage is still used for
  artifacts, job logs, telemetry logs, workspaces, and exported bundles, but
  these files are referenced by control-plane records.

This design keeps the coding agent as the search actor while making the outer
system responsible for durable state, policy, and reproducibility.

## 3. Task API and Evaluation Safeguards

The task interface exposes three public evaluation surfaces.

The verifier checks hard constraints and performs cheap screening. It answers
questions such as whether the candidate has the required interface, compiles,
passes small sanity checks, and is worth sending to a more expensive evaluator.

Diagnostic probes provide cheap process feedback that is richer than a
pass/fail verifier but less authoritative than final scoring. A probe may expose
repair-oriented signals, partial metrics, or task-specific diagnostics. This is
useful because many optimization failures are near misses whose structure can
guide the next local change.

The evaluator performs the authoritative measurement. For official submissions,
the control plane snapshots the candidate into an artifact before scoring. A
valid completed official evaluation can update the leaderboard and the
experiment incumbent.

To reduce reward hacking, evaluator internals, hidden tests, private benchmark
assets, and scoring code remain outside the agent-readable workspace. The agent
submits candidates through `eval` and observes only public feedback. This is an
explicit system boundary, not a prompt-level instruction.

Current implementation detail: `verify` and `probe` may run with approved
worker environment overlays, while official `submit` defaults to the task base
environment. Overlay-backed official submission is a future policy/runner
extension.

## 4. Control Plane, Artifacts, and Memory

The earlier prototype framed the system as a filesystem-backed archive. The
active design is different: the database is the semantic source of truth, while
filesystem artifacts are durable blobs referenced by server records.

Important current resource types include:

- `Experiment`
- `WorkerAssignment`
- `WorkerSession`
- `Environment`
- `EnvironmentOverlay`
- `Artifact`
- `Evaluation`
- `LeaderboardEntry`
- incumbent state
- `Job`
- `Finding`
- `NotebookCheckpoint`
- `TelemetryRun`
- `SharedTool`
- `KnowledgeItem`
- `NetworkAccessEvent`

Historical attempts are not yet represented by a dedicated `Attempt` table.
Today, candidate history is primarily represented by candidate artifacts,
evaluations, leaderboard entries, incumbents, events, notebook checkpoints, and
job/telemetry records. A first-class attempt/run model remains an open design
target.

Agent-authored memory is deliberately split into several forms:

- `WORKLOG.md` is local scratch state in the worker workspace.
- `notebook checkpoint` turns useful local notes into server-visible
  `NotebookCheckpoint` records.
- `finding share` creates durable agent-authored findings, including reusable
  patterns, hypotheses, failures, and insights.
- `tool publish` creates artifact-backed shared tools that later workers can
  discover and checkout.

Task-provided knowledge is separate from agent-authored memory. A task may
include curated read-only materials under `public/knowledge/` with a manifest.
These become `KnowledgeItem` records and are accessed through the `knowledge`
tool. Knowledge is part of the task definition, not a loophole around network
policy and not a worker-generated finding.

## 5. Semantic Tools

The worker should access server-owned state through semantic commands rather
than raw filesystem archaeology. The active command surface is:

```text
ctx         assignment, task, findings, artifacts, evaluations, jobs,
            environments, leaderboard, incumbent, telemetry, knowledge,
            shared tools, and network state
artifact    upload durable files/directories and checkout incumbents
eval        verify, probe, submit, status, wait
finding     share or search reusable agent-authored findings
notebook    checkpoint local notebook/worklog state
job         launch durable local, Docker, or provider-backed jobs
env         inspect task environments and request dependency overlays
telemetry   record non-official metrics and run metadata
tool        publish, list, show, checkout, and install shared tools
knowledge   list, show, and materialize task-provided context
network     inspect external internet policy and access events
```

These are capabilities, not a fixed workflow. The worker startup prompt and
skills describe the tools, but they do not impose a numbered algorithm.

The accepted near-term missing command is `trace`. Raw Codex/App Server traces
are currently written under worker workspaces, but there is not yet a
server-owned `AgentTraceBundle` resource with immutable trace artifacts and
export providers.

## 6. Environment Control

Environment control is a central part of the framework. Tests, controller code,
worker tools, task code, jobs, and evaluation should run against a coherent
declared environment instead of whichever Python happens to be first on the host
`PATH`.

Tasks declare runtime requirements through `TaskRuntimeSpec`. The control plane
resolves those declarations into `Environment` records. The current default
provider is `local_venv`, which prepares a local virtual environment, installs
declared requirements, verifies required imports, records lock information, and
exports runtime variables to worker tool wrappers.

Worker-requested dependency changes go through environment overlays. A worker
can request an overlay with `env install`; the control plane records the
requirements, applies policy, and prepares a separate environment when allowed.
This avoids silently mutating the task base environment used for official
scoring.

The target long-term provider is `docker_image`. It should use the same
`Environment` and `EnvironmentOverlay` resources while building or pulling
immutable image digests for reproducible worker and evaluation execution.

## 7. Network Control

Network policy must separate semantic control-plane access from public internet
access.

```text
control_plane_network
  Required for semantic tools such as ctx, eval, artifact, finding, notebook,
  job, env, telemetry, tool, knowledge, and network.

external_internet
  Experiment policy controlling live web search, arbitrary HTTP clients,
  package downloads outside environment policy, and answer lookup against
  public internet sources.
```

The current `codex-local` App Server path still needs coarse network access so
semantic tools can reach the control plane. When an experiment denies external
internet under this backend, the run is marked as policy-weakened rather than
pretending to be isolated.

Docker-backed jobs are stricter. Under `external_internet=deny`, local Docker
jobs run with `docker run --network none` and requests for broad networking
such as `bridge` or `host` are rejected. If a Docker-backed job or worker also
needs semantic tool access, the system uses a Unix-socket control-plane relay:
the container mounts only the relay socket, `AO_CONTROL_API_URL` points at a
`unix://...` URL, and the relay forwards only control-plane API paths to the
Flask server. This provides a narrow control-plane channel without granting
general internet access.

Network access policy and enforcement status are exposed through the `network`
tool and recorded in `NetworkAccessEvent` records.

## 8. Iteration Lifecycle

An optimization run is organized as a sequence of autonomous worker sessions,
but the framework should avoid encoding a fixed numbered workflow into prompts
or controller logic. A typical session shape is:

1. The user creates an experiment with task id, budget, config, and policy.
2. The control plane generates worker assignments, optionally assigning
   task-defined research directions.
3. A worker session starts and prepares a semantic workspace.
4. The task base environment is resolved through `EnvironmentService`.
5. The workspace receives semantic CLI wrappers, AGENTS.md, local skills,
   candidate seed files, knowledge/shared-tool directories, and runtime/network
   exports.
6. The agent reads selectively through `ctx`, `knowledge`, `finding`,
   `artifact`, `leaderboard`, `incumbent`, and related tools.
7. The agent edits candidates, runs local checks, requests verify/probe
   feedback, launches jobs, checkpoints notebooks, shares findings, publishes
   tools, and submits when useful.
8. Official submit snapshots the candidate as an artifact before scoring.
9. Completed valid official evaluations update leaderboard and incumbent state.
10. Later sessions continue from server-visible state rather than unbounded
    same-thread conversation history.

This is descriptive, not prescriptive. The agent remains responsible for
choosing the actual search behavior.

## 9. Implementation Strategy

The active implementation lives under `src/agentic_opt/`:

```text
common/          Low-level utilities.
control_plane/   Repository, services, environment control, jobs, evaluation,
                 telemetry, shared tools, task knowledge, network policy,
                 and provider adapters.
worker_tools/    Agent-facing semantic CLI.
adapter/         Codex/App Server integration, workspace preparation, and the
                 thin semantic worker loop.
web/             Flask backend and route layer.
task_api.py      Task protocol and candidate contract.
task_registry.py Task loading boundary.
```

The control plane is currently Flask + SQLite. The local filesystem stores
artifact content, job logs, telemetry logs, environment directories, worker
workspaces, and relay sockets. The SQLite database records the semantic state
that makes these files meaningful.

### 9.1 Codex Integration

For Codex, the framework uses the App Server rather than `codex exec` as the
primary integration surface. The App Server supports long-lived sessions,
turns, streamed item events, sandboxed command execution, interruption, and raw
trace capture, which are a better fit for long-horizon optimization loops than
a one-shot scripting interface.

The Codex adapter treats a worker session as an autonomous turn inside a
prepared semantic workspace. The workspace includes:

- generated `AGENTS.md`
- `.agents/skills/` with semantic tool guidance
- `bin/` wrappers for server tools
- task runtime environment exports
- network policy exports
- `WORKLOG.md`
- `local_tools/`, `shared_tools/`, `knowledge/`, `artifacts/`, and other
  workspace directories

The adapter should rely on sandbox policy, filesystem permissions, environment
control, server APIs, and provider-level network enforcement rather than prompt
obedience. Hidden grader internals and canonical durable records remain behind
the control plane.

## 10. Experimental Plan

We plan to study three classes of problems.

The first class contains lightweight optimization tasks with strict requirements
and relatively cheap verification. `tasks/circle_packing_26` is the current
concrete task for end-to-end system hardening. It exercises runtime dependency
control, candidate verification/probing/submission, leaderboard/incumbent
updates, findings/notebook behavior, multi-agent direction assignment, network
contamination controls, and reusable tools.

The second class contains benchmark-driven optimization tasks, where the score
function is a standardized public benchmark rather than a bespoke internal
metric. Representative examples include GPU kernel optimization, compiler
optimization, and solver-engineering benchmarks. The PPBench harness tasks in
this repository are examples of task packaging and benchmark loading patterns.

The third class contains domain-specific tasks with custom score functions.
Examples include internal repository CI repair, project-level memory and
retrieval optimization, incident-response assistants, and workflow agents whose
objectives cannot be fully captured by an existing benchmark.

Across these categories, we care about:

- how much useful search behavior can be delegated to the coding agent
- whether verify/probe/submit decomposition reduces benchmark cost
- whether hidden graders reduce reward hacking pressure
- whether server-owned findings, notebooks, artifacts, and shared tools support
  long-horizon improvement
- whether multi-agent search maintains exploration breadth instead of
  collapsing into one direction
- whether environment and network controls prevent misleading results
- whether traces are complete enough to reconstruct how results were found

## 11. Scope and Extensions

The framework begins with autonomous sessions over server-owned durable state,
but controlled multi-agent breadth is also a first-class experimental setting.
Different worker assignments may receive fixed human-written research
directions or coarser islands of related methods. In the current task packaging
model, a task can expose these directions through
`public/research_directions/manifest.json`, and assignment generation can
distribute workers across those directions.

This makes exploration diversity measurable. A simple first statistic is the
Shannon entropy of active workers over directions or islands. Low entropy
indicates that parallel agents have collapsed into only a few areas; high
entropy indicates that the run is maintaining breadth. Exploiter or replication
agents can be useful for chasing the frontier, but they should be analyzed
separately from explorers because their purpose is not diversity.

Natural extensions include richer frontier maintenance, asynchronous parallel
agents, stronger visualization dashboards, conditional supervisory or
critic-style interventions during stagnation, full trace bundle export,
containerized worker/evaluation providers, and post-run analysis of which method
families were actually explored. These extensions should not replace the core
principle: the outer loop provides infrastructure and constraints, while the
coding agent remains the primary search actor.

## 12. Current Gaps

Important gaps remain:

- No dedicated `Attempt` table yet.
- No first-class `AgentTraceBundle` service yet. Raw Codex traces exist, but
  server-indexed immutable trace bundles and exporters remain future work.
- `docker_image` is still a planned environment provider, although Docker job
  network enforcement and the Unix-socket control-plane relay exist.
- `codex-local` cannot truly block public internet while preserving App Server
  control-plane access; denied-internet local runs are marked policy-weakened.
- Full Docker worker runner integration remains future work.
- UI and post-run analysis dashboards are still incomplete.
