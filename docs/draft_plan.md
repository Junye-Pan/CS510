# Agentic Optimization with CLI Coding Agents

## Abstract

Recent progress in frontier language models has made CLI coding agents increasingly capable of long-horizon program modification, debugging, and iterative improvement. Yet in many existing optimization systems, the model remains confined to candidate generation, while a hand-designed outer loop determines how prior solutions are sampled, selected, mutated, and retained. We propose an agentic optimization framework that shifts substantially more of this search behavior back to the coding agent itself. The framework consists of three core components: a CLI coding agent, a task-specific score interface, and a filesystem-backed memory and archive. The outer loop defines the optimization objective, enforces safety and immutability constraints, meters evaluation cost, and archives artifacts. The agent is given broad freedom over what prior information to inspect, what solution to inherit or modify, when to run lightweight validation, when to request diagnostic probes, when to submit for authoritative scoring, and what reusable knowledge to preserve for future sessions. The score interface separates hard-constraint verification, cheap diagnostic probing, and authoritative evaluation. The filesystem stores immutable historical attempts and evaluation outputs together with a durable but lightweight knowledge layer. Rather than requiring a fixed reflection phase inside the same conversation, the runtime favors fresh autonomous sessions whose continuity comes from worklogs, queryable history, and shared reusable patterns. The central question is whether a minimally scaffolded outer loop can better expose and amplify the optimization capabilities of modern coding agents.

## 1. Introduction

As frontier language models improve at coding, the role of the model is shifting from passive code generation to active software iteration. Modern coding agents can inspect repositories, edit files, run shell commands, analyze outputs, and revise implementations over extended interactions. This makes them natural candidates for optimization settings in which progress depends on repeated cycles of inspection, modification, validation, and reflection.

Many recent systems already use language models to improve programs, but the model is often still embedded inside a fixed outer-loop search procedure. The outer loop determines how parents are selected, how mutations are formed, what history is exposed, and when evaluation is triggered. In such systems, the model acts as a powerful generator inside a largely hand-designed search scaffold.

Our goal is to push further in the agentic direction. We study a framework in which the coding agent is not limited to producing a candidate from a prescribed prompt. Instead, the agent is given tool-mediated access to a structured archive containing prior attempts, evaluation outputs, diagnostics, and reusable knowledge. The agent decides what to inspect, what to compare, what to modify, when to run local checks, when to submit, and what lessons to preserve.

The central hypothesis is that modern coding agents already possess more optimization ability than current outer-loop designs typically allow them to express. If the outer loop is reduced to the minimum necessary responsibilities—objective definition, permission control, budgeting, verification, evaluation, and archival—then more of the actual search behavior can emerge from the agent itself.

## 2. Framework Overview

The framework has three core components: an agent, a score interface, and a filesystem.

The agent is a CLI coding agent. We intentionally define this broadly: the framework should operate with existing coding agents rather than requiring a bespoke model API stack. This choice is both practical and scientific. Practically, modern coding agents already provide code editing, shell execution, tool use, and context management. Meanwhile users can simply use credits from their subscribed coding plans. Scientifically, they provide a realistic substrate for studying long-horizon optimization behavior.

The score interface is the primary optimization signal. It is task-specific and is responsible for converting a high-level research or engineering goal into measurable feedback. We do not assume that the model can reliably infer the correct metric from a vague objective. Defining the score interface remains the responsibility of the researcher or engineer. It supplies the external feedback needed for optimization while keeping private scoring logic outside the agent's workspace.

The filesystem stores previous attempts, their associated evaluation outputs, diagnostic artifacts, worklogs, and reusable knowledge that can accumulate over time. Rather than compressing all prior experience into a single prompt, the system lets the agent selectively inspect public artifacts through a small query surface.

## 3. Score Function and Evaluation Safeguards

The score interface is divided into three public surfaces.

The verifier checks hard constraints and performs cheap screening. It answers questions such as: Does the candidate compile? Does it satisfy the required interface? Does it pass safety or correctness checks on small cases? If the task uses an expensive benchmark, the verifier may run a lightweight proxy or a small benchmark subset to determine whether full evaluation is worthwhile.

Diagnostic probes provide cheap process feedback that is richer than a pass/fail verifier but cheaper and less authoritative than final scoring. A probe may expose repair-oriented signals, partial metrics, or task-specific diagnostics. Probes are useful because many optimization failures are not simply invalid submissions; they are near misses whose structure can guide the next local change.

The evaluator performs the authoritative measurement. It may run the full benchmark suite, compute final metrics, and produce detailed reports. This separation is essential whenever full scoring is expensive. It prevents invalid, malformed, or obviously weak candidates from consuming unnecessary benchmark budget.

To reduce reward hacking, the evaluator should expose only a public submission interface and public feedback, while grader internals, hidden tests, private benchmark assets, and task-specific scoring code remain outside the agent’s readable workspace. This kind of evaluator isolation already appears as an explicit safeguard in recent autonomous evolution systems, where the grader is hidden from agents to reduce opportunities for direct exploitation.

## 4. Filesystem, Permissions, and Queryability

The filesystem contains several logical regions: historical attempts, experimental results, diagnostic probes, agent worklogs, shared reusable patterns, orchestrator-managed support data, and a private evaluation area.

Historical attempts store self-contained candidate implementations. Each attempt should run independently in isolation, without hidden dependencies on previous attempts. In addition to code snapshots, an attempt may include submission metadata, diffs, shell logs, tool-call records, and other observable interaction traces.

Experimental results contain the outputs of the score interface, including verifier outputs, evaluator scores, benchmark reports, and task-specific diagnostics. These artifacts are append-only and immutable once recorded.

Durable agent-authored memory should remain lightweight but should not collapse all reuse into one artifact type. A workspace-local worklog acts as raw scratch state for ongoing reasoning. Shared patterns capture distilled reusable findings such as what worked, what failed, and what a result implies. Shared tools capture reusable executable workflow aids such as analyzers, comparators, exporters, or local diagnostic scripts. This separates noisy local thought from durable findings and durable capabilities without requiring a heavy schema for agent-authored memory.

A separate private/ region stores grader implementations, hidden tests, and any non-public benchmark assets. This area is inaccessible to the agent. The agent can submit candidates and observe feedback, but cannot inspect or modify the evaluation logic directly.

Because raw history can grow quickly, the filesystem should be complemented by a small query CLI. Rather than forcing the agent to repeatedly traverse the archive with ad hoc grep, find, and cat patterns, this CLI can expose common operations such as listing top attempts, viewing the current frontier or leaderboard, showing diffs between attempts, surfacing recent failures and probes, retrieving worklog history, searching shared patterns, and listing shared tools. The CLI is especially useful because the agent's active context should remain compact; richer raw detail remains in archived attempts, result artifacts, worklogs, patterns, and tools.

## 5. Iteration Lifecycle

An optimization run is organized as a sequence of fresh autonomous sessions.

First, the orchestrator prepares a candidate workspace. It materializes the writable candidate directory, restores the agent's worklog when appropriate, exposes query and evaluation tools, and enforces the sandbox boundary.

Second, the agent reads selectively. It may inspect the current leaderboard, recent attempts, prior diagnostics, shared patterns, shared tools, direction-specific guidance, or any approved support material. The outer loop does not prescribe a fixed inspection order.

Third, the agent works autonomously inside the candidate workspace. It may modify existing code, generate a new solution, compare against prior attempts, run tests, invoke verification, request diagnostic probes, submit for authoritative scoring, update its worklog, or share a distilled pattern.

Fourth, during the work itself, the agent may also choose to create small task-specific tools or reuse shared tools when repeated work is better encoded as executable logic than re-run manually. This should be treated as ordinary inner-loop research behavior, not as a separately orchestrated workflow phase.

Fifth, when the agent judges the candidate ready, it submits through the public evaluation interface. The host-side service archives the submitted workspace snapshot before authoritative scoring.

Sixth, the verifier filters invalid candidates before evaluator budget is consumed. If verification succeeds and budget remains, the evaluator performs authoritative scoring outside the agent-readable workspace.

The session may then end without an explicit same-thread reflection phase. The next session starts with a fresh context window, preventing unbounded conversational accumulation while preserving continuity through the filesystem, worklogs, query surfaces, shared patterns, and shared tools.

## 6. Implementation Strategy

The system should aim for a provider-neutral outer architecture with thin agent-specific adapters, while allowing the first implementation to exploit the concrete control surface of a capable coding agent.

The orchestrator is responsible for workspace creation, permission management, run bookkeeping, service lifecycle, archival, session tracking, budget enforcement, and structured event logging. It should define a common contract for supported coding agents: start a session, observe traces, interrupt unproductive work when necessary, and preserve enough filesystem state for future sessions.

Each supported coding agent then receives a dedicated adapter. The adapter translates between the orchestrator’s common interface and the concrete control surface offered by that agent. Different agents expose different mechanisms for authentication, session continuity, tool control, and structured outputs, but the outer architecture should remain stable across providers.

### 6.1 Codex Integration

For Codex, we use the App Server rather than codex exec as the primary integration surface. OpenAI documents codex exec as the non-interactive mode for scripts and CI, while the App Server is the long-lived JSON-RPC harness surface used across Codex clients. The App Server exposes sessions, turn interruption, streamed item events, and sandboxed command execution, which are a much better fit for long-horizon optimization loops than a one-shot scripting interface.

Concretely, the Codex adapter should treat a worker session as an autonomous turn inside a sandboxed workspace. The outer loop can start fresh sessions for the same logical agent slot, interrupt unproductive turns, and rely on filesystem memory rather than same-thread conversation history for continuity.

Codex also provides two instruction surfaces that are especially useful for this framework. First, AGENTS.md is loaded before work begins and is appropriate for persistent project-level rules such as filesystem permissions, submission contracts, and safety constraints. Second, repository-local skills in .agents/skills are suitable for reusable workflows such as how to query the leaderboard, how to use verifier/probe/submit feedback, and how to share reusable patterns. Skills use progressive disclosure: Codex initially loads only skill metadata and pulls the full instructions on demand, which is well aligned with token-efficient workflow composition.

The Codex adapter should rely primarily on sandbox policy and filesystem permissions, not prompt-level obedience, to enforce safety boundaries. In particular, the writable workspace should be limited to the current candidate directory and local workspace memory, while archives, private evaluation assets, and canonical durable records remain accessible only through controlled services. The agent-facing tools should therefore be thin clients to host-side services rather than direct imports of privileged evaluator code.

## 7. Experimental Plan

We plan to study three classes of problems.

The first class contains lightweight optimization tasks with strict requirements and relatively cheap verification. Representative examples include packing problems, stencil and PDE kernel optimization, and other structured families such as routing, scheduling, and graph optimization. These problems are attractive because they support rapid iteration and clean verifier/evaluator splits.

The second class contains benchmark-driven optimization tasks, where the score function is a standardized public benchmark rather than a bespoke internal metric. Representative examples include GPU kernel optimization on KernelBench, which evaluates model-generated kernels on a public suite of 250 workloads; compiler optimization with LLVM test-suite, CTMark, and CompilerGym; and solver-engineering benchmarks such as SAT, MaxSAT, and MIPLIB. LLVM test-suite provides reference outputs and collects runtime, compile-time, and code-size metrics; CTMark is the compile-time-only subset of LLVM test-suite; MIPLIB 2017 provides a 240-instance benchmark set; and MaxSAT/SAT evaluation infrastructures provide standardized public solver benchmarks.

The third class contains domain-specific tasks with custom score functions. Examples include internal repository CI repair, project-level memory and retrieval optimization, incident-response assistants, and workflow agents whose objectives cannot be fully captured by an existing benchmark. These settings are less standardized, but they are essential for understanding whether the framework is useful beyond clean research problems.

Across all three categories, we are interested in the same underlying questions: how much useful search behavior can be delegated to the agent itself, how effective verifier/probe/evaluator decomposition is at reducing benchmark cost, whether hidden graders reduce reward hacking pressure, and whether durable filesystem-based memory enables stronger long-horizon improvement than compressed history alone.

## 8. Scope and Extensions

The present framework begins with autonomous sessions over a durable filesystem memory, but the current design also treats controlled multi-agent breadth as a first-class experimental setting. In that setting, different agent slots may be assigned fixed human-written research directions or coarser islands of related methods. Agents within the same island can share more freely, while cross-island reuse of high-scoring snapshots and detailed patterns can be deliberately limited. This is not meant to make the outer loop choose ideas. It is an information-structure intervention: the system shapes what each agent can conveniently inspect while leaving the concrete research decisions to the agent.

This also makes exploration diversity measurable. A simple first statistic is the Shannon entropy of the active explorer distribution over directions or islands. Low entropy indicates that parallel agents have collapsed into only a few areas; high entropy indicates that the run is still maintaining breadth. Exploiter or replication agents can be useful for chasing the frontier, but they should be analyzed separately from explorers because their purpose is not diversity.

Natural extensions include richer frontier maintenance, asynchronous parallel agents, stronger visualization dashboards, conditional supervisory or critic-style interventions during stagnation, and post-run analysis of which method families were actually explored. We do not treat these extensions as replacements for the core principle: the outer loop should provide infrastructure and constraints, while the coding agent remains the primary search actor.
