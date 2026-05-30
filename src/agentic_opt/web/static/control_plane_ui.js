(function () {
  "use strict";

  const root = document.getElementById("ao-ui");
  if (!root) {
    return;
  }

  const apiBase = root.dataset.apiBase || "/api/v1";
  const state = {
    experiments: [],
    selectedExperimentId: root.dataset.initialExperimentId || "",
    detail: null,
    analysis: null,
    currentTab: "overview",
    loading: false,
    error: "",
  };

  const nodes = {
    experimentCount: document.getElementById("experiment-count"),
    apiHealth: document.getElementById("api-health"),
    experimentList: document.getElementById("experiment-list"),
    eyebrow: document.getElementById("experiment-eyebrow"),
    title: document.getElementById("experiment-title"),
    statusStrip: document.getElementById("status-strip"),
    tabs: Array.from(document.querySelectorAll(".ao-tab")),
    content: document.getElementById("content"),
  };

  function escapeHtml(value) {
    return String(value == null ? "" : value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function asArray(value) {
    return Array.isArray(value) ? value : [];
  }

  function asObject(value) {
    return value && typeof value === "object" && !Array.isArray(value) ? value : {};
  }

  function shortId(value) {
    const text = String(value || "");
    if (text.length <= 20) {
      return text || "-";
    }
    return `${text.slice(0, 10)}…${text.slice(-7)}`;
  }

  function fmtDate(value) {
    if (!value) {
      return "-";
    }
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
      return String(value);
    }
    return date.toLocaleString();
  }

  function fmtNumber(value, digits = 4) {
    const number = Number(value);
    if (!Number.isFinite(number)) {
      return "-";
    }
    return number.toLocaleString(undefined, { maximumFractionDigits: digits });
  }

  function statusTone(status) {
    const text = String(status || "").toLowerCase();
    if (["completed", "ready", "active", "succeeded", "pass", "passed"].includes(text)) {
      return "good";
    }
    if (["failed", "error", "cancelled", "invalid", "denied"].includes(text)) {
      return "bad";
    }
    if (["running", "queued", "pending", "weakened", "stale running"].includes(text)) {
      return "warn";
    }
    return "";
  }

  function pill(label, value, tone) {
    return `<span class="ao-pill ${tone || statusTone(value)}">${escapeHtml(label)} ${escapeHtml(value == null ? "-" : value)}</span>`;
  }

  function chip(value) {
    return `<span class="ao-chip">${escapeHtml(value == null ? "-" : value)}</span>`;
  }

  async function api(path) {
    const response = await fetch(`${apiBase}${path}`, { headers: { Accept: "application/json" } });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.error || `${response.status} ${response.statusText}`);
    }
    return payload;
  }

  async function loadExperiments() {
    state.loading = true;
    state.error = "";
    render();
    try {
      const payload = await api("/experiments");
      state.experiments = asArray(payload.experiments);
      if (!state.selectedExperimentId && state.experiments.length) {
        state.selectedExperimentId = state.experiments[0].experiment_id;
      }
      nodes.apiHealth.textContent = "online";
      await loadSelectedExperiment();
    } catch (error) {
      state.error = error.message;
      nodes.apiHealth.textContent = "offline";
      state.loading = false;
      render();
    }
  }

  async function loadSelectedExperiment() {
    if (!state.selectedExperimentId) {
      state.detail = null;
      state.analysis = null;
      state.loading = false;
      render();
      return;
    }
    state.loading = true;
    state.error = "";
    render();
    try {
      const [detail, analysis] = await Promise.all([
        api(`/experiments/${encodeURIComponent(state.selectedExperimentId)}`),
        api(`/experiments/${encodeURIComponent(state.selectedExperimentId)}/analysis`).catch((error) => ({
          error: error.message,
        })),
      ]);
      state.detail = detail;
      state.analysis = analysis && !analysis.error ? analysis : null;
      state.loading = false;
      render();
    } catch (error) {
      state.error = error.message;
      state.detail = null;
      state.analysis = null;
      state.loading = false;
      render();
    }
  }

  function selectExperiment(experimentId) {
    if (experimentId === state.selectedExperimentId) {
      return;
    }
    state.selectedExperimentId = experimentId;
    const path = `/ui/experiments/${encodeURIComponent(experimentId)}`;
    window.history.pushState({ experimentId }, "", path);
    loadSelectedExperiment();
  }

  function setTab(tab) {
    state.currentTab = tab;
    render();
  }

  function render() {
    renderSidebar();
    renderHeader();
    nodes.tabs.forEach((tab) => {
      tab.classList.toggle("is-active", tab.dataset.tab === state.currentTab);
    });
    if (state.error) {
      nodes.content.innerHTML = `<div class="ao-error">${escapeHtml(state.error)}</div>`;
      return;
    }
    if (state.loading && !state.detail) {
      nodes.content.innerHTML = '<div class="ao-loading">Loading control-plane data…</div>';
      return;
    }
    if (!state.selectedExperimentId) {
      nodes.content.innerHTML = '<div class="ao-empty">No experiments are registered in this state root.</div>';
      return;
    }
    if (!state.detail) {
      nodes.content.innerHTML = '<div class="ao-loading">Loading experiment…</div>';
      return;
    }
    const renderers = {
      overview: renderOverview,
      leaderboard: renderLeaderboard,
      analysis: renderAnalysis,
      traces: renderTraces,
      isolation: renderIsolation,
      replay: renderReplay,
    };
    nodes.content.innerHTML = (renderers[state.currentTab] || renderOverview)();
  }

  function renderSidebar() {
    nodes.experimentCount.textContent = `${state.experiments.length} experiment${state.experiments.length === 1 ? "" : "s"}`;
    if (!state.experiments.length) {
      nodes.experimentList.innerHTML = '<div class="ao-empty">Empty</div>';
      return;
    }
    nodes.experimentList.innerHTML = state.experiments
      .map((experiment) => {
        const active = experiment.experiment_id === state.selectedExperimentId ? " is-active" : "";
        return `
          <button class="ao-experiment-item${active}" type="button" data-experiment-id="${escapeHtml(experiment.experiment_id)}">
            <span class="ao-experiment-row">
              <span class="ao-experiment-id">${escapeHtml(shortId(experiment.experiment_id))}</span>
              ${chip(experiment.status || "unknown")}
            </span>
            <span class="ao-experiment-task">${escapeHtml(experiment.task_id || "unknown task")}</span>
          </button>
        `;
      })
      .join("");
  }

  function renderHeader() {
    const detail = state.detail || {};
    const experiment = asObject(detail.experiment);
    if (!experiment.experiment_id) {
      nodes.eyebrow.textContent = "No experiment selected";
      nodes.title.textContent = "Control Plane";
      nodes.statusStrip.innerHTML = "";
      return;
    }
    const incumbent = asObject(detail.incumbent);
    const budget = budgetSummary();
    const execution = executionSummary();
    const displayStatus = execution.stale ? "stale running" : experiment.status || "unknown";
    nodes.eyebrow.textContent = experiment.task_id || "experiment";
    nodes.title.textContent = experiment.experiment_id;
    nodes.statusStrip.innerHTML = [
      pill("status", displayStatus),
      pill("mode", experiment.mode || "unknown", ""),
      pill("score budget", budget.label, budget.remaining === 0 ? "good" : ""),
      pill("score", incumbent.score == null ? "-" : fmtNumber(incumbent.score, 6), incumbent.score == null ? "warn" : "good"),
      chip(`updated ${fmtDate(experiment.updated_at)}`),
    ].join("");
  }

  function renderOverview() {
    const detail = state.detail;
    const experiment = asObject(detail.experiment);
    const incumbent = asObject(detail.incumbent);
    const networkPolicy = asObject(detail.network_policy);
    const enforcement = asObject(networkPolicy.enforcement);
    const taskKnowledge = asObject(detail.task_knowledge);
    const recentEvents = asArray(detail.events).slice(0, 8);
    const metadata = asObject(experiment.metadata);
    const budget = budgetSummary();
    const execution = executionSummary();
    const activeWorkers = activeWorkerSummary();
    const leaderboardEntries = getLeaderboardEntries();
    return `
      <div class="ao-section">
        <div class="ao-metrics ao-metrics-compact">
          ${metric("Score Budget", budget.label, budget.sub)}
          ${metric("Best Score", incumbent.score == null ? "-" : fmtNumber(incumbent.score, 6), shortId(incumbent.evaluation_id))}
          ${metric("Active Worker", activeWorkers.label, activeWorkers.sub)}
          ${metric("Published Scores", leaderboardEntries.length, "leaderboard entries")}
        </div>
        <div class="ao-grid-2">
          <div class="ao-band">
            <div class="ao-band-header">
              <div class="ao-band-title">Experiment</div>
              <div class="ao-band-note">${escapeHtml(execution.stale ? "stale running" : experiment.status || "unknown")}</div>
            </div>
            ${kv({
              "Task": experiment.task_id,
              "Mode": experiment.mode,
              "Created": fmtDate(experiment.created_at),
              "Updated": fmtDate(experiment.updated_at),
              "Score budget": budget.detail,
              "Best score": incumbent.score == null ? "-" : `${fmtNumber(incumbent.score, 8)} (${shortId(incumbent.evaluation_id)})`,
              "Active worker": activeWorkers.detail,
              "State": root.dataset.stateRoot,
            })}
          </div>
          <div class="ao-band">
            <div class="ao-band-header">
              <div class="ao-band-title">Policy Snapshot</div>
              <div class="ao-band-note">${escapeHtml(enforcement.policy_weakened ? "weakened" : "strict")}</div>
            </div>
            ${kv({
              "External internet": asObject(networkPolicy.policy).external_internet,
              "Enforced": enforcement.external_internet_enforced,
              "Relay required": enforcement.control_plane_relay_required,
              "Task knowledge": `${asArray(taskKnowledge.files).length} files`,
              "Knowledge digest": taskKnowledge.digest,
            })}
          </div>
        </div>
        <div class="ao-band">
          <div class="ao-band-header">
            <div class="ao-band-title">Recent Events</div>
            <div class="ao-band-note">${recentEvents.length} shown</div>
          </div>
          ${timeline(recentEvents)}
        </div>
      </div>
    `;
  }

  function renderLeaderboard() {
    const entries = getLeaderboardEntries();
    const budget = budgetSummary();
    return `
      <div class="ao-section">
        <div class="ao-metrics">
          ${metric("Score Budget", budget.label, budget.sub)}
          ${metric("Published Scores", entries.length, "leaderboard entries")}
          ${metric("Pending Scoring", budget.pending, "queued/running submit")}
          ${metric("Remaining", budget.remaining == null ? "-" : budget.remaining, "scores to publish")}
        </div>
        <div class="ao-band">
          <div class="ao-band-header">
            <div class="ao-band-title">Leaderboard</div>
            <div class="ao-band-note">${entries.length} entries</div>
          </div>
          ${table(
            ["Rank", "Score", "Evaluation", "Artifact", "Provider", "Task Context", "Official", "Updated"],
            entries.map((entry, index) => {
              const metadata = asObject(entry.metadata);
              const taskContext = asObject(metadata.task_context || metadata.task_context_snapshot || {});
              return [
                index + 1,
                fmtNumber(entry.score, 8),
                mono(entry.evaluation_id),
                mono(entry.artifact_id),
                metadata.environment_provider || metadata.provider || "-",
                mono(taskContext.digest || metadata.task_context_digest || "-"),
                metadata.official_submit || entry.kind === "submit" ? "yes" : "no",
                fmtDate(entry.updated_at),
              ];
            })
          )}
        </div>
      </div>
    `;
  }

  function renderAnalysis() {
    const analysis = state.analysis;
    if (!analysis) {
      return '<div class="ao-empty">Run analysis is unavailable for this experiment.</div>';
    }
    const scoreSeries = asArray(analysis.score_series);
    const lineage = asArray(analysis.candidate_lineage);
    const graph = asObject(analysis.attempt_graph);
    const nodes = asArray(graph.nodes);
    const edges = asArray(graph.edges);
    return `
      <div class="ao-section">
        <div class="ao-grid-2">
          <div class="ao-band">
            <div class="ao-band-header">
              <div class="ao-band-title">Score Series</div>
              <div class="ao-band-note">${scoreSeries.length} points</div>
            </div>
            ${scoreSeries.length ? scoreLineChart(scoreSeries) : '<div class="ao-empty">No score points.</div>'}
          </div>
          <div class="ao-band">
            <div class="ao-band-header">
              <div class="ao-band-title">Attempt Graph</div>
              <div class="ao-band-note">${nodes.length} nodes, ${edges.length} edges</div>
            </div>
            ${kv({
              "Schema": analysis.schema_version,
              "Relationship count": asArray(analysis.relationships).length,
              "Trace count": asObject(analysis.summary).trace_count,
              "Leaderboard count": asObject(analysis.summary).leaderboard_count,
            })}
          </div>
        </div>
        <div class="ao-band">
          <div class="ao-band-header">
            <div class="ao-band-title">Candidate Lineage</div>
            <div class="ao-band-note">${lineage.length} attempts</div>
          </div>
          ${table(
            ["Attempt", "Parent", "Status", "Best Score", "Candidate", "Traces", "Jobs"],
            lineage.map((item) => [
              mono(item.attempt_id),
              mono(item.parent_attempt_id || "-"),
              item.status || "-",
              fmtNumber(item.best_score, 8),
              mono(item.candidate_artifact_id || "-"),
              asArray(item.trace_ids).map(shortId).join(", ") || "-",
              asArray(item.job_ids).map(shortId).join(", ") || "-",
            ])
          )}
        </div>
      </div>
    `;
  }

  function renderTraces() {
    const traces = asArray(state.detail.agent_traces);
    const exports = asArray(state.detail.trace_exports);
    return `
      <div class="ao-section">
        <div class="ao-grid-2">
          <div class="ao-band">
            <div class="ao-band-header">
              <div class="ao-band-title">Agent Traces</div>
              <div class="ao-band-note">${traces.length} traces</div>
            </div>
            ${table(
              ["Trace", "Session", "Status", "Commands", "Failed", "Artifact", "Updated"],
              traces.map((trace) => {
                const metadata = asObject(trace.metadata);
                return [
                  mono(trace.trace_id),
                  mono(trace.session_id),
                  trace.status || "-",
                  metadata.command_count ?? "-",
                  metadata.failed_command_count ?? "-",
                  mono(trace.artifact_id || "-"),
                  fmtDate(trace.updated_at),
                ];
              })
            )}
          </div>
          <div class="ao-band">
            <div class="ao-band-header">
              <div class="ao-band-title">Trace Exports</div>
              <div class="ao-band-note">${exports.length} exports</div>
            </div>
            ${table(
              ["Export", "Provider", "Status", "Traces", "Artifact", "Updated"],
              exports.map((item) => [
                mono(item.trace_export_id),
                item.provider || "-",
                item.status || "-",
                asArray(item.source_trace_ids).length,
                mono(item.artifact_id || "-"),
                fmtDate(item.updated_at),
              ])
            )}
          </div>
        </div>
        <div class="ao-band">
          <div class="ao-band-header">
            <div class="ao-band-title">External Exporters</div>
            <div class="ao-band-note">not implemented</div>
          </div>
          <div class="ao-grid-3">
            ${externalExporter("OTLP")}
            ${externalExporter("Phoenix")}
            ${externalExporter("Helicone")}
          </div>
        </div>
      </div>
    `;
  }

  function renderIsolation() {
    const detail = state.detail;
    const jobs = asArray(detail.jobs);
    const sessions = asArray(detail.sessions);
    const networkEvents = asArray(detail.network_access_events);
    const networkPolicy = asObject(detail.network_policy);
    const enforcement = asObject(networkPolicy.enforcement);
    const rows = jobs.map((job) => {
      const details = asObject(job.details);
      const network = asObject(details.network_enforcement);
      const taskContext = asObject(details.task_context_enforcement);
      return [
        mono(job.job_id),
        job.provider || "-",
        job.status || "-",
        network.external_internet || asObject(networkPolicy.policy).external_internet || "-",
        network.external_internet_enforced ?? enforcement.external_internet_enforced ?? "-",
        network.policy_weakened ?? enforcement.policy_weakened ?? "-",
        taskContext.mode || taskContext.provider || "-",
        taskContext.read_only_mount || taskContext.digest_guard || taskContext.enforced || "-",
      ];
    });
    return `
      <div class="ao-section">
        <div class="ao-metrics">
          ${metric("Sessions", sessions.length, "worker contexts")}
          ${metric("Jobs", jobs.length, "provider executions")}
          ${metric("Denied Events", networkEvents.filter((event) => event.decision === "denied").length, "network access")}
          ${metric("Policy", enforcement.policy_weakened ? "weakened" : "strict", enforcement.enforcement_mode || "-")}
        </div>
        <div class="ao-band">
          <div class="ao-band-header">
            <div class="ao-band-title">Provider Enforcement</div>
            <div class="ao-band-note">local and docker job records</div>
          </div>
          ${table(
            ["Job", "Provider", "Status", "Network", "Net Enforced", "Weakened", "Task Context", "Read-only Guard"],
            rows
          )}
        </div>
        <div class="ao-band">
          <div class="ao-band-header">
            <div class="ao-band-title">Network Events</div>
            <div class="ao-band-note">${networkEvents.length} events</div>
          </div>
          ${table(
            ["Decision", "Destination", "Access", "Reason", "Session", "Updated"],
            networkEvents.map((event) => [
              event.decision || "-",
              event.destination || "-",
              event.access_type || "-",
              event.reason || "-",
              mono(event.session_id || "-"),
              fmtDate(event.created_at || event.updated_at),
            ])
          )}
        </div>
      </div>
    `;
  }

  function renderReplay() {
    const evaluations = asArray(state.detail.evaluations);
    const replayEvaluations = evaluations.filter((evaluation) => asObject(evaluation.request).replay || asObject(evaluation.metadata).replay);
    const bundleReady = evaluations.filter((evaluation) => evaluation.artifact_id || asObject(evaluation.request).artifact_id);
    return `
      <div class="ao-section">
        <div class="ao-metrics">
          ${metric("Evaluations", evaluations.length, "bundle candidates")}
          ${metric("Replay Runs", replayEvaluations.length, "recorded")}
          ${metric("Bundle-ready", bundleReady.length, "with artifact input")}
          ${metric("Leaderboard", "off by default", "replay policy")}
        </div>
        <div class="ao-band">
          <div class="ao-band-header">
            <div class="ao-band-title">Replay Records</div>
            <div class="ao-band-note">read-only view</div>
          </div>
          ${table(
            ["Evaluation", "Status", "Kind", "Provider", "Score", "Budget Counted", "Leaderboard", "Updated"],
            evaluations.map((evaluation) => {
              const request = asObject(evaluation.request);
              const metadata = asObject(evaluation.metadata);
              return [
                mono(evaluation.evaluation_id),
                evaluation.status || "-",
                evaluation.kind || request.kind || "-",
                request.environment_provider || metadata.environment_provider || "-",
                fmtNumber(evaluation.score, 8),
                request.count_budget === false ? "no" : "yes",
                request.publish_leaderboard === false ? "no" : "yes",
                fmtDate(evaluation.updated_at),
              ];
            })
          )}
        </div>
        <div class="ao-band">
          <div class="ao-band-header">
            <div class="ao-band-title">Restore Gaps</div>
            <div class="ao-band-note">remaining implementation work</div>
          </div>
          ${table(
            ["Area", "Status"],
            [
              ["Cross-machine restore", "pending"],
              ["External artifact store restore", "pending"],
              ["Trace artifact packaging", "pending"],
              ["Docker image task context immutability", "pending"],
            ]
          )}
        </div>
      </div>
    `;
  }

  function metric(label, value, sub) {
    return `
      <div class="ao-metric">
        <div class="ao-metric-label">${escapeHtml(label)}</div>
        <div class="ao-metric-value">${escapeHtml(value == null ? "-" : value)}</div>
        <div class="ao-metric-sub">${escapeHtml(sub == null ? "" : sub)}</div>
      </div>
    `;
  }

  function getLeaderboardEntries() {
    const detailEntries = asArray((state.detail || {}).leaderboard);
    const analysisEntries = asArray(asObject(state.analysis).entities && asObject(state.analysis.entities).leaderboard_entries);
    return analysisEntries.length >= detailEntries.length ? analysisEntries : detailEntries;
  }

  function budgetSummary() {
    const detail = state.detail || {};
    const experiment = asObject(detail.experiment);
    const budget = asObject(experiment.budget);
    const total = positiveInt(budget.total_evaluator_runs || budget.evaluator_runs);
    const budgeted = asArray(detail.evaluations).filter(isBudgetedSubmitEvaluation);
    const pending = budgeted.filter((evaluation) => ["queued", "running"].includes(String(evaluation.status || ""))).length;
    const used = budgeted.length - pending;
    const effectiveUsed = used + pending;
    const remaining = total == null ? null : Math.max(0, total - effectiveUsed);
    return {
      total,
      used,
      pending,
      remaining,
      label: total == null ? `${effectiveUsed}` : `${effectiveUsed}/${total}`,
      sub: pending ? `${pending} pending submit run${pending === 1 ? "" : "s"}` : "submit evaluator runs",
      detail:
        total == null
          ? `${effectiveUsed} submit evaluator runs; verifier and probe runs excluded; no total_evaluator_runs limit configured`
          : `${used} finished submit evaluator runs of ${total}; ${pending} queued/running submit evaluations; verifier and probe runs excluded; ${remaining} remaining`,
    };
  }

  function isBudgetedSubmitEvaluation(evaluation) {
    if (["cancelled", "stopped"].includes(String(evaluation.status || ""))) {
      return false;
    }
    const request = asObject(evaluation.request);
    const kind = evaluation.kind || request.kind;
    if (!["submit", "official"].includes(String(kind || ""))) {
      return false;
    }
    if (request.count_budget === false || request.publish_leaderboard === false) {
      return false;
    }
    if (request.replay && request.publish_leaderboard !== true) {
      return false;
    }
    return true;
  }

  function executionSummary() {
    const detail = state.detail || {};
    const experiment = asObject(detail.experiment);
    const activeJobs = asArray(detail.jobs).filter((job) => ["queued", "running"].includes(String(job.status || "")));
    const activeEvaluations = asArray(detail.evaluations).filter((evaluation) => ["queued", "running"].includes(String(evaluation.status || "")));
    const activeSessions = asArray(detail.sessions).filter((session) => ["starting", "running"].includes(String(session.status || "")));
    const runningExperiment = String(experiment.status || "") === "running";
    const activeWork = activeJobs.length + activeEvaluations.length;
    const latestActiveMs = Math.max(
      0,
      ...activeJobs.map(recordTimeMs),
      ...activeEvaluations.map(recordTimeMs),
      ...activeSessions.map(recordTimeMs)
    );
    const staleAgeMs = latestActiveMs ? Date.now() - latestActiveMs : Number.POSITIVE_INFINITY;
    const stale = runningExperiment && activeWork === 0 && (activeSessions.length === 0 || staleAgeMs > 10 * 60 * 1000);
    if (stale) {
      return {
        stale: true,
        label: "stale running",
        sub: "no active job or evaluation",
        detail: `${activeSessions.length} starting/running sessions, ${activeJobs.length} active jobs, ${activeEvaluations.length} active evaluations`,
      };
    }
    if (runningExperiment) {
      return {
        stale: false,
        label: activeWork ? "active" : "running",
        sub: `${activeSessions.length} workers, ${activeJobs.length} jobs, ${activeEvaluations.length} evaluations`,
        detail: `${activeSessions.length} active sessions, ${activeJobs.length} active jobs, ${activeEvaluations.length} active evaluations`,
      };
    }
    return {
      stale: false,
      label: experiment.status || "unknown",
      sub: "experiment status",
      detail: `${activeSessions.length} active sessions, ${activeJobs.length} active jobs, ${activeEvaluations.length} active evaluations`,
    };
  }

  function activeWorkerSummary() {
    const detail = state.detail || {};
    const activeSessions = asArray(detail.sessions).filter((session) => ["starting", "running"].includes(String(session.status || "")));
    const activeJobs = asArray(detail.jobs).filter((job) => ["queued", "running"].includes(String(job.status || "")));
    const activeEvaluations = asArray(detail.evaluations).filter((evaluation) => ["queued", "running"].includes(String(evaluation.status || "")));
    const sessionCount = activeSessions.length;
    return {
      label: sessionCount,
      sub: sessionCount === 1 ? "running worker session" : "running worker sessions",
      detail: `${sessionCount} running worker session${sessionCount === 1 ? "" : "s"}; ${activeJobs.length} active jobs; ${activeEvaluations.length} active evaluations`,
    };
  }

  function positiveInt(value) {
    if (value == null || value === "") {
      return null;
    }
    const parsed = Number.parseInt(value, 10);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
  }

  function recordTimeMs(record) {
    const value = record.updated_at || record.created_at || record.timestamp;
    if (!value) {
      return 0;
    }
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? 0 : date.getTime();
  }

  function kv(items) {
    const rows = Object.entries(items)
      .map(([key, value]) => `<dt>${escapeHtml(key)}</dt><dd>${escapeHtml(value == null || value === "" ? "-" : value)}</dd>`)
      .join("");
    return `<dl class="ao-kv">${rows}</dl>`;
  }

  function mono(value) {
    return { __html: `<span class="ao-mono">${escapeHtml(value == null ? "-" : value)}</span>` };
  }

  function renderCell(cell) {
    if (cell && typeof cell === "object" && Object.prototype.hasOwnProperty.call(cell, "__html")) {
      return cell.__html;
    }
    return escapeHtml(cell == null ? "-" : cell);
  }

  function table(headers, rows) {
    if (!rows.length) {
      return '<div class="ao-empty">No records.</div>';
    }
    return `
      <div class="ao-table-wrap">
        <table>
          <thead><tr>${headers.map((header) => `<th>${escapeHtml(header)}</th>`).join("")}</tr></thead>
          <tbody>
            ${rows.map((row) => `<tr>${row.map((cell) => `<td>${renderCell(cell)}</td>`).join("")}</tr>`).join("")}
          </tbody>
        </table>
      </div>
    `;
  }

  function timeline(events) {
    if (!events.length) {
      return '<div class="ao-empty">No recent events.</div>';
    }
    return `
      <div class="ao-timeline">
        ${events
          .map(
            (event) => `
              <div class="ao-timeline-item">
                <span class="ao-dot" aria-hidden="true"></span>
                <div>
                  <div class="ao-event-title">${escapeHtml(event.summary || event.event_type || event.event_id)}</div>
                  <div class="ao-event-meta">${escapeHtml(event.event_type || "-")} · ${escapeHtml(fmtDate(event.created_at || event.updated_at))}</div>
                </div>
              </div>
            `
          )
          .join("")}
      </div>
    `;
  }

  function scoreLineChart(series) {
    const points = series
      .filter((item) => Number.isFinite(Number(item.score)));
    if (!points.length) {
      return '<div class="ao-empty">No numeric score points.</div>';
    }
    const values = points.map((item) => Number(item.score));
    const domain = niceScoreDomain(Math.min(...values), Math.max(...values));
    const width = 720;
    const height = 240;
    const pad = { top: 18, right: 20, bottom: 34, left: 70 };
    const innerWidth = width - pad.left - pad.right;
    const innerHeight = height - pad.top - pad.bottom;
    const span = Math.max(domain.max - domain.min, 1e-12);
    const xFor = (index) => pad.left + (points.length === 1 ? innerWidth / 2 : (index / (points.length - 1)) * innerWidth);
    const yFor = (score) => pad.top + ((domain.max - score) / span) * innerHeight;
    const coords = points.map((item, index) => ({
      item,
      score: Number(item.score),
      x: xFor(index),
      y: yFor(Number(item.score)),
    }));
    const linePoints = coords.map((point) => `${roundSvg(point.x)},${roundSvg(point.y)}`).join(" ");
    const yTicks = scoreTicks(domain.min, domain.max, domain.step);
    const xTicks = scoreIndexTicks(points.length);
    const bestScore = Math.max(...values);
    return `
      <div class="ao-score-chart">
        <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Score series line chart">
          <line class="ao-score-axis" x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height - pad.bottom}"></line>
          <line class="ao-score-axis" x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}"></line>
          ${yTicks
            .map((tick) => {
              const y = yFor(tick);
              return `
                <line class="ao-score-grid" x1="${pad.left}" y1="${roundSvg(y)}" x2="${width - pad.right}" y2="${roundSvg(y)}"></line>
                <text class="ao-score-y-label" x="${pad.left - 12}" y="${roundSvg(y + 4)}">${escapeHtml(fmtNumber(tick, 5))}</text>
              `;
            })
            .join("")}
          <polyline class="ao-score-line" points="${escapeHtml(linePoints)}"></polyline>
          ${coords
            .map((point, index) => {
              const isBest = point.score === bestScore ? " is-best" : "";
              const title = `#${index + 1} score ${fmtNumber(point.score, 8)} · ${fmtDate(point.item.created_at || point.item.updated_at)}`;
              return `
                <circle class="ao-score-point${isBest}" cx="${roundSvg(point.x)}" cy="${roundSvg(point.y)}" r="${isBest ? 5 : 4}">
                  <title>${escapeHtml(title)}</title>
                </circle>
              `;
            })
            .join("")}
          ${xTicks
            .map((index) => {
              const x = xFor(index);
              return `<text class="ao-score-x-label" x="${roundSvg(x)}" y="${height - 10}">#${index + 1}</text>`;
            })
            .join("")}
        </svg>
        <div class="ao-score-chart-meta">
          <span>range ${escapeHtml(fmtNumber(domain.min, 5))} to ${escapeHtml(fmtNumber(domain.max, 5))}</span>
          <span>${points.length} plotted score${points.length === 1 ? "" : "s"}</span>
        </div>
      </div>
    `;
  }

  function niceScoreDomain(min, max) {
    let domainMin = min;
    let domainMax = max;
    if (!Number.isFinite(domainMin) || !Number.isFinite(domainMax)) {
      return { min: 0, max: 1, step: 0.25 };
    }
    if (domainMin === domainMax) {
      const padding = Math.max(Math.abs(domainMin) * 0.08, 1);
      domainMin -= padding;
      domainMax += padding;
    } else {
      const padding = Math.max((domainMax - domainMin) * 0.08, 1e-9);
      domainMin -= padding;
      domainMax += padding;
    }
    const step = niceNumber((domainMax - domainMin) / 4, true);
    return {
      min: cleanNumber(Math.floor(domainMin / step) * step),
      max: cleanNumber(Math.ceil(domainMax / step) * step),
      step,
    };
  }

  function niceNumber(value, round) {
    if (!Number.isFinite(value) || value <= 0) {
      return 1;
    }
    const exponent = Math.floor(Math.log10(value));
    const fraction = value / Math.pow(10, exponent);
    let niceFraction;
    if (round) {
      if (fraction < 1.5) {
        niceFraction = 1;
      } else if (fraction < 3) {
        niceFraction = 2;
      } else if (fraction < 7) {
        niceFraction = 5;
      } else {
        niceFraction = 10;
      }
    } else if (fraction <= 1) {
      niceFraction = 1;
    } else if (fraction <= 2) {
      niceFraction = 2;
    } else if (fraction <= 5) {
      niceFraction = 5;
    } else {
      niceFraction = 10;
    }
    return niceFraction * Math.pow(10, exponent);
  }

  function scoreTicks(min, max, step) {
    const ticks = [];
    for (let value = min; value <= max + step * 0.5; value += step) {
      ticks.push(cleanNumber(value));
      if (ticks.length > 8) {
        break;
      }
    }
    return ticks;
  }

  function scoreIndexTicks(count) {
    if (count <= 1) {
      return [0];
    }
    const candidates = count <= 4 ? [0, count - 1] : [0, Math.floor((count - 1) / 2), count - 1];
    return Array.from(new Set(candidates)).sort((a, b) => a - b);
  }

  function cleanNumber(value) {
    return Number(Number(value).toPrecision(12));
  }

  function roundSvg(value) {
    return Number(value).toFixed(2);
  }

  function externalExporter(name) {
    return `
      <div class="ao-metric">
        <div class="ao-metric-label">${escapeHtml(name)}</div>
        <div class="ao-metric-value">pending</div>
        <div class="ao-metric-sub">exporter, retry, timeout, batching</div>
      </div>
    `;
  }

  nodes.experimentList.addEventListener("click", (event) => {
    const button = event.target.closest("[data-experiment-id]");
    if (button) {
      selectExperiment(button.dataset.experimentId);
    }
  });

  nodes.tabs.forEach((tab) => {
    tab.addEventListener("click", () => setTab(tab.dataset.tab));
  });

  window.addEventListener("popstate", () => {
    const match = window.location.pathname.match(/\/ui\/experiments\/([^/]+)/);
    state.selectedExperimentId = match ? decodeURIComponent(match[1]) : "";
    loadSelectedExperiment();
  });

  loadExperiments();
})();
