from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .repository import ControlPlaneRepository


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    auto_approved: bool = False
    reason: str | None = None
    estimated_cost: dict[str, Any] | None = None
    remaining_auto_approval_usd: float | None = None
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "auto_approved": self.auto_approved,
            "reason": self.reason,
            "estimated_cost": self.estimated_cost or {},
            "remaining_auto_approval_usd": self.remaining_auto_approval_usd,
            "details": self.details or {},
        }


class PolicyService:
    """Server-owned budget and approval policy.

    The shape borrows the ml-intern idea of auto-approval with a hard cost cap,
    but policy is experiment-owned rather than session-owned.
    """

    def __init__(self, repository: ControlPlaneRepository) -> None:
        self.repository = repository

    def decide_job(self, payload: dict[str, Any]) -> PolicyDecision:
        experiment_id = payload.get("experiment_id")
        experiment = self.repository.get_experiment(experiment_id) if experiment_id else None
        policy = experiment.get("policy") if experiment else {}
        budget = experiment.get("budget") if experiment else {}
        job_policy = {**(policy.get("jobs") or {}), **(payload.get("policy") or {})}
        provider = payload.get("provider") or "local"
        estimated = estimated_cost(payload)

        max_jobs = job_policy.get("max_jobs", budget.get("max_jobs"))
        if max_jobs is not None and experiment_id:
            current_jobs = self.repository.list_jobs(experiment_id=experiment_id)
            non_blocked_jobs = [item for item in current_jobs if item["status"] != "blocked"]
            if len(non_blocked_jobs) >= int(max_jobs):
                return PolicyDecision(False, reason="max_jobs_exceeded", estimated_cost=estimated, details={"max_jobs": int(max_jobs)})

        max_job_cost = job_policy.get("max_cost_usd", budget.get("max_job_cost_usd"))
        estimated_usd = estimated.get("estimated_usd")
        if estimated_usd is not None and max_job_cost is not None and float(estimated_usd) > float(max_job_cost):
            return PolicyDecision(
                False,
                reason="estimated_cost_exceeds_job_budget",
                estimated_cost=estimated,
                details={"estimated_usd": estimated_usd, "max_cost_usd": max_job_cost},
            )

        total_cap = job_policy.get("max_total_estimated_cost_usd", budget.get("max_total_estimated_cost_usd"))
        if estimated_usd is not None and total_cap is not None and experiment_id:
            spent = self.estimated_job_spend_usd(experiment_id)
            if spent + float(estimated_usd) > float(total_cap):
                return PolicyDecision(
                    False,
                    reason="estimated_cost_exceeds_total_budget",
                    estimated_cost=estimated,
                    details={"estimated_usd": estimated_usd, "spent_usd": spent, "max_total_estimated_cost_usd": total_cap},
                )

        requires_approval = bool(payload.get("requires_approval") or job_policy.get("require_approval"))
        if provider not in {"local", "local-docker"}:
            requires_approval = bool(job_policy.get("require_provider_approval", True))
        if requires_approval and payload.get("approved"):
            return PolicyDecision(True, estimated_cost=estimated)
        if requires_approval:
            auto_enabled = bool(job_policy.get("auto_approve") or budget.get("auto_approve"))
            auto_cap = job_policy.get("auto_approval_cost_cap_usd", budget.get("auto_approval_cost_cap_usd"))
            if auto_enabled:
                remaining = None if auto_cap is None or not experiment_id else max(0.0, float(auto_cap) - self.estimated_auto_approved_spend_usd(experiment_id))
                if estimated_usd is None:
                    return PolicyDecision(False, reason="approval_required_unknown_cost", estimated_cost=estimated, remaining_auto_approval_usd=remaining)
                if remaining is not None and float(estimated_usd) > remaining:
                    return PolicyDecision(
                        False,
                        reason="auto_approval_cost_cap_exceeded",
                        estimated_cost=estimated,
                        remaining_auto_approval_usd=remaining,
                    )
                return PolicyDecision(True, auto_approved=True, estimated_cost=estimated, remaining_auto_approval_usd=remaining)
            return PolicyDecision(False, reason="approval_required", estimated_cost=estimated)
        return PolicyDecision(True, estimated_cost=estimated)

    def estimated_job_spend_usd(self, experiment_id: str) -> float:
        total = 0.0
        for job in self.repository.list_jobs(experiment_id=experiment_id):
            if job["status"] == "blocked":
                continue
            total += float((job.get("cost") or {}).get("estimated_usd") or 0.0)
        return round(total, 4)

    def estimated_auto_approved_spend_usd(self, experiment_id: str) -> float:
        total = 0.0
        for job in self.repository.list_jobs(experiment_id=experiment_id):
            details = job.get("details") or {}
            if details.get("policy_decision", {}).get("auto_approved"):
                total += float((job.get("cost") or {}).get("estimated_usd") or 0.0)
        return round(total, 4)


def estimated_cost(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("estimated_cost") or payload.get("cost") or {}
    if isinstance(raw, (int, float)):
        return {"estimated_usd": float(raw)}
    if isinstance(raw, dict):
        return raw
    return {}

