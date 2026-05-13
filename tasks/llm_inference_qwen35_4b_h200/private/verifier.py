from __future__ import annotations

from pathlib import Path
from typing import Any

from .preflight import live_verifier_enabled, run_live_preflight
from .rmsnorm_live import run_rmsnorm_live_checks
from .run_artifacts import new_run_dir, write_json
from .schema import BundleManifest, ManifestValidationError, load_manifest, validate_bundle_files
from .workloads import PUBLIC_WORKLOAD_SHAPES


EXPECTED_MANIFEST_NAME = "manifest.json"


class StaticVerifier:
    def verify(self, entry_path: Path) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []
        entry_path = entry_path.resolve()
        if entry_path.name != EXPECTED_MANIFEST_NAME:
            return _failed(
                checks,
                f"entrypoint must be named {EXPECTED_MANIFEST_NAME}",
            )
        if not entry_path.exists():
            return _failed(checks, f"{EXPECTED_MANIFEST_NAME} missing")
        checks.append({"name": "entrypoint_exists", "status": "passed", "message": str(entry_path)})

        candidate_root = entry_path.parent
        try:
            manifest = load_manifest(entry_path)
            checks.append({"name": "manifest_schema", "status": "passed", "message": None})
            checks.extend(validate_bundle_files(manifest, candidate_root=candidate_root))
            checks.append({"name": "static_bundle_validation", "status": "passed", "message": None})
        except ManifestValidationError as exc:
            return _failed(checks, str(exc))

        live_checks: dict[str, Any] = {"enabled": live_verifier_enabled(), "status": "skipped", "checks": []}
        if live_verifier_enabled():
            run_dir = new_run_dir("llm_kernel_verify")
            live_checks = run_live_preflight()
            checks.append(
                {
                    "name": "live_environment_preflight",
                    "status": "passed" if live_checks.get("valid") else "failed",
                    "message": None if live_checks.get("valid") else "H200 live preflight failed",
                }
            )
            if not live_checks.get("valid"):
                return _failed(checks, "H200 live preflight failed", manifest=manifest, live_checks=live_checks)
            rmsnorm_live = run_rmsnorm_live_checks(
                entry_path=entry_path,
                manifest=manifest,
                shapes=PUBLIC_WORKLOAD_SHAPES,
                benchmark=False,
                run_dir=run_dir,
            )
            live_checks["rmsnorm"] = rmsnorm_live
            live_checks["run_dir"] = str(run_dir)
            write_json(run_dir / "verify_summary.json", {"checks": checks, "live_checks": live_checks})
            checks.append(
                {
                    "name": "live_rmsnorm_correctness",
                    "status": "passed" if rmsnorm_live.get("valid") else "failed",
                    "message": rmsnorm_live.get("error"),
                }
            )
            if not rmsnorm_live.get("valid"):
                return _failed(checks, rmsnorm_live.get("error") or "live RMSNorm check failed", manifest=manifest, live_checks=live_checks)

        bundle = _bundle_summary(manifest)
        return {
            "status": "passed",
            "valid": True,
            "checks": checks,
            "feedback": {
                "error": None,
                "public_details": {
                    "implementation_count": bundle["implementation_count"],
                    "definitions": bundle["definitions"],
                    "live_checks": live_checks,
                },
            },
            "bundle": bundle,
            "live_checks": live_checks,
        }


def _failed(
    checks: list[dict[str, Any]],
    message: str,
    *,
    manifest: BundleManifest | None = None,
    live_checks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checks.append({"name": "static_bundle_validation", "status": "failed", "message": message})
    return {
        "status": "failed",
        "valid": False,
        "checks": checks,
        "feedback": {
            "error": message,
            "public_details": {
                "implementation_count": len(manifest.implementations) if manifest else 0,
                "definitions": list(manifest.definition_names) if manifest else [],
                "live_checks": live_checks or {"enabled": live_verifier_enabled(), "status": "skipped", "checks": []},
            },
        },
        "bundle": _bundle_summary(manifest) if manifest else None,
        "live_checks": live_checks,
    }


def _bundle_summary(manifest: BundleManifest) -> dict[str, Any]:
    return {
        "schema": manifest.schema,
        "target": dict(manifest.target),
        "implementation_count": len(manifest.implementations),
        "definitions": list(manifest.definition_names),
        "implementations": [
            {
                "id": implementation.id,
                "definition": implementation.definition,
                "language": implementation.language,
                "binding": implementation.binding,
                "entry_point": implementation.entry_point,
                "priority": implementation.priority,
                "fallback": implementation.fallback,
            }
            for implementation in manifest.implementations
        ],
    }
