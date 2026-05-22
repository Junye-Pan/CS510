from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


class DockerImagePolicyError(ValueError):
    def __init__(self, reason: str, message: str, *, decision: dict[str, Any]) -> None:
        super().__init__(message)
        self.reason = reason
        self.decision = decision


@dataclass(frozen=True)
class DockerImageIdentity:
    input_ref: str
    image_id: str | None
    image_digest: str | None
    repo_digests: tuple[str, ...]
    repo_tags: tuple[str, ...]
    registries: tuple[str, ...]
    repositories: tuple[str, ...]
    immutable_references: tuple[str, ...]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "input_ref": self.input_ref,
            "image_id": self.image_id,
            "image_digest": self.image_digest,
            "repo_digests": list(self.repo_digests),
            "repo_tags": list(self.repo_tags),
            "registries": list(self.registries),
            "repositories": list(self.repositories),
            "immutable_references": list(self.immutable_references),
        }


def docker_image_identity(*, image_ref: str, image_info: dict[str, Any]) -> DockerImageIdentity:
    repo_digests = tuple(str(item) for item in (image_info.get("RepoDigests") or []) if item)
    repo_tags = tuple(str(item) for item in (image_info.get("RepoTags") or []) if item)
    image_id = str(image_info.get("Id") or "") or None
    image_digest = _docker_image_digest(image_info)
    references = (str(image_ref), *repo_digests, *repo_tags)
    registries = tuple(sorted({registry for ref in references for registry in [_registry_for_ref(ref)] if registry}))
    repositories = tuple(sorted({repo for ref in references for repo in [_repository_for_ref(ref)] if repo}))
    immutable_references = tuple(ref for ref in references if _is_immutable_ref(ref))
    return DockerImageIdentity(
        input_ref=str(image_ref),
        image_id=image_id,
        image_digest=image_digest,
        repo_digests=repo_digests,
        repo_tags=repo_tags,
        registries=registries,
        repositories=repositories,
        immutable_references=immutable_references,
    )


def evaluate_docker_image_policy(
    *,
    identity: DockerImageIdentity,
    policy: dict[str, Any] | None,
    source: str,
) -> dict[str, Any]:
    policy = dict(policy or {})
    allowed_registries = _string_set(policy.get("trusted_registries") or policy.get("allowed_registries"))
    denied_registries = _string_set(policy.get("denied_registries") or policy.get("blocked_registries"))
    allowed_prefixes = _string_list(policy.get("allowed_repo_prefixes") or policy.get("trusted_repo_prefixes"))
    denied_prefixes = _string_list(policy.get("denied_repo_prefixes") or policy.get("blocked_repo_prefixes"))
    allowed_digests = _string_set(policy.get("allowed_image_digests") or policy.get("trusted_image_digests"))
    require_repo_digest = bool(policy.get("require_repo_digest"))
    require_immutable = bool(policy.get("require_immutable", True))
    allow_local_builds = bool(policy.get("allow_local_builds", True))

    decision = {
        "allowed": True,
        "reason": None,
        "source": source,
        "policy": {
            "trusted_registries": sorted(allowed_registries),
            "denied_registries": sorted(denied_registries),
            "allowed_repo_prefixes": allowed_prefixes,
            "denied_repo_prefixes": denied_prefixes,
            "allowed_image_digests": sorted(allowed_digests),
            "require_repo_digest": require_repo_digest,
            "require_immutable": require_immutable,
            "allow_local_builds": allow_local_builds,
        },
        "identity": identity.to_jsonable(),
        "checks": [],
    }

    def check(name: str, passed: bool, message: str | None = None) -> None:
        decision["checks"].append({"name": name, "status": "passed" if passed else "failed", "message": message})

    if denied_registries:
        blocked = sorted(set(identity.registries) & denied_registries)
        check("registry_not_denied", not blocked, f"denied registries: {blocked}" if blocked else None)
        if blocked:
            return _blocked(decision, "denied_registry", f"Docker image registry is denied: {blocked[0]}")

    if allowed_registries:
        matched = bool(set(identity.registries) & allowed_registries)
        check("registry_trusted", matched, None if matched else f"registries {list(identity.registries)} are not trusted")
        if not matched:
            return _blocked(decision, "untrusted_registry", "Docker image registry is not trusted by experiment policy")

    if denied_prefixes:
        blocked_prefix = _first_matching_prefix(identity.repositories, denied_prefixes)
        check("repo_prefix_not_denied", blocked_prefix is None, f"denied repo prefix: {blocked_prefix}" if blocked_prefix else None)
        if blocked_prefix:
            return _blocked(decision, "denied_repo_prefix", f"Docker image repository matches denied prefix: {blocked_prefix}")

    if allowed_prefixes:
        matched_prefix = _first_matching_prefix(identity.repositories, allowed_prefixes)
        check("repo_prefix_allowed", matched_prefix is not None, None if matched_prefix else f"repositories {list(identity.repositories)} do not match allowed prefixes")
        if matched_prefix is None:
            return _blocked(decision, "untrusted_repo_prefix", "Docker image repository is not allowed by experiment policy")

    if allowed_digests:
        digest_candidates = {item for item in (identity.image_digest, identity.image_id) if item}
        digest_candidates.update(_digest_from_repo_digest(ref) for ref in identity.repo_digests)
        digest_candidates.discard(None)
        matched_digest = bool(digest_candidates & allowed_digests)
        check("image_digest_allowed", matched_digest, None if matched_digest else "image digest is not in trusted digest set")
        if not matched_digest:
            return _blocked(decision, "untrusted_image_digest", "Docker image digest is not trusted by experiment policy")

    has_repo_digest = bool(identity.repo_digests)
    check("repo_digest_present", has_repo_digest or not require_repo_digest, "repo digest is required" if require_repo_digest and not has_repo_digest else None)
    if require_repo_digest and not has_repo_digest:
        return _blocked(decision, "repo_digest_required", "Docker image must resolve to a registry repo digest")

    has_immutable_identity = bool(identity.immutable_references or identity.image_id or identity.image_digest)
    check("immutable_identity_present", has_immutable_identity or not require_immutable, "immutable image identity is required" if require_immutable and not has_immutable_identity else None)
    if require_immutable and not has_immutable_identity:
        return _blocked(decision, "immutable_identity_required", "Docker image must have an immutable digest or image id")

    is_local_only = identity.registries == ("local",) and not identity.repo_digests
    check("local_build_allowed", allow_local_builds or not is_local_only, "local-only image ids are disabled" if is_local_only and not allow_local_builds else None)
    if is_local_only and not allow_local_builds:
        return _blocked(decision, "local_build_denied", "Docker image policy does not allow local-only images")

    return decision


def enforce_docker_image_policy(
    *,
    identity: DockerImageIdentity,
    policy: dict[str, Any] | None,
    source: str,
) -> dict[str, Any]:
    decision = evaluate_docker_image_policy(identity=identity, policy=policy, source=source)
    if not decision["allowed"]:
        raise DockerImagePolicyError(str(decision["reason"]), str(decision["message"]), decision=decision)
    return decision


def docker_policy_from_experiment(experiment: dict[str, Any] | None, provider_config: dict[str, Any] | None = None) -> dict[str, Any]:
    provider_config = dict(provider_config or {})
    config = (experiment or {}).get("config") or {}
    policy = (experiment or {}).get("policy") or {}
    environment_policy = policy.get("environments") if isinstance(policy.get("environments"), dict) else {}
    docker_policy = policy.get("docker_image") if isinstance(policy.get("docker_image"), dict) else {}
    docker_config_policy = config.get("docker_image_policy") if isinstance(config.get("docker_image_policy"), dict) else {}
    nested_environment_policy = environment_policy.get("docker_image") if isinstance(environment_policy.get("docker_image"), dict) else {}
    inline_policy = provider_config.get("policy") if isinstance(provider_config.get("policy"), dict) else {}
    trust_policy = provider_config.get("trust_policy") if isinstance(provider_config.get("trust_policy"), dict) else {}
    result: dict[str, Any] = {}
    result.update(docker_config_policy)
    result.update(docker_policy)
    result.update(nested_environment_policy)
    result.update(inline_policy)
    result.update(trust_policy)
    return result


def _blocked(decision: dict[str, Any], reason: str, message: str) -> dict[str, Any]:
    decision["allowed"] = False
    decision["reason"] = reason
    decision["message"] = message
    return decision


def _string_set(raw: Any) -> set[str]:
    return set(_string_list(raw))


def _string_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    return [str(item) for item in raw if item is not None]


def _first_matching_prefix(repositories: tuple[str, ...], prefixes: list[str]) -> str | None:
    for prefix in prefixes:
        for repository in repositories:
            if repository == prefix or repository.startswith(prefix.rstrip("/") + "/"):
                return prefix
    return None


def _registry_for_ref(reference: str) -> str:
    name = _reference_name(reference)
    if not name or name == "local":
        return "local"
    parts = name.split("/")
    if len(parts) == 1:
        return "docker.io"
    first = parts[0]
    if "." in first or ":" in first or first == "localhost":
        return first
    return "docker.io"


def _repository_for_ref(reference: str) -> str:
    name = _reference_name(reference)
    if not name:
        return ""
    if name == "local":
        return "local"
    parts = name.split("/")
    if len(parts) == 1:
        return f"library/{name}"
    first = parts[0]
    if "." in first or ":" in first or first == "localhost":
        return "/".join(parts[1:]) or "local"
    if first == "docker.io":
        return "/".join(parts[1:]) or "local"
    return name


def _reference_name(reference: str) -> str:
    ref = reference.strip()
    if not ref:
        return ""
    if ref.startswith("sha256:"):
        return "local"
    if "/" not in ref and re.fullmatch(r"[a-fA-F0-9]{12,}", ref):
        return "local"
    name = ref.split("@", 1)[0]
    last = name.rsplit("/", 1)[-1]
    if ":" in last:
        name = name.rsplit(":", 1)[0]
    return name or "local"


def _is_immutable_ref(reference: str) -> bool:
    return "@sha256:" in reference or reference.startswith("sha256:")


def _digest_from_repo_digest(reference: str) -> str | None:
    if "@sha256:" not in reference:
        return None
    return "sha256:" + reference.split("@sha256:", 1)[1]


def _docker_image_digest(image_info: dict[str, Any]) -> str | None:
    repo_digests = image_info.get("RepoDigests") or []
    if repo_digests:
        first = str(repo_digests[0])
        if "@sha256:" in first:
            return "sha256:" + first.split("@sha256:", 1)[1]
    image_id = image_info.get("Id")
    if isinstance(image_id, str) and image_id.startswith("sha256:"):
        return image_id
    return None
