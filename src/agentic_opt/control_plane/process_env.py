from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any


SAFE_INHERITED_ENV_NAMES = {
    "CURL_CA_BUNDLE",
    "GIT_SSL_CAINFO",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LOGNAME",
    "NO_COLOR",
    "PATH",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_FILE",
    "TEMP",
    "TMP",
    "TMPDIR",
    "USER",
}

SENSITIVE_ENV_NAME_RE = re.compile(
    r"(?:"
    r"ACCESS[_-]?TOKEN|"
    r"API[_-]?KEY|"
    r"AUTH|"
    r"CODEX_HOME|"
    r"COOKIE|"
    r"CREDENTIAL|"
    r"ID[_-]?TOKEN|"
    r"PASSWORD|"
    r"PRIVATE[_-]?KEY|"
    r"REFRESH[_-]?TOKEN|"
    r"SECRET|"
    r"TOKEN"
    r")",
    re.IGNORECASE,
)


def is_sensitive_env_name(name: str) -> bool:
    return bool(SENSITIVE_ENV_NAME_RE.search(name))


def sanitize_env(env: Mapping[str, Any] | None) -> dict[str, str]:
    """Return stringified env values safe to persist in control-plane records."""

    sanitized: dict[str, str] = {}
    for key, value in (env or {}).items():
        name = str(key)
        if is_sensitive_env_name(name):
            continue
        sanitized[name] = str(value)
    return sanitized


def build_subprocess_env(overrides: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Build a minimal subprocess env instead of inheriting the full host env."""

    env = {
        name: os.environ[name]
        for name in SAFE_INHERITED_ENV_NAMES
        if name in os.environ and not is_sensitive_env_name(name)
    }
    env.update(sanitize_env(overrides))
    env.pop("PYTHONHOME", None)
    return env
