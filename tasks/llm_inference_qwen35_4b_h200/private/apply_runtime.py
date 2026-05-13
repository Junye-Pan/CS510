from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class DispatchStats:
    calls: int = 0
    hits: int = 0
    misses: int = 0
    fallbacks: int = 0
    exceptions: int = 0
    definitions: dict[str, int] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "hits": self.hits,
            "misses": self.misses,
            "fallbacks": self.fallbacks,
            "exceptions": self.exceptions,
            "definitions": dict(sorted(self.definitions.items())),
        }


class BaselineOnlyApplyRuntime:
    """MVP apply runtime that records dispatch and always takes baseline fallback."""

    def __init__(self) -> None:
        self.stats = DispatchStats()

    def dispatch(self, def_name: str, args: tuple[Any, ...], kwargs: dict[str, Any], fallback: Callable[..., Any]) -> Any:
        self.stats.calls += 1
        self.stats.misses += 1
        self.stats.fallbacks += 1
        self.stats.definitions[def_name] = self.stats.definitions.get(def_name, 0) + 1
        try:
            return fallback(*args, **kwargs)
        except Exception:
            self.stats.exceptions += 1
            raise
