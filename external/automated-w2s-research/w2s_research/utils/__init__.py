"""
Utilities for weak-to-strong research.
"""
# Import from unified hierarchical cache
from .hierarchical_cache import (
    # Core hierarchical cache
    HierarchicalCache,
    compute_hyperparam_config_key,
    # Weak artifact support
    compute_weak_artifact_cache_key,
    get_cached_weak_artifacts,
    cache_weak_artifacts,
    # Ceiling result support
    get_cached_ceiling_result,
    # Fixed baseline support (for PGR computation)
    get_fixed_weak_baseline,
    get_fixed_ceiling_baseline,
)

# Re-export BASELINE_EPOCHS from core
from w2s_research.core.config import BASELINE_EPOCHS

from .logging_utils import (
    init_weave,
    get_weave_op,
    get_weave_attributes,
    set_weave_config,
    ENABLE_WEAVE_TRACING,
)

from .remote_evaluation import (
    evaluate_predictions_remote,
    is_server_available,
)

__all__ = [
    # Hierarchical cache (unified system)
    "HierarchicalCache",
    "compute_hyperparam_config_key",
    "compute_weak_artifact_cache_key",
    "get_cached_weak_artifacts",
    "cache_weak_artifacts",
    "get_cached_ceiling_result",
    # Fixed baseline support (for PGR computation)
    "get_fixed_weak_baseline",
    "get_fixed_ceiling_baseline",
    # Baseline epochs constant
    "BASELINE_EPOCHS",
    # Weave tracing utilities
    "init_weave",
    "get_weave_op",
    "get_weave_attributes",
    "set_weave_config",
    "ENABLE_WEAVE_TRACING",
    # Remote evaluation (for AAR mode)
    "evaluate_predictions_remote",
    "is_server_available",
]
