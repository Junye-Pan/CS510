"""
Core library for weak-to-strong research.
Self-contained, no external dependencies.
"""

from .config import (
    RunConfig,
    create_run_arg_parser,
    BASELINE_EPOCHS,
)

from .data import (
    load_dataset,
    format_classification_as_causal,
    detect_aar_mode,
)

from .train import (
    train_model,
    find_latest_checkpoint,
    load_model_from_checkpoint,
    normalize_model_name_for_path,
    is_base_model,
)

from .eval import (
    evaluate_model,
    print_evaluation_results,
    generate_predictions,
    save_predictions,
    compute_metrics_from_predictions,
)

from .vllm_inference import (
    predict_batch_labels,
)

from .seed_utils import (
    set_seed,
)

__all__ = [
    # Config
    "RunConfig",
    "create_run_arg_parser",
    "BASELINE_EPOCHS",
    # Data
    "load_dataset",
    "format_classification_as_causal",
    "detect_aar_mode",
    # Training
    "train_model",
    "find_latest_checkpoint",
    "load_model_from_checkpoint",
    "normalize_model_name_for_path",
    "is_base_model",
    "evaluate_model",
    "print_evaluation_results",
    "generate_predictions",
    "save_predictions",
    "compute_metrics_from_predictions",
    # Inference
    "predict_batch_labels",
    # Seed utilities
    "set_seed",
]
