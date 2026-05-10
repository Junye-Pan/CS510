import unsloth
"""
Train Only On Confident Labels - Weak-to-Strong Generalization

This idea implements confidence-based label filtering for weak-to-strong learning.
The core hypothesis is that if the weak teacher is calibrated, uncertain predictions likely correspond to incorrect labels.

By filtering out these low-confidence samples from the training data before training
the strong model, we can improve the quality of the weak supervision signal and
potentially achieve better transfer performance.

Implementation:
1. Load pre-trained weak model artifacts (cached soft labels)
2. Filter samples where max(soft_label) is in the uncertainty range [0.35, 0.65]
3. Train strong model only on the confident (filtered) samples
4. Evaluate and compare with vanilla W2S baseline
"""
import sys
import gc
import time
from pathlib import Path

# Add w2s_research to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import wandb
from transformers import AutoTokenizer
from w2s_research.core import (
    RunConfig,
    create_run_arg_parser,
    load_dataset,
    train_model,
    find_latest_checkpoint,
    evaluate_model,
    set_seed,
    normalize_model_name_for_path,
)
from w2s_research.utils import (
    HierarchicalCache,
    compute_hyperparam_config_key,
    get_cached_weak_artifacts,
    get_fixed_weak_baseline,
    get_fixed_ceiling_baseline,
    get_cached_ceiling_result,
    evaluate_predictions_remote,
)
from w2s_research.core.data import BINARY_LABEL_TOKENS, COMPARISON_LABEL_TOKENS, detect_aar_mode


# Idea name for caching
IDEA_NAME = "train_only_on_confident_labels"

# Default confidence thresholds
DEFAULT_LOW_THRESHOLD = 0.25
DEFAULT_HIGH_THRESHOLD = 0.75


def filter_confident_samples(
    dataset,
    soft_labels,
    low_threshold: float = DEFAULT_LOW_THRESHOLD,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
):
    """
    Filter samples based on weak model confidence.

    Removes samples where the weak model's maximum class probability is in the
    uncertain range [low_threshold, high_threshold].

    Args:
        dataset: HuggingFace dataset with training samples
        soft_labels: List of [prob_class_0, prob_class_1] for each sample
        low_threshold: Lower bound of uncertainty range (default: 0.35)
        high_threshold: Upper bound of uncertainty range (default: 0.65)

    Returns:
        Tuple of:
        - filtered_dataset: Dataset with uncertain samples removed
        - filtered_soft_labels: Corresponding soft labels
        - filter_stats: Dictionary with filtering statistics
    """
    assert len(dataset) == len(soft_labels), (
        f"Dataset size ({len(dataset)}) must match soft_labels size ({len(soft_labels)})"
    )

    # Track indices to keep
    keep_indices = []
    filtered_soft_labels = []

    # Analyze confidence distribution
    confidence_values = []
    filtered_confidences = []

    for i, (soft_label) in enumerate(soft_labels):
        # Get max probability (confidence)
        max_prob = max(soft_label)
        confidence_values.append(max_prob)

        # Check if in uncertain range
        # A sample is uncertain if max_prob is in [low_threshold, high_threshold]
        # This means both classes have probability around 0.35-0.65
        is_uncertain = low_threshold <= max_prob <= high_threshold

        if not is_uncertain:
            # Keep this confident sample
            keep_indices.append(i)
            filtered_soft_labels.append(soft_label)
        else:
            # Filter out uncertain sample
            filtered_confidences.append(max_prob)

    # Compute statistics
    total_samples = len(dataset)
    kept_samples = len(keep_indices)
    filtered_samples = total_samples - kept_samples
    retention_rate = kept_samples / total_samples if total_samples > 0 else 0.0

    filter_stats = {
        "total_samples": total_samples,
        "kept_samples": kept_samples,
        "filtered_samples": filtered_samples,
        "retention_rate": retention_rate,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "avg_confidence_all": sum(confidence_values) / len(confidence_values) if confidence_values else 0.0,
        "avg_confidence_filtered": sum(filtered_confidences) / len(filtered_confidences) if filtered_confidences else 0.0,
    }

    # Log filtering statistics
    print(f"\n{'='*60}")
    print("CONFIDENCE FILTERING STATISTICS")
    print(f"{'='*60}")
    print(f"Confidence threshold range: [{low_threshold:.2f}, {high_threshold:.2f}]")
    print(f"Total samples: {total_samples}")
    print(f"Kept (confident) samples: {kept_samples} ({retention_rate*100:.1f}%)")
    print(f"Filtered (uncertain) samples: {filtered_samples} ({(1-retention_rate)*100:.1f}%)")
    print(f"\nAverage confidence (all samples): {filter_stats['avg_confidence_all']:.4f}")
    if filtered_confidences:
        print(f"Average confidence (filtered samples): {filter_stats['avg_confidence_filtered']:.4f}")
    print(f"{'='*60}\n")

    # Select the kept samples from dataset
    if kept_samples > 0:
        filtered_dataset = dataset.select(keep_indices)
    else:
        # Edge case: all samples filtered - return empty dataset
        filtered_dataset = dataset.select([])

    return filtered_dataset, filtered_soft_labels, filter_stats


def run_experiment(config: RunConfig):
    """
    Run train-only-on-confident-labels weak-to-strong experiment.

    This experiment filters out low-confidence weak labels before training
    the strong model, hypothesizing that uncertain predictions are more
    likely to be incorrect.

    Args:
        config: RunConfig with all experiment parameters

    Returns:
        Dictionary with results including weak_acc, transfer_acc, strong_acc, pgr
    """
    print("="*80)
    print("TRAIN ONLY ON CONFIDENT LABELS EXPERIMENT")
    print("="*80)
    print(f"Weak model: {config.weak_model}")
    print(f"Strong model: {config.strong_model}")
    print(f"Confidence thresholds: [{DEFAULT_LOW_THRESHOLD}, {DEFAULT_HIGH_THRESHOLD}]")
    print("="*80)

    # Extract dataset name from data_dir
    dataset_name = Path(config.data_dir).name

    # Build cache key with filtering parameters
    extra_params = {
        "conf_low": DEFAULT_LOW_THRESHOLD,
        "conf_high": DEFAULT_HIGH_THRESHOLD,
    }
    # Include train_size and test_size in cache key
    if config.train_size:
        extra_params["train_size"] = config.train_size
    if config.test_size:
        extra_params["test_size"] = config.test_size

    config_key = compute_hyperparam_config_key(
        strong_model=config.strong_model,
        weak_model=config.weak_model,
        dataset_name=dataset_name,
        epochs=config.epochs,
        lr=config.lr,
        batch_size=config.batch_size,
        scheduler=config.lr_schedule,
        loss=config.loss,
        **extra_params
    )

    # Check cache first
    cache = HierarchicalCache()
    cached_result = cache.get_seed_result(
        idea_name=IDEA_NAME,
        hyperparam_config_key=config_key,
        seed=config.seed,
        dataset_name=dataset_name,
    )

    if cached_result:
        print("\n✓ Found cached result!")
        print(f"  Weak accuracy: {cached_result['weak_acc']:.4f}")
        print(f"  Transfer accuracy: {cached_result['transfer_acc']:.4f}")
        if cached_result.get('pgr') is not None:
            print(f"  PGR: {cached_result['pgr']:.4f}")
        return cached_result

    # Set random seed for reproducibility
    set_seed(config.seed)

    # [1/5] Load data
    # Note: AAR-compliant - we don't need train_label.jsonl, only train_unlabel.jsonl and test.jsonl
    print("\n[1/5] Loading data...")
    datasets = load_dataset(config.data_dir, seed=config.seed)
    train_unlabel_ds = datasets["train_unlabel"]
    test_ds = datasets["test"]

    # Support quick testing with small train_size
    if config.train_size:
        train_unlabel_ds = train_unlabel_ds.select(range(min(config.train_size, len(train_unlabel_ds))))
    if config.test_size:
        test_ds = test_ds.select(range(min(config.test_size, len(test_ds))))

    print(f"Train (unlabeled): {len(train_unlabel_ds)} samples")
    print(f"Test: {len(test_ds)} samples")

    # [2/5] Load pre-trained weak model artifacts
    print("\n[2/5] Loading pre-trained weak model artifacts...")
    weak_artifacts = get_cached_weak_artifacts(
        weak_model=config.weak_model,
        dataset_name=dataset_name,
        seed=config.seed,
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        scheduler=config.lr_schedule,
    )

    if not weak_artifacts:
        raise RuntimeError(
            f"Weak model artifacts not found! Please run vanilla_w2s first to generate "
            f"weak model outputs for {config.weak_model} on {dataset_name}."
        )

    # Extract weak model outputs
    weak_acc = weak_artifacts["weak_acc"]
    soft_labels = weak_artifacts["soft_labels"]
    hard_label_acc = weak_artifacts["hard_label_acc"]

    # Truncate soft_labels if using train_size
    if config.train_size:
        soft_labels = soft_labels[:config.train_size]

    print(f"✓ Loaded pre-trained weak artifacts!")
    print(f"  Weak accuracy: {weak_acc:.4f}")
    print(f"  Hard weak label accuracy: {hard_label_acc:.4f}")
    print(f"  Soft labels: {len(soft_labels)} samples")

    # [3/5] Filter confident samples (THE NOVEL CONTRIBUTION)
    print("\n[3/5] Filtering confident samples (novel contribution)...")

    # Apply confidence filtering
    filtered_dataset, filtered_soft_labels, filter_stats = filter_confident_samples(
        dataset=train_unlabel_ds,
        soft_labels=soft_labels,
        low_threshold=DEFAULT_LOW_THRESHOLD,
        high_threshold=DEFAULT_HIGH_THRESHOLD,
    )

    # Check edge case: if all samples filtered, raise informative error
    if filter_stats["kept_samples"] == 0:
        raise RuntimeError(
            f"Confidence filtering removed ALL samples! "
            f"Threshold range [{DEFAULT_LOW_THRESHOLD}, {DEFAULT_HIGH_THRESHOLD}] is too narrow. "
            f"Consider widening the thresholds."
        )

    # Warning if retention is too low
    if filter_stats["retention_rate"] < 0.1:
        print(f"\n⚠ WARNING: Only {filter_stats['retention_rate']*100:.1f}% of samples retained!")
        print(f"  This may not be enough data for effective training.")
        print(f"  Consider widening confidence thresholds.")

    # Add filtered soft labels to filtered dataset
    train_filtered_with_soft = filtered_dataset.add_column("soft_label", filtered_soft_labels)
    print(f"\n✓ Filtered dataset ready: {len(train_filtered_with_soft)} samples")

    # [4/5] Train strong model on filtered data
    print("\n[4/5] Training strong model on FILTERED weak-labeled data...")
    start_time = time.time()

    # Get label token IDs for training
    temp_tokenizer = AutoTokenizer.from_pretrained(config.strong_model)

    # Detect format from dataset
    sample_columns = set(train_filtered_with_soft.column_names)
    is_binary_format = 'question' in sample_columns and 'choice' in sample_columns

    if is_binary_format:
        label_tokens = BINARY_LABEL_TOKENS
    else:
        label_tokens = COMPARISON_LABEL_TOKENS

    label_token_ids = [temp_tokenizer.encode(tok, add_special_tokens=False)[0] for tok in label_tokens]

    # Define output directory
    weak_name = normalize_model_name_for_path(config.weak_model).replace('/', '_')
    strong_name = normalize_model_name_for_path(config.strong_model).replace('/', '_')
    workspace = os.getenv("WORKSPACE_DIR", str(Path(__file__).parent.parent.parent.parent))
    strong_output_dir = f"{workspace}/results/{dataset_name}_{IDEA_NAME}/{config_key}/seed_{config.seed}"

    # Train strong model on filtered dataset
    strong_results = train_model(
        model_name=config.strong_model,
        train_dataset=train_filtered_with_soft,
        test_dataset=test_ds,
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        lr=config.lr,
        epochs=config.epochs,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_ctx=config.max_ctx,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        load_in_4bit=config.load_in_4bit,
        optimizer=config.optimizer,
        lr_scheduler_type=config.lr_schedule,
        output_dir=strong_output_dir,
        bf16=config.bf16,
        fp16=config.fp16,
        seed=config.seed,
    )
    strong_training_time = time.time() - start_time

    transfer_acc = strong_results["final_eval"]["accuracy"]
    print(f"\n✓ Transfer model test accuracy: {transfer_acc:.4f}")

    # Clean up GPU memory
    gc.collect()
    torch.cuda.empty_cache()

    # [5/5] Get ceiling performance and compute PGR
    print("\n[5/5] Getting ceiling performance and computing PGR...")

    ceiling_result = get_cached_ceiling_result(
        strong_model=config.strong_model,
        dataset_name=dataset_name,
        seed=config.seed,
        epochs=config.epochs,
        lr=config.lr,
        batch_size=config.batch_size,
        scheduler=config.lr_schedule,
    )

    if ceiling_result:
        strong_acc = ceiling_result.get("strong_acc", ceiling_result.get("strong_gt_acc", 0.0))
        print(f"✓ Strong model (ceiling) accuracy: {strong_acc:.4f}")
    else:
        print("⚠ Ceiling result not found in cache")
        strong_acc = None

    # Get fixed baselines for PGR computation
    fixed_weak_mean, fixed_weak_se, weak_num_seeds = get_fixed_weak_baseline(
        weak_model=config.weak_model,
        dataset_name=dataset_name,
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        scheduler=config.lr_schedule,
    )

    fixed_strong_mean, fixed_strong_se, strong_num_seeds = get_fixed_ceiling_baseline(
        strong_model=config.strong_model,
        dataset_name=dataset_name,
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        scheduler=config.lr_schedule,
    )

    # Compute PGR using fixed baselines
    if fixed_weak_mean is not None and fixed_strong_mean is not None and fixed_strong_mean > fixed_weak_mean:
        pgr = (transfer_acc - fixed_weak_mean) / (fixed_strong_mean - fixed_weak_mean)
        print(f"✓ Using fixed baselines: weak={fixed_weak_mean:.4f} ({weak_num_seeds} seeds), strong={fixed_strong_mean:.4f} ({strong_num_seeds} seeds)")
        print(f"✓ PGR = ({transfer_acc:.4f} - {fixed_weak_mean:.4f}) / ({fixed_strong_mean:.4f} - {fixed_weak_mean:.4f}) = {pgr:.4f}")
    else:
        pgr = None
        print(f"⚠ Fixed baselines not available yet (weak={weak_num_seeds} seeds, strong={strong_num_seeds} seeds)")
        print(f"  PGR will be computed by orchestrator after all seeds complete")

    # Print final results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Weak Model Accuracy (this seed): {weak_acc:.4f}")
    print(f"Hard Weak Label Accuracy (all samples): {hard_label_acc:.4f}")
    print(f"Samples Retained: {filter_stats['kept_samples']}/{filter_stats['total_samples']} ({filter_stats['retention_rate']*100:.1f}%)")
    if strong_acc is not None:
        print(f"Strong Model (ceiling, this seed): {strong_acc:.4f}")
    print(f"Transfer Accuracy: {transfer_acc:.4f}")
    if fixed_weak_mean is not None:
        print(f"Fixed Weak Baseline: {fixed_weak_mean:.4f} ({weak_num_seeds} seeds)")
    if fixed_strong_mean is not None:
        print(f"Fixed Strong Baseline: {fixed_strong_mean:.4f} ({strong_num_seeds} seeds)")
    if pgr is not None:
        print(f"PGR (fixed baselines): {pgr:.4f}")
    else:
        print(f"PGR: N/A (fixed baselines not available yet)")
    print(f"Training Time: {strong_training_time:.2f}s")
    print("="*80)

    # Package results
    results = {
        "idea_name": IDEA_NAME,
        "weak_model": config.weak_model,
        "strong_model": config.strong_model,
        "weak_acc": weak_acc,
        "hard_weak_label_acc": hard_label_acc,
        "strong_acc": strong_acc,
        "transfer_acc": transfer_acc,
        "pgr": pgr,
        "fixed_weak_baseline": fixed_weak_mean,
        "fixed_strong_baseline": fixed_strong_mean,
        "training_time": strong_training_time,
        # Filtering-specific metrics
        "filter_stats": filter_stats,
        "confidence_thresholds": {
            "low": DEFAULT_LOW_THRESHOLD,
            "high": DEFAULT_HIGH_THRESHOLD,
        },
        "config": {
            "epochs": config.epochs,
            "lr": config.lr,
            "batch_size": config.batch_size,
            "seed": config.seed,
            "loss": config.loss,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "load_in_4bit": config.load_in_4bit,
        },
        "detailed_eval": {
            "strong": {"final_eval": strong_results["final_eval"]},
        }
    }

    # Cache the result
    cache.set_seed_result(
        idea_name=IDEA_NAME,
        hyperparam_config_key=config_key,
        seed=config.seed,
        result_data=results,
        dataset_name=dataset_name,
    )
    print(f"\n✓ Cached result with config key: {config_key}")

    # Log to wandb
    wandb.init(
        project="generalization",
        name=f"{IDEA_NAME}_{weak_name}_{strong_name}_seed{config.seed}",
        dir=strong_output_dir,
        config={
            "experiment": IDEA_NAME,
            "weak_model": config.weak_model,
            "strong_model": config.strong_model,
            "seed": config.seed,
            "epochs": config.epochs,
            "loss": config.loss,
            "confidence_low_threshold": DEFAULT_LOW_THRESHOLD,
            "confidence_high_threshold": DEFAULT_HIGH_THRESHOLD,
        },
        reinit=True,
    )

    log_dict = {
        "weak_acc": weak_acc,
        "hard_weak_label_acc": hard_label_acc,
        "transfer_acc": transfer_acc,
        "training_time": strong_training_time,
        # Filtering metrics
        "filter/retention_rate": filter_stats["retention_rate"],
        "filter/kept_samples": filter_stats["kept_samples"],
        "filter/filtered_samples": filter_stats["filtered_samples"],
    }
    if strong_acc is not None:
        log_dict["strong_acc"] = strong_acc
    if pgr is not None:
        log_dict["pgr"] = pgr
    wandb.log(log_dict)

    wandb.finish()

    # Clean up GPU memory
    gc.collect()
    torch.cuda.empty_cache()
    print(f"✓ GPU memory cleaned up")

    return results


def main():
    parser = create_run_arg_parser(description="Run train-only-on-confident-labels W2S experiment")

    args = parser.parse_args()
    config = RunConfig.from_args(args)

    # Run experiment
    results = run_experiment(config)

    print("\nResults:", results)
    print(f"  Weak accuracy (this seed): {results['weak_acc']:.4f}")
    print(f"  Transfer accuracy: {results['transfer_acc']:.4f}")
    if results.get('strong_acc') is not None:
        print(f"  Strong GT accuracy (this seed): {results['strong_acc']:.4f}")
    if results.get('fixed_weak_baseline') is not None:
        print(f"  Fixed Weak Baseline: {results['fixed_weak_baseline']:.4f}")
    if results.get('fixed_strong_baseline') is not None:
        print(f"  Fixed Strong Baseline: {results['fixed_strong_baseline']:.4f}")
    if results.get('pgr') is not None:
        print(f"  PGR (fixed baselines): {results['pgr']:.4f}")
    else:
        print(f"  PGR: N/A (fixed baselines not available yet)")
    # Filtering-specific results
    print(f"\n  Filtering Results:")
    print(f"    Samples retained: {results['filter_stats']['kept_samples']}/{results['filter_stats']['total_samples']} ({results['filter_stats']['retention_rate']*100:.1f}%)")


if __name__ == "__main__":
    main()
