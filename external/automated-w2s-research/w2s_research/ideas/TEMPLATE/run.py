"""
Template for new research ideas - OPTIMIZED FOR REUSING PRE-TRAINED MODELS.

Since we've already trained weak and strong models (baselines), new ideas only need
to implement their novel contribution (e.g., new loss, new training procedure).

Key optimizations:
1. Load pre-trained weak model artifacts (checkpoint, soft labels, accuracy)
2. Load cached strong model ground-truth accuracy (ceiling performance)
3. Implement only your novel approach
4. Minimal training - reuse as much as possible

==============================================================================
AAR MODE (Automated Alignment Research) - AVAILABLE DATA
==============================================================================
Worker pods run in AAR mode where ground truth labels are held server-side.

DATA AVAILABLE TO WORKERS:
- train_unlabel.jsonl - Training samples (prompts + answer choices)
- test.jsonl - Test samples (prompts + answer choices)  
- Weak model checkpoint (pre-trained, available in cache)
- Weak model soft labels for all samples (pre-computed, available in cache)
- Weak model accuracy (pre-computed baseline, available in cache)
- Strong model ceiling accuracy (pre-computed baseline, available in cache)

EVALUATION:
- Generate predictions on test set
- Worker sends predictions to server via remote evaluation API
- Server computes and returns PGR (available for iteration decisions)
- Final metrics saved to results.json
==============================================================================
"""
import os
import sys
import gc
import json
from pathlib import Path

# Add w2s_research to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from w2s_research.core import (
    RunConfig,
    create_run_arg_parser,
    load_dataset,
    detect_aar_mode,  # AAR mode detection utility
    train_model,
    find_latest_checkpoint,
    load_model_from_checkpoint,
    evaluate_model,
    set_seed,
    normalize_model_name_for_path,
    generate_predictions,
)
from w2s_research.utils import (
    HierarchicalCache,
    compute_hyperparam_config_key,
    get_cached_weak_artifacts,
    get_cached_ceiling_result,
    get_fixed_weak_baseline,
    get_fixed_ceiling_baseline,
    evaluate_predictions_remote,
)


def run_experiment(config: RunConfig):
    """
    Run your custom weak-to-strong experiment.

    This template assumes:
    1. Weak model is already trained (load from cache)
    2. Strong model ceiling (trained on ground truth) is already computed (load from cache)
    3. You only need to implement your novel training procedure

    Args:
        config: RunConfig with all experiment parameters

    Returns:
        Dictionary with results including weak_acc, transfer_acc, strong_acc, pgr
        In AAR mode: includes "predictions" list and "aar_mode": True
    """
    print("="*80)
    print("YOUR CUSTOM EXPERIMENT")
    print("="*80)
    print(f"Weak model: {config.weak_model}")
    print(f"Strong model: {config.strong_model}")
    print(f"Novel approach: [DESCRIBE YOUR APPROACH HERE]")
    print("="*80)

    # Extract dataset name from data_dir
    dataset_name = Path(config.data_dir).name

    # ===========================================================================
    # AAR MODE DETECTION - Do this EARLY before any data loading
    # ===========================================================================
    aar_mode = detect_aar_mode(config.data_dir)
    if aar_mode:
        print("\n⚠️  AAR MODE DETECTED - Ground truth labels NOT available")
        print("   - Will return predictions instead of accuracy")
        print("   - PGR will be computed server-side after upload")
        print("   - DO NOT access dataset['label'] - it contains placeholder values (-1)")
    # ===========================================================================

    # Check if this exact configuration is already cached
    cache = HierarchicalCache()
    extra_params = {}
    # Include train_size and test_size in cache key to differentiate quick validation from full training
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

    cached_result = cache.get_seed_result(
        idea_name="my_custom_idea",  # TODO: Replace with your idea name
        hyperparam_config_key=config_key,
        seed=config.seed,
        dataset_name=dataset_name,
    )

    if cached_result:
        print("\n✓ Found cached result!")
        return cached_result

    # Set random seed for reproducibility
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===========================================================================
    # [1/5] LOAD DATA - Use aar_mode parameter for proper handling
    # ===========================================================================
    print("\n[1/5] Loading data...")
    # Pass aar_mode to load_dataset - this handles missing labels gracefully
    datasets = load_dataset(config.data_dir, seed=config.seed, aar_mode=aar_mode)
    train_unlabel_ds = datasets["train_unlabel"]
    test_ds = datasets["test"]

    # Support quick testing with small train_size
    if config.train_size:
        train_unlabel_ds = train_unlabel_ds.select(range(min(config.train_size, len(train_unlabel_ds))))
    if config.test_size:
        test_ds = test_ds.select(range(min(config.test_size, len(test_ds))))

    print(f"Train (unlabeled): {len(train_unlabel_ds)} samples")
    print(f"Test: {len(test_ds)} samples")
    
    # ===========================================================================
    # CRITICAL AAR MODE WARNING: DO NOT ACCESS GROUND TRUTH LABELS
    # ===========================================================================
    # In AAR mode, train_unlabel_ds["label"] and test_ds["label"] contain -1.
    # DO NOT use these for:
    #   - Computing accuracy
    #   - Determining which weak model predictions are "errors"
    #   - Any logic that requires ground truth
    #
    # WRONG (will fail in AAR mode):
    #   ground_truth = train_unlabel_ds["label"]  # All -1 in AAR mode!
    #   errors = [1 if pred != gt else 0 for pred, gt in zip(preds, ground_truth)]
    #
    # RIGHT (works in AAR mode):
    #   confidence = [max(soft) for soft in soft_labels]  # Use weak model outputs
    #   low_conf_mask = [c < 0.6 for c in confidence]  # Derived from weak model
    # ===========================================================================

    # [2/5] Load pre-trained weak model artifacts (REUSE - NO RETRAINING!)
    print("\n[2/5] Loading pre-trained weak model artifacts...")
    weak_artifacts = get_cached_weak_artifacts(
        weak_model=config.weak_model,
        dataset_name=dataset_name,
        seed=config.seed,
        batch_size=config.batch_size,
        lr=config.lr,
        scheduler=config.lr_schedule,
    )

    if not weak_artifacts:
        raise RuntimeError(
            f"Weak model artifacts not found!"
        )

    # Extract pre-trained weak model outputs
    weak_acc = weak_artifacts["weak_acc"]
    soft_labels = weak_artifacts["soft_labels"]
    if config.train_size:
        soft_labels = soft_labels[:config.train_size]
    hard_label_acc = weak_artifacts["hard_label_acc"]

    print(f"✓ Loaded pre-trained weak artifacts!")
    print(f"  Weak accuracy: {weak_acc:.4f}")
    print(f"  Hard weak label accuracy: {hard_label_acc:.4f}")
    print(f"  Soft labels: {len(soft_labels)} samples")

    # Add soft labels to unlabeled dataset
    train_unlabel_with_soft = train_unlabel_ds.add_column("soft_label", soft_labels)

    # [3/5] Load pre-trained strong model ceiling accuracy (REFERENCE - NO RETRAINING!)
    print("\n[3/5] Loading pre-trained strong model ceiling accuracy...")
    ceiling_result = get_cached_ceiling_result(
        strong_model=config.strong_model,
        dataset_name=dataset_name,
        seed=config.seed,
        lr=config.lr,
        batch_size=config.batch_size,
        scheduler=config.lr_schedule,
    )

    if not ceiling_result:
        raise RuntimeError(
            f"Strong model ceiling not found!"
        )

    strong_acc = ceiling_result["strong_acc"]
    print(f"✓ Loaded strong model ceiling accuracy: {strong_acc:.4f}")

    # [4/5] YOUR NOVEL APPROACH - Implement ONLY the unique part
    print("\n[4/5] Training strong model with YOUR NOVEL APPROACH...")
    print("TODO: Implement your custom training procedure here")
    print("Example approaches:")
    print("  - Custom loss function (e.g., confidence-weighted, entropy-based)")
    print("  - Custom label processing (e.g., filtering, reweighting)")
    print("  - Custom training procedure (e.g., curriculum learning, meta-learning)")
    print("")

    # ===========================================================================
    # MEMORY MANAGEMENT BEST PRACTICES (FOLLOW THESE TO PREVENT CUDA OOM!)
    # ===========================================================================
    # 
    # Hardware: Single H200 SXM GPU (140 GB VRAM). Must complete within 24 hours.
    #
    # 1. Use small batch sizes with gradient accumulation:
    #    batch_size=4, gradient_accumulation_steps=8 → effective batch of 32
    #
    # 2. If you load any intermediate models, clean up before training:
    #    del model, tokenizer
    #    gc.collect()
    #    torch.cuda.empty_cache()
    #
    # 3. Use torch.no_grad() for all inference operations:
    #    with torch.no_grad():
    #        predictions = model(inputs)
    #
    # 4. Process large datasets in batches during inference
    # ===========================================================================

    # TODO: Implement your approach here
    # Example skeleton:
    #
    # 1. Process weak labels with your custom method
    # processed_labels = your_custom_label_processor(soft_labels, hard_label_acc)
    #
    # 2. Train strong model with your custom loss/procedure
    #    NOTE: Use small batch_size + gradient_accumulation_steps to prevent OOM!
    #    NOTE: Pass aar_mode=aar_mode to train_model for proper evaluation handling!
    # strong_results = train_model(
    #     model_name=config.strong_model,
    #     train_dataset=train_unlabel_with_soft,  # or your processed version
    #     test_dataset=test_ds,
    #     batch_size=4,  # Keep small to prevent OOM
    #     gradient_accumulation_steps=8,  # Effective batch = 32
    #     aar_mode=aar_mode,  # IMPORTANT: Pass AAR mode flag!
    #     ...
    # )
    #
    # 3. Clean up after training (if doing multiple model operations)
    # gc.collect()
    # torch.cuda.empty_cache()

    # PLACEHOLDER - Replace with actual implementation
    # For demonstration, we'll show the AAR-mode-aware evaluation pattern
    transfer_acc = None
    predictions = None

    # ===========================================================================
    # [5/5] EVALUATION - Handle AAR mode properly
    # ===========================================================================
    if aar_mode:
        print("\n[5/5] AAR Mode - Generating predictions (accuracy computed server-side)...")
        # In AAR mode: generate predictions, don't compute accuracy
        # The server will compute accuracy using ground truth labels it has access to
        #
        # Example pattern for generating predictions:
        # predictions = generate_predictions(
        #     test_dataset=formatted_test,
        #     label_token_ids=label_token_ids,
        #     tokenizer=tokenizer,
        #     model_name=config.strong_model,
        #     lora_checkpoint=checkpoint_path,
        # )["predictions"]
        #
        # For this placeholder, we'll just set empty predictions
        predictions = []  # TODO: Replace with actual predictions from your trained model
        print(f"  Generated {len(predictions)} predictions")
        print("  Accuracy and PGR will be computed server-side")
        
        # Skip PGR computation in AAR mode
        pgr = None
        fixed_weak_mean = None
        fixed_strong_mean = None
    else:
        print("\n[5/5] Computing PGR with fixed baselines...")
        
        # Get fixed weak baseline (averaged across all cached seeds)
        fixed_weak_mean, fixed_weak_se, weak_num_seeds = get_fixed_weak_baseline(
            weak_model=config.weak_model,
            dataset_name=dataset_name,
            batch_size=config.batch_size,
            lr=config.lr,
            scheduler=config.lr_schedule,
        )
        
        # Get fixed ceiling baseline (averaged across all cached seeds)
        fixed_strong_mean, fixed_strong_se, strong_num_seeds = get_fixed_ceiling_baseline(
            strong_model=config.strong_model,
            dataset_name=dataset_name,
            batch_size=config.batch_size,
            lr=config.lr,
            scheduler=config.lr_schedule,
        )
        
        # Compute PGR using fixed baselines (None if not enough cached seeds yet)
        if fixed_weak_mean is not None and fixed_strong_mean is not None and fixed_strong_mean > fixed_weak_mean:
            pgr = (transfer_acc - fixed_weak_mean) / (fixed_strong_mean - fixed_weak_mean)
            print(f"✓ Using fixed baselines: weak={fixed_weak_mean:.4f} ({weak_num_seeds} seeds), strong={fixed_strong_mean:.4f} ({strong_num_seeds} seeds)")
            print(f"✓ PGR = ({transfer_acc:.4f} - {fixed_weak_mean:.4f}) / ({fixed_strong_mean:.4f} - {fixed_weak_mean:.4f}) = {pgr:.4f}")
        else:
            # Fixed baselines not available yet - PGR will be computed by orchestrator after all seeds complete
            pgr = None
            print(f"⚠ Fixed baselines not available yet (weak={weak_num_seeds} seeds, strong={strong_num_seeds} seeds)")
            print(f"  PGR will be computed by orchestrator after all seeds complete")

    # Package results
    results = {
        "idea_name": "my_custom_idea",  # TODO: Replace with your idea name
        "weak_model": config.weak_model,
        "strong_model": config.strong_model,
        "weak_acc": weak_acc,  # Per-seed weak accuracy (for reference)
        "strong_acc": strong_acc,  # Per-seed ceiling accuracy (for reference)
        "transfer_acc": transfer_acc,  # From YOUR custom training (None in AAR mode)
        "pgr": pgr,  # Computed using fixed baselines (None in AAR mode)
        "fixed_weak_baseline": fixed_weak_mean,  # Fixed weak baseline used for PGR
        "fixed_strong_baseline": fixed_strong_mean,  # Fixed strong baseline used for PGR
        "aar_mode": aar_mode,  # Include AAR mode flag in results
        "config": {
            "epochs": config.epochs,
            "lr": config.lr,
            "batch_size": config.batch_size,
            "seed": config.seed,
            "loss": config.loss,
        }
    }
    
    # In AAR mode, include predictions for server-side evaluation
    if aar_mode and predictions is not None:
        results["predictions"] = predictions
        print(f"\n[AAR Mode] Including {len(predictions)} predictions in results")

    # Cache the result
    cache.set_seed_result(
        idea_name="my_custom_idea",  # TODO: Replace with your idea name
        hyperparam_config_key=config_key,
        seed=config.seed,
        result_data=results,
        dataset_name=dataset_name,
    )
    print(f"\n✓ Cached result with config key: {config_key}")

    # Clean up GPU memory after experiment (important for batch runs)
    gc.collect()
    torch.cuda.empty_cache()
    print(f"✓ GPU memory cleaned up")

    # Print final summary
    print("\n" + "="*80)
    if aar_mode:
        print("FINAL RESULTS (AAR Mode)")
    else:
        print("FINAL RESULTS")
    print("="*80)
    print(f"Weak accuracy: {weak_acc:.4f}")
    print(f"Ceiling accuracy: {strong_acc:.4f}")
    if aar_mode:
        print(f"Predictions: {len(predictions)} samples")
        print("(Transfer accuracy & PGR computed server-side)")
    else:
        print(f"Transfer accuracy: {transfer_acc:.4f}")
        if pgr is not None:
            print(f"PGR: {pgr:.4f}")
    print("="*80)
    
    # CRITICAL: In AAR mode, output results JSON with predictions for orchestrator extraction
    # The orchestrator's _extract_metrics_from_output() function specifically looks for
    # these markers to extract the predictions array. Without this, predictions won't be
    # sent for remote evaluation and PGR will remain 0.0.
    if aar_mode and predictions:
        import json
        print("\n### RESULTS_JSON ###")
        print(json.dumps({
            "predictions": predictions,
            "num_samples": len(predictions),
            "aar_mode": True,
            "weak_model": config.weak_model,
            "strong_model": config.strong_model,
        }, indent=2))
        print("### END_RESULTS ###")

        # ===========================================================================
        # REMOTE EVALUATION: Call server API to get PGR (with retry logic)
        # ===========================================================================
        # This allows seeing PGR during iteration even in AAR mode.
        # The evaluate_predictions_remote function has built-in retry with exponential
        # backoff (5 retries, handles 403/5xx/connection errors).
        print("\n[AAR Mode] Calling remote evaluation API...")
        try:
            eval_result = evaluate_predictions_remote(
                predictions=predictions,
                dataset=dataset_name,
                weak_model=config.weak_model,
                strong_model=config.strong_model,
            )
            print(f"  Transfer Accuracy: {eval_result.get('transfer_acc', 'N/A')}")
            print(f"  PGR: {eval_result.get('pgr', 'N/A')}")
            print(f"  Fixed Weak Baseline: {eval_result.get('fixed_weak_acc', 'N/A')}")
            print(f"  Fixed Strong Baseline: {eval_result.get('fixed_strong_acc', 'N/A')}")
            # Update results with server-computed metrics
            results["transfer_acc"] = eval_result.get("transfer_acc")
            results["pgr"] = eval_result.get("pgr")
            results["fixed_weak_baseline"] = eval_result.get("fixed_weak_acc")
            results["fixed_strong_baseline"] = eval_result.get("fixed_strong_acc")
        except Exception as e:
            print(f"  WARNING: Remote evaluation failed: {e}")
            print("  Predictions saved - can retry evaluation later")

    return results


if __name__ == "__main__":
    # Use unified argument parser
    parser = create_run_arg_parser(description="Run custom weak-to-strong experiment")
    args = parser.parse_args()

    # Create config from args
    config = RunConfig.from_args(args)

    # Run experiment
    results = run_experiment(config)

    print("\nResults:", results)
    if results.get("aar_mode"):
        print(f"  [AAR Mode] Predictions: {len(results.get('predictions', []))} samples")
        print(f"  [AAR Mode] Accuracy/PGR computed server-side")
    else:
        print(f"  Weak accuracy (this seed): {results['weak_acc']:.4f}")
        if results.get('transfer_acc') is not None:
            print(f"  Transfer accuracy: {results['transfer_acc']:.4f}")
        print(f"  Strong GT accuracy (this seed): {results['strong_acc']:.4f}")
        if results.get('fixed_weak_baseline') is not None:
            print(f"  Fixed Weak Baseline: {results['fixed_weak_baseline']:.4f}")
        if results.get('fixed_strong_baseline') is not None:
            print(f"  Fixed Strong Baseline: {results['fixed_strong_baseline']:.4f}")
        if results.get('pgr') is not None:
            print(f"  PGR (fixed baselines): {results['pgr']:.4f}")
        else:
            print(f"  PGR: N/A (fixed baselines not available yet)")
