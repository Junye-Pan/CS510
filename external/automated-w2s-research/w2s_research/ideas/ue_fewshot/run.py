"""
UE_fewshot: Few-shot Labeling + Consistency Fixing for Weak-to-Strong Learning.

This applies few-shot labeling with random demonstrations + consistency fixing
to generate labels, then uses those labels to train a strong model.

Supports both:
- Binary tasks (question/choice format with True/False labels)
- Comparison tasks (prompt/first/second format with True/False labels for A>B claim)
"""
import unsloth
from unsloth import FastLanguageModel
import os
import time
import torch
from pathlib import Path
from collections import Counter
from datasets import Dataset as HfDataset
from w2s_research.core import (
    RunConfig,
    create_run_arg_parser,
    train_model,
    load_dataset,
)
from w2s_research.core.data import detect_aar_mode
from w2s_research.utils import HierarchicalCache, compute_hyperparam_config_key, get_cached_ceiling_result, get_cached_weak_artifacts, evaluate_predictions_remote
from w2s_research.ideas.ue_fewshot.experiment import ue_fewshot, detect_format


def run_experiment(config: RunConfig):
    """
    Run UE_fewshot experiment (auto-detects binary vs comparison format).

    Steps:
    1. Use few-shot labeling + consistency fixing to generate labels
    2. Train strong model on generated labels
    3. Evaluate and compare with ceiling performance

    Args:
        config: RunConfig with all experiment parameters

    Returns:
        Dictionary with results
    """
    # Extract dataset name from data_dir
    dataset_name = Path(config.data_dir).name

    # Detect AAR mode (workers don't have access to labeled data)
    aar_mode = detect_aar_mode(config.data_dir) or os.getenv("AAR_MODE", "false").lower() == "true"

    # Load data first to detect format
    datasets = load_dataset(config.data_dir, seed=config.seed, aar_mode=aar_mode)
    train_unlabel = datasets["train_unlabel"]
    test_ds = datasets["test"]

    # Detect format from data
    format_type = detect_format(train_unlabel)

    print("="*80)
    print(f"UE_FEWSHOT: Few-shot + Consistency Fixing ({format_type.upper()} format)")
    print("="*80)
    print(f"Strong Model: {config.strong_model}")
    print(f"Loss: {config.loss}")
    print(f"Epochs: {config.epochs}")
    print(f"Seed: {config.seed}")
    print(f"Demo schedule: zero-shot -> {' -> '.join(f'{n}-shot' for n in config.num_demos)}")
    print(f"Number of iterations: {config.num_iterations} (zero-shot + {len(config.num_demos)} few-shot)")
    print("="*80)

    if aar_mode:
        print("[AAR Mode] Detected - workers should not have access to labeled data")

    # Check cache first
    cache = HierarchicalCache()
    # Convert num_demos list to string for cache key (e.g., "2_4_16")
    num_demos_str = "_".join(str(n) for n in config.num_demos)
    extra_params = {"task_type": format_type, "num_demos": num_demos_str, "num_iterations": config.num_iterations}
    if config.train_size:
        extra_params["train_size"] = config.train_size
    if config.test_size:
        extra_params["test_size"] = config.test_size
    config_key = compute_hyperparam_config_key(
        strong_model=config.strong_model,
        weak_model=None,  # UE doesn't use a weak model
        dataset_name=dataset_name,
        epochs=config.epochs,
        lr=config.lr,
        batch_size=config.batch_size,
        scheduler=config.lr_schedule,
        loss=config.loss,
        **extra_params
    )

    # Create output directory matching cache structure
    base_output_dir = f"./results/{dataset_name}_ue_fewshot/{config_key}/seed_{config.seed}"
    print(f"Output Directory: {base_output_dir}")

    cached_result = cache.get_seed_result(
        idea_name="ue_fewshot",
        hyperparam_config_key=config_key,
        seed=config.seed,
        dataset_name=dataset_name,
    )
    if cached_result:
        print("\n>>> Found cached result for ue_fewshot!")
        if cached_result.get('transfer_acc') is not None:
            print(f"  Transfer accuracy: {cached_result['transfer_acc']:.4f}")
        if cached_result.get('pgr') is not None:
            print(f"  PGR: {cached_result['pgr']:.4f}")
        return cached_result

    # Truncate if needed
    if config.train_size and len(train_unlabel) > config.train_size:
        train_unlabel = train_unlabel.select(range(config.train_size))
    if config.test_size and len(test_ds) > config.test_size:
        test_ds = test_ds.select(range(config.test_size))

    print(f"\n[1/4] Loaded data")
    print(f"  Train (unlabeled): {len(train_unlabel)} examples")
    print(f"  Test: {len(test_ds)} examples")

    # Step 2: Apply UE_fewshot to generate labels (handles consistency groups internally)
    print("\n[2/4] Applying UE_fewshot to generate labels...")
    start_time = time.time()

    # Cache dir for per-iteration caching
    iteration_cache_dir = f"{base_output_dir}/iteration_cache"

    labeled_dataset, iteration_logs = ue_fewshot(
        unlabeled_dataset=train_unlabel,
        pretrained_model_name=config.strong_model,
        num_demos=config.num_demos,
        max_ctx=config.max_ctx,
        seed=config.seed,
        num_iterations=config.num_iterations,
        cache_dir=iteration_cache_dir,
    )

    labeling_time = time.time() - start_time
    torch.cuda.empty_cache()

    print(f"\n>>> UE_fewshot complete ({labeling_time:.2f}s)")
    print(f"  Generated labels for {len(labeled_dataset)} examples")
    print(f"  Label distribution: {Counter([labeled_dataset[i]['label'] for i in range(len(labeled_dataset))])}")

    # Step 3: Train strong model on generated labels
    print("\n[3/4] Training strong model on generated labels...")
    if aar_mode:
        print("[AAR Mode] Will return predictions instead of accuracy")
    start_time = time.time()

    train_results = train_model(
        model_name=config.strong_model,
        train_dataset=labeled_dataset,
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
        output_dir=f"{base_output_dir}/final_strong",
        seed=config.seed,
        bf16=config.bf16,
        fp16=config.fp16,
        aar_mode=aar_mode,
    )

    training_time = time.time() - start_time

    # Handle results based on mode
    if aar_mode:
        predictions = train_results.get("predictions", [])
        transfer_acc = None  # Will be computed server-side
        print(f"\n>>> Generated {len(predictions)} predictions")
    else:
        predictions = None
        transfer_acc = train_results["final_eval"]["accuracy"]
        print(f"\n>>> Strong model test accuracy: {transfer_acc:.4f}")

    # Step 4: Get ceiling performance (skip in AAR mode)
    if aar_mode:
        print("\n[4/4] Skipping baseline loading (AAR mode - computed server-side)")
        ceiling_acc = None
        weak_acc = None
        pgr = None
    else:
        print("\n[4/4] Getting baseline performances...")
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
            ceiling_acc = ceiling_result["strong_acc"]
            print(f"  Ceiling accuracy: {ceiling_acc:.4f} (cached)")
        else:
            print(f"  Ceiling not cached. Run train_ceiling first.")
            ceiling_acc = None

        # Load weak model accuracy for PGR calculation
        weak_acc = None
        if config.weak_model:
            weak_artifacts = get_cached_weak_artifacts(
                weak_model=config.weak_model,
                dataset_name=dataset_name,
                seed=config.seed,
                epochs=config.epochs,
                batch_size=config.batch_size,
                lr=config.lr,
                scheduler=config.lr_schedule,
            )
            if weak_artifacts:
                weak_acc = weak_artifacts["weak_acc"]
                print(f"  Weak accuracy: {weak_acc:.4f} (cached)")

        # Calculate PGR (Performance Gap Recovered)
        if ceiling_acc is not None and weak_acc is not None and ceiling_acc > weak_acc:
            pgr = (transfer_acc - weak_acc) / (ceiling_acc - weak_acc)
        else:
            pgr = None

    # Calculate metrics
    total_time = labeling_time + training_time

    print("\n" + "="*80)
    if aar_mode:
        print("RESULTS (AAR Mode)")
    else:
        print("RESULTS")
    print("="*80)
    if aar_mode:
        print(f"Predictions:        {len(predictions)} samples")
        print("(Transfer accuracy & PGR computed server-side)")
    else:
        print(f"Transfer Accuracy:  {transfer_acc:.4f}")
        print(f"Weak Accuracy:      {weak_acc:.4f if weak_acc else 'N/A'}")
        print(f"Ceiling Accuracy:   {ceiling_acc:.4f if ceiling_acc else 'N/A'}")
        print(f"PGR:                {pgr:.4f if pgr else 'N/A'}")
    print(f"Time: {total_time:.2f}s (Label: {labeling_time:.2f}s, Train: {training_time:.2f}s)")
    print("="*80)

    # Return results
    results = {
        "idea_name": "ue_fewshot",
        "strong_model": config.strong_model,
        "weak_model": config.weak_model,
        "strong_acc": ceiling_acc,
        "weak_acc": weak_acc,
        "transfer_acc": transfer_acc,
        "pgr": pgr,
        "ceiling_acc": ceiling_acc,
        "training_time": training_time,
        "labeling_time": labeling_time,
        "total_time": total_time,
        "aar_mode": aar_mode,
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
            "num_demos": config.num_demos,
            "num_iterations": config.num_iterations,
            "task_type": format_type,
        },
        "detailed_eval": train_results.get("final_eval", {}),
        "labeling_logs": iteration_logs,
    }

    # In AAR mode, include predictions for server-side evaluation
    if aar_mode and predictions:
        results["predictions"] = predictions
        print(f"[AAR Mode] Including {len(predictions)} predictions in results")

    # Cache the result
    cache.set_seed_result(
        idea_name="ue_fewshot",
        hyperparam_config_key=config_key,
        seed=config.seed,
        result_data=results,
        dataset_name=dataset_name,
    )
    print(f"\n>>> Cached result with key: {config_key}")

    # In AAR mode, output results JSON with predictions for orchestrator extraction
    if aar_mode and predictions:
        import json
        print("\n### RESULTS_JSON ###")
        print(json.dumps({
            "predictions": predictions,
            "num_samples": len(predictions),
            "aar_mode": True,
            "strong_model": config.strong_model,
        }, indent=2))
        print("### END_RESULTS ###")

        # Remote evaluation: Call server API to get PGR (with retry logic)
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
            results["transfer_acc"] = eval_result.get("transfer_acc")
            results["pgr"] = eval_result.get("pgr")
        except Exception as e:
            print(f"  WARNING: Remote evaluation failed: {e}")

    return results


def main():
    parser = create_run_arg_parser()
    parser.add_argument(
        "--num-demos",
        type=int,
        nargs='+',
        default=[2, 4, 16],
        help="List of few-shot demo counts for each iteration after zero-shot. "
             "E.g., [2, 4, 16] means: zero-shot -> 2-shot -> 4-shot -> 16-shot (default: [2, 4, 16])"
    )
    args = parser.parse_args()
    config = RunConfig.from_args(args)

    # Add few-shot specific config
    config.num_demos = args.num_demos  # List of demo counts per iteration
    config.num_iterations = len(args.num_demos) + 1  # +1 for initial zero-shot

    # Run UE_fewshot (auto-detects format)
    results = run_experiment(config)


if __name__ == "__main__":
    main()
