"""
UE_zeroshot: Zero-shot Labeling + Consistency Fixing for Weak-to-Strong Learning.

This applies the zero-shot initialization step of ICM to generate labels,
then uses those labels to train a strong model.

Supports both:
- Binary tasks (question/choice format with True/False labels)
- Preference tasks (prompt/first/second format with A/B labels) - e.g., helpfulness dataset
"""
import unsloth
import torch
from unsloth import FastLanguageModel
import time
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
from w2s_research.ideas.ue_zeroshot.experiment import ue_zeroshot, detect_format, balance_by_confidence


def run_experiment(config: RunConfig, balance_per_subset: bool = False):
    """
    Run UE_zeroshot experiment (auto-detects binary vs preference format).

    Steps:
    1. Load data
    2. Apply UE_zeroshot labeling + consistency fixing to generate labels
    3. Train strong model on generated labels
    4. Evaluate and compare with ceiling performance

    Args:
        config: RunConfig with all experiment parameters
        balance_per_subset: If True, balance each subset (subset + difficulty) separately

    Returns:
        Dictionary with results
    """
    # Detect AAR mode (workers don't have access to labeled data)
    import os
    aar_mode = detect_aar_mode(config.data_dir) or os.getenv("AAR_MODE", "false").lower() == "true"

    # Load data first to detect format
    datasets = load_dataset(config.data_dir, seed=config.seed, aar_mode=aar_mode)
    train_unlabel = datasets["train_unlabel"]
    test_ds = datasets["test"]

    # Detect format from data
    format_type = detect_format(train_unlabel)
    dataset_name = Path(config.data_dir).name

    if aar_mode:
        print("[AAR Mode] Detected - workers should not have access to labeled data")

    print("="*80)
    print(f"UE_ZEROSHOT: Zero-shot + Consistency Fixing ({format_type.upper()} format)")
    print("="*80)
    print(f"Strong Model: {config.strong_model}")
    print(f"Loss: {config.loss}")
    print(f"Epochs: {config.epochs}")
    print(f"Seed: {config.seed}")
    print("="*80)

    # Check cache
    cache = HierarchicalCache()
    extra_params = {"task_type": format_type}
    if config.train_size:
        extra_params["train_size"] = config.train_size
    if config.test_size:
        extra_params["test_size"] = config.test_size

    config_key = compute_hyperparam_config_key(
        strong_model=config.strong_model,
        weak_model=None,
        dataset_name=dataset_name,
        epochs=config.epochs,
        lr=config.lr,
        batch_size=config.batch_size,
        scheduler=config.lr_schedule,
        loss=config.loss,
        **extra_params
    )

    idea_name = "ue_zeroshot"
    base_output_dir = f"./results/{dataset_name}_{idea_name}/{config_key}/seed_{config.seed}"
    print(f"Output Directory: {base_output_dir}")

    cached_result = cache.get_seed_result(
        idea_name=idea_name,
        hyperparam_config_key=config_key,
        seed=config.seed,
        dataset_name=dataset_name,
    )
    if cached_result:
        print(f"\n>>> Found cached result!")
        print(f"  Transfer accuracy: {cached_result['transfer_acc']:.4f}")
        if cached_result.get('pgr') is not None:
            print(f"  PGR: {cached_result['pgr']:.4f}")
        return cached_result

    # Truncate if needed
    if config.train_size and len(train_unlabel) > config.train_size:
        train_unlabel = train_unlabel.select(range(config.train_size))
    if config.test_size and len(test_ds) > config.test_size:
        test_ds = test_ds.select(range(config.test_size))

    print(f"\n[1/5] Loaded data")
    print(f"  Train (unlabeled): {len(train_unlabel)} examples")
    print(f"  Test: {len(test_ds)} examples")

    # Apply UE_zeroshot (unified function handles both formats)
    print(f"\n[2/5] Applying UE_zeroshot...")
    start_time = time.time()

    labeled_dataset, iteration_logs, confidences = ue_zeroshot(
        unlabeled_dataset=train_unlabel,
        pretrained_model_name=config.strong_model,
        max_ctx=config.max_ctx,
        return_confidences=True,
    )

    labeling_time = time.time() - start_time
    torch.cuda.empty_cache()

    print(f"\n>>> UE_zeroshot complete ({labeling_time:.2f}s)")
    print(f"  Generated labels for {len(labeled_dataset)} examples")
    print(f"  Label distribution: {Counter([labeled_dataset[i]['label'] for i in range(len(labeled_dataset))])}")

    # Apply data balancing if labels are imbalanced
    print(f"\n[3/5] Checking label balance...")
    labeled_dataset, confidences, balance_info = balance_by_confidence(
        dataset=labeled_dataset,
        confidences=confidences,
        imbalance_threshold=1.1,
        balance_per_subset=balance_per_subset,
    )
    iteration_logs.append({"stage": "balance", **balance_info})
    print(f"  Label distribution: {Counter([labeled_dataset[i]['label'] for i in range(len(labeled_dataset))])}")

    # Train strong model
    print("\n[4/5] Training strong model on generated labels...")
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

    # Get ceiling and weak performance (skip in AAR mode)
    if aar_mode:
        print("\n[5/5] Skipping baseline loading (AAR mode - computed server-side)")
        ceiling_acc = None
        weak_acc = None
        pgr = None
    else:
        print("\n[5/5] Getting baseline performances...")
        ceiling_result = get_cached_ceiling_result(
            strong_model=config.strong_model,
            dataset_name=dataset_name,
            seed=config.seed,
            epochs=config.epochs,
            lr=config.lr,
            batch_size=config.batch_size,
            scheduler=config.lr_schedule,
        )

        ceiling_acc = None
        if ceiling_result:
            ceiling_acc = ceiling_result["strong_acc"]
            print(f"  Ceiling accuracy: {ceiling_acc:.4f} (cached)")
        else:
            print(f"  Ceiling not cached. Run train_ceiling first.")

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

        # Calculate PGR
        pgr = None
        if ceiling_acc is not None and weak_acc is not None and ceiling_acc > weak_acc:
            pgr = (transfer_acc - weak_acc) / (ceiling_acc - weak_acc)

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

    results = {
        "idea_name": idea_name,
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
            "task_type": format_type,
        },
        "detailed_eval": train_results.get("final_eval", {}),
        "labeling_logs": iteration_logs,
    }

    # In AAR mode, include predictions for server-side evaluation
    if aar_mode and predictions:
        results["predictions"] = predictions
        print(f"[AAR Mode] Including {len(predictions)} predictions in results")

    cache.set_seed_result(
        idea_name=idea_name,
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
    parser.add_argument("--balance-per-subset", action="store_true",
                       help="Balance each subset (subset + difficulty) separately instead of the whole dataset")
    args = parser.parse_args()
    config = RunConfig.from_args(args)
    return run_experiment(config, balance_per_subset=args.balance_per_subset)


if __name__ == "__main__":
    main()
