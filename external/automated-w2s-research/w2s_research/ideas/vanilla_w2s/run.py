"""
Vanilla weak-to-strong generalization baseline with causal LM.

This implementation loads pre-trained baselines (weak model + ceiling) and only
trains the transfer model. Baselines are trained with 5 epochs and cached.

Pipeline:
1. Load cached weak model artifacts (soft labels, accuracy) - trained with 5 epochs
2. Load cached ceiling accuracy (strong model on ground truth) - trained with 5 epochs
3. Train transfer model (strong model on weak labels) - epochs from config
4. Compute PGR using fixed baselines
"""
import unsloth
import sys
import os
import time
import gc
from pathlib import Path
import wandb
import torch
from w2s_research.core import (
    RunConfig,
    create_run_arg_parser,
    load_dataset,
    train_model,
    find_latest_checkpoint,
    evaluate_model,
    normalize_model_name_for_path,
    set_seed,
)
from w2s_research.utils import (
    HierarchicalCache,
    compute_hyperparam_config_key,
    get_cached_weak_artifacts,
    get_cached_ceiling_result,
    get_fixed_weak_baseline,
    get_fixed_ceiling_baseline,
    BASELINE_EPOCHS,
    evaluate_predictions_remote,
)
from w2s_research.core.data import format_classification_as_causal, detect_aar_mode
from transformers import AutoTokenizer

from .loss import LogconfLossTrainer


def run_experiment(config: RunConfig):
    """
    Run vanilla weak-to-strong experiment.
    
    Baselines (weak model, ceiling) are loaded from cache (trained with BASELINE_EPOCHS=5).
    Only transfer model is trained with config.epochs.
    """
    print("="*80)
    print("VANILLA WEAK-TO-STRONG EXPERIMENT")
    print("="*80)
    print(f"Weak model: {config.weak_model}")
    print(f"Strong model: {config.strong_model}")
    print(f"Baseline epochs: {BASELINE_EPOCHS} (fixed)")
    print(f"Transfer epochs: {config.epochs}")
    print("="*80)
    
    dataset_name = Path(config.data_dir).name
    cache = HierarchicalCache()
    
    # Config key for THIS experiment (uses config.epochs for transfer)
    extra_params = {}
    if config.logconf_warmup_frac is not None:
        extra_params["logconf-warmup-frac"] = config.logconf_warmup_frac
    if config.train_size:
        extra_params["train_size"] = config.train_size
    if config.test_size:
        extra_params["test_size"] = config.test_size
        
    config_key = compute_hyperparam_config_key(
        strong_model=config.strong_model,
        weak_model=config.weak_model,
        dataset_name=dataset_name,
        epochs=config.epochs,  # Transfer epochs
        lr=config.lr,
        batch_size=config.batch_size,
        scheduler=config.lr_schedule,
        loss=config.loss,
        **extra_params
    )
    
    # Check if this exact experiment is already cached
    cached_result = cache.get_seed_result(
        idea_name="vanilla_w2s",
        hyperparam_config_key=config_key,
        seed=config.seed,
        dataset_name=dataset_name,
    )
    
    if cached_result:
        print("\n✓ Found cached result!")
        print(f"  Transfer accuracy: {cached_result['transfer_acc']:.4f}")
        if cached_result.get('pgr') is not None:
            print(f"  PGR: {cached_result['pgr']:.4f}")
        return cached_result
    
    set_seed(config.seed)
    print(f"\n✓ Set random seed: {config.seed}")
    
    # [1/4] Load data
    print("\n[1/4] Loading data...")
    # Detect AAR mode (workers don't have access to labeled data)
    aar_mode = detect_aar_mode(config.data_dir) or os.getenv("AAR_MODE", "false").lower() == "true"
    if aar_mode:
        print("[AAR Mode] Detected - workers should not have access to labeled data")
    
    datasets = load_dataset(config.data_dir, seed=config.seed, aar_mode=aar_mode)
    train_label_ds = datasets["train_label"]
    train_unlabel_ds = datasets["train_unlabel"]
    test_ds = datasets["test"]
    
    if config.train_size:
        if train_label_ds is not None:
            train_label_ds = train_label_ds.select(range(config.train_size))
        train_unlabel_ds = train_unlabel_ds.select(range(config.train_size))
    if config.test_size:
        test_ds = test_ds.select(range(config.test_size))
    
    if train_label_ds is not None:
        print(f"Train (labeled): {len(train_label_ds)}")
    else:
        print(f"Train (labeled): None (AAR mode - no labeled data available)")
    print(f"Train (unlabeled): {len(train_unlabel_ds)}")
    print(f"Test: {len(test_ds)}")
    
    # [2/4] Load cached weak artifacts
    print(f"\n[2/4] Loading cached weak artifacts...")
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
            f"Weak artifacts not found! Please train weak teacher models first.\n"
            f"  weak_model={config.weak_model}\n"
            f"  dataset={dataset_name}\n"
            f"  seed={config.seed}"
        )
    
    weak_acc = weak_artifacts["weak_acc"]
    soft_labels = weak_artifacts["soft_labels"]
    hard_label_acc = weak_artifacts["hard_label_acc"]
    
    if config.train_size:
        soft_labels = soft_labels[:config.train_size]
    
    print(f"✓ Loaded weak artifacts!")
    print(f"  Weak accuracy: {weak_acc:.4f}")
    print(f"  Hard label accuracy: {hard_label_acc:.4f}")
    print(f"  Soft labels: {len(soft_labels)} samples")
    
    train_unlabel_with_soft = train_unlabel_ds.add_column("soft_label", soft_labels)
    
    # [3/4] Load cached ceiling (default: BASELINE_EPOCHS=5)
    print(f"\n[3/4] Loading cached ceiling...")
    ceiling_result = get_cached_ceiling_result(
        strong_model=config.strong_model,
        dataset_name=dataset_name,
        seed=config.seed,
        # epochs defaults to BASELINE_EPOCHS
        lr=config.lr,
        batch_size=config.batch_size,
        scheduler=config.lr_schedule,
    )
    
    if not ceiling_result:
        raise RuntimeError(
            f"Ceiling not found! Please run train_baselines.py first.\n"
            f"  strong_model={config.strong_model}\n"
            f"  dataset={dataset_name}\n"
            f"  seed={config.seed}"
        )
    
    strong_acc = ceiling_result["strong_acc"]
    print(f"✓ Loaded ceiling accuracy: {strong_acc:.4f}")
    
    # [4/4] Train transfer model (with config.epochs)
    print(f"\n[4/4] Training transfer model (epochs={config.epochs})...")
    
    weak_name = normalize_model_name_for_path(config.weak_model).replace('/', '_')
    strong_name = normalize_model_name_for_path(config.strong_model).replace('/', '_')
    strong_output_dir = f"./results/{dataset_name}_vanilla_w2s/{config_key}/seed_{config.seed}"
    
    # Setup trainer for custom loss
    temp_tokenizer = AutoTokenizer.from_pretrained(config.strong_model)
    
    sample_columns = set(train_unlabel_with_soft.column_names)
    is_binary_format = 'question' in sample_columns and 'choice' in sample_columns
    
    if is_binary_format:
        from w2s_research.core.data import BINARY_LABEL_TOKENS
        label_tokens = BINARY_LABEL_TOKENS
    else:
        from w2s_research.core.data import COMPARISON_LABEL_TOKENS
        label_tokens = COMPARISON_LABEL_TOKENS
    
    label_token_ids = [temp_tokenizer.encode(tok, add_special_tokens=False)[0] for tok in label_tokens]
    
    trainer_class = None
    trainer_kwargs = None
    if config.loss == 'logconf':
        trainer_class = LogconfLossTrainer
        trainer_kwargs = {"label_token_ids": label_token_ids, "warmup_frac": config.logconf_warmup_frac}
    
    # AAR mode already detected above
    if aar_mode:
        print("[AAR Mode] Will return predictions instead of accuracy")
    
    start_time = time.time()
    strong_results = train_model(
        model_name=config.strong_model,
        train_dataset=train_unlabel_with_soft,
        test_dataset=test_ds,
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        lr=config.lr,
        epochs=config.epochs,  # Transfer epochs from config
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
        trainer_class=trainer_class,
        trainer_kwargs=trainer_kwargs,
        bf16=config.bf16,
        fp16=config.fp16,
        seed=config.seed,
        aar_mode=aar_mode,
    )
    training_time = time.time() - start_time
    
    # Handle results based on mode
    if aar_mode:
        # AAR mode: predictions only, no accuracy
        predictions = strong_results.get("predictions", [])
        transfer_acc = None  # Will be computed server-side
        print(f"\n✓ Generated {len(predictions)} predictions")
        print(f"  Training time: {training_time:.2f}s")
    else:
        predictions = None
        transfer_acc = strong_results["final_eval"]["accuracy"]
        print(f"\n✓ Transfer accuracy: {transfer_acc:.4f}")
        print(f"  Training time: {training_time:.2f}s")
    
    # Compute PGR using fixed baselines (averaged across all seeds)
    # In AAR mode, PGR will be computed server-side
    if aar_mode:
        print("\n[5/5] Skipping PGR computation (AAR mode - computed server-side)")
        pgr = None
        fixed_weak_mean = None
        fixed_strong_mean = None
    else:
        print("\n[5/5] Computing PGR with fixed baselines...")
        
        fixed_weak_mean, fixed_weak_se, weak_num_seeds = get_fixed_weak_baseline(
            weak_model=config.weak_model,
            dataset_name=dataset_name,
            # epochs defaults to BASELINE_EPOCHS
            batch_size=config.batch_size,
            lr=config.lr,
            scheduler=config.lr_schedule,
        )
        
        fixed_strong_mean, fixed_strong_se, strong_num_seeds = get_fixed_ceiling_baseline(
            strong_model=config.strong_model,
            dataset_name=dataset_name,
            # epochs defaults to BASELINE_EPOCHS
            batch_size=config.batch_size,
            lr=config.lr,
            scheduler=config.lr_schedule,
        )
        
        if fixed_weak_mean is not None and fixed_strong_mean is not None and fixed_strong_mean > fixed_weak_mean:
            pgr = (transfer_acc - fixed_weak_mean) / (fixed_strong_mean - fixed_weak_mean)
            print(f"✓ Fixed baselines: weak={fixed_weak_mean:.4f} ({weak_num_seeds} seeds), strong={fixed_strong_mean:.4f} ({strong_num_seeds} seeds)")
            print(f"✓ PGR = ({transfer_acc:.4f} - {fixed_weak_mean:.4f}) / ({fixed_strong_mean:.4f} - {fixed_weak_mean:.4f}) = {pgr:.4f}")
        else:
            pgr = None
            print(f"⚠ Fixed baselines not available (weak={weak_num_seeds} seeds, strong={strong_num_seeds} seeds)")
    
    # Package results
    results = {
        "idea_name": "vanilla_w2s",
        "weak_model": config.weak_model,
        "strong_model": config.strong_model,
        "weak_acc": weak_acc,
        "hard_weak_label_acc": hard_label_acc,
        "strong_acc": strong_acc,
        "transfer_acc": transfer_acc,
        "pgr": pgr,
        "fixed_weak_baseline": fixed_weak_mean,
        "fixed_strong_baseline": fixed_strong_mean,
        "training_time": training_time,
        "aar_mode": aar_mode,
        "config": {
            "epochs": config.epochs,
            "baseline_epochs": BASELINE_EPOCHS,
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
            "weak": {"final_eval": {"accuracy": weak_acc}},
            "strong": {"final_eval": strong_results["final_eval"]},
            "ceiling": ceiling_result.get("detailed_eval", {}).get("ceiling", {"final_eval": {"accuracy": strong_acc}}),
        }
    }
    
    # In AAR mode, include predictions for server-side evaluation
    if aar_mode and predictions:
        results["predictions"] = predictions
        print(f"[AAR Mode] Including {len(predictions)} predictions in results")
    
    # Cache result
    cache.set_seed_result(
        idea_name="vanilla_w2s",
        hyperparam_config_key=config_key,
        seed=config.seed,
        result_data=results,
        dataset_name=dataset_name,
    )
    print(f"\n✓ Cached result: {config_key}")
    
    # Log to wandb
    wandb.init(
        project="generalization",
        name=f"vanilla_w2s_{weak_name}_{strong_name}_e{config.epochs}_seed{config.seed}",
        dir=strong_output_dir,
        config={
            "experiment": "vanilla_w2s",
            "weak_model": config.weak_model,
            "strong_model": config.strong_model,
            "seed": config.seed,
            "epochs": config.epochs,
            "baseline_epochs": BASELINE_EPOCHS,
            "loss": config.loss,
        },
        reinit=True,
    )
    
    log_dict = {
        "weak_acc": weak_acc,
        "hard_weak_label_acc": hard_label_acc,
        "strong_acc": strong_acc,
        "training_time": training_time,
        "aar_mode": aar_mode,
    }
    if transfer_acc is not None:
        log_dict["transfer_acc"] = transfer_acc
    if pgr is not None:
        log_dict["pgr"] = pgr
    wandb.log(log_dict)
    wandb.finish()
    
    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()
    
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
    
    # In AAR mode, output results JSON with predictions for orchestrator extraction
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

        # Remote evaluation: Call server API to get PGR (with retry logic)
        # Has built-in retry with exponential backoff (5 retries, handles 403/5xx errors)
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
            # Update results with server-computed metrics
            results["transfer_acc"] = eval_result.get("transfer_acc")
            results["pgr"] = eval_result.get("pgr")
        except Exception as e:
            print(f"  WARNING: Remote evaluation failed: {e}")
            print("  Predictions saved - can retry evaluation later")

    return results


def main():
    parser = create_run_arg_parser(description="Run vanilla W2S experiment")
    args = parser.parse_args()
    config = RunConfig.from_args(args)
    
    results = run_experiment(config)


if __name__ == "__main__":
    main()
