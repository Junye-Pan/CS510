"""
Critic Training for Weak-to-Strong Generalization.

This approach uses critique generation to improve weak-to-strong learning.
The key insight: Train critic to write helpful critiques that expose
flaws in incorrect answers, helping a weak judge make better decisions.

Pipeline:
1. Load data (train_labeled, train_unlabeled, test)
1.5. (Optional) SFT strong model on weak model critiques
2. Generate critiques for train_labeled AND test set with strong model
3. Train weak judge on train_labeled with critiques
4. Generate critiques for train_unlabeled
5. Create GRPO dataset with prompts and reference critiques
6. Train critic with GRPO (rewards computed on-the-fly based on judge predictions)
7. Generate critiques for test set with GRPO-trained critic
8. Evaluate judge with GRPO-trained critic on test set
9. Generate critiques for train_unlabeled with GRPO-trained critic
10. Use judge to generate labels on train_unlabeled
11. SFT strong model on judge-labeled data and evaluate on test set
This file orchestrates the pipeline by calling step functions from the steps/ directory.
Helper functions are in utils.py.
"""
import unsloth
from pathlib import Path
import json
import time
import uuid
import wandb
from transformers import AutoTokenizer

from w2s_research.core import (
    RunConfig,
    create_run_arg_parser,
    load_dataset,
    find_latest_checkpoint,
)
from w2s_research.core.data import JUDGE_PROMPT_VERSION
from w2s_research.core.seed_utils import set_seed
from w2s_research.utils import (
    HierarchicalCache,
    compute_hyperparam_config_key,
    get_cached_ceiling_result,
    get_cached_weak_artifacts,
    get_fixed_weak_baseline,
    get_fixed_ceiling_baseline,
    evaluate_predictions_remote,
)
from w2s_research.core.data import detect_aar_mode

from w2s_research.ideas.critic.steps import (
    sft_strong_model_on_weak_critiques,
    generate_critiques_for_labeled_and_test,
    train_judge_model_step,
    generate_critiques_for_unlabeled,
    create_grpo_dataset_step,
    train_critic_with_grpo,
    generate_critiques_test_grpo,
    evaluate_judge_with_grpo_critiques,
    generate_critiques_unlabeled_grpo,
    generate_judge_labels,
    sft_strong_model_on_judge_labels,
)

from w2s_research.ideas.critic.utils import check_training_checkpoint


def run_experiment(config: RunConfig):
    """
    Run critic training experiment.

    Args:
        config: Run configuration
            - Judge training uses judge_epochs, lr, batch_size, etc.
            - Final SFT training uses epochs, lr, batch_size, etc.
            - GRPO training uses GRPO-specific params (grpo_epochs, grpo_lr, grpo_batch_size)

    Returns:
        Results dictionary with step-by-step metrics
    """
    print("="*80)
    print("CRITIC TRAINING EXPERIMENT")
    print("="*80) 
    print(f"Weak model (judge): {config.weak_model}")
    print(f"Strong model (critic): {config.strong_model}")
    print(f"Step 1.5 SFT: epochs={config.sft_critic_epochs}, lr={config.sft_critic_lr}")
    print(f"Judge training: epochs={config.judge_epochs}, lr={config.sft_critic_lr}, batch_size={config.batch_size}")
    print(f"GRPO training: epochs={config.grpo_epochs}, lr={config.grpo_lr}, batch_size={config.grpo_batch_size}")
    print(f"Final SFT: epochs={config.epochs}, lr={config.lr}")
    print("="*80)

    # Extract dataset name
    dataset_name = Path(config.data_dir).name

    # AAR mode detection (ground truth labels held server-side)
    aar_mode = detect_aar_mode(config.data_dir)
    if aar_mode:
        print("\n⚠️  AAR MODE DETECTED - Ground truth labels NOT available")
        print("   NOTE: Critic pipeline requires labels for judge training.")
        print("   AAR mode support for critic is limited - will use remote evaluation for final PGR.")

    # Save original model names for baseline lookups (before they get modified to checkpoint paths)
    original_strong_model = config.strong_model
    original_weak_model = config.weak_model

    # Check cache
    cache = HierarchicalCache()
    config_key = compute_hyperparam_config_key(
        strong_model=config.strong_model,
        weak_model=config.weak_model,
        dataset_name=dataset_name,
        # Standard training params (must match run_idea_multi_seed.py)
        epochs=config.epochs,
        lr=config.lr,
        batch_size=config.batch_size,
        scheduler=config.lr_schedule,
        loss=config.loss,
        # Critic-specific params (only the ones we're sweeping)
        grpo_lr=config.grpo_lr,
        grpo_max_steps=config.grpo_max_steps,
        grpo_kl_penalty=config.grpo_kl_penalty,
        sft_critic_num_samples=config.sft_critic_num_samples,
    )

    # Force rerun: clear cache and results (if requested)
    if config.force_rerun:
        print("\n⚠️  Force rerun enabled - clearing cache and results...")
        import shutil

        # Delete cache entry
        cache.delete_seed_result(
            idea_name="critic",
            hyperparam_config_key=config_key,
            seed=config.seed,
            dataset_name=dataset_name,
        )

        # Delete results directory (includes intermediate checkpoints and all output dirs)
        # This also deletes experiment_uid.txt, so a new uid will be generated
        results_dir = Path(f"./results/{dataset_name}_critic/{config_key}/seed_{config.seed}")
        if results_dir.exists():
            print(f"  → Deleting results directory: {results_dir}")
            shutil.rmtree(results_dir)
        
        print("  ✓ Cache and results cleared")

    # Check cache
    cached_result = cache.get_seed_result(
        idea_name="critic",
        hyperparam_config_key=config_key,
        seed=config.seed,
        dataset_name=dataset_name,
    )

    if cached_result:
        print("\n✓ Found cached result for critic!")
        print(json.dumps(cached_result, indent=2, default=str))
        return cached_result

    # Set random seed for reproducibility
    set_seed(config.seed)

    # Try to load pre-trained weak model artifacts (for PGR calculation)
    print("\n[Loading] Weak model artifacts (for PGR)...")
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
        print(f"✓ Loaded pre-trained weak artifacts!")
        print(f"  Weak accuracy: {weak_acc:.4f}")
    else:
        weak_acc = None
        print("⚠ Weak model artifacts not found - PGR will use fixed baselines only")

    # Try to load pre-trained strong model ceiling accuracy (for PGR calculation)
    print("\n[Loading] Strong model ceiling accuracy (for PGR)...")
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
        strong_acc = ceiling_result["strong_acc"]
        print(f"✓ Loaded strong model ceiling accuracy: {strong_acc:.4f}")
    else:
        strong_acc = None
        print("⚠ Strong model ceiling not found - PGR will use fixed baselines only")

    # Create results directory for intermediate checkpoints and outputs
    # Aligned with other ideas: use results/ for everything
    results_dir = Path(f"./results/{dataset_name}_critic/{config_key}/seed_{config.seed}")
    if not results_dir.exists():
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nCreated results directory: {results_dir}")
    else:
        print(f"\nUsing existing results directory: {results_dir}")

    # Generate or load experiment UID for wandb grouping
    # This allows all wandb runs (judge, grpo, sft) to be grouped together
    uid_file = results_dir / "experiment_uid.txt"
    if uid_file.exists():
        experiment_uid = uid_file.read_text().strip()
        print(f"Loaded experiment UID: {experiment_uid}")
    else:
        experiment_uid = uuid.uuid4().hex[:8]
        uid_file.write_text(experiment_uid)
        print(f"Generated new experiment UID: {experiment_uid}")
    
    # Create wandb group name with model info for better identification
    weak_name = config.weak_model.replace("/", "_").split("_")[-1]  # e.g., "Qwen1.5-0.5B-Chat"
    strong_name = config.strong_model.replace("/", "_").split("_")[-1]  # e.g., "Qwen3-4B-Base"
    wandb_group_name = f"{weak_name}+{strong_name}_{experiment_uid}"
    print(f"Wandb group: {wandb_group_name}")

    # Load data
    print("\n[1/11] Loading data...")
    datasets = load_dataset(config.data_dir, seed=config.seed)

    train_label_ds = datasets["train_label"]
    train_unlabel_ds = datasets["train_unlabel"]
    test_ds = datasets["test"]

    # Set sizes separately for train_labeled and train_unlabeled
    # None or -1 (converted to None) means full dataset
    if config.train_labeled_size is not None:
        train_label_ds = train_label_ds.select(range(config.train_labeled_size))
    if config.train_size is not None:
        train_unlabel_ds = train_unlabel_ds.select(range(config.train_size))
    if config.test_size is not None:
        test_ds = test_ds.select(range(config.test_size))

    print(f"Train (labeled, for judge): {len(train_label_ds)} samples")
    print(f"Train (unlabeled, for GRPO): {len(train_unlabel_ds)} samples")
    print(f"Test: {len(test_ds)} samples")

    # Initialize tokenizers
    strong_tokenizer = AutoTokenizer.from_pretrained(config.strong_model)

    # [STEP 1.5] (Optional) SFT strong model on weak model critiques
    sft_critic_checkpoint, step1_5_metrics = sft_strong_model_on_weak_critiques(
        config=config,
        train_unlabel_ds=train_unlabel_ds,
        test_ds=test_ds,
        strong_tokenizer=strong_tokenizer,
        checkpoint_dir=results_dir,
        dataset_name=dataset_name,
        config_key=config_key,
        experiment_uid=experiment_uid,
        wandb_group=wandb_group_name,
    )
    
    # If step 1.5 produced a checkpoint, use it as the strong model for subsequent steps
    if sft_critic_checkpoint:
        print(f"\n  → Using SFT checkpoint from step 1.5 as strong model: {sft_critic_checkpoint}")
        config.strong_model = sft_critic_checkpoint
        # Re-initialize tokenizer in case the model path changed (though it shouldn't for merged checkpoints)
        strong_tokenizer = AutoTokenizer.from_pretrained(config.strong_model)

    # [STEP 2] Generate critiques for train_labeled AND test set with strong model
    critiques_labeled, critiques_test, critique_gen_time = generate_critiques_for_labeled_and_test(
        config=config,
        train_label_ds=train_label_ds,
        test_ds=test_ds,
        strong_tokenizer=strong_tokenizer,
        checkpoint_dir=results_dir,
    )

    # [STEP 3] Train weak judge on train_labeled with critiques
    judge_results, judge_output_dir, judge_training_time = train_judge_model_step(
        config=config,
        train_label_ds=train_label_ds,
        test_ds=test_ds,
        critiques_labeled=critiques_labeled,
        critiques_test=critiques_test,
        dataset_name=dataset_name,
        config_key=config_key,
        checkpoint_dir=results_dir,
        experiment_uid=experiment_uid,
        wandb_group=wandb_group_name,
    )
    judge_baseline_acc = judge_results["final_eval"]["accuracy"]

    # [STEP 4] Generate critiques for train_unlabeled
    critiques_unlabeled, prompts_unlabeled, num_samples, critique_unlabel_gen_time = generate_critiques_for_unlabeled(
        config=config,
        train_unlabel_ds=train_unlabel_ds,
        strong_tokenizer=strong_tokenizer,
        checkpoint_dir=results_dir,
    )

    # Initialize weak tokenizer for judge predictions during GRPO training
    weak_tokenizer = AutoTokenizer.from_pretrained(config.weak_model)

    judge_checkpoint = find_latest_checkpoint(judge_output_dir)
    if not judge_checkpoint:
        raise RuntimeError(f"No judge checkpoint found in {judge_output_dir}")

    # [STEP 5] Create GRPO dataset
    # Rewards are computed on-the-fly during GRPO training based on judge predictions
    grpo_dataset = create_grpo_dataset_step(
        config=config,
        train_unlabel_ds=train_unlabel_ds,
        critiques_unlabeled=critiques_unlabeled,
        prompts_unlabeled=prompts_unlabeled,
        strong_tokenizer=strong_tokenizer,
        checkpoint_dir=results_dir,
    )

    # [STEP 6] Train critic with GRPO (VERL handles wandb initialization internally)
    # Include experiment_uid in output_dir for wandb grouping
    critic_grpo_output_dir = f"./results/{dataset_name}_critic/{config_key}/seed_{config.seed}/grpo_{experiment_uid}"
    step6_checkpoint = results_dir / "step6_grpo_training.json"

    # Check if GRPO training is already complete
    # Use find_latest_verl_checkpoint for VERL checkpoints (global_step_* format)
    from w2s_research.ideas.critic.utils import find_latest_verl_checkpoint
    step6_data = check_training_checkpoint(step6_checkpoint)
    grpo_checkpoint = find_latest_verl_checkpoint(critic_grpo_output_dir)
    if step6_data and grpo_checkpoint:
        print(f"  → Loading from checkpoint: {step6_checkpoint}")
        print(f"  → Found VERL checkpoint: {grpo_checkpoint}")
        grpo_training_time = step6_data.get('training_time', 0.0)
        print(f"  ✓ GRPO training already complete (cached)")
    else:

        start_time = time.time()
        grpo_results = train_critic_with_grpo(
            model_name=config.strong_model,  # May be SFT checkpoint from step 1.5 if enabled
            grpo_dataset=grpo_dataset,
            original_dataset=train_unlabel_ds,
            judge_model_name=config.weak_model,
            judge_checkpoint=judge_checkpoint,
            judge_tokenizer=weak_tokenizer,
            config=config,
            output_dir=critic_grpo_output_dir,
            config_key=config_key,
            experiment_uid=experiment_uid,
            wandb_group=wandb_group_name,
        )

        grpo_training_time = grpo_results.get('training_time', time.time() - start_time)

        # Save checkpoint
        step6_data = {
            'training_time': grpo_training_time,
            'output_dir': critic_grpo_output_dir,
        }
        with open(step6_checkpoint, 'w') as f:
            json.dump(step6_data, f, indent=2)
        print(f"  ✓ Saved checkpoint: {step6_checkpoint}")

    # [STEP 7] Generate new critiques with GRPO-trained critic
    critiques_test_grpo = generate_critiques_test_grpo(
        config=config,
        test_ds=test_ds,
        strong_tokenizer=strong_tokenizer,
        critic_grpo_output_dir=critic_grpo_output_dir,
        checkpoint_dir=results_dir,
        config_key=config_key,
    )

    # [STEP 8] Evaluate judge with GRPO-trained critic
    judge_grpo_acc = evaluate_judge_with_grpo_critiques(
        config=config,
        test_ds=test_ds,
        critiques_test_grpo=critiques_test_grpo,
        weak_tokenizer=weak_tokenizer,
        judge_checkpoint=judge_checkpoint,
        checkpoint_dir=results_dir,
        judge_baseline_acc=judge_baseline_acc,
    )
    
    # Calculate judge improvement from GRPO-trained critiques
    if judge_grpo_acc is not None:
        improvement = judge_grpo_acc - judge_baseline_acc
        improvement_pct = (improvement / judge_baseline_acc) * 100 if judge_baseline_acc > 0 else 0
    else:
        improvement = None
        improvement_pct = None

    # [STEP 9] Generate critiques for train_unlabeled with GRPO-trained critic
    critiques_unlabeled_grpo = generate_critiques_unlabeled_grpo(
        config=config,
        train_unlabel_ds=train_unlabel_ds,
        strong_tokenizer=strong_tokenizer,
        critic_grpo_output_dir=critic_grpo_output_dir,
        checkpoint_dir=results_dir,
    )

    # [STEP 10] Use judge to generate labels on train_unlabeled
    judge_predictions_unlabeled = generate_judge_labels(
        config=config,
        train_unlabel_ds=train_unlabel_ds,
        critiques_unlabeled_grpo=critiques_unlabeled_grpo,
        weak_tokenizer=weak_tokenizer,
        judge_checkpoint=judge_checkpoint,
        checkpoint_dir=results_dir,
    )

    # [STEP 11] SFT strong model on judge-labeled data and evaluate
    transfer_acc, sft_training_time = sft_strong_model_on_judge_labels(
        config=config,
        train_unlabel_ds=train_unlabel_ds,
        test_ds=test_ds,
        judge_predictions_unlabeled=judge_predictions_unlabeled,
        dataset_name=dataset_name,
        config_key=config_key,
        checkpoint_dir=results_dir,
        experiment_uid=experiment_uid,
        wandb_group=wandb_group_name,
    )

    # Compute PGR using fixed baselines (averaged across all cached seeds)
    # Use ORIGINAL model names for baseline lookups (not the checkpoint paths)
    fixed_weak_mean, fixed_weak_se, weak_num_seeds = get_fixed_weak_baseline(
        weak_model=original_weak_model,
        dataset_name=dataset_name,
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        scheduler=config.lr_schedule,
    )
    
    fixed_strong_mean, fixed_strong_se, strong_num_seeds = get_fixed_ceiling_baseline(
        strong_model=original_strong_model,
        dataset_name=dataset_name,
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        scheduler=config.lr_schedule,
    )
    
    if fixed_weak_mean is not None and fixed_strong_mean is not None and fixed_strong_mean > fixed_weak_mean:
        pgr = (transfer_acc - fixed_weak_mean) / (fixed_strong_mean - fixed_weak_mean)
        print(f"✓ PGR (fixed baselines): {pgr:.4f} (weak={fixed_weak_mean:.4f}, strong={fixed_strong_mean:.4f})")
    else:
        # Fixed baselines not available yet - PGR will be computed by orchestrator
        pgr = None
        print(f"⚠ Fixed baselines not available yet (weak={weak_num_seeds} seeds, strong={strong_num_seeds} seeds)")
        print(f"  PGR will be computed by orchestrator after all seeds complete")

    # Build comprehensive results dictionary
    step1_5_training_time = step1_5_metrics.get('training_time', 0.0) if step1_5_metrics.get('enabled', False) else 0.0
    total_time = step1_5_training_time + judge_training_time + grpo_training_time + sft_training_time
    results = {
        "idea_name": "critic",
        "experiment_uid": experiment_uid,  # Unique ID for wandb grouping
        "weak_model": original_weak_model,  # Use original model names, not checkpoint paths
        "strong_model": original_strong_model,
        "dataset_name": dataset_name,
        # Standard fields (required for analysis/comparison)
        "weak_acc": weak_acc,  # From cached weak model artifacts
        "strong_acc": strong_acc,  # From cached ceiling
        "transfer_acc": transfer_acc,  # From step 11 SFT training
        "pgr": pgr,  # Computed using fixed baselines
        "fixed_weak_baseline": fixed_weak_mean,
        "fixed_strong_baseline": fixed_strong_mean,
        # Step-by-step metrics
        "step1_data_loading": {
            "train_labeled_samples": len(train_label_ds),
            "train_unlabeled_samples": len(train_unlabel_ds),
            "test_samples": len(test_ds),
        },
        "step1_5_sft_critic_on_weak_demos": step1_5_metrics,
        "step2_critique_generation": {
            "generation_time": critique_gen_time,
            "num_critiques_labeled": len(critiques_labeled),
            "num_critiques_test": len(critiques_test),
        },
        "step3_judge_training": {
            "judge_baseline_acc": judge_baseline_acc,
            "training_time": judge_training_time,
        },
        "step4_critique_generation_unlabeled": {
            "generation_time": critique_unlabel_gen_time,
            "num_critiques_unlabeled": len(critiques_unlabeled),
            "num_samples_per_prompt": num_samples,
        },
        "step5_grpo_dataset": {
            "dataset_size": len(grpo_dataset),
        },
        "step6_grpo_training": {
            "training_time": grpo_training_time,
            "epochs": config.grpo_epochs,
            "lr": config.grpo_lr,
            "batch_size": config.grpo_batch_size,
        },
        "step7_critique_generation_test_grpo": {
            "num_critiques_test_grpo": len(critiques_test_grpo),
        },
        "step8_judge_evaluation": {
            "judge_grpo_acc": judge_grpo_acc,
            "improvement": improvement,
            "improvement_pct": improvement_pct,
        },
        "step9_critique_generation_unlabeled_grpo": {
            "num_critiques_unlabeled_grpo": len(critiques_unlabeled_grpo),
        },
        "step10_judge_label_generation": {
            "num_predictions": len(judge_predictions_unlabeled),
        },
        "step11_sft_training": {
            "transfer_acc": transfer_acc,
            "training_time": sft_training_time,
        },
        "total_training_time": total_time,
        "config": {
            "judge_epochs": config.judge_epochs,
            "sft_epochs": config.epochs,
            "grpo_epochs": config.grpo_epochs,
            "sft_critic_lr": config.sft_critic_lr,  # Used for Step 1.5 and Judge training
            "lr": config.lr,  # Used for Final SFT (transfer model)
            "grpo_lr": config.grpo_lr,
            "batch_size": config.batch_size,
            "grpo_batch_size": config.grpo_batch_size,
            "seed": config.seed,
        },
    }
    
    # Print results to console (pretty-printed)
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE - RESULTS")
    print("="*80)
    print(json.dumps(results, indent=2, default=str))
    print("="*80)
    
    # Log to wandb (summary run)
    base_name = f"{config_key}_seed{config.seed}"
    wandb.init(
        project="generalization",
        name=f"{base_name}_summary",
        group=wandb_group_name,  # Group all runs together (includes model names)
        dir=str(results_dir),  # Save wandb logs to the results directory
        config=results['config'],
    )
    
    log_dict = {
        "weak_acc": results['weak_acc'],
        "strong_acc": results['strong_acc'],
        "transfer_acc": results['transfer_acc'],
        "judge_baseline_acc": results['step3_judge_training']['judge_baseline_acc'],
        "judge_grpo_acc": results['step8_judge_evaluation']['judge_grpo_acc'],
        "judge_improvement": results['step8_judge_evaluation']['improvement'],
        "judge_improvement_pct": results['step8_judge_evaluation']['improvement_pct'],
        "total_training_time": results['total_training_time'],
        "step1_5_training_time": step1_5_training_time,
        "judge_training_time": results['step3_judge_training']['training_time'],
        "grpo_training_time": results['step6_grpo_training']['training_time'],
        "sft_training_time": results['step11_sft_training']['training_time'],
    }
    if results['pgr'] is not None:
        log_dict["pgr"] = results['pgr']  # Computed using fixed baselines
    wandb.log(log_dict)
    wandb.finish()

    # Add AAR mode flag to results
    results["aar_mode"] = aar_mode

    # NOTE: Full AAR mode support for critic requires refactoring step 11 to return
    # predictions instead of computing accuracy locally. GT labels are only needed for
    # train_label (judge training), which is allowed in AAR mode as weak supervision.
    # train_unlabel and test don't need GT labels - they use judge-generated pseudo-labels.
    if aar_mode:
        print("\n[AAR Mode] Critic pipeline computed accuracy locally.")
        print("  Full AAR mode (prediction-based eval) requires step 11 refactoring.")

    # Cache result
    cache.set_seed_result(
        idea_name="critic",
        hyperparam_config_key=config_key,
        seed=config.seed,
        result_data=results,
        dataset_name=dataset_name,
    )
    print(f"\n✓ Cached result with config key: {config_key}")

    return results


if __name__ == "__main__":
    parser = create_run_arg_parser(description="Run critic training experiment")
    args = parser.parse_args()
    config = RunConfig.from_args(args)
    
    results = run_experiment(config)
    
    print("\n✓ Experiment completed successfully!")
    print(f"  Transfer accuracy: {results['transfer_acc']:.4f}")
    if results.get('fixed_strong_baseline') is not None:
        print(f"  Strong GT accuracy (fixed): {results['fixed_strong_baseline']:.4f}")
    elif results['strong_acc'] is not None:
        print(f"  Strong GT accuracy: {results['strong_acc']:.4f}")
    else:
        print(f"  Strong GT accuracy: N/A (run ceiling to cache)")
    if results['pgr'] is not None:
        print(f"  PGR (fixed baselines): {results['pgr']:.4f}")
    else:
        print(f"  PGR: N/A (fixed baselines not available yet)")
