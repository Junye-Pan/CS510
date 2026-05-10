"""
Step 3: Train weak judge on train_labeled with critiques.

Uses train_model from core which may use unsloth.
Supports both A/B comparison format and binary format.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Union
import json
import time
import gc
import torch

from datasets import Dataset

from w2s_research.core import RunConfig, train_model, find_latest_checkpoint
from w2s_research.ideas.critic.utils import (
    format_judge_dataset,
    format_judge_dataset_binary,
    check_training_checkpoint,
    log_gpu_memory,
)


def detect_binary_format(dataset: Dataset) -> bool:
    """Detect if dataset is in binary format (question/choice) vs comparison format (prompt/first/second)."""
    if len(dataset) > 0 and 'choice' in dataset.column_names:
        return True
    return False


def train_judge_model_step(
    config: RunConfig,
    train_label_ds: Dataset,
    test_ds: Dataset,
    critiques_labeled: Union[List[Tuple[str, str]], List[str]],
    critiques_test: Union[List[Tuple[str, str]], List[str]],
    dataset_name: str,
    config_key: str,
    checkpoint_dir: Path,
    experiment_uid: str = "",
    wandb_group: str = "",  # Separate wandb group name (can include model names)
) -> Tuple[Dict, str, float]:
    """
    Step 3: Train weak judge on train_labeled with critiques.
    
    Supports both formats:
    - Comparison (A/B): critiques are List[Tuple[str, str]] - (critique_a, critique_b) per sample
    - Binary: critiques are List[str] - one critique per sample
    
    Args:
        config: Run configuration
        train_label_ds: Training labeled dataset
        test_ds: Test dataset
        critiques_labeled: Critiques for train_labeled
        critiques_test: Critiques for test set
        dataset_name: Dataset name
        config_key: Hyperparameter config key (aligned with other ideas)
        checkpoint_dir: Directory for checkpoints
        experiment_uid: Unique experiment ID for wandb grouping (appended to output_dir)
        wandb_group: Separate wandb group name (can include model names)
        
    Returns:
        Tuple of (judge_results, judge_output_dir, training_time)
    """
    print("\n[3/11] Training weak judge with critiques...")
    
    # Detect format
    is_binary = detect_binary_format(train_label_ds)
    if is_binary:
        print("  → Detected binary format")
    else:
        print("  → Detected comparison format")
    
    # Note: train_model will initialize wandb itself

    step3_checkpoint = checkpoint_dir / "step3_judge_training.json"
    # Include experiment_uid in output_dir name so wandb run name includes it
    uid_suffix = f"_{experiment_uid}" if experiment_uid else ""
    judge_output_dir = f"./results/{dataset_name}_critic/{config_key}/seed_{config.seed}/judge{uid_suffix}"

    # Check if judge training is already complete
    step3_data = check_training_checkpoint(step3_checkpoint)
    if step3_data and find_latest_checkpoint(judge_output_dir):
        print(f"  → Loading from checkpoint: {step3_checkpoint}")
        judge_baseline_acc = step3_data['judge_baseline_acc']
        judge_training_time = step3_data.get('training_time', 0.0)
        print(f"  ✓ Judge baseline accuracy (cached): {judge_baseline_acc:.4f}")
        
        # Return mock results dict for compatibility
        judge_results = {
            "final_eval": {"accuracy": judge_baseline_acc}
        }
    else:
        start_time = time.time()

        # Format datasets with critiques based on format
        if is_binary:
            train_judge_ds = format_judge_dataset_binary(train_label_ds, critiques_labeled)
            test_judge_ds = format_judge_dataset_binary(test_ds, critiques_test)
        else:
            train_judge_ds = format_judge_dataset(train_label_ds, critiques_labeled)
            test_judge_ds = format_judge_dataset(test_ds, critiques_test)

        # Train judge
        # Note: use_binary_format is auto-detected by format_classification_as_causal
        # based on whether the dataset has 'question'/'choice' fields
        trainer_kwargs = {"use_judge_template": True}
        
        judge_results = train_model(
            model_name=config.weak_model,
            train_dataset=train_judge_ds,
            test_dataset=test_judge_ds,
            batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            lr=config.sft_critic_lr,  # Use sft_critic_lr for judge training (higher than final SFT lr)
            epochs=config.judge_epochs,  # Use judge_epochs for judge training
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            max_ctx=config.critic_judge_max_ctx,  # Use longer context for judge with critiques
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            load_in_4bit=config.load_in_4bit,
            optimizer=config.optimizer,
            lr_scheduler_type=config.lr_schedule,
            output_dir=judge_output_dir,
            seed=config.seed,
            trainer_kwargs=trainer_kwargs,
            wandb_group=wandb_group or experiment_uid,  # Group related runs together
            save_total_limit=1,  # Only keep last checkpoint to save disk space
        )

        judge_baseline_acc = judge_results["final_eval"]["accuracy"]
        judge_training_time = time.time() - start_time

        print(f"✓ Judge baseline accuracy: {judge_baseline_acc:.4f}")

        # Save checkpoint
        step3_data = {
            'judge_baseline_acc': judge_baseline_acc,
            'training_time': judge_training_time,
            'output_dir': judge_output_dir,
            'format': 'binary' if is_binary else 'comparison',
        }
        with open(step3_checkpoint, 'w') as f:
            json.dump(step3_data, f, indent=2)
        print(f"  ✓ Saved checkpoint: {step3_checkpoint}")
    
    # Print metrics
    print(f"\n📊 [STEP 3 METRICS]")
    print(f"  • Judge baseline accuracy: {judge_baseline_acc:.4f}")
    print(f"  • Training time: {judge_training_time:.2f}s")
    
    # Note: train_model handles wandb logging and finishing
    
    # Print GPU memory status
    log_gpu_memory("[STEP 3 END] ")
    
    # Cleanup GPU memory
    print("\nCleaning up GPU memory after judge training...")
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
    print("✓ GPU memory cleaned")

    return judge_results, judge_output_dir, judge_training_time

