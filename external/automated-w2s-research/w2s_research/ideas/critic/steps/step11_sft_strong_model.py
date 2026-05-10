"""
Step 11: SFT strong model on judge-labeled data and evaluate.

Uses train_model from core which may use unsloth.
"""
from pathlib import Path
from typing import List, Tuple
import json
import time
import gc
import torch

from datasets import Dataset

from w2s_research.core import RunConfig, train_model, find_latest_checkpoint
from w2s_research.core.data import BINARY_LABEL_TOKENS, COMPARISON_LABEL_TOKENS
from w2s_research.ideas.critic.utils import (
    check_training_checkpoint,
    log_gpu_memory,
)


def detect_binary_format(dataset: Dataset) -> bool:
    """Detect if dataset is in binary format (question/choice) vs comparison format (prompt/first/second)."""
    if len(dataset) > 0 and 'choice' in dataset.column_names:
        return True
    return False


def sft_strong_model_on_judge_labels(
    config: RunConfig,
    train_unlabel_ds: Dataset,
    test_ds: Dataset,
    judge_predictions_unlabeled: List[str],
    dataset_name: str,
    config_key: str,
    checkpoint_dir: Path,
    experiment_uid: str = "",
    wandb_group: str = "",  # Separate wandb group name (can include model names)
) -> Tuple[float, float]:
    """
    Step 11: SFT strong model on judge-labeled data and evaluate.
    
    Supports both formats:
    - Comparison (A/B): predictions are 'A' or 'B'
    - Binary: predictions are 'True' or 'False' (True=correct, False=incorrect, no space prefix)
    
    Args:
        config: Run configuration
        train_unlabel_ds: Training unlabeled dataset
        test_ds: Test dataset
        judge_predictions_unlabeled: Judge predictions from step 10
        dataset_name: Dataset name
        config_key: Hyperparameter config key (aligned with other ideas)
        checkpoint_dir: Directory for checkpoints
        experiment_uid: Unique experiment ID for wandb grouping (appended to output_dir)
        
    Returns:
        Tuple of (strong_test_acc, sft_training_time)
    """
    print("\n[11/11] SFT Strong Model on Judge-Labeled Data...")
    
    # Detect format
    is_binary = detect_binary_format(train_unlabel_ds)
    if is_binary:
        print("  → Detected binary format")
    else:
        print("  → Detected comparison format")
    
    # Create labeled dataset for SFT
    if is_binary:
        # Binary format: BINARY_LABEL_TOKENS = ["False", "True"]
        # index 0 = "False" (incorrect), index 1 = "True" (correct) - matches bool->int conversion
        label_tokens = BINARY_LABEL_TOKENS  # ["False", "True"]
        # Map judge predictions to label indices: "False" -> 0, "True" -> 1
        judge_label_indices = [label_tokens.index(pred) if pred in label_tokens else 1 for pred in judge_predictions_unlabeled]
        
        sft_train_data = []
        for i, item in enumerate(train_unlabel_ds):
            judge_label = judge_predictions_unlabeled[i]
            # Direct mapping: "False" -> 0, "True" -> 1
            label_int = label_tokens.index(judge_label) if judge_label in label_tokens else 1
            sft_train_data.append({
                "question": item["question"],
                "choice": item["choice"],
                "label": label_int,  # 0=False (incorrect), 1=True (correct) - matches BINARY_LABEL_TOKENS
            })
        
        sft_train_ds = Dataset.from_list(sft_train_data)
        print(f"✓ Created SFT training dataset with {len(sft_train_ds)} samples")
        print(f"  Label distribution: {sum(1 for x in judge_label_indices if x == 1)} True (correct), {sum(1 for x in judge_label_indices if x == 0)} False (incorrect)")
    else:
        # Comparison format: COMPARISON_LABEL_TOKENS[0] = "A" -> 0, COMPARISON_LABEL_TOKENS[1] = "B" -> 1
        label_tokens = COMPARISON_LABEL_TOKENS  # ["A", "B"] - no space prefix
        judge_label_indices = [label_tokens.index(pred) if pred in label_tokens else 0 for pred in judge_predictions_unlabeled]
        
        sft_train_data = []
        for i, item in enumerate(train_unlabel_ds):
            judge_label = judge_predictions_unlabeled[i]
            label_int = 0 if judge_label == label_tokens[0] else 1  # label_tokens[0] = " A"
            sft_train_data.append({
                "prompt": item["prompt"],
                "first": item["first"],
                "second": item["second"],
                "label": label_int,  # Convert 'A'/'B' to 0/1
            })
        
        sft_train_ds = Dataset.from_list(sft_train_data)
        print(f"✓ Created SFT training dataset with {len(sft_train_ds)} samples")
        print(f"  Label distribution: {sum(1 for x in judge_label_indices if x == 0)} A, {sum(1 for x in judge_label_indices if x == 1)} B")
    
    # Include experiment_uid in output_dir name so wandb run name includes it
    uid_suffix = f"_{experiment_uid}" if experiment_uid else ""
    sft_output_dir = f"./results/{dataset_name}_critic/{config_key}/seed_{config.seed}/sft{uid_suffix}"
    
    # Check if SFT is already complete
    step11_checkpoint = checkpoint_dir / "step11_sft_training.json"
    step11_data = check_training_checkpoint(step11_checkpoint)
    use_cached_sft = False
    if step11_data and find_latest_checkpoint(sft_output_dir):
        cached_test_size = step11_data.get('test_size', None)
        if cached_test_size == config.test_size:
            print(f"  → Loading from checkpoint: {step11_checkpoint}")
            strong_test_acc = step11_data['strong_test_acc']
            sft_training_time = step11_data.get('training_time', 0.0)
            print(f"  ✓ Strong model test accuracy (cached): {strong_test_acc:.4f}")
            print(f"  ✓ Test size: {len(test_ds)} samples (matches cache)")
            use_cached_sft = True
        else:
            print(f"  ⚠️  Cached result has test_size={cached_test_size}, current is {config.test_size}")
            print(f"  → Re-running SFT training with test_size={config.test_size}")
    
    if not use_cached_sft:
        start_time = time.time()
        
        # Check and reset PyTorch CUDA allocator config
        # Step 6 may have set expandable_segments:False in Ray workers,
        # but the main process allocator state might still be affected
        import os
        current_alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        print(f"\n{'='*60}")
        print("GPU MEMORY STATE BEFORE STEP 11")
        print(f"{'='*60}")
        print(f"Current PYTORCH_CUDA_ALLOC_CONF: {current_alloc_conf if current_alloc_conf else '(not set)'}")
        
        # Explicitly set expandable_segments:True for better large block allocation
        # This helps Unsloth load models even if memory is fragmented
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        print("  → Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for step 11")
        
        # Check detailed memory state before loading model
        stats = log_gpu_memory("[BEFORE MODEL LOAD] ", include_nvidia_smi=True)
        
        # Check if reserved memory is high (indicates fragmentation)
        if stats.get('reserved_gb', 0) > 10:
            print(f"\n⚠️  WARNING: High reserved memory ({stats['reserved_gb']:.2f} GB)")
            print("   This may prevent large contiguous allocations")
            print("   Attempting to release reserved memory...")
        
        # Aggressive cleanup to release reserved memory and reset allocator state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
            # Reset memory stats to clear allocator state
            torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()
        
        # Check again after cleanup
        stats_after = log_gpu_memory("[AFTER CLEANUP] ", include_nvidia_smi=True)
        if stats.get('reserved_gb', 0) > 10:
            print(f"   Reserved memory after cleanup: {stats_after.get('reserved_gb', 0):.2f} GB")
            if stats_after.get('reserved_gb', 0) < stats.get('reserved_gb', 0):
                print("   ✓ Reserved memory reduced")
            else:
                print("   ⚠️  Reserved memory not reduced (may indicate fragmentation)")
        
        print(f"{'='*60}\n")
        print("✓ Memory allocator config reset and cleanup complete")
        
        # Train strong model
        # Use gradient_accumulation_steps=4 to reduce memory usage
        # This helps avoid OOM when running after GRPO training (step 6)
        # which uses expandable_segments:False causing memory fragmentation
        # Note: use_binary_format is auto-detected by format_classification_as_causal
        # based on whether the dataset has 'question'/'choice' fields
        trainer_kwargs = {}
        
        sft_results = train_model(
            model_name=config.strong_model,
            train_dataset=sft_train_ds,
            test_dataset=test_ds,  # Evaluate on test set
            batch_size=config.batch_size // 4,
            lr=config.lr,
            epochs=config.epochs,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            max_ctx=config.max_ctx,  
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            load_in_4bit=config.load_in_4bit,
            output_dir=sft_output_dir,
            optimizer=config.optimizer,
            lr_scheduler_type=config.lr_schedule,
            gradient_accumulation_steps=4,  # Reduce memory usage to avoid OOM after GRPO
            bf16=config.bf16,
            seed=config.seed,
            wandb_group=wandb_group or experiment_uid,  # Group related runs together
            trainer_kwargs=trainer_kwargs,
            save_total_limit=1,  # Only keep last checkpoint to save disk space
        )
        
        strong_test_acc = sft_results["final_eval"]["accuracy"]
        sft_training_time = time.time() - start_time
        
        print(f"✓ Strong model test accuracy: {strong_test_acc:.4f}")
        
        # Save checkpoint
        step11_data = {
            'strong_test_acc': strong_test_acc,
            'training_time': sft_training_time,
            'output_dir': sft_output_dir,
            'test_size': config.test_size,
            'test_samples': len(test_ds),
            'format': 'binary' if is_binary else 'comparison',
        }
        with open(step11_checkpoint, 'w') as f:
            json.dump(step11_data, f, indent=2)
        print(f"  ✓ Saved checkpoint: {step11_checkpoint}")
        print(f"  ✓ Test size: {len(test_ds)} samples")
    else:
        sft_training_time = step11_data.get('training_time', 0.0)
    
    # Print metrics
    print(f"\n📊 [STEP 11 METRICS]")
    print(f"  • Strong model test accuracy: {strong_test_acc:.4f}")
    print(f"  • Training time: {sft_training_time:.2f}s")
    print(f"  • Note: train_model handles wandb logging internally")
    
    # Print GPU memory status
    log_gpu_memory("[STEP 11 END] ")
    
    # Cleanup GPU memory
    print("\nCleaning up GPU memory after SFT training...")
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
    print("✓ GPU memory cleaned")

    return strong_test_acc, sft_training_time

