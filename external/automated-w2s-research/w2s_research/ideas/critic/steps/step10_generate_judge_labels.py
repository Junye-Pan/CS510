"""
Step 10: Use judge to generate labels on train_unlabeled.

Uses vLLM for inference. No unsloth dependencies.
Supports both A/B comparison format and binary format.
"""
from pathlib import Path
from typing import List, Union
import json
import gc
import torch

from datasets import Dataset
from transformers import AutoTokenizer

from w2s_research.core import RunConfig
from w2s_research.core.data import BINARY_LABEL_TOKENS, COMPARISON_LABEL_TOKENS
from w2s_research.ideas.critic.utils import (
    format_judge_dataset,
    format_judge_dataset_binary,
    get_judge_predictions_vllm,
    get_judge_predictions_vllm_binary,
    log_gpu_memory,
)


def detect_binary_format(dataset: Dataset) -> bool:
    """Detect if dataset is in binary format (question/choice) vs comparison format (prompt/first/second)."""
    if len(dataset) > 0 and 'choice' in dataset.column_names:
        return True
    return False


def generate_judge_labels(
    config: RunConfig,
    train_unlabel_ds: Dataset,
    critiques_unlabeled_grpo: Union[List[tuple], List[str]],
    weak_tokenizer: AutoTokenizer,
    judge_checkpoint: str,
    checkpoint_dir: Path,
) -> List[str]:
    """
    Step 10: Use judge to generate labels on train_unlabeled.
    
    Supports both formats:
    - Comparison (A/B): Returns List[str] of 'A' or 'B'
    - Binary: Returns List[str] of 'True' or 'False' (True=correct, False=incorrect, no space prefix)
    
    Args:
        config: Run configuration
        train_unlabel_ds: Training unlabeled dataset
        critiques_unlabeled_grpo: Critiques generated with GRPO-trained critic
        weak_tokenizer: Tokenizer for weak judge model
        judge_checkpoint: Path to judge checkpoint
        checkpoint_dir: Directory for checkpoints
        
    Returns:
        List of judge predictions
    """
    print("\n[10/11] Using judge to generate labels on train_unlabeled...")
    
    # Detect format
    is_binary = detect_binary_format(train_unlabel_ds)
    if is_binary:
        print("  → Detected binary format")
        step10_checkpoint = checkpoint_dir / "step10_judge_labels_unlabeled_binary.json"
    else:
        print("  → Detected comparison format")
        step10_checkpoint = checkpoint_dir / "step10_judge_labels_unlabeled.json"
    
    if step10_checkpoint.exists():
        print(f"  → Loading from checkpoint: {step10_checkpoint}")
        with open(step10_checkpoint, 'r') as f:
            step10_data = json.load(f)
        judge_predictions_unlabeled = step10_data['judge_predictions']
        print(f"  ✓ Loaded {len(judge_predictions_unlabeled)} judge predictions from checkpoint")
    else:
        if is_binary:
            # Binary format
            train_unlabel_judge_ds = format_judge_dataset_binary(train_unlabel_ds, critiques_unlabeled_grpo)
            
            judge_predictions_unlabeled = get_judge_predictions_vllm_binary(
                model_name=config.weak_model,
                lora_checkpoint=judge_checkpoint,
                tokenizer=weak_tokenizer,
                dataset=train_unlabel_judge_ds,
                max_ctx=config.critic_judge_max_ctx,
            )
            
            # Calculate judge accuracy on unlabeled data (if labels exist)
            if "label" in train_unlabel_ds[0]:
                label_tokens = BINARY_LABEL_TOKENS  # ["False", "True"] - index 0=False (incorrect), index 1=True (correct)
                # Judge predicts "True" (correct) or "False" (incorrect)
                # Map to indices: "False" -> 0, "True" -> 1
                judge_label_indices = [label_tokens.index(pred) if pred in label_tokens else 1 for pred in judge_predictions_unlabeled]
                ground_truth_labels = [train_unlabel_ds[i]["label"] for i in range(len(train_unlabel_ds))]
                # ground_truth is bool: True=correct, False=incorrect
                # Convert to indices matching BINARY_LABEL_TOKENS: 0=False (incorrect), 1=True (correct)
                ground_truth_indices = [1 if gt else 0 for gt in ground_truth_labels]
                
                judge_unlabel_acc = sum(pred == gt for pred, gt in zip(judge_label_indices, ground_truth_indices)) / len(ground_truth_indices)
                print(f"✓ Judge accuracy on unlabeled data: {judge_unlabel_acc:.4f}")
        else:
            # Comparison format
            train_unlabel_judge_ds = format_judge_dataset(train_unlabel_ds, critiques_unlabeled_grpo)
            
            judge_predictions_unlabeled = get_judge_predictions_vllm(
                model_name=config.weak_model,
                lora_checkpoint=judge_checkpoint,
                tokenizer=weak_tokenizer,
                dataset=train_unlabel_judge_ds,
                max_ctx=config.critic_judge_max_ctx,
            )
            
            # Calculate judge accuracy on unlabeled data (if labels exist)
            if "label" in train_unlabel_ds[0]:
                label_tokens = COMPARISON_LABEL_TOKENS  # ["A", "B"] - no space prefix
                judge_label_indices = [label_tokens.index(pred) if pred in label_tokens else 0 for pred in judge_predictions_unlabeled]
                ground_truth_labels = [train_unlabel_ds[i]["label"] for i in range(len(train_unlabel_ds))]
                if isinstance(ground_truth_labels[0], str):
                    # Ground truth might be "A"/"B", normalize to match label_tokens
                    gt_normalized = [label_tokens[0] if gt == 'A' or gt == label_tokens[0] else label_tokens[1] if gt == 'B' or gt == label_tokens[1] else label_tokens[0] for gt in ground_truth_labels]
                    ground_truth_indices = [label_tokens.index(gt) if gt in label_tokens else 0 for gt in gt_normalized]
                else:
                    ground_truth_indices = ground_truth_labels
                
                judge_unlabel_acc = sum(pred == gt for pred, gt in zip(judge_label_indices, ground_truth_indices)) / len(ground_truth_indices)
                print(f"✓ Judge accuracy on unlabeled data: {judge_unlabel_acc:.4f}")
        
        print(f"✓ Generated {len(judge_predictions_unlabeled)} judge predictions")
        
        # Save checkpoint
        step10_data = {
            'format': 'binary' if is_binary else 'comparison',
            'judge_predictions': judge_predictions_unlabeled,
        }
        with open(step10_checkpoint, 'w') as f:
            json.dump(step10_data, f, indent=2)
        print(f"  ✓ Saved checkpoint: {step10_checkpoint}")
    
    # Print metrics
    print(f"\n📊 [STEP 10 METRICS]")
    print(f"  • Judge predictions generated: {len(judge_predictions_unlabeled)}")
    
    # Count label distribution
    if is_binary:
        label_counts = {token: 0 for token in BINARY_LABEL_TOKENS}  # ["True", "False"]
    else:
        label_counts = {token: 0 for token in COMPARISON_LABEL_TOKENS}  # ["A", "B"]
    for pred in judge_predictions_unlabeled:
        if pred in label_counts:
            label_counts[pred] += 1
    print(f"  • Label distribution: {label_counts}")
    
    # Print GPU memory status
    log_gpu_memory("[STEP 10 END] ")
    
    # Cleanup GPU memory after step 10 (vLLM inference) before step 11 (SFT training)
    # Step 11 needs to load the strong model for training, so we need to ensure vLLM memory is fully released
    # This is the CRITICAL cleanup before SFT training - GPU should be completely empty
    print("\nCleaning up GPU memory before step 11 (SFT training)...")
    
    # Final verification that GPU is clean
    print("\n" + "="*60)
    print("FINAL GPU STATE CHECK BEFORE STEP 11")
    print("="*60)
    final_stats = log_gpu_memory("[FINAL CHECK] ")
    if final_stats.get('allocated_gb', 0) > 0.5:
        print(f"\n🚨 CRITICAL: GPU not clean! {final_stats['allocated_gb']:.2f} GB still allocated")
        print("   Step 11 may fail with OOM. Consider:")
        print("   - Reducing batch_size")
        print("   - Reducing max_ctx")
        print("   - Checking for memory leaks in previous steps")
    else:
        print(f"\n✅ GPU is clean and ready for step 11 (allocated: {final_stats.get('allocated_gb', 0):.2f} GB)")
    print("="*60 + "\n")
    
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
    print("✓ GPU memory cleaned")

    return judge_predictions_unlabeled

