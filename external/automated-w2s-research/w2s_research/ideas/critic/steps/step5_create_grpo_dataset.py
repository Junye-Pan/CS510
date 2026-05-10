"""
Step 5: Create GRPO dataset.

Uses utils.create_grpo_dataset_for_critic or create_grpo_dataset_for_critic_binary.
Supports both A/B comparison format and binary format.
"""
from pathlib import Path
from typing import List, Tuple, Union
import json

from datasets import Dataset
from transformers import AutoTokenizer

from w2s_research.core import RunConfig
from w2s_research.ideas.critic.utils import (
    create_grpo_dataset_for_critic,
    create_grpo_dataset_for_critic_binary,
    log_gpu_memory
)


def detect_binary_format(dataset: Dataset) -> bool:
    """Detect if dataset is in binary format (question/choice) vs comparison format (prompt/first/second)."""
    if len(dataset) > 0 and 'choice' in dataset.column_names:
        return True
    return False


def create_grpo_dataset_step(
    config: RunConfig,
    train_unlabel_ds: Dataset,
    critiques_unlabeled: Union[List[Tuple], List[str]],
    prompts_unlabeled: Union[List[Tuple], List[list]],
    strong_tokenizer: AutoTokenizer,
    checkpoint_dir: Path,
) -> Dataset:
    """
    Step 5: Create GRPO dataset.
    
    Supports both formats:
    - Comparison (A/B): Creates 2 prompts per question (one for A, one for B)
    - Binary: Creates 1 prompt per question
    
    IMPORTANT: rewards are computed on-the-fly during GRPO training based on judge predictions.
    
    Args:
        config: Run configuration
        train_unlabel_ds: Training unlabeled dataset
        critiques_unlabeled: Critiques from step 4
        prompts_unlabeled: Prompts from step 4 (raw messages)
        strong_tokenizer: Tokenizer for strong model
        checkpoint_dir: Directory for checkpoints
        
    Returns:
        GRPO dataset with prompts and reference critiques
    """
    print("\n[5/11] Creating GRPO dataset...")
    print("  → Rewards will be computed on-the-fly during GRPO training based on judge predictions")
    
    # Detect format
    is_binary = detect_binary_format(train_unlabel_ds)
    if is_binary:
        print("  → Detected binary format - creating 1 prompt per question")
        step5_checkpoint = checkpoint_dir / "step5_grpo_dataset_binary.json"
    else:
        print("  → Detected comparison format - creating 2 prompts per question (A and B)")
        step5_checkpoint = checkpoint_dir / "step5_grpo_dataset.json"
    
    if step5_checkpoint.exists():
        print(f"  → Loading from checkpoint: {step5_checkpoint}")
        with open(step5_checkpoint, 'r') as f:
            step5_data = json.load(f)
        grpo_dataset = Dataset.from_list(step5_data['grpo_dataset'])
        print(f"  ✓ Loaded GRPO dataset with {len(grpo_dataset)} prompts")
    else:
        if is_binary:
            # Binary format: one prompt per question
            grpo_dataset = create_grpo_dataset_for_critic_binary(
                dataset=train_unlabel_ds,
                critiques=critiques_unlabeled,  # List of critique strings
                prompts=prompts_unlabeled,  # List of raw message dicts
                tokenizer=strong_tokenizer,
            )
            print(f"Created GRPO dataset with {len(grpo_dataset)} prompts (1 per question)")
        else:
            # Comparison format: two prompts per question (A and B)
            grpo_dataset = create_grpo_dataset_for_critic(
                dataset=train_unlabel_ds,
                critiques=critiques_unlabeled,  # List of (critique_a, critique_b) tuples
                prompts=prompts_unlabeled,  # List of (messages_a, messages_b) tuples
                tokenizer=strong_tokenizer,
            )
            print(f"Created GRPO dataset with {len(grpo_dataset)} prompts (2 per question: A and B)")
        
        print(f"  Reference critiques stored directly in dataset entries")
        
        # Save checkpoint
        step5_data = {
            'format': 'binary' if is_binary else 'comparison',
            'grpo_dataset': list(grpo_dataset),
        }
        with open(step5_checkpoint, 'w') as f:
            json.dump(step5_data, f, indent=2)
        print(f"  ✓ Saved checkpoint: {step5_checkpoint}")
    
    # Print metrics
    print(f"\n📊 [STEP 5 METRICS]")
    print(f"  • GRPO dataset size: {len(grpo_dataset)} samples")
    
    # Print GPU memory status
    log_gpu_memory("[STEP 5 END] ")

    return grpo_dataset

