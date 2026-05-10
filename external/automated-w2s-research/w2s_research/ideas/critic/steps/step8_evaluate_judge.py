"""
Step 8: Evaluate judge with GRPO-trained critic.

Uses vLLM for inference. No unsloth dependencies.
Supports both A/B comparison format and binary format.
"""
from pathlib import Path
from typing import Optional, Union, List, Tuple
import json
import gc
import torch

from datasets import Dataset
from transformers import AutoTokenizer

from w2s_research.core import RunConfig, find_latest_checkpoint, evaluate_model
from w2s_research.core.data import format_classification_as_causal
from w2s_research.ideas.critic.utils import (
    format_judge_dataset,
    format_judge_dataset_binary,
    log_gpu_memory
)


def detect_binary_format(dataset: Dataset) -> bool:
    """Detect if dataset is in binary format (question/choice) vs comparison format (prompt/first/second)."""
    if len(dataset) > 0 and 'choice' in dataset.column_names:
        return True
    return False


def evaluate_judge_with_grpo_critiques(
    config: RunConfig,
    test_ds: Dataset,
    critiques_test_grpo: Union[List[Tuple[str, str]], List[str]],
    weak_tokenizer: AutoTokenizer,
    judge_checkpoint: str,
    checkpoint_dir: Path,
    judge_baseline_acc: Optional[float] = None,
) -> Optional[float]:
    """
    Step 8: Evaluate judge with GRPO-trained critic.
    
    Supports both formats:
    - Comparison (A/B): critiques_test_grpo is List[Tuple[str, str]]
    - Binary: critiques_test_grpo is List[str]
    
    Args:
        config: Run configuration
        test_ds: Test dataset
        critiques_test_grpo: Critiques generated with GRPO-trained critic
        weak_tokenizer: Tokenizer for weak judge model
        judge_checkpoint: Path to judge checkpoint
        checkpoint_dir: Directory for checkpoints
        
    Returns:
        Judge accuracy with GRPO critiques, or None if step was skipped
    """
    # Detect format
    is_binary = detect_binary_format(test_ds)
    if is_binary:
        step8_checkpoint = checkpoint_dir / "step8_judge_eval_binary.json"
    else:
        step8_checkpoint = checkpoint_dir / "step8_judge_eval.json"
    
    # Skip step 8 if checkpoint exists
    if step8_checkpoint.exists():
        print("\n[8/11] Skipping judge evaluation (checkpoint exists)...")
        
        # Load from checkpoint
        print(f"  → Loading judge evaluation result from checkpoint: {step8_checkpoint}")
        with open(step8_checkpoint, 'r') as f:
            step8_data = json.load(f)
        judge_grpo_acc = step8_data.get('judge_grpo_acc')
        if judge_grpo_acc is not None:
            print(f"  ✓ Loaded judge test accuracy (with GRPO critiques): {judge_grpo_acc:.4f}")
        else:
            print(f"  ⚠️  Checkpoint exists but missing judge_grpo_acc, using None")
        return judge_grpo_acc
    else:
        print("\n[8/11] Evaluating judge with GRPO-trained critic...")
        if is_binary:
            print("  → Using binary format")
        else:
            print("  → Using comparison format")

        # Format dataset for evaluation based on format
        if is_binary:
            test_judge_grpo_ds = format_judge_dataset_binary(test_ds, critiques_test_grpo)
        else:
            test_judge_grpo_ds = format_judge_dataset(test_ds, critiques_test_grpo)
        
        # Use judge template with appropriate format
        # Note: use_binary_format is auto-detected by format_classification_as_causal
        # based on whether the dataset has 'question'/'choice' fields
        test_formatted, template, label_tokens = format_classification_as_causal(
            test_judge_grpo_ds,
            weak_tokenizer,
            max_ctx=config.critic_judge_max_ctx,
            use_judge_template=True,
        )
        label_token_ids = [weak_tokenizer.encode(tok, add_special_tokens=False)[0] for tok in label_tokens]

        # Evaluate with longer context for critiques
        judge_grpo_eval = evaluate_model(
            test_dataset=test_formatted,
            label_token_ids=label_token_ids,
            tokenizer=weak_tokenizer,
            model_name=config.weak_model,
            lora_checkpoint=judge_checkpoint,
            original_dataset=test_ds,
            max_model_len=config.critic_judge_max_ctx,
        )

        judge_grpo_acc = judge_grpo_eval["accuracy"]
        print(f"✓ Judge test accuracy (with GRPO critiques): {judge_grpo_acc:.4f}")
        
        # Save checkpoint for future resumption
        step8_data = {
            'judge_grpo_acc': judge_grpo_acc,
            'format': 'binary' if is_binary else 'comparison',
        }
        with open(step8_checkpoint, 'w') as f:
            json.dump(step8_data, f, indent=2)
        print(f"  ✓ Saved checkpoint: {step8_checkpoint}")
    
    # Print metrics
    print(f"\n📊 [STEP 8 METRICS]")
    if judge_grpo_acc is not None:
        if judge_baseline_acc is not None:
            improvement = judge_grpo_acc - judge_baseline_acc
            improvement_pct = (improvement / judge_baseline_acc) * 100 if judge_baseline_acc > 0 else 0
            print(f"  • Judge accuracy (with GRPO critiques): {judge_grpo_acc:.4f}")
            print(f"  • Improvement over baseline: {improvement:+.4f} ({improvement_pct:+.2f}%)")
        else:
            print(f"  • Judge accuracy (with GRPO critiques): {judge_grpo_acc:.4f}")
    else:
        print(f"  • Judge accuracy: N/A (step 8 was skipped)")

    # Print GPU memory status
    log_gpu_memory("[STEP 8 END] ")

    # Cleanup GPU memory before next steps
    print("\nCleaning up GPU memory after judge evaluation...")
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
    print("✓ GPU memory cleaned")

    return judge_grpo_acc
