"""
Step 7: Generate critiques with GRPO-trained critic.

Uses vLLM for fast batch inference. No unsloth dependencies.
Supports both A/B comparison format and binary format.
"""
from pathlib import Path
from typing import List, Tuple, Union
import json
import gc
import torch

from datasets import Dataset
from transformers import AutoTokenizer

from w2s_research.core import RunConfig
from w2s_research.ideas.critic.utils import (
    generate_critiques_vllm,
    generate_critiques_vllm_binary,
    log_gpu_memory,
    find_latest_verl_checkpoint,
)


def detect_binary_format(dataset: Dataset) -> bool:
    """Detect if dataset is in binary format (question/choice) vs comparison format (prompt/first/second)."""
    if len(dataset) > 0 and 'choice' in dataset.column_names:
        return True
    return False


def generate_critiques_test_grpo(
    config: RunConfig,
    test_ds: Dataset,
    strong_tokenizer: AutoTokenizer,
    critic_grpo_output_dir: str,
    checkpoint_dir: Path,
    config_key: str,
) -> Union[List[Tuple[str, str]], List[str]]:
    """
    Step 7: Generate new critiques with GRPO-trained critic.
    
    Supports both formats:
    - Comparison (A/B): Returns List[Tuple[str, str]] - (critique_a, critique_b) per sample
    - Binary: Returns List[str] - one critique per sample
    
    Args:
        config: Run configuration
        test_ds: Test dataset
        strong_tokenizer: Tokenizer for strong model
        critic_grpo_output_dir: Output directory for GRPO training
        checkpoint_dir: Directory for checkpoints
        config_key: Config key for experiment name
        
    Returns:
        List of critiques (format depends on dataset format)
    """
    print("\n[7/11] Generating critiques with GRPO-trained critic...")
    
    # Detect format
    is_binary = detect_binary_format(test_ds)
    if is_binary:
        print("  → Detected binary format")
        step7_checkpoint = checkpoint_dir / "step7_critiques_test_grpo_binary.json"
    else:
        print("  → Detected comparison format")
        step7_checkpoint = checkpoint_dir / "step7_critiques_test_grpo.json"

    # Find GRPO checkpoint
    grpo_checkpoint = find_latest_verl_checkpoint(
        output_dir=critic_grpo_output_dir,
        return_lora_path=True,
    )
    if not grpo_checkpoint:
        raise RuntimeError(
            f"No GRPO LoRA checkpoint found in {critic_grpo_output_dir}"
        )
    print(f"  → Found GRPO LoRA checkpoint: {grpo_checkpoint}")
    
    if step7_checkpoint.exists():
        print(f"  → Loading from checkpoint: {step7_checkpoint}")
        with open(step7_checkpoint, 'r') as f:
            step7_data = json.load(f)
        
        if is_binary:
            critiques_test_grpo = [item['critique'] for item in step7_data['critiques']]
        else:
            critiques_test_grpo = [(item['critique_a'], item['critique_b']) for item in step7_data['critiques']]
        print(f"  ✓ Loaded {len(critiques_test_grpo)} critiques from checkpoint")
    else:
        if is_binary:
            # Binary format
            critiques_test_grpo = generate_critiques_vllm_binary(
                model_name=config.strong_model,
                lora_checkpoint=grpo_checkpoint,
                tokenizer=strong_tokenizer,
                dataset=test_ds,
                max_new_tokens=config.gen_max_tokens,
                temperature=config.gen_temperature,
                top_p=config.gen_top_p,
                top_k=config.gen_top_k,
                min_p=config.gen_min_p,
                repetition_penalty=config.gen_repetition_penalty,
                presence_penalty=config.gen_presence_penalty,
            )
            
            # Save checkpoint
            step7_data = {
                'format': 'binary',
                'critiques': [{'critique': c} for c in critiques_test_grpo],
            }
        else:
            # Comparison format
            critiques_test_grpo = generate_critiques_vllm(
                model_name=config.strong_model,
                lora_checkpoint=grpo_checkpoint,
                tokenizer=strong_tokenizer,
                dataset=test_ds,
                max_new_tokens=config.gen_max_tokens,
                temperature=config.gen_temperature,
                top_p=config.gen_top_p,
                top_k=config.gen_top_k,
                min_p=config.gen_min_p,
                repetition_penalty=config.gen_repetition_penalty,
                presence_penalty=config.gen_presence_penalty,
            )
            
            # Save checkpoint
            step7_data = {
                'format': 'comparison',
                'critiques': [
                    {'critique_a': ca, 'critique_b': cb}
                    for ca, cb in critiques_test_grpo
                ],
            }
        
        with open(step7_checkpoint, 'w') as f:
            json.dump(step7_data, f, indent=2)
        print(f"  ✓ Saved checkpoint: {step7_checkpoint}")
    
    # Print metrics
    print(f"\n📊 [STEP 7 METRICS]")
    print(f"  • Critiques generated (test, GRPO-trained): {len(critiques_test_grpo)}")
    
    # Print GPU memory status
    log_gpu_memory("[STEP 7 END] ")
    
    # Cleanup GPU memory
    print("\nCleaning up GPU memory after critique generation...")
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
    print("✓ GPU memory cleaned")

    return critiques_test_grpo

