"""
Step 4: Generate critiques for train_unlabeled.

Uses vLLM for fast batch inference.
Supports both A/B comparison format and binary format.
"""
from pathlib import Path
from typing import List, Tuple, Union
import json
import time
import gc
import torch

from datasets import Dataset
from transformers import AutoTokenizer

from w2s_research.core import RunConfig
from w2s_research.ideas.critic.utils import (
    generate_critiques_vllm, 
    generate_critiques_vllm_binary,
    log_gpu_memory
)


def detect_binary_format(dataset: Dataset) -> bool:
    """Detect if dataset is in binary format (question/choice) vs comparison format (prompt/first/second)."""
    if len(dataset) > 0 and 'choice' in dataset.column_names:
        return True
    return False


def generate_critiques_for_unlabeled(
    config: RunConfig,
    train_unlabel_ds: Dataset,
    strong_tokenizer: AutoTokenizer,
    checkpoint_dir: Path,
) -> Tuple[Union[List[Tuple], List[str]], Union[List[Tuple], List[list]], int, float]:
    """
    Step 4: Generate critiques for train_unlabeled.
    
    Supports both formats:
    - Comparison (A/B): Returns (List[Tuple], List[Tuple], ...) - (critique_a, critique_b) per sample
    - Binary: Returns (List[str], List[list], ...) - one critique per sample
    
    Args:
        config: Run configuration
        train_unlabel_ds: Training unlabeled dataset
        strong_tokenizer: Tokenizer for strong model
        checkpoint_dir: Directory for checkpoints
        
    Returns:
        Tuple of (critiques_unlabeled, prompts_unlabeled, num_samples, generation_time)
    """
    print("\n[4/11] Generating critiques for train_unlabeled...")
    
    # Detect format
    is_binary = detect_binary_format(train_unlabel_ds)
    if is_binary:
        print("  → Detected binary format - generating one critique per sample")
        step4_checkpoint = checkpoint_dir / "step4_critiques_unlabeled_binary.json"
    else:
        print("  → Detected comparison format - generating critiques for options A and B")
        step4_checkpoint = checkpoint_dir / "step4_critiques_unlabeled.json"
    
    critique_unlabel_gen_time = 0.0
    
    if step4_checkpoint.exists():
        print(f"  → Loading from checkpoint: {step4_checkpoint}")
        with open(step4_checkpoint, 'r') as f:
            step4_data = json.load(f)
        
        if is_binary:
            # Binary format: single critique per sample
            critiques_unlabeled = [item['critique'] for item in step4_data['entries']]
            prompts_unlabeled = [item['critique_messages'] for item in step4_data['entries']]
        else:
            # Comparison format: (critique_a, critique_b) per sample
            critiques_unlabeled = []
            prompts_unlabeled = []
            for item in step4_data['entries']:
                critiques_unlabeled.append((item['critique_a'], item['critique_b']))
                prompts_unlabeled.append((item['critique_messages_a'], item['critique_messages_b']))
        
        num_samples = step4_data.get('num_samples', 1)
        critique_unlabel_gen_time = step4_data.get('generation_time', 0.0)
        print(f"  ✓ Loaded {len(critiques_unlabeled)} critique sets from checkpoint")
    else:
        start_time = time.time()
        num_samples = 1  # Number of critique samples per option
        
        if is_binary:
            # Binary format: generate one critique per sample
            critiques_unlabeled, prompts_unlabeled = generate_critiques_vllm_binary(
                model_name=config.strong_model,
                lora_checkpoint=None,
                tokenizer=strong_tokenizer,
                dataset=train_unlabel_ds,
                max_new_tokens=config.gen_max_tokens,
                return_prompts=True,
                temperature=config.gen_temperature,
                top_p=config.gen_top_p,
                top_k=config.gen_top_k,
                min_p=config.gen_min_p,
                repetition_penalty=config.gen_repetition_penalty,
                presence_penalty=config.gen_presence_penalty,
            )
            critique_unlabel_gen_time = time.time() - start_time
            print(f"✓ Generated {len(critiques_unlabeled)} critiques in {critique_unlabel_gen_time:.2f}s")
            
            # Save checkpoint with binary schema
            step4_data = {
                'format': 'binary',
                'entries': [
                    {
                        'question': item['question'],
                        'choice': item['choice'],
                        'label': item['label'],
                        'critique_messages': msg,
                        'critique': c,
                    }
                    for item, c, msg in zip(train_unlabel_ds, critiques_unlabeled, prompts_unlabeled)
                ],
                'num_samples': num_samples,
                'generation_time': critique_unlabel_gen_time,
            }
        else:
            # Comparison format: generate critiques for both options
            critiques_unlabeled, prompts_unlabeled = generate_critiques_vllm(
                model_name=config.strong_model,
                lora_checkpoint=None,
                tokenizer=strong_tokenizer,
                dataset=train_unlabel_ds,
                max_new_tokens=config.gen_max_tokens,
                return_prompts=True,
                temperature=config.gen_temperature,
                top_p=config.gen_top_p,
                top_k=config.gen_top_k,
                min_p=config.gen_min_p,
                repetition_penalty=config.gen_repetition_penalty,
                presence_penalty=config.gen_presence_penalty,
            )
            critique_unlabel_gen_time = time.time() - start_time
            print(f"✓ Generated {len(critiques_unlabeled)} critique sets in {critique_unlabel_gen_time:.2f}s")
            
            # Save checkpoint with comparison schema
            step4_data = {
                'format': 'comparison',
                'entries': [
                    {
                        'question': item['prompt'],
                        'answer_a': item['first'],
                        'answer_b': item['second'],
                        'ground_truth': None,  # Unlabeled data has no ground truth
                        'critique_messages_a': msg_a,
                        'critique_messages_b': msg_b,
                        'critique_a': ca,
                        'critique_b': cb,
                    }
                    for item, (ca, cb), (msg_a, msg_b) in zip(train_unlabel_ds, critiques_unlabeled, prompts_unlabeled)
                ],
                'num_samples': num_samples,
                'generation_time': critique_unlabel_gen_time,
            }
        
        with open(step4_checkpoint, 'w') as f:
            json.dump(step4_data, f, indent=2)
        print(f"  ✓ Saved checkpoint: {step4_checkpoint}")
    
    # Print metrics
    print(f"\n📊 [STEP 4 METRICS]")
    print(f"  • Critiques generated (train_unlabeled): {len(critiques_unlabeled)}")
    print(f"  • Number of samples per prompt: {num_samples}")
    print(f"  • Generation time: {critique_unlabel_gen_time:.2f}s")
    
    # Print GPU memory status
    log_gpu_memory("[STEP 4 END] ")
    
    # Cleanup GPU memory
    print("\nCleaning up GPU memory after critique generation...")
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
    print("✓ GPU memory cleaned")

    return critiques_unlabeled, prompts_unlabeled, num_samples, critique_unlabel_gen_time

