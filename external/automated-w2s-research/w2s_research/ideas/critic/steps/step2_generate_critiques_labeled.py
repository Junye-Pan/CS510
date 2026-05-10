"""
Step 2: Generate critiques for train_labeled and test set with strong model.

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


def generate_critiques_for_labeled_and_test(
    config: RunConfig,
    train_label_ds: Dataset,
    test_ds: Dataset,
    strong_tokenizer: AutoTokenizer,
    checkpoint_dir: Path,
) -> Tuple[Union[List[Tuple[str, str]], List[str]], Union[List[Tuple[str, str]], List[str]], float]:
    """
    Step 2: Generate critiques for train_labeled AND test set with strong model.
    
    Supports both formats:
    - Comparison (A/B): Returns List[Tuple[str, str]] - (critique_a, critique_b) per sample
    - Binary: Returns List[str] - one critique per sample
    
    Args:
        config: Run configuration
        train_label_ds: Training labeled dataset
        test_ds: Test dataset
        strong_tokenizer: Tokenizer for strong model
        checkpoint_dir: Directory for checkpoints
        
    Returns:
        Tuple of (critiques_labeled, critiques_test, generation_time)
    """
    print("\n[2/11] Generating critiques for train_labeled and test set...")
    
    # Detect format
    is_binary = detect_binary_format(train_label_ds)
    if is_binary:
        print("  → Detected binary format (question/choice)")
        step2_checkpoint = checkpoint_dir / "step2_critiques_labeled_binary.json"
    else:
        print("  → Detected comparison format (prompt/first/second)")
        step2_checkpoint = checkpoint_dir / "step2_critiques_labeled.json"
    
    critique_gen_time = 0.0
    
    if step2_checkpoint.exists():
        print(f"  → Loading from checkpoint: {step2_checkpoint}")
        with open(step2_checkpoint, 'r') as f:
            step2_data = json.load(f)
        
        if is_binary:
            # Binary format: single critique per sample
            critiques_labeled = [item['critique'] for item in step2_data['critiques_labeled']]
            critiques_test = [item['critique'] for item in step2_data['critiques_test']]
        else:
            # Comparison format: (critique_a, critique_b) per sample
            critiques_labeled = []
            critiques_test = []
            for item in step2_data['critiques_labeled']:
                if 'critique_a' in item:
                    critiques_labeled.append((item['critique_a'], item['critique_b']))
                else:
                    critiques_labeled.append((item.get('critique_a', ''), item.get('critique_b', '')))
            for item in step2_data['critiques_test']:
                if 'critique_a' in item:
                    critiques_test.append((item['critique_a'], item['critique_b']))
                else:
                    critiques_test.append((item.get('critique_a', ''), item.get('critique_b', '')))
        
        critique_gen_time = step2_data.get('generation_time', 0.0)
        print(f"  ✓ Loaded {len(critiques_labeled)} train critiques and {len(critiques_test)} test critiques from checkpoint")
    else:
        start_time = time.time()
        
        if is_binary:
            # Binary format: use binary critique generation
            print("  → Generating critiques for train_labeled (binary)...")
            critiques_labeled, messages_labeled = generate_critiques_vllm_binary(
                model_name=config.strong_model,
                lora_checkpoint=None,
                tokenizer=strong_tokenizer,
                dataset=train_label_ds,
                max_new_tokens=config.gen_max_tokens,
                return_prompts=True,
                temperature=config.gen_temperature,
                top_p=config.gen_top_p,
                top_k=config.gen_top_k,
                min_p=config.gen_min_p,
                repetition_penalty=config.gen_repetition_penalty,
                presence_penalty=config.gen_presence_penalty,
            )

            print("  → Generating critiques for test set (binary)...")
            critiques_test, messages_test = generate_critiques_vllm_binary(
                model_name=config.strong_model,
                lora_checkpoint=None,
                tokenizer=strong_tokenizer,
                dataset=test_ds,
                max_new_tokens=config.gen_max_tokens,
                return_prompts=True,
                temperature=config.gen_temperature,
                top_p=config.gen_top_p,
                top_k=config.gen_top_k,
                min_p=config.gen_min_p,
                repetition_penalty=config.gen_repetition_penalty,
                presence_penalty=config.gen_presence_penalty,
            )

            critique_gen_time = time.time() - start_time
            print(f"✓ Generated critiques in {critique_gen_time:.2f}s")
            
            # Save checkpoint with binary schema
            step2_data = {
                'format': 'binary',
                'critiques_labeled': [
                    {
                        'question': item['question'],
                        'choice': item['choice'],
                        'label': item['label'],
                        'critique_messages': msg,
                        'critique': c,
                    }
                    for item, c, msg in zip(train_label_ds, critiques_labeled, messages_labeled)
                ],
                'critiques_test': [
                    {
                        'question': item['question'],
                        'choice': item['choice'],
                        'label': item['label'],
                        'critique_messages': msg,
                        'critique': c,
                    }
                    for item, c, msg in zip(test_ds, critiques_test, messages_test)
                ],
                'generation_time': critique_gen_time,
            }
        else:
            # Comparison format: use original critique generation
            print("  → Generating critiques for train_labeled...")
            critiques_labeled, messages_labeled = generate_critiques_vllm(
                model_name=config.strong_model,
                lora_checkpoint=None,
                tokenizer=strong_tokenizer,
                dataset=train_label_ds,
                max_new_tokens=config.gen_max_tokens,
                return_prompts=True,
                temperature=config.gen_temperature,
                top_p=config.gen_top_p,
                top_k=config.gen_top_k,
                min_p=config.gen_min_p,
                repetition_penalty=config.gen_repetition_penalty,
                presence_penalty=config.gen_presence_penalty,
            )

            print("  → Generating critiques for test set...")
            critiques_test, messages_test = generate_critiques_vllm(
                model_name=config.strong_model,
                lora_checkpoint=None,
                tokenizer=strong_tokenizer,
                dataset=test_ds,
                max_new_tokens=config.gen_max_tokens,
                return_prompts=True,
                temperature=config.gen_temperature,
                top_p=config.gen_top_p,
                top_k=config.gen_top_k,
                min_p=config.gen_min_p,
                repetition_penalty=config.gen_repetition_penalty,
                presence_penalty=config.gen_presence_penalty,
            )

            critique_gen_time = time.time() - start_time
            print(f"✓ Generated critiques in {critique_gen_time:.2f}s")
            
            # Save checkpoint with comparison schema
            step2_data = {
                'format': 'comparison',
                'critiques_labeled': [
                    {
                        'question': item['prompt'],
                        'answer_a': item['first'],
                        'answer_b': item['second'],
                        'ground_truth': item['label'],
                        'critique_messages_a': msg_a,
                        'critique_messages_b': msg_b,
                        'critique_a': ca,
                        'critique_b': cb,
                    }
                    for item, (ca, cb), (msg_a, msg_b) in zip(train_label_ds, critiques_labeled, messages_labeled)
                ],
                'critiques_test': [
                    {
                        'question': item['prompt'],
                        'answer_a': item['first'],
                        'answer_b': item['second'],
                        'ground_truth': item['label'],
                        'critique_messages_a': msg_a,
                        'critique_messages_b': msg_b,
                        'critique_a': ca,
                        'critique_b': cb,
                    }
                    for item, (ca, cb), (msg_a, msg_b) in zip(test_ds, critiques_test, messages_test)
                ],
                'generation_time': critique_gen_time,
            }
        
        with open(step2_checkpoint, 'w') as f:
            json.dump(step2_data, f, indent=2)
        print(f"  ✓ Saved checkpoint: {step2_checkpoint}")
    
    # Print metrics
    print(f"\n📊 [STEP 2 METRICS]")
    print(f"  • Critiques generated (train_labeled): {len(critiques_labeled)}")
    print(f"  • Critiques generated (test): {len(critiques_test)}")
    print(f"  • Generation time: {critique_gen_time:.2f}s")
    
    # Print GPU memory status
    log_gpu_memory("[STEP 2 END] ")
    
    # Cleanup GPU memory
    print("\nCleaning up GPU memory after critique generation...")
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
    print("✓ GPU memory cleaned")

    return critiques_labeled, critiques_test, critique_gen_time

