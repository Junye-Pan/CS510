"""
UE_fewshot: Few-shot Labeling + Consistency Fixing for Binary and Comparison Tasks.

This implements a simplified version of the ICM algorithm using few-shot demonstrations:
1. For each example, sample random examples (excluding itself) as demonstrations
2. Assign balanced random labels to demonstrations
3. Use these as in-context demonstrations for labeling
4. Apply consistency fixing to enforce logical constraints

Supports both:
- Binary tasks (question/choice format with True/False labels)
- Comparison tasks (prompt/first/second format with True/False labels for A>B claim)
"""
import unsloth
import torch
import numpy as np
import gc
import random
from typing import List, Dict, Tuple, Optional
from datasets import Dataset
from transformers import AutoTokenizer
from collections import Counter

from w2s_research.core import predict_batch_labels
from w2s_research.core.data import (
    BINARY_ZERO_SHOT_LABEL_TOKENS,
    BINARY_ZERO_SHOT_TEMPLATE,
    COMPARISON_ZERO_SHOT_TEMPLATE,
    COMPARISON_ZERO_SHOT_LABEL_TOKENS,
    format_classification_as_causal,
)

# Import shared functions from ue_zeroshot
from w2s_research.ideas.ue_zeroshot.experiment import (
    detect_format,
    build_consistency_groups,
    compute_consistency,
    consistency_fix,
    prepare_binary_dataset,
    prepare_preference_dataset,
    filter_to_original,
    compute_label_accuracy,
)
from w2s_research.ideas.ue_zeroshot.math_eval_tools import is_correct


# =============================================================================
# FEW-SHOT FORMATTING FUNCTIONS
# =============================================================================

def format_fewshot_demonstrations(
    demo_examples: List[Dict],
    demo_labels: List[int],
    format_type: str,
) -> str:
    """
    Format few-shot demonstration examples.

    Args:
        demo_examples: List of example dicts
        demo_labels: List of labels for each demo
        format_type: "binary" or "preference"

    Returns:
        Formatted string with demonstrations
    """
    if format_type == "binary":
        template = BINARY_ZERO_SHOT_TEMPLATE
        label_tokens = BINARY_ZERO_SHOT_LABEL_TOKENS
        demos = []
        for example, label in zip(demo_examples, demo_labels):
            demo_str = template.format(
                question=example['question'],
                choice=example['choice']
            ) + label_tokens[label]
            demos.append(demo_str)
    else:  # preference/comparison
        template = COMPARISON_ZERO_SHOT_TEMPLATE
        label_tokens = COMPARISON_ZERO_SHOT_LABEL_TOKENS
        demos = []
        for example, label in zip(demo_examples, demo_labels):
            demo_str = template.format(
                prompt=example['prompt'],
                first=example['first'],
                second=example['second']
            ) + label_tokens[label]
            demos.append(demo_str)

    return "\n\n".join(demos) + "\n\n"


def format_example_text(example: Dict, format_type: str) -> str:
    """Format a single example's input text (without label)."""
    if format_type == "binary":
        return BINARY_ZERO_SHOT_TEMPLATE.format(
            question=example["question"],
            choice=example["choice"]
        )
    else:  # preference
        return COMPARISON_ZERO_SHOT_TEMPLATE.format(
            prompt=example["prompt"],
            first=example["first"],
            second=example["second"]
        )


def get_label_tokens(format_type: str) -> List[str]:
    """Get label tokens for the given format."""
    if format_type == "binary":
        return BINARY_ZERO_SHOT_LABEL_TOKENS
    else:
        return COMPARISON_ZERO_SHOT_LABEL_TOKENS


def format_classification_as_causal_fewshot(
    dataset: Dataset,
    tokenizer,
    all_examples: Dataset,
    format_type: str,
    num_demos: int = 4,
    max_ctx: int = 131072,
    seed: int = 42,
    consistency_groups: Optional[List[List[int]]] = None,
) -> Tuple[Dataset, str, List[str]]:
    """
    Format classification dataset with few-shot demonstrations.

    For each example, sample `num_demos` random examples (excluding itself
    and examples from the same consistency group) and use their labels.

    Args:
        dataset: Dataset with label field
        tokenizer: Tokenizer for encoding
        all_examples: Full dataset to sample demonstrations from
        format_type: "binary" or "preference"
        num_demos: Number of demonstration examples (default: 4)
        max_ctx: Maximum context length
        seed: Random seed for reproducibility
        consistency_groups: Optional list of index lists for excluding same-group demos

    Returns:
        Formatted dataset, template string, label tokens
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    label_tokens = get_label_tokens(format_type)
    n_total = len(all_examples)

    # Build index-to-group mapping for efficient lookup
    index_to_group_indices = {}
    if consistency_groups:
        for group in consistency_groups:
            group_set = set(group)
            for idx in group:
                index_to_group_indices[idx] = group_set

    # Check if labels are available
    has_labels = all_examples[0].get("label") is not None and all_examples[0].get("label") != -1

    # Pre-compute indices by label for balanced sampling
    if has_labels:
        indices_by_label = {0: [], 1: []}
        for i in range(n_total):
            label = all_examples[i]["label"]
            if label in indices_by_label:
                indices_by_label[label].append(i)

    def base_format_example(example, idx):
        # Get indices to exclude: current index + same consistency group
        excluded_indices = {idx}
        if idx in index_to_group_indices:
            excluded_indices = index_to_group_indices[idx]

        if has_labels:
            # Sample balanced demonstrations from predicted labels
            n_per_class = num_demos // 2
            demo_indices = []
            for label in [0, 1]:
                available = [i for i in indices_by_label[label] if i not in excluded_indices]
                sampled = rng.sample(available, min(n_per_class, len(available)))
                demo_indices.extend(sampled)
            rng.shuffle(demo_indices)
            demo_examples = [all_examples[i] for i in demo_indices]
            demo_labels = [all_examples[i]["label"] for i in demo_indices]
        else:
            # No labels yet - sample random examples with balanced random labels
            available_indices = [i for i in range(n_total) if i not in excluded_indices]
            demo_indices = rng.sample(available_indices, min(num_demos, len(available_indices)))
            demo_examples = [all_examples[i] for i in demo_indices]
            n_true = num_demos // 2
            n_false = num_demos - n_true
            demo_labels = [1] * n_true + [0] * n_false
            np_rng.shuffle(demo_labels)

        # Format demonstrations
        demonstrations = format_fewshot_demonstrations(demo_examples, demo_labels, format_type)

        # Get the input text
        input_text = demonstrations + format_example_text(example, format_type)

        # Get hard label
        hard_label = example["label"]
        if isinstance(hard_label, bool):
            hard_label = 1 if hard_label else 0

        if 'soft_label' in example:
            hard_label = example['soft_label'].index(max(example['soft_label']))

        label_text = label_tokens[hard_label]

        # Get special tokens
        bos_id = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
        eos_id = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

        # Tokenize label first (we MUST keep this)
        label_ids = tokenizer.encode(label_text, add_special_tokens=False)

        # Calculate available space for prompt
        reserved_tokens = len(bos_id) + len(label_ids) + len(eos_id)
        available_for_input = max_ctx - reserved_tokens

        # Tokenize input text and truncate if needed
        input_ids = tokenizer.encode(input_text, add_special_tokens=False)

        if len(input_ids) > available_for_input:
            input_ids = input_ids[:available_for_input]

        # Build full sequence: [BOS] + input + label + [EOS]
        full_ids = bos_id + input_ids + label_ids + eos_id

        # Create labels for loss computation
        prompt_length = len(bos_id) + len(input_ids)
        labels = [-100] * prompt_length + label_ids + eos_id

        result = {
            "input_ids": full_ids,
            "labels": labels,
            "prompt_length": prompt_length,
        }

        if 'soft_label' in example:
            result["soft_labels"] = example["soft_label"]
        return result

    # Apply formatting with index tracking
    formatted_data = []
    for idx, example in enumerate(dataset):
        formatted_data.append(base_format_example(example, idx))

    formatted = Dataset.from_list(formatted_data)

    return formatted, format_example_text.__doc__, label_tokens


# =============================================================================
# LABELING FUNCTIONS
# =============================================================================

def label_with_model_vllm_fewshot(
    dataset: Dataset,
    model_name: str,
    lora_checkpoint: Optional[str],
    tokenizer,
    all_examples: Dataset,
    format_type: str,
    num_demos: int = 4,
    max_ctx: int = 131072,
    seed: int = 42,
    consistency_groups: Optional[List[List[int]]] = None,
) -> Tuple[Dataset, List[List[float]]]:
    """
    Label dataset using vLLM with few-shot demonstrations.

    Args:
        dataset: Unlabeled dataset
        model_name: Base model name
        lora_checkpoint: Path to LoRA checkpoint directory (or None for base model)
        tokenizer: Tokenizer
        all_examples: Full dataset to sample demonstrations from
        format_type: "binary" or "preference"
        num_demos: Number of demonstration examples
        max_ctx: Maximum context length
        seed: Random seed
        consistency_groups: Optional list of index lists for excluding same-group demos

    Returns:
        Tuple of (labeled_dataset, probabilities)
    """
    # Create temporary dataset with dummy labels for formatting
    temp_data = []
    for example in dataset:
        temp_example = dict(example)
        temp_example["label"] = 0  # Dummy label
        temp_data.append(temp_example)
    temp_dataset = Dataset.from_list(temp_data)

    # Format dataset with few-shot demonstrations
    formatted_dataset, _, label_tokens = format_classification_as_causal_fewshot(
        temp_dataset,
        tokenizer,
        all_examples=all_examples,
        format_type=format_type,
        num_demos=num_demos,
        max_ctx=max_ctx,
        seed=seed,
        consistency_groups=consistency_groups,
    )

    label_tokens = get_label_tokens(format_type)
    label_token_ids = [tokenizer.encode(label_token, add_special_tokens=False)[0] for label_token in label_tokens]

    # Use vLLM for fast inference
    result = predict_batch_labels(
        model_name=model_name,
        formatted_dataset=formatted_dataset,
        label_token_ids=label_token_ids,
        tokenizer=tokenizer,
        lora_checkpoint=lora_checkpoint,
        return_probabilities=True,
        max_model_len=max_ctx,
    )
    predictions = result['predictions']
    probabilities = result['probabilities']

    # Create labeled dataset
    labeled_data = []
    for i, example in enumerate(dataset):
        labeled_example = dict(example)
        labeled_example["label"] = predictions[i]
        labeled_data.append(labeled_example)

    result_dataset = Dataset.from_list(labeled_data)

    # Cleanup
    del predictions, labeled_data, temp_data, temp_dataset, formatted_dataset
    torch.cuda.empty_cache()
    gc.collect()

    return result_dataset, probabilities


def label_with_model_vllm_zeroshot(
    dataset: Dataset,
    model_name: str,
    lora_checkpoint: Optional[str],
    tokenizer,
    format_type: str,
    max_ctx: int = 131072,
) -> Tuple[Dataset, List[List[float]]]:
    """
    Label dataset using vLLM with zero-shot prompting.

    Args:
        dataset: Unlabeled dataset
        model_name: Base model name
        lora_checkpoint: Path to LoRA checkpoint directory (or None for base model)
        tokenizer: Tokenizer
        format_type: "binary" or "preference"
        max_ctx: Maximum context length

    Returns:
        Tuple of (labeled_dataset, probabilities)
    """
    # Create temporary dataset with dummy labels for formatting
    temp_data = []
    for example in dataset:
        temp_example = dict(example)
        temp_example["label"] = 0  # Dummy label
        temp_data.append(temp_example)
    temp_dataset = Dataset.from_list(temp_data)

    # Format dataset for causal LM (zero-shot)
    formatted_dataset, template, label_tokens = format_classification_as_causal(
        temp_dataset,
        tokenizer,
        max_ctx=max_ctx,
        zero_shot=True,
        use_chat_template=False,
    )

    label_token_ids = [tokenizer.encode(label_token, add_special_tokens=False)[0] for label_token in label_tokens]

    # Use vLLM for fast inference
    result = predict_batch_labels(
        model_name=model_name,
        formatted_dataset=formatted_dataset,
        label_token_ids=label_token_ids,
        tokenizer=tokenizer,
        lora_checkpoint=lora_checkpoint,
        return_probabilities=True,
        max_model_len=max_ctx,
    )
    predictions = result['predictions']
    probabilities = result['probabilities']

    # Create labeled dataset
    labeled_data = []
    for i, example in enumerate(dataset):
        labeled_example = dict(example)
        labeled_example["label"] = predictions[i]
        labeled_data.append(labeled_example)

    result_dataset = Dataset.from_list(labeled_data)

    # Cleanup
    del predictions, labeled_data, temp_data, temp_dataset, formatted_dataset
    torch.cuda.empty_cache()
    gc.collect()

    return result_dataset, probabilities


# =============================================================================
# MAIN UE_FEWSHOT FUNCTION
# =============================================================================

def ue_fewshot(
    unlabeled_dataset: Dataset,
    pretrained_model_name: str,
    num_demos: List[int] = [2, 4, 16],
    max_ctx: int = 131072,
    seed: int = 42,
    num_iterations: Optional[int] = None,
    cache_dir: Optional[str] = None,
    ground_truth_dataset: Optional[Dataset] = None,
) -> Tuple[Dataset, List[Dict]]:
    """
    UE_fewshot: Iterative few-shot labeling + consistency fixing.

    Automatically detects format (binary vs preference) and applies appropriate pipeline:
    - Binary: build consistency groups from consistency_id
    - Preference: augment with flipped pairs, apply consistency fixing, filter augmented

    At each iteration:
    1. Generate labels:
       - Iteration 0: Zero-shot prompting (initialization)
       - Iteration i (i > 0): Few-shot prompting with num_demos[i-1] demonstrations
    2. Apply consistency fixing with max probability

    Args:
        unlabeled_dataset: Dataset (binary or preference format)
        pretrained_model_name: Name of pretrained model
        num_demos: List of demo counts for each iteration after zero-shot
        max_ctx: Maximum context length
        seed: Random seed
        num_iterations: Number of iterations (default: len(num_demos) + 1)
        cache_dir: Directory to cache iteration results
        ground_truth_dataset: Optional ground truth for accuracy tracking

    Returns:
        labeled_dataset: Dataset with learned labels
        logs: List of dicts with metrics
    """
    import json
    import os

    # Detect format and choose preparation function
    format_type = detect_format(unlabeled_dataset)
    prepare_fn = prepare_preference_dataset if format_type == "preference" else prepare_binary_dataset

    # Compute num_iterations from num_demos if not provided
    if num_iterations is None:
        num_iterations = len(num_demos) + 1  # +1 for zero-shot

    print("="*80)
    print(f"UE_FEWSHOT: Zero-shot Init + Iterative Few-shot Labeling ({format_type.upper()} format)")
    print("="*80)
    print(f"Model: {pretrained_model_name}")
    print(f"Unlabeled examples: {len(unlabeled_dataset)}")
    print(f"Demo schedule: zero-shot -> {' -> '.join(f'{n}-shot' for n in num_demos)}")
    print(f"Number of iterations: {num_iterations} (zero-shot + {len(num_demos)} few-shot)")
    print(f"Seed: {seed}")
    print("="*80)

    # Load tokenizer
    print(f"  Loading tokenizer from: {pretrained_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    logs = []

    # Step 1: Prepare dataset (augment for preference, no-op for binary)
    print("\n[0/N] Preparing dataset...")
    working_dataset, original_indices = prepare_fn(unlabeled_dataset)
    consistency_groups = build_consistency_groups(working_dataset)

    # Prepare ground truth similarly for accuracy tracking
    working_gt = None
    if ground_truth_dataset is not None:
        working_gt, _ = prepare_fn(ground_truth_dataset)

    # Initialize dataset
    current_dataset = working_dataset

    # Setup cache directory if provided
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    # Check for cached iterations to resume from
    start_iteration = 0
    if cache_dir:
        for i in range(num_iterations - 1, -1, -1):
            cache_file = os.path.join(cache_dir, f"iteration_{i + 1}.json")
            if os.path.exists(cache_file):
                print(f"\n>>> Found cached iteration {i + 1}, loading...")
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                current_dataset = Dataset.from_list(cached['dataset'])
                logs = cached['logs']
                start_iteration = i + 1
                print(f">>> Resuming from iteration {start_iteration + 1}")
                break

    for iteration in range(start_iteration, num_iterations):
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration + 1}/{num_iterations}")
        if iteration > 0:
            print('Initial label distribution:', Counter([current_dataset[i]['label'] for i in range(len(current_dataset))]))
        print(f"{'='*80}")

        # Step 1: Labeling (zero-shot for iteration 0, few-shot for subsequent iterations)
        if iteration == 0:
            print(f"\n[{iteration + 1}.1] Zero-shot labeling (initialization)...")
            labeled_dataset, probabilities = label_with_model_vllm_zeroshot(
                dataset=current_dataset,
                model_name=pretrained_model_name,
                lora_checkpoint=None,
                tokenizer=tokenizer,
                format_type=format_type,
                max_ctx=max_ctx,
            )
        else:
            # Get demo count for this iteration (iteration 1 uses num_demos[0], etc.)
            current_num_demos = num_demos[iteration - 1]
            print(f"\n[{iteration + 1}.1] {current_num_demos}-shot labeling...")
            labeled_dataset, probabilities = label_with_model_vllm_fewshot(
                dataset=current_dataset,
                model_name=pretrained_model_name,
                lora_checkpoint=None,
                tokenizer=tokenizer,
                all_examples=current_dataset,  # Uses current labels for demos
                format_type=format_type,
                num_demos=current_num_demos,
                max_ctx=max_ctx,
                seed=seed + iteration,
                consistency_groups=consistency_groups,
            )

        # Compute metrics
        label_acc = compute_label_accuracy(labeled_dataset, working_gt)
        label_cons = compute_consistency(labeled_dataset, consistency_groups)

        if label_acc.get('accuracy') is not None:
            print(f"  Accuracy after labeling: {label_acc['accuracy']:.4f}")
        if label_cons.get("consistency") is not None:
            print(f"  Consistency after labeling: {label_cons['consistency']:.4f}")
        print(f"  Label distribution: {Counter([labeled_dataset[i]['label'] for i in range(len(labeled_dataset))])}")

        logs.append({
            "stage": f"iteration_{iteration + 1}_labeling",
            "iteration": iteration + 1,
            "num_demos": 0 if iteration == 0 else num_demos[iteration - 1],
            "accuracy": label_acc.get('accuracy'),
            "consistency": label_cons.get('consistency'),
            "consistent_groups": label_cons.get('consistent_groups'),
            "total_groups": label_cons.get('total_groups'),
        })

        # Step 2: Consistency fixing
        print(f"\n[{iteration + 1}.2] Consistency fixing...")
        labeled_dataset = consistency_fix(labeled_dataset, consistency_groups, probabilities)

        # Compute metrics after fixing
        fix_acc = compute_label_accuracy(labeled_dataset, working_gt)
        fix_cons = compute_consistency(labeled_dataset, consistency_groups)

        if fix_acc.get('accuracy') is not None:
            print(f"  Accuracy after fixing: {fix_acc['accuracy']:.4f}")
        if fix_cons.get("consistency") is not None:
            print(f"  Consistency after fixing: {fix_cons['consistency']:.4f}")
        print(f"  Label distribution: {Counter([labeled_dataset[i]['label'] for i in range(len(labeled_dataset))])}")

        logs.append({
            "stage": f"iteration_{iteration + 1}_fixed",
            "iteration": iteration + 1,
            "num_demos": 0 if iteration == 0 else num_demos[iteration - 1],
            "accuracy": fix_acc.get('accuracy'),
            "consistency": fix_cons.get('consistency'),
            "consistent_groups": fix_cons.get('consistent_groups'),
            "total_groups": fix_cons.get('total_groups'),
        })

        # Update current_dataset for next iteration
        current_dataset = labeled_dataset

        # Cache iteration results
        if cache_dir:
            cache_file = os.path.join(cache_dir, f"iteration_{iteration + 1}.json")
            cache_data = {
                'dataset': [dict(ex) for ex in labeled_dataset],
                'logs': logs,
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
            print(f"  >>> Cached iteration {iteration + 1} to {cache_file}")

    # Filter to original indices (for preference: removes flipped pairs)
    labeled_dataset = filter_to_original(labeled_dataset, original_indices)

    # Get first and last iteration metrics for summary
    first_cons = logs[0].get('consistency') if logs else None
    final_cons = logs[-1].get('consistency') if logs else None

    print("\n" + "="*80)
    print("UE_FEWSHOT COMPLETE")
    print("="*80)
    if first_cons is not None and final_cons is not None:
        print(f"Consistency: {first_cons:.4f} -> {final_cons:.4f} (delta {final_cons - first_cons:+.4f})")
    print(f"Output: {len(labeled_dataset)} labeled examples")
    print("="*80)

    return labeled_dataset, logs


