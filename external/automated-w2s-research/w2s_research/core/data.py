import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import Dataset as HfDataset
from transformers import AutoTokenizer
from datasets import Dataset
import numpy as np
from jinja2 import Template

import wandb
WANDB_AVAILABLE = True


COMPARISON_TEMPLATE = "Question: {prompt}\nAnswer A: {first}\nAnswer B: {second}\nLabel:"
COMPARISON_LABEL_TOKENS = ["A", "B"]  # No leading space - chat template adds \n after assistant

COMPARISON_ZERO_SHOT_TEMPLATE = "Question: {prompt}\nAnswer A: {first}\nAnswer B: {second}\nClaim: Answer A is more helpful than Answer B.\nI think this Claim is (True or False):"
COMPARISON_ZERO_SHOT_LABEL_TOKENS = [" True", " False"]
# Judge prompt template: formats question, options, and critiques for judge training
# V1: Simple version - prone to reward hacking (critic learns to over-criticize)
JUDGE_PROMPT_VERSION = "v1"  # Update this when changing JUDGE_PROMPT_TEMPLATE
JUDGE_PROMPT_TEMPLATE_PATH = Path(__file__).parent.parent / "ideas" / "critic" / "judge_prompt.jinja2"
with open(JUDGE_PROMPT_TEMPLATE_PATH, "r") as f:
    JUDGE_PROMPT_TEMPLATE = Template(f.read())

# Binary judge prompt template: for binary format (question/choice/critique -> True/False)
BINARY_JUDGE_PROMPT_TEMPLATE_PATH = Path(__file__).parent.parent / "ideas" / "critic" / "binary_judge_prompt.jinja2"
with open(BINARY_JUDGE_PROMPT_TEMPLATE_PATH, "r") as f:
    BINARY_JUDGE_PROMPT_TEMPLATE = Template(f.read())

# New format: T/F prediction for single choice
BINARY_TEMPLATE = "Question: {question}\nAnswer: {choice}\nLabel:"
BINARY_LABEL_TOKENS = ["False", "True"]  # No leading space - chat template adds \n after assistant

BINARY_ZERO_SHOT_TEMPLATE = "Question: {question}\nProposed Answer: {choice}\nClaim: This answer is correct.\nI think this claim is (True or False):"
BINARY_ZERO_SHOT_LABEL_TOKENS = ["False", "True"]  # No leading space - chat template adds \n after assistant

def load_jsonl(file_path: str, require_labels: bool = True) -> List[Dict[str, Any]]:
    """
    Load JSONL file into list of dictionaries.
    
    Args:
        file_path: Path to the JSONL file
        require_labels: If True (default), raises error if labels are missing.
                       If False, allows missing labels (sets label to -1 for AAR mode).
    
    Returns:
        List of dictionaries with normalized labels (0/1 for present, -1 for missing)
    """
    raw_data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            raw_data.append(item)
    
    if not raw_data:
        return []
    
    # Check if this is the new binary format (has "question" and "choice" fields)
    has_question = "question" in raw_data[0]
    has_choice = "choice" in raw_data[0]
    is_binary_format = has_question and has_choice
    
    # Check if labels are present
    has_labels = "label" in raw_data[0]
    
    if not has_labels and require_labels:
        raise ValueError(
            f"Labels are missing from {file_path} but require_labels=True. "
            f"For AAR mode (worker pods), use require_labels=False."
        )
    
    if is_binary_format:
        # New binary format: question/choice with boolean label
        # Normalize field names and convert boolean to 0/1
        data = []
        for item in raw_data:
            if "label" in item:
                # Convert boolean label to 0/1
                # False (incorrect) -> 0 (False token), True (correct) -> 1 (True token)
                if isinstance(item.get("label"), bool):
                    item['label'] = 1 if item["label"] is True else 0
            else:
                # AAR mode: no labels available, use -1 as sentinel
                item['label'] = -1
            data.append(item)
        return data
    
    else:
        # Old format: prompt/first/second with A/B label
        data = []
        for item in raw_data:
            if "label" in item:
                # Convert A/B string labels to 0/1
                if isinstance(item.get("label"), str):
                    item['label'] = 0 if item["label"] == "A" else 1
            else:
                # AAR mode: no labels available, use -1 as sentinel
                item['label'] = -1
            data.append(item)
        return data


def detect_aar_mode(data_dir: str) -> bool:
    """
    Detect if we're running in AAR (Automated Alignment Research) mode.
    
    AAR mode means ground truth labels are NOT available:
    - train_label.jsonl does not exist
    - train_unlabel.jsonl has label=-1 for all samples
    - test.jsonl has label=-1 for all samples
    
    AAR mode is enabled if:
    1. AAR_MODE environment variable is set to "true", OR
    2. train_unlabel.jsonl has no "label" field or label=-1
    
    Args:
        data_dir: Path to data directory containing the JSONL files
        
    Returns:
        True if in AAR mode (no ground truth available), False otherwise
        
    Example:
        >>> aar_mode = detect_aar_mode(config.data_dir)
        >>> if aar_mode:
        ...     print("AAR mode - will return predictions instead of accuracy")
        ...     # DO NOT access dataset["label"] - it contains -1 placeholders
    """
    import os
    import json
    from pathlib import Path
    
    # Check environment variable first
    if os.getenv("AAR_MODE", "false").lower() == "true":
        return True
    
    # Auto-detect by checking if labels exist in train_unlabel.jsonl
    train_unlabel_path = Path(data_dir) / "train_unlabel.jsonl"
    if train_unlabel_path.exists():
        with open(train_unlabel_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                first_item = json.loads(first_line)
                # No label field or label is -1 means AAR mode
                if "label" not in first_item or first_item.get("label") == -1:
                    return True
    return False


def load_dataset(
    data_dir: str,
    train_label_file: str = "train_label.jsonl",
    train_unlabel_file: str = "train_unlabel.jsonl",
    test_file: str = "test.jsonl",
    seed: int = 42,
    ue: bool = False,
    aar_mode: bool = False,
) -> Dict[str, HfDataset]:
    """
    Load and format dataset.
    
    Args:
        data_dir: Path to data directory
        train_label_file: Filename for labeled training data
        train_unlabel_file: Filename for unlabeled training data
        test_file: Filename for test data
        seed: Random seed (unused, kept for compatibility)
        ue: Whether loading for unsupervised elicitation (unused, kept for compatibility)
        aar_mode: If True, allows missing labels in train_unlabel and test files.
                 train_label file will be skipped entirely in AAR mode.
                 Labels will be set to -1 for missing values.
    
    Returns:
        Dictionary with 'train_label', 'train_unlabel', 'test' datasets.
        In aar_mode, 'train_label' will be None.
    """
    data_path = Path(data_dir)

    # In AAR mode, skip train_label entirely (workers shouldn't have access)
    if aar_mode:
        print("[AAR Mode] Skipping train_label.jsonl - workers should not have access to labeled data")
        train_label_data = []
    else:
        train_label_path = data_path / train_label_file
        if train_label_path.exists():
            train_label_data = load_jsonl(train_label_path, require_labels=True)
        else:
            print(f"[Warning] {train_label_file} not found at {train_label_path}")
            train_label_data = []
    
    # Load train_unlabel - labels optional in AAR mode
    train_unlabel_data = load_jsonl(
        data_path / train_unlabel_file, 
        require_labels=not aar_mode
    )
    
    # Load test - labels optional in AAR mode
    test_data = load_jsonl(
        data_path / test_file, 
        require_labels=not aar_mode
    )

    # Convert to HF datasets
    datasets = {
        "train_label": HfDataset.from_list(train_label_data) if train_label_data else None,
        "train_unlabel": HfDataset.from_list(train_unlabel_data),
        "test": HfDataset.from_list(test_data),
    }

    return datasets


def format_classification_as_causal(
    dataset: Dataset,
    tokenizer,
    max_ctx: int = 512,
    zero_shot: bool = False,
    use_judge_template: bool = False,
    dataset_name: str = "unknown",  # For WandB logging context
    use_chat_template: bool = True,  # Whether to use chat template (False for base models)
) -> Dataset:
    """
    Format dataset for causal LM classification training.
    
    Args:
        dataset: Dataset with 'prompt', 'first', 'second', 'label' fields
                (for judge template: also needs 'critique_a', 'critique_b' fields)
                OR 'question', 'choice', 'label' fields (binary format)
        tokenizer: Tokenizer to use
        max_ctx: Maximum context length (must be 8192)
        zero_shot: Whether to use zero-shot template
        use_judge_template: Whether to use JUDGE_PROMPT_TEMPLATE (formats from question, options, critiques)
        use_chat_template: Whether to use chat template formatting. Set to False for base models
                          to avoid loss spikes from chat template formatting.
    
    Returns:
        Formatted dataset with 'input_ids', 'labels', 'prompt_length' fields
    """
    # Allow max_ctx up to 10240 (8192 + 1024*2) for longer contexts (e.g., judge prompts with critiques)
    assert max_ctx <= 10240, f"max_ctx must be <= 10240, got {max_ctx}"

    # Detect format: check if dataset has "question" and "choice" (binary format)
    # or "prompt", "first", "second" (A/B format)
    is_binary_format = False
    if len(dataset) > 0:
        sample = dataset[0]
        is_binary_format = "question" in sample and "choice" in sample

    if use_judge_template:
        if is_binary_format:
            # Binary judge format: True/False prediction with critique
            template = BINARY_JUDGE_PROMPT_TEMPLATE  # Jinja2 Template object
            label_tokens = BINARY_LABEL_TOKENS  # True/False labels
        else:
            # A/B judge format
            template = JUDGE_PROMPT_TEMPLATE  # Jinja2 Template object
            label_tokens = COMPARISON_LABEL_TOKENS  # A/B labels
    elif is_binary_format:
        # New binary format: True/False prediction
        if zero_shot:
            template = BINARY_ZERO_SHOT_TEMPLATE
            label_tokens = BINARY_ZERO_SHOT_LABEL_TOKENS
        else:
            template = BINARY_TEMPLATE
            label_tokens = BINARY_LABEL_TOKENS
    elif zero_shot:
        template = COMPARISON_ZERO_SHOT_TEMPLATE
        label_tokens = COMPARISON_ZERO_SHOT_LABEL_TOKENS
    else:
        template = COMPARISON_TEMPLATE
        label_tokens = COMPARISON_LABEL_TOKENS
        
    def base_format_example(example):
        # Get the input text (should end with "Label:")
        if is_binary_format:
            input_text = template.format(
                question=example["question"],
                choice=example["choice"]
            )
        else:
            input_text = template.format(
                prompt=example["prompt"],
                first=example["first"],
                second=example["second"]
            )

        # Get hard label (0 or 1 for binary classification)
        hard_label = example["label"]
        if 'soft_label' in example:
            hard_label = example['soft_label'].index(max(example['soft_label']))

        label_text = label_tokens[hard_label]

        # Get special tokens
        bos_id = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
        eos_id = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

        # Tokenize label first (we MUST keep this)
        label_ids = tokenizer.encode(label_text, add_special_tokens=False)

        # Tokenize "Label:" separately to ensure it's always preserved
        input_text_without_label = input_text.rstrip()
        if input_text_without_label.endswith("Label:"):
            input_text_without_label = input_text_without_label[:-6]  # Remove "Label:
            label_prompt = "Label:"
            label_prompt_ids = tokenizer.encode(label_prompt, add_special_tokens=False)
        else:
            label_prompt = ""
            label_prompt_ids = []
        # Calculate available space for prompt (excluding "Label:" which we'll add back)
        reserved_tokens = len(bos_id) + len(label_prompt_ids) + len(label_ids) + len(eos_id)
        available_for_input = max_ctx - reserved_tokens

        # Remove "Label:" from input text for separate handling      
        # Tokenize input text (without "Label:") and truncate if needed
        input_ids = tokenizer.encode(input_text_without_label, add_special_tokens=False)

        if len(input_ids) > available_for_input:
            input_ids = input_ids[:available_for_input]

        # Build full sequence: [BOS] + input + "Label:" + label + [EOS]
        # This ensures "Label:" is ALWAYS present, even after truncation
        full_ids = bos_id + input_ids + label_prompt_ids + label_ids + eos_id

        # Create labels for loss computation
        prompt_length = len(bos_id) + len(input_ids) + len(label_prompt_ids)
        labels = [-100] * prompt_length + label_ids + eos_id

        result = {
            "input_ids": full_ids,
            "labels": labels,
            "prompt_length": prompt_length,
        }

        # Preserve soft labels if present
        if 'soft_label' in example:
            result["soft_labels"] = example["soft_label"]
        # Preserve sample index if present (for curriculum learning)
        if 'sample_idx' in example:
            result["sample_idx"] = example["sample_idx"]
        return result
    
    
    def format_example(example):
        # Get hard label (0 or 1 for binary classification)
        hard_label = example["label"]
        if 'soft_label' in example:
            hard_label = example['soft_label'].index(max(example['soft_label']))

        label_text = label_tokens[hard_label]

        # Construct the user message content
        if use_judge_template:
            if is_binary_format:
                # Binary judge template: format from question, choice, and critique
                # Dataset should have: question, choice, critique, label
                user_content = template.render(
                    question=example["question"],
                    choice=example["choice"],
                    critique=example.get("critique", ""),
                )
            else:
                # A/B Judge template: format from question, options, and critiques
                # Dataset should have: prompt (question), first (option_a), second (option_b), 
                # critique_a, critique_b, label
                user_content = template.render(
                    question=example["prompt"],
                    option_a=example["first"],
                    option_b=example["second"],
                    critique_a=example.get("critique_a", ""),
                    critique_b=example.get("critique_b", ""),
                )
        elif is_binary_format:
            # Binary format: question/choice
            user_content = template.format(
                question=example["question"],
                choice=example["choice"]
            )
        else:
            # Standard template: format prompt, first, second
            user_content = template.format(
                prompt=example["prompt"],
                first=example["first"],
                second=example["second"]
            )
        # Prepare messages for chat template
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": label_text}
        ]
        prompt_messages = [{"role": "user", "content": user_content}]
        
        # Apply chat template
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Post-process: Remove <think> tags that Qwen3 tokenizer auto-adds
        # The tokenizer adds exactly "<think>\n\n</think>\n\n" before assistant content
        # Use exact string replacement to preserve any leading spaces in labels
        formatted_text = formatted_text.replace('<think>\n\n</think>\n\n', '')
        
        # Now tokenize the formatted text
        full_encoding = tokenizer(
            formatted_text,
            add_special_tokens=False,  # Chat template already adds them
        )
        full_ids = full_encoding["input_ids"]
        
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        label_ids = tokenizer.encode(label_text + formatted_text[len(prompt_text):].split(label_text)[-1], add_special_tokens=False)
        
        prompt_length = len(full_ids) - len(label_ids) # A/B<|im_end|>
        
        # Filter out samples exceeding max_ctx instead of truncating
        # (truncation can corrupt code samples)
        if len(full_ids) > max_ctx:
            return None  # Will be filtered out
        
        # Create labels for loss computation
        # -100 means ignore in loss (standard PyTorch convention)
        # Only compute loss on the assistant's response
        labels = [-100] * prompt_length + full_ids[prompt_length:]
        result = {
            "input_ids": full_ids,
            "labels": labels,
            "prompt_length": prompt_length,
        }

        # Preserve soft labels if present
        if 'soft_label' in example:
            result["soft_labels"] = example["soft_label"]
        # Preserve sample index if present (for curriculum learning)
        if 'sample_idx' in example:
            result["sample_idx"] = example["sample_idx"]

        return result

    # Choose formatting function based on use_chat_template
    if use_chat_template:
        format_fn = format_example
        format_desc = "Formatting for causal LM with chat template"
    else:
        format_fn = base_format_example
        format_desc = "Formatting for causal LM (base model, no chat template)"
        print("📝 Using base format (no chat template) for base model training")
    
    # Track filtering statistics
    filter_stats = {
        "total_samples": 0,
        "filtered_samples": 0,
        "filtered_lengths": [],
    }
    
    def format_example_with_filtering(example):
        filter_stats["total_samples"] += 1
        result = format_fn(example)
        
        if result is None:
            # Sample exceeded max_ctx
            filter_stats["filtered_samples"] += 1
            # We don't have the exact length here, but we know it exceeded max_ctx
            return None
        
        return result
    
    # Apply formatting and filter out None results
    original_size = len(dataset)
    formatted = dataset.map(
        format_example_with_filtering,
        remove_columns=dataset.column_names,
        desc=format_desc,
    )
    
    # Filter out None entries (samples that exceeded max_ctx)
    formatted = formatted.filter(lambda x: x["input_ids"] is not None)
    filtered_count = original_size - len(formatted)
    
    if filtered_count > 0:
        print(f"⚠️  Filtered {filtered_count}/{original_size} samples exceeding max_ctx={max_ctx} ({100*filtered_count/original_size:.1f}%)")
    else:
        print(f"✓ All {original_size} samples fit within max_ctx={max_ctx}")
    

    print(formatted[0])
    print(tokenizer.decode(formatted[0]['input_ids']))
    return formatted, template, label_tokens
