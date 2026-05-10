"""
UE_zeroshot: Zero-shot Labeling + Consistency Fixing for Binary and Preference Tasks.

This implements a simplified version of the ICM algorithm:
1. Use zero-shot prompting to generate initial labels (the initialization step of ICM)
2. Apply consistency fixing to enforce logical constraints

No iterative refinement is performed - this is exactly the initialization step.

For preference/comparison tasks (e.g., helpfulness):
1. Augment dataset with flipped pairs (A>B and B>A share same consistency_id)
2. Zero-shot predict on augmented dataset
3. Apply consistency fixing using max probability
4. Filter out augmented pairs, keep only original pairs with generated labels
"""
import unsloth
import torch
import numpy as np
import gc
from typing import List, Dict, Tuple, Optional, Callable
from datasets import Dataset
from transformers import AutoTokenizer
from collections import Counter

from w2s_research.core import predict_batch_labels
from w2s_research.core.data import format_classification_as_causal
from w2s_research.ideas.ue_zeroshot.math_eval_tools import is_correct


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def detect_format(dataset: Dataset) -> str:
    """Detect if dataset is binary (question/choice) or preference (prompt/first/second)."""
    if len(dataset) == 0:
        raise ValueError("Empty dataset")
    sample = dataset[0]
    if "question" in sample and "choice" in sample:
        return "binary"
    elif "prompt" in sample and "first" in sample and "second" in sample:
        return "preference"
    else:
        raise ValueError(f"Unknown dataset format. Sample keys: {list(sample.keys())}")


def compute_label_accuracy(dataset: Dataset, ground_truth: Optional[Dataset] = None) -> Dict[str, float]:
    """Compute label accuracy against ground truth if available."""
    if ground_truth is None or len(dataset) != len(ground_truth):
        return {"accuracy": None, "correct": None, "total": None}

    correct = sum(1 for i in range(len(dataset)) if dataset[i]["label"] == ground_truth[i]["label"])
    total = len(dataset)
    return {"accuracy": correct / total if total > 0 else 0.0, "correct": correct, "total": total}


# =============================================================================
# DATASET PREPARATION (FORMAT-SPECIFIC)
# =============================================================================

def prepare_binary_dataset(dataset: Dataset) -> Tuple[Dataset, List[int]]:
    """Prepare binary dataset - no augmentation needed, just return as-is."""
    return dataset, list(range(len(dataset)))


def prepare_preference_dataset(dataset: Dataset) -> Tuple[Dataset, List[int]]:
    """
    Prepare preference dataset by augmenting with flipped pairs.
    Sets consistency_key to "A>B" and "B>A" for use in unified consistency logic.
    """
    augmented = []
    original_indices = []

    for i, example in enumerate(dataset):
        # Original pair
        orig = dict(example)
        orig["consistency_id"] = i
        orig["consistency_key"] = "A>B"
        orig["is_flipped"] = False
        augmented.append(orig)
        original_indices.append(len(augmented) - 1)

        # Flipped pair
        flipped = {
            "prompt": example["prompt"],
            "first": example["second"],
            "second": example["first"],
            "label": 1 - example["label"],
            "consistency_id": i,
            "consistency_key": "B>A",
            "is_flipped": True,
        }
        for key in example:
            if key not in flipped:
                flipped[key] = example[key]
        augmented.append(flipped)

    print(f"Augmented preference dataset: {len(dataset)} -> {len(augmented)} examples")
    return Dataset.from_list(augmented), original_indices


def build_consistency_groups(dataset: Dataset) -> List[List[int]]:
    """Build consistency groups based on consistency_id (works for both formats)."""
    cid_to_indices = {}
    for i, example in enumerate(dataset):
        cid = example.get("consistency_id", i)
        if cid not in cid_to_indices:
            cid_to_indices[cid] = []
        cid_to_indices[cid].append(i)

    groups = list(cid_to_indices.values())
    sizes = [len(g) for g in groups]
    print(f"Built consistency groups: {len(dataset)} examples in {len(groups)} groups")
    print(f"  Group size: min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.1f}")
    return groups


def group_by_consistency_key(dataset: Dataset, group: List[int]) -> Tuple[Dict[str, List[int]], bool]:
    """
    Group indices by consistency_key within a consistency group.

    For binary: groups by answer (via consistency_key or math equivalence)
    For preference: groups by "A>B" vs "B>A"

    Returns:
        key_to_indices: Dict mapping keys to list of indices
        use_math_equiv: Whether math equivalence should be used for key comparison
    """
    # Check if consistency_key exists
    if len(group) > 0 and "consistency_key" in dataset[group[0]]:
        key_to_indices = {}
        for idx in group:
            key = str(dataset[idx].get("consistency_key", idx))
            if key not in key_to_indices:
                key_to_indices[key] = []
            key_to_indices[key].append(idx)
        return key_to_indices, False
    else:
        # Fallback for math datasets: group by mathematical equivalence
        answer_groups = []
        for idx in group:
            answer = dataset[idx].get("answer", str(idx))
            found = False
            for rep, indices in answer_groups:
                if is_correct(answer, rep):
                    indices.append(idx)
                    found = True
                    break
            if not found:
                answer_groups.append((answer, [idx]))
        return {rep: indices for rep, indices in answer_groups}, True


def keys_match(key1: str, key2: str, use_math_equiv: bool) -> bool:
    """Check if two keys match (using math equivalence if needed)."""
    if use_math_equiv:
        return is_correct(key1, key2)
    return key1 == key2


def compute_consistency(dataset: Dataset, groups: List[List[int]]) -> Dict[str, float]:
    """
    Compute logical consistency (works for both formats).

    Consistency means: within each group, at most one key has label=1 (True).
    - For binary: at most one answer is True
    - For preference: exactly one of A>B or B>A is True (they must be opposite)
    """
    if not groups:
        return {"consistency": None, "consistent_groups": None, "total_groups": 0}

    consistent = 0
    for group in groups:
        if len(group) < 2:
            consistent += 1
            continue

        key_to_indices, _ = group_by_consistency_key(dataset, group)

        # Check: same key → same label
        is_consistent = True
        for indices in key_to_indices.values():
            if len(indices) > 1:
                labels = [dataset[idx]["label"] for idx in indices]
                if len(set(labels)) > 1:
                    is_consistent = False
                    break

        if not is_consistent:
            continue

        # Check: at most one key has True
        true_count = sum(1 for indices in key_to_indices.values() if dataset[indices[0]]["label"] == 1)
        if true_count > 1:
            is_consistent = False

        if is_consistent:
            consistent += 1

    return {
        "consistency": consistent / len(groups) if groups else 0.0,
        "consistent_groups": consistent,
        "total_groups": len(groups)
    }


def consistency_fix(
    dataset: Dataset,
    groups: List[List[int]],
    probabilities: List[List[float]],
) -> Dataset:
    """
    Apply consistency fixing using max probability.

    For each group:
    1. Group by consistency_key
    2. Find the key with highest P(True) across all examples
    3. Assign that key's examples to True, all others to False

    Args:
        dataset: Dataset with labels
        groups: Consistency groups
        probabilities: List of [P(False), P(True)] for each example
    """
    new_labels = list(dataset["label"])

    for group in groups:
        if len(group) < 2:
            continue

        key_to_indices, use_math_equiv = group_by_consistency_key(dataset, group)

        # For each key, find the max P(True) among its examples
        key_info = {}  # key -> (best_idx, max_p_true)
        for key, indices in key_to_indices.items():
            best_idx, max_p_true = max(
                [(idx, probabilities[idx][1]) for idx in indices],
                key=lambda x: x[1]
            )
            key_info[key] = (best_idx, max_p_true)

        # Pick the key with highest P(True) as winner
        winner_key = max(key_info.keys(), key=lambda k: key_info[k][1])

        # Assign labels: winner key gets True, others get False
        for key, indices in key_to_indices.items():
            label = 1 if keys_match(key, winner_key, use_math_equiv) else 0
            for idx in indices:
                new_labels[idx] = label

    return Dataset.from_list([{**dict(dataset[i]), "label": new_labels[i]} for i in range(len(dataset))])


def filter_to_original(dataset: Dataset, original_indices: List[int]) -> Dataset:
    """Filter dataset to keep only original indices (for preference: removes flipped pairs)."""
    filtered = []
    for idx in original_indices:
        example = dict(dataset[idx])
        # Remove augmentation metadata
        for key in ["consistency_id", "consistency_key", "is_flipped"]:
            example.pop(key, None)
        filtered.append(example)

    if len(filtered) != len(dataset):
        print(f"Filtered to original pairs: {len(dataset)} -> {len(filtered)} examples")
    return Dataset.from_list(filtered)


# =============================================================================
# ZERO-SHOT LABELING
# =============================================================================

def label_with_model_zeroshot(
    dataset: Dataset,
    model_name: str,
    tokenizer,
    max_ctx: int = 4096,
) -> Tuple[Dataset, List[List[float]]]:
    """Zero-shot labeling using vLLM (works for both binary and preference formats)."""
    # Create temp dataset with dummy labels
    temp_data = [{**dict(ex), "label": 0} for ex in dataset]
    temp_dataset = Dataset.from_list(temp_data)

    # Format for causal LM
    formatted, template, label_tokens = format_classification_as_causal(
        temp_dataset,
        tokenizer,
        max_ctx=max_ctx,
        zero_shot=True,
        use_chat_template=False,
    )

    label_token_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in label_tokens]

    # vLLM inference
    result = predict_batch_labels(
        model_name=model_name,
        formatted_dataset=formatted,
        label_token_ids=label_token_ids,
        tokenizer=tokenizer,
        lora_checkpoint=None,
        return_probabilities=True,
    )

    # Create labeled dataset
    labeled = [{**dict(dataset[i]), "label": result['predictions'][i]} for i in range(len(dataset))]

    del temp_data, temp_dataset, formatted
    torch.cuda.empty_cache()
    gc.collect()

    return Dataset.from_list(labeled), result['probabilities']


# =============================================================================
# UNIFIED UE_ZEROSHOT FUNCTION
# =============================================================================

def ue_zeroshot(
    unlabeled_dataset: Dataset,
    pretrained_model_name: str,
    max_ctx: int = 4096,
    ground_truth_dataset: Optional[Dataset] = None,
    return_confidences: bool = True,
) -> Tuple[Dataset, List[Dict], Optional[List[float]]]:
    """
    UE_zeroshot: Zero-shot labeling + consistency fixing.

    Automatically detects format (binary vs preference) and applies appropriate pipeline:
    - Binary: build consistency groups from consistency_id, apply consistency fixing
    - Preference: augment with flipped pairs, apply consistency fixing, filter augmented

    Args:
        unlabeled_dataset: Dataset (binary or preference format)
        pretrained_model_name: Name of pretrained model
        max_ctx: Maximum context length
        ground_truth_dataset: Optional ground truth for accuracy tracking
        return_confidences: If True, return confidence scores for each example

    Returns:
        labeled_dataset: Dataset with learned labels
        logs: List of dicts with metrics
        confidences: List of confidence scores (max probability) for each example (if return_confidences=True)
    """
    # Detect format and choose preparation function
    format_type = detect_format(unlabeled_dataset)
    prepare_fn = prepare_preference_dataset if format_type == "preference" else prepare_binary_dataset

    print("="*80)
    print(f"UE_ZEROSHOT: Zero-shot + Consistency Fixing ({format_type.upper()} format)")
    print("="*80)
    print(f"Model: {pretrained_model_name}")
    print(f"Unlabeled examples: {len(unlabeled_dataset)}")
    print("="*80)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    logs = []

    # Step 1: Prepare dataset (augment for preference, no-op for binary)
    print("\n[1/3] Preparing dataset...")
    working_dataset, original_indices = prepare_fn(unlabeled_dataset)
    consistency_groups = build_consistency_groups(working_dataset)

    # Prepare ground truth similarly for accuracy tracking
    working_gt = None
    if ground_truth_dataset is not None:
        working_gt, _ = prepare_fn(ground_truth_dataset)

    # Step 2: Zero-shot labeling
    print("\n[2/3] Zero-shot labeling...")
    labeled_dataset, probabilities = label_with_model_zeroshot(
        dataset=working_dataset,
        model_name=pretrained_model_name,
        tokenizer=tokenizer,
        max_ctx=max_ctx,
    )

    init_acc = compute_label_accuracy(labeled_dataset, working_gt)
    init_cons = compute_consistency(labeled_dataset, consistency_groups)

    if init_acc['accuracy'] is not None:
        print(f"  Zero-shot accuracy: {init_acc['accuracy']:.4f} ({init_acc['correct']}/{init_acc['total']})")
    print(f"  Zero-shot consistency: {init_cons['consistency']:.4f} ({init_cons['consistent_groups']}/{init_cons['total_groups']})")
    print(f"  Label distribution: {Counter([labeled_dataset[i]['label'] for i in range(len(labeled_dataset))])}")

    logs.append({
        "stage": "zeroshot_labeling",
        "accuracy": init_acc.get('accuracy'),
        "consistency": init_cons['consistency'],
        "consistent_groups": init_cons['consistent_groups'],
        "total_groups": init_cons['total_groups'],
    })

    # Step 3: Consistency fixing
    # Apply consistency fixing: assign the key with highest P(True) as True
    print("\n[3/3] Consistency fixing...")
    labeled_dataset = consistency_fix(labeled_dataset, consistency_groups, probabilities)

    final_acc = compute_label_accuracy(labeled_dataset, working_gt)
    final_cons = compute_consistency(labeled_dataset, consistency_groups)

    if final_acc['accuracy'] is not None:
        print(f"  Final accuracy: {final_acc['accuracy']:.4f} ({final_acc['correct']}/{final_acc['total']})")
    print(f"  Final consistency: {final_cons['consistency']:.4f} ({final_cons['consistent_groups']}/{final_cons['total_groups']})")
    print(f"  Label distribution: {Counter([labeled_dataset[i]['label'] for i in range(len(labeled_dataset))])}")

    logs.append({
        "stage": "consistency_fixed",
        "accuracy": final_acc.get('accuracy'),
        "consistency": final_cons['consistency'],
        "consistent_groups": final_cons['consistent_groups'],
        "total_groups": final_cons['total_groups'],
    })

    # Compute confidence scores (max probability for each example)
    # For the predicted label, confidence is P(label)
    confidences_all = []
    for i, prob in enumerate(probabilities):
        label = labeled_dataset[i]["label"]
        confidence = prob[label]  # Probability of the assigned label
        confidences_all.append(confidence)

    # Filter to original indices (for preference: removes flipped pairs)
    labeled_dataset = filter_to_original(labeled_dataset, original_indices)
    confidences = [confidences_all[i] for i in original_indices] if return_confidences else None

    if ground_truth_dataset is not None and format_type == "preference":
        orig_acc = compute_label_accuracy(labeled_dataset, ground_truth_dataset)
        print(f"  Original pairs accuracy: {orig_acc['accuracy']:.4f} ({orig_acc['correct']}/{orig_acc['total']})")
        logs.append({"stage": "final_original_only", **orig_acc})

    print("\n" + "="*80)
    print("UE_ZEROSHOT COMPLETE")
    print("="*80)
    print(f"Consistency: {init_cons['consistency']:.4f} -> {final_cons['consistency']:.4f}")
    print(f"Output: {len(labeled_dataset)} labeled examples")
    print("="*80)

    return labeled_dataset, logs, confidences


# =============================================================================
# DATA BALANCING BY CONFIDENCE
# =============================================================================

def _is_preference_format(dataset: Dataset) -> bool:
    """Check if dataset is in preference format (has first/second fields)."""
    if len(dataset) == 0:
        return False
    sample = dataset[0]
    return "first" in sample and "second" in sample


def _balance_preference_by_flipping(
    dataset: Dataset,
    confidences: List[float],
    imbalance_threshold: float = 1.0,
) -> Tuple[Dataset, List[float], Dict[str, any]]:
    """
    Balance preference dataset by flipping A/B order for majority examples.

    For preference datasets, instead of downsampling, we flip the order of first/second
    to flip the label. This preserves all data while achieving balance.

    Note: Confidence is not used for selection since flipping preserves all information -
    we just randomly select which examples to flip.

    Args:
        dataset: Preference dataset with first/second fields
        confidences: Confidence score for each example (not used for selection)
        imbalance_threshold: Threshold ratio for considering dataset imbalanced

    Returns:
        balanced_dataset: Balanced dataset (same size, some examples flipped)
        balanced_confidences: Updated confidences
        balance_info: Dict with balancing statistics
    """
    labels = [dataset[i]["label"] for i in range(len(dataset))]
    label_counts = Counter(labels)

    count_0 = label_counts.get(0, 0)
    count_1 = label_counts.get(1, 0)

    balance_info = {
        "original_count_0": count_0,
        "original_count_1": count_1,
        "original_total": len(dataset),
        "was_balanced": False,
        "method": "flip",
        "imbalance_ratio": max(count_0, count_1) / max(min(count_0, count_1), 1),
    }

    min_count = min(count_0, count_1)
    max_count = max(count_0, count_1)

    if min_count == 0:
        print(f"  [Balance] Warning: One class has 0 examples, cannot balance")
        balance_info["error"] = "one_class_empty"
        # Add original indices for consistency
        data_with_idx = [{**dict(dataset[i]), "_original_idx": i} for i in range(len(dataset))]
        return Dataset.from_list(data_with_idx), confidences, balance_info

    if max_count <= imbalance_threshold * min_count:
        print(f"  [Balance] Dataset is balanced (ratio {max_count/min_count:.2f} <= {imbalance_threshold})")
        # Add original indices for consistency
        data_with_idx = [{**dict(dataset[i]), "_original_idx": i} for i in range(len(dataset))]
        return Dataset.from_list(data_with_idx), confidences, balance_info

    print(f"  [Balance] Imbalanced preference dataset detected (ratio {max_count/min_count:.2f} > {imbalance_threshold})")
    print(f"    Class 0: {count_0}, Class 1: {count_1}")

    # Determine how many to flip to balance
    majority_label = 0 if count_0 > count_1 else 1
    num_to_flip = (max_count - min_count) // 2  # Flip half the difference to balance

    # Get majority class indices and randomly select which to flip
    majority_indices = [i for i in range(len(dataset)) if labels[i] == majority_label]
    # Use deterministic selection (first N) for reproducibility
    indices_to_flip = set(majority_indices[:num_to_flip])

    # Build balanced dataset
    balanced_data = []
    balanced_confidences = []

    for i in range(len(dataset)):
        example = dict(dataset[i])
        conf = confidences[i]

        if i in indices_to_flip:
            # Flip the order: swap first and second, flip label
            flipped = {
                **example,
                "first": example["second"],
                "second": example["first"],
                "label": 1 - example["label"],
                "_original_idx": i,
                "_was_flipped": True,
            }
            balanced_data.append(flipped)
            # Confidence for flipped example: 1 - original confidence
            balanced_confidences.append(1.0 - conf)
        else:
            example["_original_idx"] = i
            example["_was_flipped"] = False
            balanced_data.append(example)
            balanced_confidences.append(conf)

    balanced_dataset = Dataset.from_list(balanced_data)

    # Update balance info
    new_labels = [balanced_dataset[i]["label"] for i in range(len(balanced_dataset))]
    new_counts = Counter(new_labels)

    balance_info.update({
        "was_balanced": True,
        "majority_label": majority_label,
        "flipped_count": num_to_flip,
        "final_count_0": new_counts.get(0, 0),
        "final_count_1": new_counts.get(1, 0),
        "final_total": len(balanced_dataset),
        "final_imbalance_ratio": max(new_counts.values()) / max(min(new_counts.values()), 1),
    })

    print(f"    Flipped {num_to_flip} examples from class {majority_label}")
    print(f"    Final: Class 0: {balance_info['final_count_0']}, Class 1: {balance_info['final_count_1']}")

    return balanced_dataset, balanced_confidences, balance_info


def _balance_binary_by_downsampling(
    dataset: Dataset,
    confidences: List[float],
    imbalance_threshold: float = 1.1,
    balance_per_subset: bool = False,
) -> Tuple[Dataset, List[float], Dict[str, any]]:
    """
    Balance binary dataset by downsampling majority class based on confidence.

    Removes low-confidence examples from the majority class first.

    Args:
        dataset: Binary dataset with labels (0 or 1)
        confidences: Confidence score for each example
        imbalance_threshold: Threshold ratio for considering dataset imbalanced
        balance_per_subset: If True, balance each subset (subset + difficulty) separately

    Returns:
        balanced_dataset: Balanced dataset (smaller size)
        balanced_confidences: Confidences for balanced dataset
        balance_info: Dict with balancing statistics
    """
    # Compute subset keys for each example
    subset_keys = [dataset[i].get("subset", "unknown") + dataset[i].get("difficulty", "") for i in range(len(dataset))]
    original_subset_dist = Counter(subset_keys)

    labels = [dataset[i]["label"] for i in range(len(dataset))]
    label_counts = Counter(labels)

    count_0 = label_counts.get(0, 0)
    count_1 = label_counts.get(1, 0)

    balance_info = {
        "original_count_0": count_0,
        "original_count_1": count_1,
        "original_total": len(dataset),
        "was_balanced": False,
        "method": "confidence_downsample_per_subset" if balance_per_subset else "confidence_downsample",
        "imbalance_ratio": max(count_0, count_1) / max(min(count_0, count_1), 1),
        "original_subset_distribution": dict(original_subset_dist),
        "balance_per_subset": balance_per_subset,
    }

    # If balance_per_subset is enabled, process each subset separately
    if balance_per_subset:
        print(f"  [Balance] Balancing per subset (balance_per_subset=True)")

        # Group indices by subset
        subset_to_indices = {}
        for i, sk in enumerate(subset_keys):
            if sk not in subset_to_indices:
                subset_to_indices[sk] = []
            subset_to_indices[sk].append(i)

        all_keep_indices = []
        subset_balance_info = {}

        for subset_name in sorted(subset_to_indices.keys()):
            indices = subset_to_indices[subset_name]
            subset_labels = [labels[i] for i in indices]
            subset_label_counts = Counter(subset_labels)

            s_count_0 = subset_label_counts.get(0, 0)
            s_count_1 = subset_label_counts.get(1, 0)
            s_min_count = min(s_count_0, s_count_1)
            s_max_count = max(s_count_0, s_count_1)

            print(f"    Subset '{subset_name}': Class 0={s_count_0}, Class 1={s_count_1}")

            if s_min_count == 0:
                print(f"      -> One class empty, keeping all {len(indices)} examples")
                all_keep_indices.extend(indices)
                subset_balance_info[subset_name] = {
                    "original_count_0": s_count_0,
                    "original_count_1": s_count_1,
                    "final_count_0": s_count_0,
                    "final_count_1": s_count_1,
                    "removed": 0,
                    "error": "one_class_empty",
                }
                continue

            if s_max_count <= imbalance_threshold * s_min_count:
                print(f"      -> Already balanced (ratio {s_max_count/s_min_count:.2f}), keeping all {len(indices)} examples")
                all_keep_indices.extend(indices)
                subset_balance_info[subset_name] = {
                    "original_count_0": s_count_0,
                    "original_count_1": s_count_1,
                    "final_count_0": s_count_0,
                    "final_count_1": s_count_1,
                    "removed": 0,
                }
                continue

            # Identify majority/minority for this subset
            s_majority_label = 0 if s_count_0 > s_count_1 else 1
            s_target_count = int(s_min_count * imbalance_threshold)

            s_majority_indices = [i for i in indices if labels[i] == s_majority_label]
            s_minority_indices = [i for i in indices if labels[i] != s_majority_label]

            # Sort majority by confidence (ascending) and keep top s_target_count
            s_majority_with_conf = [(i, confidences[i]) for i in s_majority_indices]
            s_majority_with_conf.sort(key=lambda x: x[1])
            s_keep_majority = [i for i, _ in s_majority_with_conf[-s_target_count:]]

            s_keep_indices = s_minority_indices + s_keep_majority
            all_keep_indices.extend(s_keep_indices)

            s_removed = len(s_majority_indices) - s_target_count
            s_final_0 = sum(1 for i in s_keep_indices if labels[i] == 0)
            s_final_1 = sum(1 for i in s_keep_indices if labels[i] == 1)

            print(f"      -> Removed {s_removed} low-confidence class {s_majority_label} examples")
            print(f"      -> Final: Class 0={s_final_0}, Class 1={s_final_1}")

            subset_balance_info[subset_name] = {
                "original_count_0": s_count_0,
                "original_count_1": s_count_1,
                "final_count_0": s_final_0,
                "final_count_1": s_final_1,
                "removed": s_removed,
                "majority_label": s_majority_label,
            }

        # Build balanced dataset from all kept indices
        keep_indices = sorted(all_keep_indices)
        balanced_data = []
        for i in keep_indices:
            example = dict(dataset[i])
            example["_original_idx"] = i
            balanced_data.append(example)
        balanced_confidences = [confidences[i] for i in keep_indices]
        balanced_dataset = Dataset.from_list(balanced_data)

        # Update balance info
        new_labels = [balanced_dataset[i]["label"] for i in range(len(balanced_dataset))]
        new_counts = Counter(new_labels)
        final_subset_dist = Counter([balanced_dataset[i].get("subset", "unknown") + balanced_dataset[i].get("difficulty", "") for i in range(len(balanced_dataset))])

        total_removed = len(dataset) - len(balanced_dataset)
        balance_info.update({
            "was_balanced": total_removed > 0,
            "removed_count": total_removed,
            "final_count_0": new_counts.get(0, 0),
            "final_count_1": new_counts.get(1, 0),
            "final_total": len(balanced_dataset),
            "final_imbalance_ratio": max(new_counts.values()) / max(min(new_counts.values()), 1) if min(new_counts.values()) > 0 else float('inf'),
            "final_subset_distribution": dict(final_subset_dist),
            "per_subset_balance_info": subset_balance_info,
        })

        print(f"  [Balance] Per-subset balancing complete:")
        print(f"    Total: {len(dataset)} -> {len(balanced_dataset)} (removed {total_removed})")
        print(f"    Final: Class 0={balance_info['final_count_0']}, Class 1={balance_info['final_count_1']}")

        return balanced_dataset, balanced_confidences, balance_info

    # Original global balancing logic
    min_count = min(count_0, count_1)
    max_count = max(count_0, count_1)

    if min_count == 0:
        print(f"  [Balance] Warning: One class has 0 examples, cannot balance")
        balance_info["error"] = "one_class_empty"
        balance_info["final_subset_distribution"] = dict(original_subset_dist)
        # Add original indices for consistency
        data_with_idx = [{**dict(dataset[i]), "_original_idx": i} for i in range(len(dataset))]
        return Dataset.from_list(data_with_idx), confidences, balance_info

    if max_count <= imbalance_threshold * min_count:
        print(f"  [Balance] Dataset is balanced (ratio {max_count/min_count:.2f} <= {imbalance_threshold})")
        balance_info["final_subset_distribution"] = dict(original_subset_dist)
        # Add original indices for consistency
        data_with_idx = [{**dict(dataset[i]), "_original_idx": i} for i in range(len(dataset))]
        return Dataset.from_list(data_with_idx), confidences, balance_info

    print(f"  [Balance] Imbalanced binary dataset detected (ratio {max_count/min_count:.2f} > {imbalance_threshold})")
    print(f"    Class 0: {count_0}, Class 1: {count_1}")
    print(f"    Subset distribution before balancing: {dict(original_subset_dist)}")

    # Identify majority and minority classes
    majority_label = 0 if count_0 > count_1 else 1
    # Downsample majority to within threshold (keep more data than exact balance)
    target_count = int(min_count * imbalance_threshold)

    # Separate indices by label
    majority_indices = [i for i in range(len(dataset)) if labels[i] == majority_label]
    minority_indices = [i for i in range(len(dataset)) if labels[i] != majority_label]

    # Sort majority indices by confidence (ascending) - low confidence first
    majority_with_conf = [(i, confidences[i]) for i in majority_indices]
    majority_with_conf.sort(key=lambda x: x[1])  # Sort by confidence ascending

    # Keep top `target_count` high-confidence examples from majority class
    keep_majority_indices = [i for i, _ in majority_with_conf[-target_count:]]

    # Combine minority (keep all) + downsampled majority
    keep_indices = sorted(minority_indices + keep_majority_indices)

    # Build balanced dataset (preserve original index for accuracy computation)
    balanced_data = []
    for i in keep_indices:
        example = dict(dataset[i])
        example["_original_idx"] = i
        balanced_data.append(example)
    balanced_confidences = [confidences[i] for i in keep_indices]
    balanced_dataset = Dataset.from_list(balanced_data)

    # Update balance info
    new_labels = [balanced_dataset[i]["label"] for i in range(len(balanced_dataset))]
    new_counts = Counter(new_labels)

    # Compute final subset distribution after balancing (combining subset + difficulty)
    final_subset_dist = Counter([balanced_dataset[i].get("subset", "unknown") + balanced_dataset[i].get("difficulty", "") for i in range(len(balanced_dataset))])

    removed_count = len(majority_indices) - target_count
    balance_info.update({
        "was_balanced": True,
        "majority_label": majority_label,
        "removed_count": removed_count,
        "final_count_0": new_counts.get(0, 0),
        "final_count_1": new_counts.get(1, 0),
        "final_total": len(balanced_dataset),
        "final_imbalance_ratio": max(new_counts.values()) / max(min(new_counts.values()), 1),
        "final_subset_distribution": dict(final_subset_dist),
    })

    print(f"    Removed {removed_count} low-confidence examples from class {majority_label}")
    print(f"    Final: Class 0: {balance_info['final_count_0']}, Class 1: {balance_info['final_count_1']}")
    print(f"    Subset distribution after balancing: {dict(final_subset_dist)}")

    # Log subset-level filtering impact
    print(f"    Subset filtering summary:")
    for subset in sorted(set(original_subset_dist.keys()) | set(final_subset_dist.keys())):
        orig = original_subset_dist.get(subset, 0)
        final = final_subset_dist.get(subset, 0)
        removed = orig - final
        pct_removed = (removed / orig * 100) if orig > 0 else 0
        print(f"      {subset}: {orig} -> {final} (removed {removed}, {pct_removed:.1f}%)")

    return balanced_dataset, balanced_confidences, balance_info


def balance_by_confidence(
    dataset: Dataset,
    confidences: List[float],
    imbalance_threshold: float = 1.1,
    balance_per_subset: bool = False,
) -> Tuple[Dataset, List[float], Dict[str, any]]:
    """
    Balance dataset by confidence-based filtering/flipping.

    For preference datasets (with first/second fields):
        - Flips the order of A/B for low-confidence majority examples
        - Preserves all data while achieving balance

    For binary datasets:
        - Downsamples majority class by removing low-confidence examples
        - Reduces dataset size

    Args:
        dataset: Dataset with labels (0 or 1)
        confidences: Confidence score for each example (higher = more confident)
        imbalance_threshold: Threshold ratio for considering dataset imbalanced (default 1.1)
        balance_per_subset: If True, balance each subset (subset + difficulty) separately
                           (only applies to binary datasets)

    Returns:
        balanced_dataset: Balanced dataset
        balanced_confidences: Confidences for balanced dataset
        balance_info: Dict with balancing statistics
    """
    if _is_preference_format(dataset):
        return _balance_preference_by_flipping(dataset, confidences, imbalance_threshold)
    else:
        return _balance_binary_by_downsampling(dataset, confidences, imbalance_threshold, balance_per_subset)
