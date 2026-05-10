"""
Evaluation functions for causal LM and sequence classification models.

Supports two modes:
1. Standard mode: Requires ground truth labels, computes accuracy metrics
2. AAR mode: Returns predictions only, no ground truth required
"""
import json
import torch
import numpy as np
from collections import Counter
from pathlib import Path
from typing import Dict, Any, Optional, List
from datasets import Dataset
from .vllm_inference import predict_batch_labels



def evaluate_model(
    test_dataset: Dataset,
    label_token_ids: list,
    tokenizer,
    model_name: str,
    lora_checkpoint: Optional[str] = None,
    original_dataset: Optional[Dataset] = None,
    max_model_len: Optional[int] = None,
    gpu_memory_utilization: float = 0.8,
) -> Dict[str, Any]:
    """
    Evaluate model on test dataset with detailed metrics using vLLM.

    Args:
        test_dataset: Test dataset (formatted for causal LM with input_ids/labels)
        label_token_ids: List of token IDs for label tokens (e.g., [token_id_A, token_id_B])
        tokenizer: Tokenizer
        model_name: Base model name (e.g., "unsloth/Qwen2.5-0.5B-Instruct")
        lora_checkpoint: Path to LoRA checkpoint (optional, for evaluation with LoRA adapters)
        original_dataset: Optional original dataset with 'label' field (before formatting).
                         If provided, true labels are extracted from here. Otherwise,
                         decoded from test_dataset's token IDs.
        gpu_memory_utilization: Fraction of GPU memory to use for vLLM (default: 0.8)

    Returns:
        Dictionary with evaluation metrics including:
        - accuracy: Overall accuracy
        - pred_distribution: Distribution of predicted labels
        - label_distribution: Distribution of true labels
        - per_subset: Per-subset metrics (if 'subset' key exists in dataset)
    """
    if test_dataset is None:
        return {
            "accuracy": 0.0,
            "pred_distribution": {},
            "label_distribution": {},
            "per_subset": {},
        }

    # Check if dataset has 'subset' key
    has_subset = len(original_dataset) > 0 and 'subset' in original_dataset[0]

    result = predict_batch_labels(
        model_name=model_name,
        formatted_dataset=test_dataset,
        label_token_ids=label_token_ids,
        tokenizer=tokenizer,
        lora_checkpoint=lora_checkpoint,
        return_probabilities=False,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    predictions = result['predictions']

    # Extract true labels from original dataset if available (more efficient)
    true_labels = [original_dataset[i]["label"] for i in range(len(original_dataset))]
    subset_names = [original_dataset[i]["subset"] for i in range(len(original_dataset))] if has_subset else []
      # Compute overall metrics
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    correct = (predictions == true_labels).sum()
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0.0

    # Compute distributions (convert numpy types to Python int for JSON serialization)
    pred_distribution = {int(k): int(v) for k, v in Counter(predictions).items()}
    label_distribution = {int(k): int(v) for k, v in Counter(true_labels).items()}

    results = {
        "accuracy": accuracy,
        "correct": int(correct),
        "total": int(total),
        "pred_distribution": pred_distribution,
        "label_distribution": label_distribution,
    }

    # Compute per-subset metrics if subset information is available
    if has_subset and len(subset_names) > 0:
        subset_names_array = np.array(subset_names)
        unique_subsets = np.unique(subset_names_array)

        per_subset = {}
        for subset_name in unique_subsets:
            mask = (subset_names_array == subset_name)
            subset_preds = predictions[mask]
            subset_labels = true_labels[mask]

            subset_correct = (subset_preds == subset_labels).sum()
            subset_total = len(subset_preds)
            subset_accuracy = subset_correct / subset_total if subset_total > 0 else 0.0

            # Subset distributions (convert numpy types to Python int for JSON serialization)
            subset_pred_dist = {int(k): int(v) for k, v in Counter(subset_preds).items()}
            subset_label_dist = {int(k): int(v) for k, v in Counter(subset_labels).items()}

            per_subset[subset_name] = {
                "accuracy": subset_accuracy,
                "correct": int(subset_correct),
                "total": subset_total,
                "pred_distribution": subset_pred_dist,
                "label_distribution": subset_label_dist,
            }

        results["per_subset"] = per_subset

    return results


def generate_predictions(
    test_dataset: Dataset,
    label_token_ids: list,
    tokenizer,
    model_name: str,
    lora_checkpoint: Optional[str] = None,
    max_model_len: Optional[int] = None,
    gpu_memory_utilization: float = 0.8,
) -> Dict[str, Any]:
    """
    Generate predictions on test dataset WITHOUT requiring ground truth labels.
    
    This is the AAR-compliant evaluation function for worker pods. It returns
    only predictions, which can be uploaded to S3 for server-side evaluation.

    Args:
        test_dataset: Test dataset (formatted for causal LM with input_ids/labels)
        label_token_ids: List of token IDs for label tokens (e.g., [token_id_A, token_id_B])
        tokenizer: Tokenizer
        model_name: Base model name (e.g., "unsloth/Qwen2.5-0.5B-Instruct")
        lora_checkpoint: Path to LoRA checkpoint (optional, for evaluation with LoRA adapters)
        max_model_len: Maximum model context length for vLLM
        gpu_memory_utilization: Fraction of GPU memory to use for vLLM (default: 0.8)

    Returns:
        Dictionary with:
        - predictions: List of predicted labels (0 or 1)
        - num_samples: Number of samples evaluated
        - pred_distribution: Distribution of predicted labels
    """
    if test_dataset is None or len(test_dataset) == 0:
        return {
            "predictions": [],
            "num_samples": 0,
            "pred_distribution": {},
        }

    result = predict_batch_labels(
        model_name=model_name,
        formatted_dataset=test_dataset,
        label_token_ids=label_token_ids,
        tokenizer=tokenizer,
        lora_checkpoint=lora_checkpoint,
        return_probabilities=False,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    predictions = result['predictions']
    
    # Convert numpy types to Python int for JSON serialization
    predictions_list = [int(p) for p in predictions]
    pred_distribution = {int(k): int(v) for k, v in Counter(predictions).items()}

    return {
        "predictions": predictions_list,
        "num_samples": len(predictions_list),
        "pred_distribution": pred_distribution,
    }


def save_predictions(
    predictions: List[int],
    output_path: str,
    model_name: str,
    lora_checkpoint: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save predictions to a JSON file for AAR server-side evaluation.
    
    Args:
        predictions: List of predicted labels (0 or 1)
        output_path: Path to save predictions JSON file
        model_name: Model name used for predictions
        lora_checkpoint: LoRA checkpoint path (if used)
        extra_metadata: Additional metadata to include
        
    Returns:
        Path to saved predictions file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Build predictions document
    predictions_doc = {
        "predictions": predictions,
        "num_samples": len(predictions),
        "pred_distribution": {int(k): int(v) for k, v in Counter(predictions).items()},
        "model_name": model_name,
        "lora_checkpoint": lora_checkpoint,
    }
    
    # Add extra metadata if provided
    if extra_metadata:
        predictions_doc["metadata"] = extra_metadata
    
    with open(output_file, 'w') as f:
        json.dump(predictions_doc, f, indent=2)
    
    print(f"✓ Saved predictions to {output_file}")
    return str(output_file)


def print_evaluation_results(results: Dict[str, Any], prefix: str = ""):
    """Pretty print evaluation results."""
    print(f"\n{prefix}{'='*80}")
    print(f"{prefix}Evaluation Results")
    print(f"{prefix}{'='*80}")
    print(f"{prefix}Overall Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")

    print(f"\n{prefix}Prediction Distribution:")
    total = results['total']
    for label, count in results['pred_distribution'].items():
        pct = count / total if total > 0 else 0.0
        print(f"{prefix}  label_{label}: {count} ({pct:.2%})")

    print(f"\n{prefix}True Label Distribution:")
    for label, count in results['label_distribution'].items():
        pct = count / total if total > 0 else 0.0
        print(f"{prefix}  label_{label}: {count} ({pct:.2%})")

    if 'per_subset' in results:
        print(f"\n{prefix}Per-Subset Metrics:")
        for subset_name, subset_results in results['per_subset'].items():
            print(f"\n{prefix}  Subset '{subset_name}':")
            print(f"{prefix}    Accuracy: {subset_results['accuracy']:.4f} ({subset_results['correct']}/{subset_results['total']})")
            print(f"{prefix}    Pred Distribution: {subset_results['pred_distribution']}")
            print(f"{prefix}    Label Distribution: {subset_results['label_distribution']}")

    print(f"{prefix}{'='*80}\n")


def compute_metrics_from_predictions(
    predictions: List[int],
    ground_truth: List[int],
    fixed_weak_acc: Optional[float] = None,
    fixed_strong_acc: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute evaluation metrics from predictions and ground truth labels.
    
    This is the SERVER-SIDE function for AAR. It takes predictions from
    worker pods and computes metrics using locally stored ground truth.
    
    Args:
        predictions: List of predicted labels (0 or 1) from worker pod
        ground_truth: List of ground truth labels (0 or 1) from server storage
        fixed_weak_acc: Fixed weak baseline accuracy (for PGR computation)
        fixed_strong_acc: Fixed strong/ceiling baseline accuracy (for PGR computation)
        
    Returns:
        Dictionary with:
        - transfer_acc: Transfer accuracy (predictions vs ground truth)
        - pgr: Performance Gap Recovery (if baselines provided)
        - correct: Number of correct predictions
        - total: Total number of predictions
        - pred_distribution: Distribution of predictions
        - label_distribution: Distribution of ground truth labels
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"Predictions ({len(predictions)}) and ground truth ({len(ground_truth)}) "
            f"must have the same length"
        )
    
    if len(predictions) == 0:
        return {
            "transfer_acc": 0.0,
            "pgr": None,
            "correct": 0,
            "total": 0,
            "pred_distribution": {},
            "label_distribution": {},
        }
    
    # Convert to numpy for efficient computation
    predictions_arr = np.array(predictions)
    ground_truth_arr = np.array(ground_truth)
    
    # Compute accuracy
    correct = int((predictions_arr == ground_truth_arr).sum())
    total = len(predictions)
    transfer_acc = correct / total
    
    # Compute PGR if baselines are provided
    pgr = None
    if fixed_weak_acc is not None and fixed_strong_acc is not None:
        if fixed_strong_acc > fixed_weak_acc:
            pgr = (transfer_acc - fixed_weak_acc) / (fixed_strong_acc - fixed_weak_acc)
        else:
            # Edge case: strong not better than weak (shouldn't happen normally)
            pgr = None
    
    # Compute distributions
    pred_distribution = {int(k): int(v) for k, v in Counter(predictions).items()}
    label_distribution = {int(k): int(v) for k, v in Counter(ground_truth).items()}
    
    return {
        "transfer_acc": transfer_acc,
        "pgr": pgr,
        "correct": correct,
        "total": total,
        "pred_distribution": pred_distribution,
        "label_distribution": label_distribution,
        "fixed_weak_acc": fixed_weak_acc,
        "fixed_strong_acc": fixed_strong_acc,
    }
