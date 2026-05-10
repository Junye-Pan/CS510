#!/usr/bin/env python3
"""
Evaluate critiques by comparing them against reference critiques from Claude 3.5 Haiku.

This script:
1. Samples test set examples
2. Generates reference critiques using Claude 3.5 Haiku
3. Generates critiques from a given checkpoint/model
4. Uses Claude Haiku 4.5 to compare quality (judge)
5. Calculates win-tie-rate and other metrics
6. Saves results
7. Optionally generates summary markdown from results

Usage:
    # Evaluate critiques and generate summary automatically
    python -m w2s_research.ideas.critic.evaluate_critiques --data-dir <path> --model <model> --output <output.json> --generate-summary

    # Generate summary from existing results
    python -m w2s_research.ideas.critic.evaluate_critiques --generate-summary-only --output-dir <dir>
"""

import unsloth
import argparse
import hashlib
import time
import json
import os
import random
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    Anthropic = None
    ANTHROPIC_AVAILABLE = False

from datasets import Dataset
from jinja2 import Template
from transformers import AutoTokenizer

from w2s_research.core import load_dataset
from w2s_research.ideas.critic.utils import (
    generate_critiques_vllm,
    generate_critique_prompt,
)

# Load critique evaluation prompt template (single source of truth)
CRITIQUE_EVALUATION_PROMPT_TEMPLATE_PATH = Path(__file__).parent / "critique_evaluation_prompt.jinja2"
with open(CRITIQUE_EVALUATION_PROMPT_TEMPLATE_PATH, "r") as f:
    CRITIQUE_EVALUATION_PROMPT_TEMPLATE = Template(f.read())


def create_reference_critique_request(
    problem_idx: int,
    option: str,  # 'A' or 'B'
    question: str,
    option_a: str,
    option_b: str,
) -> Dict:
    """Create a batch request for generating a reference critique."""
    messages = generate_critique_prompt(question, option_a, option_b, option)
    
    # Extract system message and user messages
    system_message = None
    user_messages = []
    for msg in messages:
        if msg['role'] == 'system':
            system_message = msg['content']
        else:
            user_messages.append(msg)
    
    request = {
        "custom_id": f"ref_{problem_idx}_{option}",
        "params": {
            "model": "claude-3-5-haiku-20241022",
            "max_tokens": 4096,
            "messages": user_messages
        }
    }
    if system_message:
        request["params"]["system"] = system_message
    
    return request


def wait_for_batch_completion(client: Anthropic, batch_id: str, check_interval: int = 30) -> Dict:
    """Wait for a batch to complete and return results."""
    print(f"  Waiting for batch {batch_id} to complete...")
    
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        
        # Get counts
        total = batch.request_counts.processing + batch.request_counts.succeeded + batch.request_counts.errored + batch.request_counts.canceled + batch.request_counts.expired
        succeeded = batch.request_counts.succeeded
        errored = batch.request_counts.errored
        processing = batch.request_counts.processing
        
        print(f"  Status: {status} | Succeeded: {succeeded}/{total} | Errors: {errored} | Processing: {processing}")
        
        if status == "ended":
            print(f"  ✓ Batch completed!")
            break
        
        time.sleep(check_interval)
    
    # Retrieve results
    results = {}
    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        if result.result.type == "succeeded":
            message = result.result.message
            text = message.content[0].text if message.content else ""
            results[custom_id] = {"success": True, "text": text}
        else:
            results[custom_id] = {"success": False, "error": str(result.result)}
    
    return results


def get_reference_critiques_cache_path(
    data_dir: str,
    split_name: str,
    num_samples: int,
    reference_model: str = "claude-3-5-haiku-20241022"
) -> Path:
    """Get the cache path for reference critiques."""
    # Extract dataset name from data_dir
    dataset_name = Path(data_dir).name
    
    # Normalize model name for filename (replace / with _)
    model_name_normalized = reference_model.replace("/", "_").replace("-", "_")
    
    # Create cache directory
    cache_dir = Path("data") / "reference_critiques"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create cache key: dataset_name_split_model_n{num_samples}
    # Format: {dataset_name}_{split_name}_{model_name_normalized}_n{num_samples}
    cache_key = f"{dataset_name}_{split_name}_{model_name_normalized}_n{num_samples}"
    
    cache_file = cache_dir / f"{cache_key}.json"
    return cache_file


def load_cached_reference_critiques(
    cache_path: Path,
    problems: List[Dict]
) -> Optional[List[Dict]]:
    """Load cached reference critiques if they exist and match the problems."""
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'r') as f:
            cached_data = json.load(f)
        
        # Verify that cached problems match current problems
        cached_problems = cached_data.get('problems', [])
        if len(cached_problems) != len(problems):
            print(f"  Cache mismatch: different number of problems ({len(cached_problems)} vs {len(problems)})")
            return None
        
        # Check if problems match (by prompt hash)
        for i, (cached_prob, current_prob) in enumerate(zip(cached_problems, problems)):
            cached_prompt_hash = cached_prob.get('prompt_hash')
            current_prompt_hash = hashlib.md5(current_prob['prompt'].encode()).hexdigest()
            if cached_prompt_hash != current_prompt_hash:
                print(f"  Cache mismatch: problem {i} differs")
                return None
        
        print(f"✓ Loaded {len(cached_data['critiques'])} cached reference critiques from: {cache_path}")
        return cached_data['critiques']
    except Exception as e:
        print(f"  Warning: Failed to load cache: {e}")
        return None


def save_reference_critiques_cache(
    cache_path: Path,
    problems: List[Dict],
    critiques: List[Dict]
):
    """Save reference critiques to cache."""
    try:
        # Store problems with hashes for verification
        problems_with_hashes = [
            {
                'prompt': p['prompt'],
                'first': p['first'],
                'second': p['second'],
                'label': p['label'],
                'prompt_hash': hashlib.md5(p['prompt'].encode()).hexdigest()
            }
            for p in problems
        ]
        
        cache_data = {
            'problems': problems_with_hashes,
            'critiques': critiques
        }
        
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        print(f"✓ Saved {len(critiques)} reference critiques to cache: {cache_path}")
    except Exception as e:
        print(f"  Warning: Failed to save cache: {e}")


def generate_reference_critiques_batch(
    client: Anthropic,
    problems: List[Dict],
    cache_path: Optional[Path] = None,
    check_interval: int = 30
) -> List[Dict]:
    """Generate reference critiques using Anthropic Batch API (50% cheaper)."""
    # Try to load from cache first
    if cache_path:
        cached_critiques = load_cached_reference_critiques(cache_path, problems)
        if cached_critiques is not None:
            return cached_critiques
    
    # Create batch requests
    print(f"\nCreating batch requests for {len(problems)} problems...")
    requests = []
    for idx, problem in enumerate(problems):
        # Request for critique A
        requests.append(create_reference_critique_request(
            idx, 'A', problem['prompt'], problem['first'], problem['second']
        ))
        # Request for critique B
        requests.append(create_reference_critique_request(
            idx, 'B', problem['prompt'], problem['first'], problem['second']
        ))
    
    print(f"  Total requests: {len(requests)}")
    print(f"  Submitting batch to Anthropic Batch API (50% cheaper than real-time)...")
    
    # Submit batch
    batch = client.messages.batches.create(requests=requests)
    print(f"  Batch ID: {batch.id}")
    
    # Wait for completion
    results = wait_for_batch_completion(client, batch.id, check_interval)
    
    # Parse results into critique pairs
    all_critiques = []
    for idx in range(len(problems)):
        critique_a_id = f"ref_{idx}_A"
        critique_b_id = f"ref_{idx}_B"
        
        critique_a = ""
        critique_b = ""
        
        if critique_a_id in results and results[critique_a_id]["success"]:
            critique_a = results[critique_a_id]["text"]
        else:
            print(f"  Warning: Failed to generate critique A for problem {idx + 1}")
        
        if critique_b_id in results and results[critique_b_id]["success"]:
            critique_b = results[critique_b_id]["text"]
        else:
            print(f"  Warning: Failed to generate critique B for problem {idx + 1}")
        
        all_critiques.append({
            'critique_a': critique_a,
            'critique_b': critique_b
        })
    
    # Save to cache
    if cache_path:
        save_reference_critiques_cache(cache_path, problems, all_critiques)
    
    return all_critiques


def generate_model_critiques(
    model_name: str,
    checkpoint_path: Optional[str],
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 0.99,
) -> List[Tuple[str, str]]:
    """Generate critiques using a model (with optional checkpoint)."""
    print(f"\nGenerating critiques using model: {model_name}")
    
    # Handle merged models: if checkpoint_path is a directory with config.json, treat it as a model path
    if checkpoint_path:
        checkpoint_path_obj = Path(checkpoint_path)
        if checkpoint_path_obj.is_dir() and (checkpoint_path_obj / "config.json").exists():
            # This is a merged model directory, use it as the model path
            print(f"  Using merged model directory: {checkpoint_path}")
            actual_model = checkpoint_path
            actual_checkpoint = None
        else:
            # This is a LoRA checkpoint
            print(f"  LoRA checkpoint: {checkpoint_path}")
            actual_model = model_name
            actual_checkpoint = checkpoint_path
    else:
        print("  Using base model (no checkpoint)")
        actual_model = model_name
        actual_checkpoint = None
    
    critiques = generate_critiques_vllm(
        model_name=actual_model,
        lora_checkpoint=actual_checkpoint,
        tokenizer=tokenizer,
        dataset=dataset,
        max_new_tokens=max_tokens,
        return_prompts=False,
        temperature=temperature,
        top_p=top_p,
        top_k=None,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
    )
    
    print(f"✓ Generated {len(critiques)} critique pairs")
    return critiques


def create_judge_request(
    problem_idx: int,
    option: str,  # 'A' or 'B'
    problem: Dict,
    solution_text: str,
    reference_critique: str,
    comparison_critique: str,
    is_correct_solution: bool,
) -> Dict:
    """Create a batch request for judging critiques."""
    prompt = CRITIQUE_EVALUATION_PROMPT_TEMPLATE.render(
        problem=problem['prompt'],
        solution=solution_text,
        ground_truth='CORRECT' if is_correct_solution else 'INCORRECT',
        reference_critique=reference_critique,
        comparison_critique=comparison_critique,
    )
    
    return {
        "custom_id": f"judge_{problem_idx}_{option}_{is_correct_solution}",
        "params": {
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 2048,
            "system": "You are an expert evaluator of mathematical critiques.",
            "messages": [{"role": "user", "content": prompt}]
        }
    }


def compare_critiques_batch(
    client: Anthropic,
    problems: List[Dict],
    reference_critiques: List[Dict],
    comparison_critiques: List[Tuple[str, str]],
    max_problems: Optional[int] = None,
    check_interval: int = 30
) -> List[Dict]:
    """Compare critiques using Anthropic Batch API (50% cheaper)."""
    num_problems = min(
        len(problems),
        len(reference_critiques),
        len(comparison_critiques),
        max_problems if max_problems else len(problems)
    )
    
    print(f"\nCreating judge batch requests for {num_problems} problems...")
    
    # Build all requests
    requests = []
    request_metadata = []  # Track problem_idx and option for each request
    
    for i in range(num_problems):
        problem = problems[i]
        label = problem['label']
        first_is_correct = label in ['A', 'a', '0', 0]
        second_is_correct = label in ['B', 'b', '1', 1]
        
        # Request for critique A comparison
        requests.append(create_judge_request(
            i, 'A', problem, problem['first'],
            reference_critiques[i]['critique_a'],
            comparison_critiques[i][0],
            first_is_correct
        ))
        request_metadata.append({'problem_idx': i, 'option': 'A', 'is_correct': first_is_correct})
        
        # Request for critique B comparison
        requests.append(create_judge_request(
            i, 'B', problem, problem['second'],
            reference_critiques[i]['critique_b'],
            comparison_critiques[i][1],
            second_is_correct
        ))
        request_metadata.append({'problem_idx': i, 'option': 'B', 'is_correct': second_is_correct})
    
    print(f"  Total requests: {len(requests)}")
    print(f"  Submitting batch to Anthropic Batch API (50% cheaper than real-time)...")
    
    # Submit batch
    batch = client.messages.batches.create(requests=requests)
    print(f"  Batch ID: {batch.id}")
    
    # Wait for completion
    batch_results = wait_for_batch_completion(client, batch.id, check_interval)
    
    # Parse results
    results_by_problem = {}
    for idx, meta in enumerate(request_metadata):
        problem_idx = meta['problem_idx']
        option = meta['option']
        is_correct = meta['is_correct']
        custom_id = f"judge_{problem_idx}_{option}_{is_correct}"
        
        if problem_idx not in results_by_problem:
            results_by_problem[problem_idx] = {
                'problem_idx': problem_idx,
                'problem_prompt': problems[problem_idx]['prompt'],
                'correct_label': problems[problem_idx]['label'],
                'comparison_a': None,
                'comparison_b': None,
            }
        
        result = None
        if custom_id in batch_results and batch_results[custom_id]["success"]:
            response_text = batch_results[custom_id]["text"].strip()
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                    result['is_correct_solution'] = is_correct
                except json.JSONDecodeError:
                    print(f"  Warning: Could not parse JSON for problem {problem_idx} option {option}")
        
        if option == 'A':
            results_by_problem[problem_idx]['comparison_a'] = result
        else:
            results_by_problem[problem_idx]['comparison_b'] = result
    
    # Convert to list
    results = [results_by_problem[i] for i in range(num_problems)]
    return results


def calculate_statistics(
    results: List[Dict],
    reference_critiques: Optional[List[Dict]] = None,
    comparison_critiques: Optional[List[Tuple[str, str]]] = None
) -> Dict:
    """Calculate statistics from comparison results."""
    stats = {
        'total_comparisons': 0,
        'reference_wins': 0,
        'comparison_wins': 0,
        'ties': 0,
        'reference_avg_quality': 0,
        'comparison_avg_quality': 0,
        'reference_avg_fluency': 0,
        'comparison_avg_fluency': 0,
        'reference_avg_math': 0,
        'comparison_avg_math': 0,
        'reference_avg_fluency': 0,
        'comparison_avg_fluency': 0,
        'reference_avg_code_mixing': 0,
        'comparison_avg_code_mixing': 0,
        'reference_avg_repetition': 0,
        'comparison_avg_repetition': 0,
        'reference_avg_length': 0,
        'comparison_avg_length': 0,
        'reference_error_identification': {
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0,
        },
        'comparison_error_identification': {
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0,
        },
    }
    
    quality_scores_ref = []
    quality_scores_comp = []
    math_scores_ref = []
    math_scores_comp = []
    fluency_scores_ref = []
    fluency_scores_comp = []
    code_mixing_scores_ref = []
    code_mixing_scores_comp = []
    repetition_scores_ref = []
    repetition_scores_comp = []
    
    # Calculate critique lengths
    if reference_critiques:
        ref_lengths = []
        for crit in reference_critiques:
            # Each critique has critique_a and critique_b
            if crit.get('critique_a'):
                ref_lengths.append(len(crit['critique_a']))
            if crit.get('critique_b'):
                ref_lengths.append(len(crit['critique_b']))
        if ref_lengths:
            stats['reference_avg_length'] = sum(ref_lengths) / len(ref_lengths)
    
    if comparison_critiques:
        comp_lengths = []
        for crit_a, crit_b in comparison_critiques:
            if crit_a:
                comp_lengths.append(len(crit_a))
            if crit_b:
                comp_lengths.append(len(crit_b))
        if comp_lengths:
            stats['comparison_avg_length'] = sum(comp_lengths) / len(comp_lengths)
    
    def safe_int(value, default=0):
        return value if isinstance(value, int) else default
    
    for result in results:
        for key in ['comparison_a', 'comparison_b']:
            comp = result.get(key)
            if comp and comp.get('comparison'):
                stats['total_comparisons'] += 1
                winner = comp['comparison'].get('winner', 'tie')
                
                if winner == 'reference':
                    stats['reference_wins'] += 1
                elif winner == 'comparison':
                    stats['comparison_wins'] += 1
                else:
                    stats['ties'] += 1
                
                # Extract scores from nested JSON structure
                ref_critique = comp.get('reference_critique') or {}
                comp_critique = comp.get('comparison_critique') or {}
                comparison = comp.get('comparison') or {}
                
                # Extract nested fields
                ref_fluency = ref_critique.get('fluency') or {}
                comp_fluency = comp_critique.get('fluency') or {}
                ref_math = ref_critique.get('mathematical_correctness') or {}
                comp_math = comp_critique.get('mathematical_correctness') or {}
                
                # Overall quality scores
                if ref_critique.get('overall_quality') is not None:
                    quality_scores_ref.append(ref_critique['overall_quality'])
                if comp_critique.get('overall_quality') is not None:
                    quality_scores_comp.append(comp_critique['overall_quality'])
                
                # Math correctness scores
                if ref_math.get('score') is not None:
                    math_scores_ref.append(ref_math['score'])
                if comp_math.get('score') is not None:
                    math_scores_comp.append(comp_math['score'])
                
                # Fluency scores
                if ref_fluency.get('score') is not None:
                    fluency_scores_ref.append(ref_fluency['score'])
                if comp_fluency.get('score') is not None:
                    fluency_scores_comp.append(comp_fluency['score'])
                
                # Code mixing (convert boolean to 0/1)
                if ref_fluency.get('has_code_mixing') is not None:
                    code_mixing_scores_ref.append(1 if ref_fluency['has_code_mixing'] else 0)
                if comp_fluency.get('has_code_mixing') is not None:
                    code_mixing_scores_comp.append(1 if comp_fluency['has_code_mixing'] else 0)
                
                # Repetition (convert boolean to 0/1)
                if ref_fluency.get('has_repetition') is not None:
                    repetition_scores_ref.append(1 if ref_fluency['has_repetition'] else 0)
                if comp_fluency.get('has_repetition') is not None:
                    repetition_scores_comp.append(1 if comp_fluency['has_repetition'] else 0)
                
                # Compute confusion matrix from valid_errors/invalid_errors lists + ground truth
                is_correct = comp.get('is_correct_solution', False)
                
                # Reference critique confusion matrix
                ref_valid = len(ref_math.get('valid_errors') or [])
                ref_invalid = len(ref_math.get('invalid_errors') or [])
                if is_correct:
                    # Solution is correct: FP = invalid claims, TN = 1 if no claims
                    stats['reference_error_identification']['false_positives'] += ref_invalid
                    if ref_valid == 0 and ref_invalid == 0:
                        stats['reference_error_identification']['true_negatives'] += 1
                else:
                    # Solution is incorrect: TP = valid errors found, FN = 1 if no valid errors found
                    stats['reference_error_identification']['true_positives'] += ref_valid
                    stats['reference_error_identification']['false_positives'] += ref_invalid
                    if ref_valid == 0:
                        stats['reference_error_identification']['false_negatives'] += 1
                
                # Comparison critique confusion matrix
                comp_valid = len(comp_math.get('valid_errors') or [])
                comp_invalid = len(comp_math.get('invalid_errors') or [])
                if is_correct:
                    # Solution is correct: FP = invalid claims, TN = 1 if no claims
                    stats['comparison_error_identification']['false_positives'] += comp_invalid
                    if comp_valid == 0 and comp_invalid == 0:
                        stats['comparison_error_identification']['true_negatives'] += 1
                else:
                    # Solution is incorrect: TP = valid errors found, FN = 1 if no valid errors found
                    stats['comparison_error_identification']['true_positives'] += comp_valid
                    stats['comparison_error_identification']['false_positives'] += comp_invalid
                    if comp_valid == 0:
                        stats['comparison_error_identification']['false_negatives'] += 1
    
    if quality_scores_ref:
        stats['reference_avg_quality'] = sum(quality_scores_ref) / len(quality_scores_ref)
    if quality_scores_comp:
        stats['comparison_avg_quality'] = sum(quality_scores_comp) / len(quality_scores_comp)
    if math_scores_ref:
        stats['reference_avg_math'] = sum(math_scores_ref) / len(math_scores_ref)
    if math_scores_comp:
        stats['comparison_avg_math'] = sum(math_scores_comp) / len(math_scores_comp)
    if fluency_scores_ref:
        stats['reference_avg_fluency'] = sum(fluency_scores_ref) / len(fluency_scores_ref)
    if fluency_scores_comp:
        stats['comparison_avg_fluency'] = sum(fluency_scores_comp) / len(fluency_scores_comp)
    if code_mixing_scores_ref:
        stats['reference_avg_code_mixing'] = sum(code_mixing_scores_ref) / len(code_mixing_scores_ref)
    if code_mixing_scores_comp:
        stats['comparison_avg_code_mixing'] = sum(code_mixing_scores_comp) / len(code_mixing_scores_comp)
    if repetition_scores_ref:
        stats['reference_avg_repetition'] = sum(repetition_scores_ref) / len(repetition_scores_ref)
    if repetition_scores_comp:
        stats['comparison_avg_repetition'] = sum(repetition_scores_comp) / len(repetition_scores_comp)
    
    # Calculate precision, recall, F1
    for error_id in ['reference_error_identification', 'comparison_error_identification']:
        eid = stats[error_id]
        tp = eid['true_positives']
        fp = eid['false_positives']
        tn = eid['true_negatives']
        fn = eid['false_negatives']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
        
        eid['precision'] = precision
        eid['recall'] = recall
        eid['f1'] = f1
        eid['accuracy'] = accuracy
    
    return stats


# Summary generation functions (from generate_summary.py)
def load_comparison_results(output_dir: Path) -> dict:
    """Load all comparison result JSON files from the output directory."""
    models = {}
    json_files_found = list(output_dir.glob("*.json"))
    print(f"  Found {len(json_files_found)} JSON files in {output_dir}")

    for json_file in json_files_found:
        if json_file.name == "summary.json":
            continue
        try:
            with open(json_file) as f:
                data = json.load(f)
            # Map file names to model names (must match MODEL_ORDER)
            file_to_model = {
                "base_model.json": "Strong Base Model",
                "sft_critic_model.json": "Sft Critic Model",
                "grpo_trained_model.json": "Grpo Trained Model"
            }
            model_name = file_to_model.get(json_file.name, json_file.stem.replace("_", " ").title())
            models[model_name] = data
            print(f"  ✓ Loaded {json_file.name} as '{model_name}'")
        except Exception as e:
            print(f"  Warning: Failed to load {json_file}: {e}")
            import traceback
            traceback.print_exc()

    return models


def generate_summary_markdown(models: dict, output_dir: Path) -> str:
    """Generate the summary markdown content."""
    # Define consistent model order
    MODEL_ORDER = ["Strong Base Model", "Sft Critic Model", "Grpo Trained Model"]
    ordered_models = [(name, models[name]) for name in MODEL_ORDER if name in models]

    # Build markdown
    md = []
    md.append("# Critique Comparison Summary\n")
    md.append(f"**Output Directory:** `{output_dir}`\n")

    # Statistics table - reorganized by category
    md.append("## Statistics\n")
    
    # Main summary table
    md.append("### Overall Summary\n")
    md.append("| Model | Win Rate | Overall Quality | Math Score | Fluency Score | Avg Length |")
    md.append("|-------|----------|-----------------|------------|---------------|------------|")

    # Add reference row (from first model's data)
    first_model = list(models.values())[0]
    stats = first_model.get('statistics', {})
    config = first_model.get('config', {})
    num_samples = config.get('num_samples', 0)
    total = stats.get('total_comparisons', 0)
    if total > 0:
        ref_wins = stats.get('reference_wins', 0)
        ref_quality = stats.get('reference_avg_quality', 0)
        ref_math = stats.get('reference_avg_math', 0)
        ref_fluency = stats.get('reference_avg_fluency', 0)
        ref_length = stats.get('reference_avg_length', 0)
        md.append(f"| **Reference (Haiku 3.5)** | {ref_wins/total*100:.1f}% ({ref_wins}/{total}) | {ref_quality:.2f}/10 | {ref_math:.2f}/10 | {ref_fluency:.2f}/10 | {ref_length:.0f} chars |")

    # Add comparison models (in order)
    for model_name, data in ordered_models:
        stats = data.get('statistics', {})
        total = stats.get('total_comparisons', 0)
        if total > 0:
            comp_wins = stats.get('comparison_wins', 0)
            comp_quality = stats.get('comparison_avg_quality', 0)
            comp_math = stats.get('comparison_avg_math', 0)
            comp_fluency = stats.get('comparison_avg_fluency', 0)
            comp_length = stats.get('comparison_avg_length', 0)
            md.append(f"| {model_name} | {comp_wins/total*100:.1f}% ({comp_wins}/{total}) | {comp_quality:.2f}/10 | {comp_math:.2f}/10 | {comp_fluency:.2f}/10 | {comp_length:.0f} chars |")
    
    md.append("")
    
    # Length details table
    md.append("### Length Details\n")
    md.append("| Model | Avg Length (chars) |")
    md.append("|-------|-------------------|")
    
    stats = first_model.get('statistics', {})
    total = stats.get('total_comparisons', 0)
    if total > 0:
        ref_length = stats.get('reference_avg_length', 0)
        md.append(f"| **Reference (Haiku 3.5)** | {ref_length:.0f} |")
    
    for model_name, data in ordered_models:
        stats = data.get('statistics', {})
        total = stats.get('total_comparisons', 0)
        if total > 0:
            comp_length = stats.get('comparison_avg_length', 0)
            md.append(f"| {model_name} | {comp_length:.0f} |")
    
    md.append("")
    
    # Fluency details table
    md.append("### Fluency Details\n")
    md.append("| Model | Fluency Score | Code Mixing Rate | Repetition Rate |")
    md.append("|-------|---------------|------------------|-----------------|")
    
    stats = first_model.get('statistics', {})
    total = stats.get('total_comparisons', 0)
    if total > 0:
        ref_fluency = stats.get('reference_avg_fluency', 0)
        ref_code_mixing = stats.get('reference_avg_code_mixing', 0)
        ref_repetition = stats.get('reference_avg_repetition', 0)
        md.append(f"| **Reference (Haiku 3.5)** | {ref_fluency:.2f}/10 | {ref_code_mixing*100:.1f}% | {ref_repetition*100:.1f}% |")
    
    for model_name, data in ordered_models:
        stats = data.get('statistics', {})
        total = stats.get('total_comparisons', 0)
        if total > 0:
            comp_fluency = stats.get('comparison_avg_fluency', 0)
            comp_code_mixing = stats.get('comparison_avg_code_mixing', 0)
            comp_repetition = stats.get('comparison_avg_repetition', 0)
            md.append(f"| {model_name} | {comp_fluency:.2f}/10 | {comp_code_mixing*100:.1f}% | {comp_repetition*100:.1f}% |")
    
    md.append("")
    
    # Mathematical correctness table (confusion matrix)
    md.append("### Mathematical Correctness (Error Identification)\n")
    md.append("| Model | Math Score | TP | FP | TN | FN | Precision | Recall | F1 |")
    md.append("|-------|------------|----|----|----|----|-----------|--------|-----|")
    
    stats = first_model.get('statistics', {})
    total = stats.get('total_comparisons', 0)
    if total > 0:
        ref_math = stats.get('reference_avg_math', 0)
        ref_eid = stats.get('reference_error_identification', {})
        ref_tp = ref_eid.get('true_positives', 0)
        ref_fp = ref_eid.get('false_positives', 0)
        ref_tn = ref_eid.get('true_negatives', 0)
        ref_fn = ref_eid.get('false_negatives', 0)
        ref_p = ref_eid.get('precision', 0)
        ref_r = ref_eid.get('recall', 0)
        ref_f1 = ref_eid.get('f1', 0)
        md.append(f"| **Reference (Haiku 3.5)** | {ref_math:.2f}/10 | {ref_tp} | {ref_fp} | {ref_tn} | {ref_fn} | {ref_p:.3f} | {ref_r:.3f} | {ref_f1:.3f} |")
    
    for model_name, data in ordered_models:
        stats = data.get('statistics', {})
        total = stats.get('total_comparisons', 0)
        if total > 0:
            comp_math = stats.get('comparison_avg_math', 0)
            comp_eid = stats.get('comparison_error_identification', {})
            comp_tp = comp_eid.get('true_positives', 0)
            comp_fp = comp_eid.get('false_positives', 0)
            comp_tn = comp_eid.get('true_negatives', 0)
            comp_fn = comp_eid.get('false_negatives', 0)
            comp_p = comp_eid.get('precision', 0)
            comp_r = comp_eid.get('recall', 0)
            comp_f1 = comp_eid.get('f1', 0)
            md.append(f"| {model_name} | {comp_math:.2f}/10 | {comp_tp} | {comp_fp} | {comp_tn} | {comp_fn} | {comp_p:.3f} | {comp_r:.3f} | {comp_f1:.3f} |")

    md.append("")

    # Add example comparison section - show all models for the same problem
    md.append("## Example Comparison\n")

    # Find the first valid example index
    example_idx = None
    first_model_data = list(models.values())[0]
    for i, result in enumerate(first_model_data.get('results', [])):
        if result.get('comparison_a') and result.get('comparison_b'):
            example_idx = i
            break

    if example_idx is not None:
        # Get problem info from first model
        first_result = first_model_data['results'][example_idx]
        problem_prompt = first_result.get('problem_prompt', '')
        correct_label = first_result.get('correct_label', '')
        
        # Get problems array (contains first/second options)
        problems_list = first_model_data.get('problems', [])
        option_a_text = ''
        option_b_text = ''
        if example_idx < len(problems_list):
            option_a_text = problems_list[example_idx].get('first', '')
            option_b_text = problems_list[example_idx].get('second', '')
        
        # Convert label to Option A/B format
        if correct_label in ['0', 0, 'A', 'a']:
            correct_option = 'A'
        elif correct_label in ['1', 1, 'B', 'b']:
            correct_option = 'B'
        else:
            correct_option = str(correct_label)
        
        # Get reference critiques
        ref_critiques = first_model_data.get('reference_critiques', [])
        
        md.append(f"### Problem\n")
        md.append(f"```\n{problem_prompt}\n```\n")
        md.append(f"**Correct Option:** {correct_option}\n")
        
        # Determine which option to show (the INCORRECT one is more interesting for critique)
        # If A is correct, show B; if B is correct, show A
        if correct_option == 'A':
            shown_option = 'B'
            shown_option_text = option_b_text
            critique_key = 'critique_b'
            result_key = 'comparison_b'
        else:
            shown_option = 'A'
            shown_option_text = option_a_text
            critique_key = 'critique_a'
            result_key = 'comparison_a'
        
        # Show the incorrect solution text (being critiqued)
        if shown_option_text:
            md.append(f"### Option {shown_option} Solution (INCORRECT - being critiqued)\n")
            md.append(f"```\n{shown_option_text}\n```\n")
        
        # Reference critique (same for all models, use first model's data)
        if example_idx < len(ref_critiques):
            ref_crit = ref_critiques[example_idx]
            md.append(f"### Reference Critique (Claude 3.5 Haiku)\n")
            md.append(f"**Critique of Option {shown_option}:**\n")
            ref_critique_text = ref_crit.get(critique_key, 'N/A')
            md.append(f"```\n{ref_critique_text}\n```\n")
        
        # Show each model's critique and judge analysis (in order)
        for model_name, data in ordered_models:
            results = data.get('results', [])
            comp_critiques = data.get('comparison_critiques', [])
            
            if example_idx >= len(results) or example_idx >= len(comp_critiques):
                continue
            
            result = results[example_idx]
            comp_crit = comp_critiques[example_idx]
            
            md.append(f"---\n")
            md.append(f"### {model_name}\n")
            
            # Model's critique for the incorrect option
            md.append(f"**Critique (Option {shown_option}):**\n")
            comp_critique_text = comp_crit.get(critique_key, 'N/A')
            md.append(f"```\n{comp_critique_text}\n```\n")
            
            # Judge analysis for the incorrect option
            comp_result = result.get(result_key, {})
            if comp_result and comp_result.get('comparison'):
                comparison = comp_result.get('comparison', {})
                winner = comparison.get('winner', 'N/A')
                reasoning = comparison.get('reasoning', 'N/A')
                
                ref_analysis = comp_result.get('reference_critique', {})
                comp_analysis = comp_result.get('comparison_critique', {})
                
                # Extract nested fields
                ref_fluency = ref_analysis.get('fluency', {})
                comp_fluency = comp_analysis.get('fluency', {})
                ref_math = ref_analysis.get('mathematical_correctness', {})
                comp_math = comp_analysis.get('mathematical_correctness', {})
                
                md.append(f"**Judge Analysis:**\n")
                md.append(f"- **Winner:** {winner}")
                md.append(f"- **Reference Overall Quality:** {ref_analysis.get('overall_quality', 'N/A')}/10")
                md.append(f"- **{model_name} Overall Quality:** {comp_analysis.get('overall_quality', 'N/A')}/10")
                md.append(f"- **Reference Math Score:** {ref_math.get('score', 'N/A')}/10")
                md.append(f"- **{model_name} Math Score:** {comp_math.get('score', 'N/A')}/10")
                md.append(f"- **Reference Fluency:** {ref_fluency.get('score', 'N/A')}/10")
                md.append(f"- **{model_name} Fluency:** {comp_fluency.get('score', 'N/A')}/10")
                ref_code_mixing = ref_fluency.get('has_code_mixing', 'N/A')
                comp_code_mixing = comp_fluency.get('has_code_mixing', 'N/A')
                ref_mixing_text = 'yes' if ref_code_mixing is True else 'no' if ref_code_mixing is False else 'N/A'
                comp_mixing_text = 'yes' if comp_code_mixing is True else 'no' if comp_code_mixing is False else 'N/A'
                ref_repetition = ref_fluency.get('has_repetition', 'N/A')
                comp_repetition = comp_fluency.get('has_repetition', 'N/A')
                ref_repetition_text = 'yes' if ref_repetition is True else 'no' if ref_repetition is False else 'N/A'
                comp_repetition_text = 'yes' if comp_repetition is True else 'no' if comp_repetition is False else 'N/A'
                md.append(f"- **Reference Code Mixing:** {ref_mixing_text}")
                md.append(f"- **{model_name} Code Mixing:** {comp_mixing_text}")
                md.append(f"- **Reference Repetition:** {ref_repetition_text}")
                md.append(f"- **{model_name} Repetition:** {comp_repetition_text}")
                md.append(f"- **Reference Valid Errors:** {ref_math.get('valid_errors', [])}")
                md.append(f"- **Reference Invalid Errors:** {ref_math.get('invalid_errors', [])}")
                md.append(f"- **{model_name} Valid Errors:** {comp_math.get('valid_errors', [])}")
                md.append(f"- **{model_name} Invalid Errors:** {comp_math.get('invalid_errors', [])}\n")
                md.append(f"\n**Reasoning:**\n> {reasoning}\n")

    # Critique Evaluation Prompt section at the end - read from Jinja2 template (single source of truth)
    with open(CRITIQUE_EVALUATION_PROMPT_TEMPLATE_PATH, "r") as f:
        critique_evaluation_prompt_text = f.read()

    md.append("---\n")
    md.append("## Appendix: Critique Evaluation Prompt\n")
    md.append("The following prompt template is used by Claude Haiku 4.5 to evaluate critiques:\n")
    md.append("\n```\n")
    md.append(critique_evaluation_prompt_text)
    md.append("\n```\n")

    return '\n'.join(md)


def print_statistics_table(models: dict):
    """Print statistics table to console with hierarchical structure."""
    MODEL_ORDER = ["Strong Base Model", "Sft Critic Model", "Grpo Trained Model"]
    ordered_models = [(name, models[name]) for name in MODEL_ORDER if name in models]

    print("\n" + "="*80)
    print("STATISTICS SUMMARY")
    print("="*80)

    # Reference row
    first_model = list(models.values())[0]
    stats = first_model.get('statistics', {})
    config = first_model.get('config', {})
    total = stats.get('total_comparisons', 0)
    
    if total > 0:
        ref_eid = stats.get('reference_error_identification', {})
        print("\n📊 OVERALL SUMMARY")
        print("-"*70)
        print(f"{'Model':<25} {'Win Rate':>10} {'Overall':>10} {'Math':>10} {'Fluency':>10} {'Length':>10}")
        print("-"*70)
        print(f"{'Reference (Haiku 3.5)':<25} {stats.get('reference_wins', 0)/total*100:>9.1f}% {stats.get('reference_avg_quality', 0):>9.2f} {stats.get('reference_avg_math', 0):>9.2f} {stats.get('reference_avg_fluency', 0):>9.2f} {stats.get('reference_avg_length', 0):>9.0f}")
        
        for model_name, data in ordered_models:
            s = data.get('statistics', {})
            t = s.get('total_comparisons', 0)
            if t > 0:
                print(f"{model_name:<25} {s.get('comparison_wins', 0)/t*100:>9.1f}% {s.get('comparison_avg_quality', 0):>9.2f} {s.get('comparison_avg_math', 0):>9.2f} {s.get('comparison_avg_fluency', 0):>9.2f} {s.get('comparison_avg_length', 0):>9.0f}")
        
        print("\n📏 LENGTH DETAILS")
        print("-"*40)
        print(f"{'Model':<25} {'Avg Length':>15}")
        print("-"*40)
        print(f"{'Reference (Haiku 3.5)':<25} {stats.get('reference_avg_length', 0):>14.0f} chars")
        
        for model_name, data in ordered_models:
            s = data.get('statistics', {})
            t = s.get('total_comparisons', 0)
            if t > 0:
                print(f"{model_name:<25} {s.get('comparison_avg_length', 0):>14.0f} chars")
        
        print("\n📝 FLUENCY DETAILS")
        print("-"*60)
        print(f"{'Model':<25} {'Fluency':>10} {'CodeMix%':>10} {'Repeat%':>10}")
        print("-"*60)
        print(f"{'Reference (Haiku 3.5)':<25} {stats.get('reference_avg_fluency', 0):>9.2f} {stats.get('reference_avg_code_mixing', 0)*100:>9.1f} {stats.get('reference_avg_repetition', 0)*100:>9.1f}")
        
        for model_name, data in ordered_models:
            s = data.get('statistics', {})
            t = s.get('total_comparisons', 0)
            if t > 0:
                print(f"{model_name:<25} {s.get('comparison_avg_fluency', 0):>9.2f} {s.get('comparison_avg_code_mixing', 0)*100:>9.1f} {s.get('comparison_avg_repetition', 0)*100:>9.1f}")
        
        print("\n🔢 MATHEMATICAL CORRECTNESS (Error Identification)")
        print("-"*70)
        print(f"{'Model':<25} {'Math':>6} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5} {'Prec':>6} {'Rec':>6} {'F1':>6}")
        print("-"*70)
        print(f"{'Reference (Haiku 3.5)':<25} {stats.get('reference_avg_math', 0):>5.2f} {ref_eid.get('true_positives', 0):>5} {ref_eid.get('false_positives', 0):>5} {ref_eid.get('true_negatives', 0):>5} {ref_eid.get('false_negatives', 0):>5} {ref_eid.get('precision', 0):>5.3f} {ref_eid.get('recall', 0):>5.3f} {ref_eid.get('f1', 0):>5.3f}")
        
        for model_name, data in ordered_models:
            s = data.get('statistics', {})
            t = s.get('total_comparisons', 0)
            if t > 0:
                comp_eid = s.get('comparison_error_identification', {})
                print(f"{model_name:<25} {s.get('comparison_avg_math', 0):>5.2f} {comp_eid.get('true_positives', 0):>5} {comp_eid.get('false_positives', 0):>5} {comp_eid.get('true_negatives', 0):>5} {comp_eid.get('false_negatives', 0):>5} {comp_eid.get('precision', 0):>5.3f} {comp_eid.get('recall', 0):>5.3f} {comp_eid.get('f1', 0):>5.3f}")
    
    print("="*80)


def main(args):
    """Main function using Anthropic Batch API (50% cheaper)."""
    # Handle summary-only mode
    if args.generate_summary_only:
        output_dir = Path(args.output_dir)
        if not output_dir.exists():
            print(f"Error: Output directory does not exist: {output_dir}")
            return 1

        # Load comparison results
        models = load_comparison_results(output_dir)
        if not models:
            print("  No comparison results found")
            print(f"  Expected JSON files in: {output_dir}")
            return 0

        # Generate and save summary markdown
        summary_content = generate_summary_markdown(models, output_dir)
        summary_file = output_dir / "summary.md"
        with open(summary_file, 'w') as f:
            f.write(summary_content)
        print(f"✓ Summary saved to: {summary_file}")

        # Print statistics to console
        print_statistics_table(models)
        return 0
    
    # Normal evaluation mode
    # Check anthropic is available
    if not ANTHROPIC_AVAILABLE:
        print("Error: anthropic package not installed. Install with: uv pip install anthropic")
        return 1
    
    # Get API key
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        return 1
    
    client = Anthropic(api_key=api_key)
    
    # Load test dataset
    print("="*80)
    print("EVALUATING CRITIQUES (using Anthropic Batch API - 50% cheaper)")
    print("="*80)
    print(f"Data dir: {args.data_dir}")
    print(f"Model: {args.model}")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
    print(f"Num samples: {args.num_samples}")
    print(f"Batch check interval: {args.batch_check_interval}s")
    print("="*80)
    
    print("\n[1/5] Loading test dataset...")
    datasets = load_dataset(args.data_dir)
    test_ds = datasets["test"]
    
    # Sample test examples
    random.seed(args.seed)
    if args.num_samples < len(test_ds):
        sample_indices = random.sample(range(len(test_ds)), args.num_samples)
        test_samples = test_ds.select(sample_indices)
    else:
        test_samples = test_ds
    
    print(f"✓ Loaded {len(test_samples)} test samples")
    
    # Convert to list for easier processing
    problems = [test_samples[i] for i in range(len(test_samples))]
    
    # Generate reference critiques (with caching)
    print("\n[2/5] Generating reference critiques using Claude 3.5 Haiku (Batch API)...")
    reference_model = "claude-3-5-haiku-20241022"
    cache_path = get_reference_critiques_cache_path(
        args.data_dir, 
        split_name="test", 
        num_samples=args.num_samples,
        reference_model=reference_model
    )
    reference_critiques = generate_reference_critiques_batch(
        client, problems, cache_path=cache_path, check_interval=args.batch_check_interval
    )
    print(f"✓ Generated/loaded {len(reference_critiques)} reference critique pairs")
    
    # Generate model critiques
    print("\n[3/5] Generating critiques from model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dataset = Dataset.from_list(problems)
    comparison_critiques = generate_model_critiques(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        tokenizer=tokenizer,
        dataset=dataset,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    
    # Compare critiques
    print("\n[4/5] Comparing critiques with Claude Haiku 4.5 (judge) (Batch API)...")
    results = compare_critiques_batch(
        client,
        problems,
        reference_critiques,
        comparison_critiques,
        max_problems=args.max_problems,
        check_interval=args.batch_check_interval
    )
    print(f"✓ Completed {len(results)} comparisons")
    
    # Calculate statistics
    print("\n[5/5] Calculating statistics...")
    stats = calculate_statistics(results, reference_critiques, comparison_critiques)
    
    # Save results
    output_data = {
        'config': {
            'model': args.model,
            'checkpoint': args.checkpoint,
            'data_dir': args.data_dir,
            'num_samples': args.num_samples,
            'seed': args.seed,
            'max_tokens': args.max_tokens,
            'temperature': args.temperature,
            'top_p': args.top_p,
        },
        'statistics': stats,
        'results': results,
        'problems': [
            {
                'prompt': p['prompt'],
                'first': p['first'],
                'second': p['second'],
                'label': p['label']
            } for p in problems
        ],
        'reference_critiques': reference_critiques,
        'comparison_critiques': [
            {'critique_a': a, 'critique_b': b} for a, b in comparison_critiques
        ],
    }
    
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary in structured format (matching summary.md)
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    total = stats['total_comparisons']
    ref_eid = stats['reference_error_identification']
    comp_eid = stats['comparison_error_identification']
    
    # Overall Summary
    print("\n┌─ OVERALL SUMMARY ─────────────────────────────────────────────────────────────┐")
    print(f"│ Total comparisons: {total}")
    if total > 0:
        ref_win_pct = stats['reference_wins']/total*100
        comp_win_pct = stats['comparison_wins']/total*100
        tie_pct = stats['ties']/total*100
        print(f"│")
        print(f"│ {'Model':<25} {'Win Rate':<15} {'Quality':<12} {'Math':<12} {'Fluency':<12}")
        print(f"│ {'-'*25} {'-'*15} {'-'*12} {'-'*12} {'-'*12}")
        print(f"│ {'Reference (Haiku 3.5)':<25} {ref_win_pct:>5.1f}% ({stats['reference_wins']}/{total})  {stats['reference_avg_quality']:>5.2f}/10    {stats.get('reference_avg_math', 0):>5.2f}/10    {stats['reference_avg_fluency']:>5.2f}/10")
        print(f"│ {'Comparison model':<25} {comp_win_pct:>5.1f}% ({stats['comparison_wins']}/{total})  {stats['comparison_avg_quality']:>5.2f}/10    {stats.get('comparison_avg_math', 0):>5.2f}/10    {stats['comparison_avg_fluency']:>5.2f}/10")
        print(f"│ {'Ties':<25} {tie_pct:>5.1f}% ({stats['ties']}/{total})")
    print("└───────────────────────────────────────────────────────────────────────────────┘")
    
    # Fluency Details
    print("\n┌─ FLUENCY DETAILS ─────────────────────────────────────────────────────────────┐")
    print(f"│ {'Model':<25} {'Fluency':<12} {'Code Mixing':<15} {'Repetition':<15} {'Avg Length':<12}")
    print(f"│ {'-'*25} {'-'*12} {'-'*15} {'-'*15} {'-'*12}")
    print(f"│ {'Reference (Haiku 3.5)':<25} {stats['reference_avg_fluency']:>5.2f}/10    {stats['reference_avg_code_mixing']*100:>5.1f}%          {stats['reference_avg_repetition']*100:>5.1f}%          {stats['reference_avg_length']:>8.0f}")
    print(f"│ {'Comparison model':<25} {stats['comparison_avg_fluency']:>5.2f}/10    {stats['comparison_avg_code_mixing']*100:>5.1f}%          {stats['comparison_avg_repetition']*100:>5.1f}%          {stats['comparison_avg_length']:>8.0f}")
    print("└───────────────────────────────────────────────────────────────────────────────┘")
    
    # Mathematical Correctness (Error Identification)
    print("\n┌─ MATHEMATICAL CORRECTNESS (Error Identification) ───────────────────────────────┐")
    print(f"│ {'Model':<25} {'Math':<8} {'TP':<5} {'FP':<5} {'TN':<5} {'FN':<5} {'Prec':<8} {'Recall':<8} {'F1':<8}")
    print(f"│ {'-'*25} {'-'*8} {'-'*5} {'-'*5} {'-'*5} {'-'*5} {'-'*8} {'-'*8} {'-'*8}")
    print(f"│ {'Reference (Haiku 3.5)':<25} {stats.get('reference_avg_math', 0):>5.2f}   {ref_eid['true_positives']:<5} {ref_eid['false_positives']:<5} {ref_eid['true_negatives']:<5} {ref_eid['false_negatives']:<5} {ref_eid['precision']:>6.3f}   {ref_eid['recall']:>6.3f}   {ref_eid['f1']:>6.3f}")
    print(f"│ {'Comparison model':<25} {stats.get('comparison_avg_math', 0):>5.2f}   {comp_eid['true_positives']:<5} {comp_eid['false_positives']:<5} {comp_eid['true_negatives']:<5} {comp_eid['false_negatives']:<5} {comp_eid['precision']:>6.3f}   {comp_eid['recall']:>6.3f}   {comp_eid['f1']:>6.3f}")
    print("└────────────────────────────────────────────────────────────────────────────────┘")
    
    print(f"\n✓ Results saved to {output_file}")
    
    # Generate summary if requested
    if args.generate_summary:
        output_dir = output_file.parent
        print(f"\n[6/6] Generating summary markdown...")
        # Load the single result as a model
        models = {args.model or "Comparison Model": output_data}
        summary_content = generate_summary_markdown(models, output_dir)
        summary_file = output_dir / "summary.md"
        with open(summary_file, 'w') as f:
            f.write(summary_content)
        print(f"✓ Summary saved to: {summary_file}")
    
    print("="*80)
    return 0


def run():
    """Main entry point (CLI)."""
    parser = argparse.ArgumentParser(
        description="Evaluate critiques by comparing against reference critiques from Claude 3.5 Haiku"
    )
    
    # Summary-only mode
    parser.add_argument(
        '--generate-summary-only',
        action='store_true',
        help='Generate summary markdown from existing results in output directory (skip evaluation)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory containing JSON result files (for --generate-summary-only mode)'
    )
    
    # Evaluation mode arguments
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model name (e.g., Qwen/Qwen3-4B-Base)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint (LoRA adapter or merged model). If None, uses base model.'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file path'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=20,
        help='Number of test samples to evaluate (default: 20)'
    )
    parser.add_argument(
        '--max-problems',
        type=int,
        default=None,
        help='Maximum number of problems to process (default: all)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=1024,
        help='Max tokens for critique generation (default: 1024)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Temperature for critique generation (default: 1.0)'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.99,
        help='Top-p for critique generation (default: 0.99)'
    )
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=10,
        help='Maximum concurrent API requests (default: 10, used for legacy mode)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--batch-check-interval',
        type=int,
        default=30,
        help='Interval in seconds to check batch status (default: 30)'
    )
    parser.add_argument(
        '--generate-summary',
        action='store_true',
        help='Generate summary markdown after evaluation'
    )
    
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.generate_summary_only:
        if not args.output_dir:
            parser.error("--output-dir is required when using --generate-summary-only")
    else:
        if not args.data_dir:
            parser.error("--data-dir is required for evaluation mode")
        if not args.model:
            parser.error("--model is required for evaluation mode")
        if not args.output:
            parser.error("--output is required for evaluation mode")
    
    return main(args)


if __name__ == "__main__":
    exit(run())
