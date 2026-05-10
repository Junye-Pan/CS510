"""
Utility functions for critic training.

This module provides functions for generating critiques and creating GRPO datasets.
These functions are extracted to avoid unsloth dependencies.
"""
from typing import List, Dict, Tuple, Optional
from datasets import Dataset
from pathlib import Path
from jinja2 import Template
import json
import gc
import torch


# Simplified critic prompt - everything in user message, no system message
# No binary True/False requirement - just ask for critique
# This works better with small models that don't distinguish system/user well
CRITIC_SYSTEM_MESSAGE = None  # Not used anymore - everything goes in user message

# Load critique prompt template
CRITIQUE_PROMPT_TEMPLATE_PATH = Path(__file__).parent / "critique_prompt.jinja2"
with open(CRITIQUE_PROMPT_TEMPLATE_PATH, "r") as f:
    CRITIQUE_PROMPT_TEMPLATE = Template(f.read())


def generate_critique_prompt(question: str, option_a: str, option_b: str, answer: str) -> list:
    """
    Generate messages for the model to critique an answer (A/B comparison format).
    Chat template will always be applied to these messages.

    IMPORTANT: The model only sees the question and the specific answer being critiqued,
    NOT both options. This prevents the model from making relative comparisons.
    
    NOTE: We use only user message (no system message) for better compatibility
    with small models and limited SFT data.
    
    Uses "grader" framing which tested best (5/5 success rate) at reducing sycophancy
    and encouraging error detection. Key elements:
    - "grading an exam" context
    - "student often makes mistakes" sets expectation
    - "What errors did the student make?" assumes errors exist
    
    The prompt template is defined in critique_prompt.jinja2 for easy editing.

    Args:
        question: The question text
        option_a: First option
        option_b: Second option
        answer: The answer to critique ('A' or 'B')

    Returns:
        List of message dicts: [{"role": "user", "content": "..."}]
    """
    option_text = option_a if answer == 'A' else option_b
    
    # Use the Jinja2 template for single source of truth
    user_content = CRITIQUE_PROMPT_TEMPLATE.render(
        question=question,
        solution=option_text
    )
    
    return [
        {"role": "user", "content": user_content}
    ]


def generate_critique_prompt_binary(question: str, choice: str) -> list:
    """
    Generate messages for the model to critique a single answer (binary format).
    Chat template will always be applied to these messages.
    
    NOTE: We use only user message (no system message) because:
    1. Small models don't distinguish system/user well
    2. Limited SFT data can't teach this distinction
    3. Simpler format works better
    
    Uses "grader" framing which tested best (5/5 success rate) at reducing sycophancy
    and encouraging error detection. Key elements:
    - "grading an exam" context
    - "student often makes mistakes" sets expectation
    - "What errors did the student make?" assumes errors exist
    
    The prompt template is defined in critique_prompt.jinja2 for easy editing.

    Args:
        question: The question text
        choice: The proposed answer/solution to critique

    Returns:
        List of message dicts: [{"role": "user", "content": "..."}]
    """
    # Use the Jinja2 template for single source of truth
    user_content = CRITIQUE_PROMPT_TEMPLATE.render(
        question=question,
        solution=choice
    )
    
    return [
        {"role": "user", "content": user_content}
    ]


def generate_critiques_vllm(
    model_name: str,
    lora_checkpoint: Optional[str],
    tokenizer,
    dataset: Dataset,
    max_new_tokens: int = 1024,
    batch_size: int = 16,
    return_prompts: bool = False,
    temperature: float = 1.0,
    top_p: float = 0.99,
    top_k: Optional[int] = None,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
) -> List[Tuple[str, str]]:
    """
    Generate critiques for both answers using vLLM for batch inference.

    IMPORTANT: Each critique is generated independently - the model only sees
    the question and one answer at a time, never both options simultaneously.

    Args:
        model_name: Base model name
        lora_checkpoint: Path to LoRA checkpoint (if any)
        tokenizer: Tokenizer
        dataset: Dataset with 'prompt', 'first', 'second' fields
        max_new_tokens: Max tokens to generate
        batch_size: Batch size for inference
        return_prompts: If True, also return the raw messages used (to avoid regenerating)

    Returns:
        If return_prompts=False: List of (critique_a, critique_b) tuples (strings)
        If return_prompts=True: Tuple of (critiques, prompts) where:
            - critiques: List of (critique_a, critique_b) tuples (strings)
            - prompts: List of (messages_a, messages_b) tuples where each is a list of message dicts
                      (raw messages, NOT templated strings - VERL will apply template)
    """
    print(f"\nGenerating critiques with vLLM...")
    print("Note: Each critique sees only the question and one answer (no comparisons)")
    
    # Always apply chat template to messages
    print(f"[Critique Generation] Using chat template format")
    if tokenizer.chat_template is None:
        print(f"  Warning: No chat template found, will use raw messages")

    # Prepare prompts for both options
    # IMPORTANT: Each prompt only contains question + one answer
    prompts_a = []  # Templated strings for vLLM generation
    prompts_b = []
    messages_list_a = []  # Raw messages for return when return_prompts=True
    messages_list_b = []

    for item in dataset:
        # Generate messages (system + user)
        messages_a = generate_critique_prompt(
            item['prompt'], item['first'], item['second'], 'A'
        )
        messages_b = generate_critique_prompt(
            item['prompt'], item['first'], item['second'], 'B'
        )
        
        # Store raw messages for return (to avoid double templating in VERL)
        if return_prompts:
            messages_list_a.append(messages_a)
            messages_list_b.append(messages_b)
        
        # Always apply chat template to messages for vLLM generation
        prompt_a = tokenizer.apply_chat_template(
            messages_a,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_b = tokenizer.apply_chat_template(
            messages_b,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts_a.append(prompt_a)
        prompts_b.append(prompt_b)

    # Generate critiques using vLLM
    from vllm import LLM, SamplingParams

    # Stop generation at </answer_correct> to avoid continuing after the final tag
    # Using stop (string) with include_stop_str_in_output=True to include the tag in output
    sampling_params_kwargs = {
        "temperature": temperature,
        "max_tokens": max_new_tokens,
        "top_p": top_p,
        "min_p": min_p,
        "repetition_penalty": repetition_penalty
    }
    if top_k is not None:
        sampling_params_kwargs["top_k"] = top_k
    if presence_penalty != 0.0:
        sampling_params_kwargs["presence_penalty"] = presence_penalty
    
    sampling_params = SamplingParams(**sampling_params_kwargs)

    # Load model with error handling and cleanup
    llm = None
    try:
        if lora_checkpoint:
            # Read LoRA rank from checkpoint to avoid dimension mismatch
            import json
            from pathlib import Path
            adapter_config_path = Path(lora_checkpoint) / "adapter_config.json"
            if adapter_config_path.exists():
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                    lora_rank = adapter_config.get("r", 32)  # Default to 64 if not found
            else:
                lora_rank = 32  # Fallback default
            print(f"Using LoRA rank from checkpoint: {lora_rank}")
            
            llm = LLM(
                model=model_name,
                enable_lora=True,
                max_lora_rank=lora_rank,  # Use rank from checkpoint
                tensor_parallel_size=1,
                gpu_memory_utilization=0.85,  # Leave some headroom
            )
            from vllm.lora.request import LoRARequest
            lora_request = LoRARequest("critic", 1, lora_checkpoint)
        else:
            llm = LLM(
                model=model_name, 
                tensor_parallel_size=1,
                gpu_memory_utilization=0.85,  # Leave some headroom
            )
            lora_request = None

        # Generate critiques for option A
        print(f"Generating critiques for option A...")
        outputs_a = llm.generate(prompts_a, sampling_params, lora_request=lora_request)
        critiques_a = [output.outputs[0].text.strip() for output in outputs_a]
        
        # Generate critiques for option B
        print(f"Generating critiques for option B...")
        outputs_b = llm.generate(prompts_b, sampling_params, lora_request=lora_request)
        critiques_b = [output.outputs[0].text.strip() for output in outputs_b]
    finally:
        # Clean up vLLM engine to free GPU memory using proper shutdown
        if llm is not None:
            try:
                print("Cleaning up vLLM engine...")
                # Use vLLM's proper shutdown mechanism to terminate worker processes
                if hasattr(llm, 'llm_engine') and llm.llm_engine is not None:
                    engine = llm.llm_engine
                    if hasattr(engine, 'engine_core') and engine.engine_core is not None:
                        print("  → Calling engine_core.shutdown()...")
                        engine.engine_core.shutdown()
                        print("  ✓ engine_core shutdown complete")
                
                # Delete the object and clean up
                del llm
                import torch
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                print("✓ vLLM engine cleanup complete")
            except Exception as e:
                print(f"Warning: Error cleaning up vLLM engine: {e}")

    # Zip critiques: [(item0_critique_a, item0_critique_b), ...]
    critiques = list(zip(critiques_a, critiques_b))

    print(f"✓ Generated {len(prompts_a)} items")

    if return_prompts:
        # Return raw messages (not templated strings) to avoid double templating in VERL
        # VERL's SingleTurnAgentLoop will apply the chat template itself
        prompts = list(zip(messages_list_a, messages_list_b))
        return critiques, prompts
    else:
        return critiques


def generate_critiques_vllm_binary(
    model_name: str,
    lora_checkpoint: Optional[str],
    tokenizer,
    dataset: Dataset,
    max_new_tokens: int = 1024,
    batch_size: int = 16,
    return_prompts: bool = False,
    temperature: float = 1.0,
    top_p: float = 0.99,
    top_k: Optional[int] = None,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
) -> List[str]:
    """
    Generate critiques for single answers using vLLM for batch inference (binary format).

    Args:
        model_name: Base model name
        lora_checkpoint: Path to LoRA checkpoint (if any)
        tokenizer: Tokenizer
        dataset: Dataset with 'question', 'choice' fields
        max_new_tokens: Max tokens to generate
        batch_size: Batch size for inference
        return_prompts: If True, also return the raw messages used
        temperature, top_p, etc.: Generation parameters

    Returns:
        If return_prompts=False: List of critique strings
        If return_prompts=True: Tuple of (critiques, prompts) where:
            - critiques: List of critique strings
            - prompts: List of raw message dicts (NOT templated - VERL will apply template)
    """
    print(f"\nGenerating critiques with vLLM (binary format)...")
    
    # Always apply chat template to messages
    print(f"[Critique Generation] Using chat template format")
    if tokenizer.chat_template is None:
        print(f"  Warning: No chat template found, will use raw messages")

    # Prepare prompts for each answer
    prompts = []  # Templated strings for vLLM generation
    messages_list = []  # Raw messages for return when return_prompts=True

    for item in dataset:
        # Generate messages (system + user)
        messages = generate_critique_prompt_binary(
            item['question'], item['choice']
        )
        
        # Store raw messages for return (to avoid double templating in VERL)
        if return_prompts:
            messages_list.append(messages)
        
        # Always apply chat template to messages for vLLM generation
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)

    # Generate critiques using vLLM
    from vllm import LLM, SamplingParams

    sampling_params_kwargs = {
        "temperature": temperature,
        "max_tokens": max_new_tokens,
        "top_p": top_p,
        "min_p": min_p,
        "repetition_penalty": repetition_penalty
    }
    if top_k is not None:
        sampling_params_kwargs["top_k"] = top_k
    if presence_penalty != 0.0:
        sampling_params_kwargs["presence_penalty"] = presence_penalty
    
    sampling_params = SamplingParams(**sampling_params_kwargs)

    # Load model with error handling and cleanup
    llm = None
    try:
        if lora_checkpoint:
            # Read LoRA rank from checkpoint
            import json
            from pathlib import Path
            adapter_config_path = Path(lora_checkpoint) / "adapter_config.json"
            if adapter_config_path.exists():
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                    lora_rank = adapter_config.get("r", 32)
            else:
                lora_rank = 32
            print(f"Using LoRA rank from checkpoint: {lora_rank}")
            
            llm = LLM(
                model=model_name,
                enable_lora=True,
                max_lora_rank=lora_rank,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.85,
            )
            from vllm.lora.request import LoRARequest
            lora_request = LoRARequest("critic", 1, lora_checkpoint)
        else:
            llm = LLM(
                model=model_name, 
                tensor_parallel_size=1,
                gpu_memory_utilization=0.85,
            )
            lora_request = None

        # Generate critiques
        print(f"Generating critiques for {len(prompts)} samples...")
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
        critiques = [output.outputs[0].text.strip() for output in outputs]
    finally:
        # Clean up vLLM engine
        if llm is not None:
            try:
                print("Cleaning up vLLM engine...")
                if hasattr(llm, 'llm_engine') and llm.llm_engine is not None:
                    engine = llm.llm_engine
                    if hasattr(engine, 'engine_core') and engine.engine_core is not None:
                        print("  → Calling engine_core.shutdown()...")
                        engine.engine_core.shutdown()
                        print("  ✓ engine_core shutdown complete")
                
                del llm
                import torch
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                print("✓ vLLM engine cleanup complete")
            except Exception as e:
                print(f"Warning: Error cleaning up vLLM engine: {e}")

    print(f"✓ Generated {len(critiques)} critiques")

    if return_prompts:
        return critiques, messages_list
    else:
        return critiques


def create_grpo_dataset_for_critic(
    dataset: Dataset,
    critiques: List[Tuple[str, str]],  # List of (critique_a, critique_b) tuples (strings)
    prompts: Optional[List[Tuple[list, list]]] = None,  # Optional: (messages_a, messages_b) tuples (raw messages, not templated)
    tokenizer=None,
    random_sample_option: bool = True,  # If True, randomly sample A or B for each prompt
    seed: Optional[int] = None,  # Random seed for option sampling
) -> Dataset:
    """
    Create GRPO dataset for critic training.
    
    By default, each question generates 2 prompts (one for option A, one for option B).
    If random_sample_option=True, randomly samples one option (A or B) per question.
    Each prompt includes a reference critique from the other option.
    
    Args:
        dataset: Original dataset with questions and options
        critiques: List of (critique_a, critique_b) tuples (strings)
        prompts: Optional pre-computed raw messages (messages_a, messages_b) tuples.
                 Each message is a list of dicts: [{"role": "system", "content": "..."}, ...]
                 NOT templated strings! VERL will apply the chat template.
        tokenizer: Tokenizer (for generating messages if prompts not provided)
        random_sample_option: If True, randomly sample one option (A or B) per question instead of using both
        seed: Random seed for option sampling (for reproducibility)
    
    Returns:
        Dataset with 'prompt' field containing raw messages (list of dicts) for GRPO training.
        VERL's SingleTurnAgentLoop will apply the chat template correctly.
    """
    import random
    
    if seed is not None:
        random.seed(seed)
    
    grpo_data = []
    
    for i, item in enumerate(dataset):
        critique_a, critique_b = critiques[i]  # Always strings (step4 normalizes when loading)
        
        # Get prompts if provided, otherwise generate them
        if prompts:
            # Prompts are now raw messages (list of dicts) from generate_critiques_vllm
            messages_a, messages_b = prompts[i]
        else:
            # Generate raw messages (not templated) - VERL will apply template
            messages_a = generate_critique_prompt(
                item['prompt'], item['first'], item['second'], 'A'
            )
            messages_b = generate_critique_prompt(
                item['prompt'], item['first'], item['second'], 'B'
            )
        
        # Randomly sample one option if requested, otherwise use both
        # IMPORTANT: Use the OTHER option's critique as reference (for judge prompt)
        # When training on option A, we generate a new critique for A and use B's critique as reference
        # When training on option B, we generate a new critique for B and use A's critique as reference
        if random_sample_option:
            # Randomly choose A or B
            chosen_option = random.choice(['A', 'B'])
            if chosen_option == 'A':
                grpo_data.append({
                    'prompt': messages_a,  # Raw messages (list of dicts)
                    'critique_other_option': critique_b,  # Use B's critique as reference for other option
                    'option': 'A',
                })
            else:
                grpo_data.append({
                    'prompt': messages_b,  # Raw messages (list of dicts)
                    'critique_other_option': critique_a,  # Use A's critique as reference for other option
                    'option': 'B',
                })
        else:
            # Store raw messages (not templated strings) so VERL can apply template correctly
            # This prevents double templating when VERL's SingleTurnAgentLoop applies the template
            grpo_data.append({
                'prompt': messages_a,  # Raw messages (list of dicts)
                'critique_other_option': critique_b,  # Use B's critique as reference for other option
                'option': 'A',
            })
            grpo_data.append({
                'prompt': messages_b,  # Raw messages (list of dicts)
                'critique_other_option': critique_a,  # Use A's critique as reference for other option
                'option': 'B',
            })
    
    return Dataset.from_list(grpo_data)


def create_grpo_dataset_for_critic_binary(
    dataset: Dataset,
    critiques: List[str],  # List of critique strings (one per sample)
    prompts: Optional[List[list]] = None,  # Optional: raw messages (not templated)
    tokenizer=None,
    seed: Optional[int] = None,
) -> Dataset:
    """
    Create GRPO dataset for critic training (binary format).
    
    In binary format, each question has ONE answer and ONE critique.
    During RL, the critic gets rewarded if pointing out problems
    helps the judge correctly identify wrong answers.
    
    Args:
        dataset: Original dataset with questions and single answers (binary format)
        critiques: List of critique strings (one per sample)
        prompts: Optional pre-computed raw messages.
                 Each message is a list of dicts: [{"role": "system", "content": "..."}, ...]
        tokenizer: Tokenizer (for generating messages if prompts not provided)
        seed: Random seed (for reproducibility)
    
    Returns:
        Dataset with 'prompt' field containing raw messages for GRPO training.
    """
    import random
    
    if seed is not None:
        random.seed(seed)
    
    grpo_data = []
    
    for i, item in enumerate(dataset):
        critique = critiques[i]
        
        # Get prompts if provided, otherwise generate them
        if prompts:
            messages = prompts[i]
        else:
            # Generate raw messages (not templated) - VERL will apply template
            messages = generate_critique_prompt_binary(
                item['question'], item['choice']
            )
        
        # For binary format: no "other option" critique
        # The reward will be based on whether critique helps identify wrong answers
        grpo_data.append({
            'prompt': messages,  # Raw messages (list of dicts)
            'question': item['question'],
            'choice': item['choice'],
            'label': item['label'],  # bool: True=correct, False=incorrect (converted to int: 0=False, 1=True)
            'reference_critique': critique,  # Initial critique for reference
        })
    
    return Dataset.from_list(grpo_data)


def format_judge_dataset_binary(
    dataset: Dataset,
    critiques: List[str],
) -> Dataset:
    """
    Add critique field to dataset for judge training (binary format).
    
    The actual formatting will be done by format_classification_as_causal()
    using BINARY_JUDGE_PROMPT_TEMPLATE when use_judge_template=True (binary format is auto-detected).

    Args:
        dataset: Original dataset with 'question', 'choice', 'label' fields
        critiques: List of critique strings (one per sample)

    Returns:
        Dataset with added 'critique' field
    """
    formatted = []

    for i, item in enumerate(dataset):
        critique = critiques[i]

        formatted.append({
            'question': item['question'],
            'choice': item['choice'],
            'label': item['label'],
            'critique': critique,
        })

    return Dataset.from_list(formatted)


def log_gpu_memory(prefix: str = "", include_nvidia_smi: bool = True) -> dict:
    """Log detailed GPU memory usage for debugging.
    
    Args:
        prefix: Optional prefix for the log message
        include_nvidia_smi: If True, also query nvidia-smi for GPU-wide memory
                           (shows memory from ALL processes, not just current one)
        
    Returns:
        Dictionary with memory stats
    """
    import torch
    import subprocess
    import os
    
    if not torch.cuda.is_available():
        print(f"{prefix}CUDA not available")
        return {}
    
    # Get memory stats for CURRENT PROCESS only
    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    reserved = torch.cuda.memory_reserved() / 1024**3  # GB
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
    
    # Get device properties for total memory
    device = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
    free = total - reserved
    
    stats = {
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'max_allocated_gb': max_allocated,
        'total_gb': total,
        'free_gb': free,
    }
    
    print(f"{prefix}GPU Memory Status (current process):")
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved:  {reserved:.2f} GB")
    print(f"  Max Allocated: {max_allocated:.2f} GB")
    print(f"  Total:     {total:.2f} GB")
    print(f"  Free (from reserved): {free:.2f} GB")
    
    # Also get GPU-wide memory usage via nvidia-smi
    if include_nvidia_smi:
        try:
            # Get the GPU index we're using
            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
            gpu_idx = cuda_visible.split(',')[0] if cuda_visible else '0'
            
            # Query nvidia-smi for actual GPU memory usage
            result = subprocess.run(
                ['nvidia-smi', f'--id={gpu_idx}', '--query-gpu=memory.used,memory.free,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                if len(parts) == 3:
                    gpu_used = float(parts[0].strip()) / 1024  # MiB to GB
                    gpu_free = float(parts[1].strip()) / 1024
                    gpu_total = float(parts[2].strip()) / 1024
                    stats['nvidia_used_gb'] = gpu_used
                    stats['nvidia_free_gb'] = gpu_free
                    stats['nvidia_total_gb'] = gpu_total
                    print(f"  [nvidia-smi] GPU-wide: used={gpu_used:.2f} GB, free={gpu_free:.2f} GB")
            
            # Also check what processes are using GPU memory
            result = subprocess.run(
                ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                print(f"  [nvidia-smi] Processes using GPU:")
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.strip().split(',')
                        if len(parts) >= 2:
                            pid = parts[0].strip()
                            mem_mib = parts[1].strip()
                            print(f"    PID {pid}: {float(mem_mib)/1024:.2f} GB")
        except Exception as e:
            print(f"  [nvidia-smi] Could not query: {e}")
    
    return stats

def check_training_checkpoint(checkpoint_file: Path) -> Optional[Dict]:
    """
    Check if a training checkpoint exists and load it.

    Args:
        checkpoint_file: Path to checkpoint JSON file

    Returns:
        Checkpoint data if valid, None otherwise
    """
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            # Validate required fields exist
            if data and isinstance(data, dict):
                return data
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load checkpoint {checkpoint_file}: {e}")
    return None


def format_judge_dataset(
    dataset: Dataset,
    critiques: List[Tuple[str, str]],
) -> Dataset:
    """
    Add critique fields to dataset for judge training.
    
    The actual formatting will be done by format_classification_as_causal()
    using JUDGE_PROMPT_TEMPLATE when use_judge_template=True.

    Args:
        dataset: Original dataset with 'prompt', 'first', 'second', 'label' fields
        critiques: List of (critique_a, critique_b) tuples

    Returns:
        Dataset with added 'critique_a' and 'critique_b' fields
    """
    formatted = []

    for i, item in enumerate(dataset):
        critique_a, critique_b = critiques[i]

        formatted.append({
            'prompt': item['prompt'],  # Keep original question
            'first': item['first'],     # Keep option A
            'second': item['second'],   # Keep option B
            'label': item['label'],
            'critique_a': critique_a,
            'critique_b': critique_b,
        })

    return Dataset.from_list(formatted)


def get_judge_predictions_vllm(
    model_name: str,
    lora_checkpoint: str,
    tokenizer,
    dataset: Dataset,
    gpu_memory_utilization: float = 0.8,
    max_ctx: int = 10240,  # Context length for judge prompts (8192 + 1024*2 for two critiques)
) -> List[str]:
    """
    Get judge predictions using vLLM.

    Args:
        model_name: Base model name
        lora_checkpoint: Judge checkpoint
        tokenizer: Tokenizer
        dataset: Dataset with judge prompts
        gpu_memory_utilization: GPU memory utilization (0.0-1.0, default 0.8)
        max_ctx: Maximum context length (default 10240)

    Returns:
        List of predictions ('A' or 'B')
    """
    from w2s_research.core.data import format_classification_as_causal
    from w2s_research.core import predict_batch_labels
    
    print(f"\nGetting judge predictions with vLLM...")

    # Format dataset for causal LM
    # Use judge template because the judge prompt is already fully formatted
    formatted_dataset, template, label_tokens = format_classification_as_causal(
        dataset,
        tokenizer,
        max_ctx=max_ctx,
        use_judge_template=True,  # Use judge template (prompt already fully formatted)
    )

    # Debug: Check label_tokens
    print(f"DEBUG: label_tokens type: {type(label_tokens)}, value: {label_tokens}")
    if not isinstance(label_tokens, list):
        raise TypeError(f"label_tokens is {type(label_tokens)}, expected list. Value: {label_tokens}")

    # Get label token IDs
    label_token_ids = []
    for tok in label_tokens:
        encoded = tokenizer.encode(tok, add_special_tokens=False)
        if len(encoded) == 0:
            raise ValueError(f"Token '{tok}' encoded to empty list")
        label_token_ids.append(encoded[0])
    print(f"DEBUG: label_token_ids: {label_token_ids}")

    # Use vLLM for fast inference
    result = predict_batch_labels(
        model_name=model_name,
        formatted_dataset=formatted_dataset,
        label_token_ids=label_token_ids,
        tokenizer=tokenizer,
        lora_checkpoint=lora_checkpoint,
        return_probabilities=False,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_ctx,  # Pass max_ctx to vLLM to handle longer prompts
    )

    # Debug: Check result type
    if not isinstance(result, dict):
        print(f"ERROR: predict returned {type(result)}, expected dict")
        print(f"Result value: {result}")
        raise TypeError(f"predict returned {type(result)}, expected dict")
    
    if 'predictions' not in result:
        print(f"ERROR: result dict missing 'predictions' key. Keys: {result.keys()}")
        print(f"Result: {result}")
        raise KeyError(f"result dict missing 'predictions' key. Keys: {result.keys()}")
    
    predictions = result['predictions']  # List of integers (0 or 1)
    
    # Debug: Check predictions type
    if not isinstance(predictions, list):
        print(f"ERROR: predictions is {type(predictions)}, expected list")
        print(f"Predictions value: {predictions}")
        raise TypeError(f"predictions is {type(predictions)}, expected list")
    
    # Convert integer predictions to 'A' or 'B' strings
    # label_tokens should be ['A', 'B'], so index 0 = 'A', index 1 = 'B'
    predictions_str = [label_tokens[pred] for pred in predictions]

    print(f"✓ Generated {len(predictions_str)} judge predictions")

    return predictions_str


def get_judge_predictions_vllm_binary(
    model_name: str,
    lora_checkpoint: str,
    tokenizer,
    dataset: Dataset,
    gpu_memory_utilization: float = 0.8,
    max_ctx: int = 10240,
    cached_engine=None,
) -> List[str]:
    """
    Get judge predictions using vLLM (binary format: True/False).

    Args:
        model_name: Base model name
        lora_checkpoint: Judge checkpoint
        tokenizer: Tokenizer
        dataset: Dataset with judge prompts (binary format)
        gpu_memory_utilization: GPU memory utilization (0.0-1.0)
        max_ctx: Maximum context length
        cached_engine: Optional CachedVLLMEngine instance

    Returns:
        List of predictions ('True' or 'False') - True=correct, False=incorrect (no space prefix)
    """
    from w2s_research.core.data import format_classification_as_causal
    from w2s_research.core import predict_batch_labels
    
    if cached_engine is None:
        print(f"\nGetting judge predictions with vLLM (binary format)...")
    else:
        print(f"\nGetting judge predictions with cached vLLM engine (binary format)...")

    # Format dataset for causal LM with binary format
    # Note: use_binary_format is auto-detected by format_classification_as_causal
    # based on whether the dataset has 'question'/'choice' fields
    formatted_dataset, template, label_tokens = format_classification_as_causal(
        dataset,
        tokenizer,
        max_ctx=max_ctx,
        use_judge_template=True,
    )

    # Debug: Check label_tokens
    print(f"DEBUG: label_tokens type: {type(label_tokens)}, value: {label_tokens}")
    if not isinstance(label_tokens, list):
        raise TypeError(f"label_tokens is {type(label_tokens)}, expected list. Value: {label_tokens}")

    # Get label token IDs
    label_token_ids = []
    for tok in label_tokens:
        encoded = tokenizer.encode(tok, add_special_tokens=False)
        if len(encoded) == 0:
            raise ValueError(f"Token '{tok}' encoded to empty list")
        label_token_ids.append(encoded[0])
    print(f"DEBUG: label_token_ids: {label_token_ids}")

    # Use cached engine if provided, otherwise create new one
    if cached_engine is not None:
        result = cached_engine.predict(
            formatted_dataset=formatted_dataset,
            label_token_ids=label_token_ids,
            return_probabilities=False,
        )
    else:
        result = predict_batch_labels(
            model_name=model_name,
            formatted_dataset=formatted_dataset,
            label_token_ids=label_token_ids,
            tokenizer=tokenizer,
            lora_checkpoint=lora_checkpoint,
            return_probabilities=False,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_ctx,
        )

    # Debug: Check result type
    if not isinstance(result, dict):
        print(f"ERROR: predict returned {type(result)}, expected dict")
        raise TypeError(f"predict returned {type(result)}, expected dict")
    
    if 'predictions' not in result:
        raise KeyError(f"result dict missing 'predictions' key. Keys: {result.keys()}")
    
    predictions = result['predictions']  # List of integers (0 or 1)
    
    # Convert integer predictions to 'True' or 'False' strings (no space prefix)
    # label_tokens should be ['False', 'True'], so index 0 = 'False' (incorrect), index 1 = 'True' (correct)
    predictions_str = [label_tokens[pred] for pred in predictions]

    print(f"✓ Generated {len(predictions_str)} judge predictions (binary)")

    return predictions_str


def find_latest_verl_checkpoint(output_dir: str, return_lora_path: bool = True) -> Optional[str]:
    """
    Find the latest VERL checkpoint in the output directory.
    
    VERL saves checkpoints as: {default_local_dir}/global_step_{step}/actor/
    LoRA adapters are saved in: {default_local_dir}/global_step_{step}/actor/lora_adapter/
    
    Args:
        output_dir: Base output directory (trainer.default_local_dir)
        return_lora_path: If True, return path to lora_adapter subdirectory (for vLLM).
                         If False, return path to actor directory.
        
    Returns:
        Path to latest LoRA adapter directory (if return_lora_path=True) or actor directory (if False),
        or None if not found
    """
    from pathlib import Path
    
    checkpoint_base = Path(output_dir)
    
    if not checkpoint_base.exists():
        return None
    
    # Use VERL's tracker file to find latest checkpoint
    try:
        from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
        latest_global_step = find_latest_ckpt_path(str(checkpoint_base))
        if latest_global_step:
            actor_checkpoint = Path(latest_global_step) / "actor"
            if actor_checkpoint.exists():
                if return_lora_path:
                    lora_adapter_path = actor_checkpoint / "lora_adapter"
                    if lora_adapter_path.exists() and (lora_adapter_path / "adapter_config.json").exists():
                        return str(lora_adapter_path)
                else:
                    return str(actor_checkpoint)
    except Exception:
        pass
    
    return None


def merge_lora_checkpoint(base_model_name: str, lora_checkpoint: str, output_dir: str, merged_model_name: str = "merged_judge_model") -> str:
    """
    Merge LoRA checkpoint into base model and save as a full model.
    
    This is required for GenRM mode because VERL's RewardModelManager
    expects a full model, not a LoRA adapter.
    
    Args:
        base_model_name: Base model name (e.g., "Qwen/Qwen1.5-0.5B")
        lora_checkpoint: Path to LoRA checkpoint directory
        output_dir: Directory to save merged model
        merged_model_name: Name for the merged model directory (default: "merged_judge_model")
        
    Returns:
        str: Path to merged model directory
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    merged_path = Path(output_dir) / merged_model_name
    
    # Check if already merged
    if merged_path.exists() and (merged_path / "config.json").exists():
        print(f"  → Using cached merged model: {merged_path}")
        return str(merged_path)
    
    print(f"  → Merging LoRA checkpoint into base model...")
    print(f"    Base model: {base_model_name}")
    print(f"    LoRA checkpoint: {lora_checkpoint}")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # Load and merge LoRA
    model = PeftModel.from_pretrained(base_model, lora_checkpoint)
    model = model.merge_and_unload()
    
    # Save merged model
    merged_path.mkdir(parents=True, exist_ok=True)
    
    # Save tokenizer - use default <|endoftext|> as EOS (token 151643)
    # This matches the SFT training data where we replaced <|im_end|> with <|endoftext|>
    # Base model knows <|endoftext|> from pre-training, so it can generate it correctly
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    print(f"  → Using default tokenizer.eos_token = '{tokenizer.eos_token}' (token_id={tokenizer.eos_token_id})")
    
    model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    
    print(f"  ✓ Merged model saved to: {merged_path}")
    
    # Clean up to free memory
    del model
    del base_model
    gc.collect()
    torch.cuda.empty_cache()
    
    return str(merged_path)

