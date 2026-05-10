"""
vLLM-based inference for fast batch prediction with LoRA support.

This module provides high-performance inference using vLLM, with support for:
- Base models (pretrained models without adapters)
- LoRA-adapted models (fine-tuned with LoRA)
- Batch processing with efficient GPU utilization
- Probability extraction for specific label tokens
"""
import torch
import json
import os
from typing import List, Dict, Optional, Tuple
from datasets import Dataset
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

def get_lora_rank_from_checkpoint(lora_checkpoint: str) -> int:
    """
    Read LoRA rank from adapter_config.json in checkpoint directory.
    
    Args:
        lora_checkpoint: Path to LoRA checkpoint directory
        
    Returns:
        LoRA rank (r parameter)
    """
    adapter_config_path = Path(lora_checkpoint) / "adapter_config.json"
    if not adapter_config_path.exists():
        # Fallback: try parent directory (some checkpoints have adapter_config in parent)
        adapter_config_path = Path(lora_checkpoint).parent / "adapter_config.json"
        if not adapter_config_path.exists():
            print(f"Warning: adapter_config.json not found in {lora_checkpoint}, using default rank=32")
            return 32
    
    with open(adapter_config_path, 'r') as f:
        adapter_config = json.load(f)
    
    lora_rank = adapter_config.get("r", 32)  # Default to 32 if not found
    print(f"Detected LoRA rank from checkpoint: {lora_rank}")
    return lora_rank


def predict_batch_labels(
    model_name: str,
    formatted_dataset: Dataset,
    label_token_ids: List[int],
    tokenizer,
    lora_checkpoint: Optional[str] = None,
    return_probabilities: bool = False,
    return_confidences: bool = False,
    current_labels: Optional[List[int]] = None,
    gpu_memory_utilization: float = 0.8,
    max_model_len: Optional[int] = None,
) -> Dict:
    """
    Convenience function for one-off vLLM inference with automatic batch sizing.

    This function creates a VLLMInferenceEngine, runs inference, and cleans up.
    For multiple inference calls with the same model, create a VLLMInferenceEngine
    instance and reuse it.

    Note: vLLM automatically determines optimal batch size based on available GPU memory.

    Args:
        model_name: Base model name or path to LoRA checkpoint
        formatted_dataset: Dataset with 'input_ids', 'labels', and 'prompt_length'
        label_token_ids: List of token IDs for label tokens
        tokenizer: Tokenizer
        lora_checkpoint: Optional path to LoRA checkpoint directory
        return_probabilities: Whether to return probability distributions
        return_confidences: Whether to return confidence scores
        current_labels: Current labels (required if return_confidences=True)
        gpu_memory_utilization: GPU memory utilization (0.0-1.0, default 0.8)
        max_model_len: Maximum sequence length (default: 8300, or auto-detect from model config)

    Returns:
        Dictionary with predictions, probabilities, and/or confidences
    """

    print('model name = ', model_name)
    print('lora checkpoint = ', lora_checkpoint)
    print('label token ids = ', label_token_ids)
    
    # Get LoRA rank from checkpoint if LoRA is used
    max_lora_rank = 32  # Default
    if lora_checkpoint is not None:
        max_lora_rank = get_lora_rank_from_checkpoint(lora_checkpoint)
    
    # Determine max_model_len: use provided value, or default to 8300
    if max_model_len is None:
        max_model_len = 8300
    
    print(f"Using max_model_len={max_model_len} for vLLM")
    
    # Create engine
    llm = LLM(
        model=model_name,
        max_model_len=max_model_len,
        tensor_parallel_size=1,
        enable_lora=lora_checkpoint is not None,
        max_loras=1,
        max_lora_rank=max_lora_rank,  # Use rank from checkpoint to avoid dimension mismatch
        enforce_eager=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    
    prompts = []
    for i in range(len(formatted_dataset)):
        example = formatted_dataset[i]
        # Use only the prompt part (up to prompt_length)
        prompt_length = example["prompt_length"]
        offset = 0
        if tokenizer.bos_token_id is not None and example["input_ids"][0] == tokenizer.bos_token_id:
            offset = 1
        prompt_ids = example["input_ids"][offset:prompt_length]
        prompts.append(tokenizer.decode(prompt_ids))
    
    sampling_params = SamplingParams(
        max_tokens=1,  # Only generate the label token
        temperature=0.0,  # Greedy decoding
        logprobs=20,  # Request logprobs for top-10 tokens
    )
    
    if lora_checkpoint is not None:
        lora_request = LoRARequest("eval_adapter", 1, lora_checkpoint)
        all_outputs = llm.generate(
            prompts,
            sampling_params,
            lora_request=lora_request,
        )
    else:
        all_outputs = llm.generate(
            prompts,
            sampling_params,
        )

    # Extract predictions and probabilities
    predictions = []
    probabilities = [] if return_probabilities else None
    confidences = [] if return_confidences else None

    for i, output in enumerate(all_outputs):
        # Get logprobs for the generated token
        token_logprobs = output.outputs[0].logprobs[0]

        # Extract logprobs for our label tokens
        label_logprobs = []
        for token_id in label_token_ids:
            if token_id in token_logprobs:
                label_logprobs.append(token_logprobs[token_id].logprob)
            else:
                # If token not in top-k, assign very low logprob
                label_logprobs.append(-100.0)

        # Convert logprobs to probabilities using softmax
        label_logprobs_tensor = torch.tensor(label_logprobs)
        label_probs = torch.softmax(label_logprobs_tensor, dim=0)

        # Get prediction (argmax)
        pred = label_probs.argmax().item()
        predictions.append(pred)

        # Store probabilities if requested
        if return_probabilities:
            probabilities.append(label_probs.cpu().numpy().tolist())

        # Store confidence if requested
        if return_confidences:
            conf = label_probs[current_labels[i]].item()
            confidences.append(conf)

    # Prepare result dictionary
    result = {'predictions': predictions}
    if return_probabilities:
        result['probabilities'] = probabilities
    if return_confidences:
        result['confidences'] = confidences

    # Thorough cleanup of vLLM resources using proper shutdown
    print("Cleaning up vLLM engine...")
    
    # Step 1: Use vLLM's proper shutdown mechanism to terminate worker processes
    try:
        if hasattr(llm, 'llm_engine') and llm.llm_engine is not None:
            engine = llm.llm_engine
            if hasattr(engine, 'engine_core') and engine.engine_core is not None:
                print("  → Calling engine_core.shutdown()...")
                engine.engine_core.shutdown()
                print("  ✓ engine_core shutdown complete")
    except Exception as e:
        print(f"  • engine_core.shutdown() failed: {e}")
    
    # Step 2: Delete the LLM object
    del llm
    
    # Step 3: Force garbage collection
    import gc
    gc.collect()
    
    # Step 4: Empty CUDA cache
    torch.cuda.empty_cache()
    
    # Step 5: Synchronize to ensure all CUDA operations complete
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Step 6: Try to clean up any distributed process groups from vLLM
    try:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception:
        pass  # May not be initialized
    
    # Step 7: Final garbage collection
    gc.collect()
    torch.cuda.empty_cache()
    
    print("✓ vLLM engine cleanup complete")

    return result
