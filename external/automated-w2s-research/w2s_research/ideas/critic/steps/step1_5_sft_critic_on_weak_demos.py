"""
Step 1.5: SFT strong model on weak model critiques.

This step:
1. Samples N samples from train_unlabeled
2. Generates critiques using weak instruct model
3. Formats critiques as SFT training dataset
4. SFTs the strong base model on these demonstrations
5. Returns checkpoint path for use in subsequent steps
"""
from pathlib import Path
from typing import Tuple, Dict, Optional
import json
import time
import gc
import torch
import random

from datasets import Dataset
from transformers import AutoTokenizer

from w2s_research.core import RunConfig, find_latest_checkpoint
from w2s_research.ideas.critic.utils import (
    generate_critiques_vllm,
    generate_critiques_vllm_binary,
    log_gpu_memory,
    check_training_checkpoint,
    merge_lora_checkpoint,
)


def detect_binary_format(dataset: Dataset) -> bool:
    """Detect if dataset is in binary format (question/choice) vs comparison format (prompt/first/second)."""
    if len(dataset) > 0 and 'choice' in dataset.column_names:
        return True
    return False



def train_model_for_generation(
    model_name: str,
    train_dataset: Dataset,
    batch_size: int = 4,
    wandb_group: Optional[str] = None,
    lr: float = 2e-4,
    epochs: int = 1,
    warmup_ratio: float = 0.0,
    weight_decay: float = 0.01,
    max_ctx: int = 512,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    load_in_4bit: bool = False,
    output_dir: str = "./results",
    optimizer: str = "adamw_8bit",
    lr_scheduler_type: str = "linear",
    gradient_accumulation_steps: int = 1,
    bf16: bool = False,
    fp16: bool = False,
    seed: int = 42,
) -> Dict:
    """
    Train model for generation tasks (SFT) using Unsloth.
    
    This function is similar to train_model but accepts pre-formatted datasets
    with 'input_ids', 'labels', 'prompt_length' fields (for generation tasks).
    Unlike train_model, it does NOT reformat the dataset.
    
    Args:
        model_name: Name of model to load
        train_dataset: Pre-formatted training dataset with 'input_ids', 'labels', 'prompt_length'
        batch_size: Batch size
        lr: Learning rate
        epochs: Number of epochs
        warmup_ratio: Warmup ratio
        weight_decay: Weight decay
        max_ctx: Maximum context length
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        load_in_4bit: Whether to use 4-bit quantization
        output_dir: Output directory
        optimizer: Optimizer type
        lr_scheduler_type: Learning rate scheduler type
        gradient_accumulation_steps: Gradient accumulation steps
        bf16: Whether to use bfloat16
        fp16: Whether to use float16
        seed: Random seed
    
    Returns:
        Training results dictionary
    """
    from unsloth import FastLanguageModel
    from transformers import TrainingArguments
    from trl import SFTTrainer
    import wandb
    from w2s_research.core.seed_utils import set_seed
    
    # Set random seed
    set_seed(seed)
    
    # Initialize wandb only if not already initialized
    # This allows the calling code to initialize wandb with its own config
    if wandb.run is None:
        run_name = Path(output_dir).name
        wandb.init(
            project="generalization",
            name=run_name,
            group=wandb_group,  # Group related runs together
            dir=output_dir,  # Save wandb logs to the output directory
            config={
                "model_name": model_name,
                "batch_size": batch_size,
                "learning_rate": lr,
                "epochs": epochs,
                "warmup_ratio": warmup_ratio,
                "weight_decay": weight_decay,
                "max_ctx": max_ctx,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "load_in_4bit": load_in_4bit,
                "optimizer": optimizer,
                "lr_scheduler_type": lr_scheduler_type,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "bf16": bf16,
                "fp16": fp16,
                "seed": seed,
            }
        )
        should_finish_wandb = True  # Only finish if we initialized it
    else:
        should_finish_wandb = False  # Don't finish if caller initialized it
    
    # Validate precision mode
    if bf16 and fp16:
        raise ValueError("Cannot use both bf16 and fp16. Choose one or neither (fp32).")
    
    # Determine dtype
    dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else None)
    
    # Load model with Unsloth
    model, _ = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_ctx,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    # NOTE: We use <|endoftext|> as EOS (base model's native EOS, token 151643)
    # instead of <|im_end|> (token 151645) because base model knows how to generate <|endoftext|>
    # The SFT data will have <|endoftext|> at the end instead of <|im_end|>
    
    # Apply LoRA (standard modules, no lm_head needed since we use native EOS)
    print(f"Training {model_name} with LoRA (r={lora_r}) for generation task")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
        use_rslora=True,
        use_dora=False,
        loftq_config=None,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=1,  # Only keep the last checkpoint to save disk space
        adam_beta1=0.9,
        adam_beta2=0.95,
        gradient_checkpointing=True,
        bf16=bf16,
        fp16=fp16,
        report_to="wandb",
        lr_scheduler_type=lr_scheduler_type,
        optim=optimizer,
        seed=seed,
        data_seed=seed,
        remove_unused_columns=False,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    
    # Custom data collator for padding variable-length sequences
    def data_collator(features):
        """Collate and pad variable-length sequences."""
        max_length = max(len(f["input_ids"]) for f in features)
        
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        
        for f in features:
            input_ids = f["input_ids"]
            labels = f["labels"]
            
            padding_length = max_length - len(input_ids)
            
            padded_input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
            padded_labels = labels + [-100] * padding_length
            attention_mask = [1] * len(input_ids) + [0] * padding_length
            
            batch["input_ids"].append(padded_input_ids)
            batch["labels"].append(padded_labels)
            batch["attention_mask"].append(attention_mask)
        
        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
        }
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        data_collator=data_collator,
        packing=False,
        dataset_num_proc=None,
        max_seq_length=max_ctx,
        args=training_args,
    )
    
    # Train
    print(f"Training {model_name} with LoRA (r={lora_r}) for generation task")
    train_result = trainer.train()
    
    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    del model
    torch.cuda.empty_cache()
    
    # Only finish wandb if we initialized it
    if should_finish_wandb:
        wandb.finish()
    
    return {
        "train_result": train_result,
        "tokenizer": tokenizer,
    }


def validate_critique(critique: str) -> bool:
    """
    Validate that critique contains the expected format.
    
    Returns:
        True if critique contains <answer>True</answer> or <answer>False</answer>
    """
    return "<answer>True</answer>" in critique or "<answer>False</answer>" in critique


def create_sft_dataset_from_critiques(
    dataset: Dataset,
    critiques: list,
    prompts: list,
    tokenizer,
    max_sft_samples: int = 1000,
    return_valid_samples: bool = False,
):
    """
    Create SFT training dataset from critique prompts and generated critiques.
    First flattens both options (2x dataset), then downsamples to max_sft_samples.
    
    Args:
        dataset: Original dataset with questions and options
        critiques: List of (critique_a, critique_b) tuples (strings)
        prompts: List of (messages_a, messages_b) tuples where each is a list of message dicts
        tokenizer: Tokenizer for formatting
        max_sft_samples: Maximum number of SFT samples
        return_valid_samples: If True, also return samples list for saving JSON
    
    Returns:
        If return_valid_samples=False: Dataset formatted for SFT training with 'input_ids', 'labels', 'prompt_length' fields
        If return_valid_samples=True: Tuple of (Dataset, samples_list) where samples_list contains dicts with:
            - sample_idx: Original index in dataset
            - option: "A" or "B"
            - item: Original dataset item
            - critique: Critique text
            - messages: Raw messages list
    """
    # Step 1: Flatten both options to create 2x dataset
    flattened_samples = []
    for i, (critique_a, critique_b) in enumerate(critiques):
        item = dataset[i]
        messages_a, messages_b = prompts[i]
        
        # Add option A sample
        flattened_samples.append({
            "sample_idx": i,
            "option": "A",
            "item": item,
            "critique": critique_a,
            "messages": messages_a,
        })
        
        # Add option B sample
        flattened_samples.append({
            "sample_idx": i,
            "option": "B",
            "item": item,
            "critique": critique_b,
            "messages": messages_b,
        })
    
    print(f"  → Flattened to {len(flattened_samples)} samples (2x {len(critiques)} questions)")
    
    # Step 2: Downsample to max_sft_samples
    samples_to_use = flattened_samples
    if len(samples_to_use) > max_sft_samples:
        random.seed(42)  # Fixed seed for reproducibility
        samples_to_use = random.sample(samples_to_use, max_sft_samples)
        print(f"  → Downsampled to {len(samples_to_use)} SFT samples")
    
    # Step 3: Create SFT dataset from samples
    sft_data = []
    for sample in samples_to_use:
        messages = sample["messages"]
        critique = sample["critique"]
        
        # Create training example: messages (system + user) -> assistant (critique)
        messages_with_response = messages + [
            {"role": "assistant", "content": critique}
        ]
        
        # Apply chat template to get formatted text
        formatted = tokenizer.apply_chat_template(
            messages_with_response,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        # FIX: Remove empty <think> tags that Qwen3's chat template adds unconditionally
        # This prevents train/inference mismatch since vLLM doesn't add these tags during inference
        # Use exact string replacement to preserve any leading spaces in labels
        formatted = formatted.replace('<think>\n\n</think>\n\n', '')
        
        # FIX: Replace ONLY the LAST <|im_end|> with <|endoftext|> (the one at end of assistant response)
        # Base model knows <|endoftext|> (token 151643) as EOS from pre-training,
        # but doesn't know <|im_end|> (token 151645) which only appears in chat templates
        # This ensures the model can learn to generate the stop token correctly
        formatted = formatted.replace('<|im_end|>', '<|endoftext|>')
        
        # Remove trailing newline
        formatted = formatted.rstrip('\n')
        
        # Tokenize
        encoding = tokenizer(
            formatted,
            add_special_tokens=False,  # Chat template already adds them
        )
        
        # Get prompt length (everything except the assistant response)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
                
        prompt_encoding = tokenizer(
            prompt,
            add_special_tokens=False,
        )
        
        # Create labels: -100 for prompt tokens, actual tokens for response
        labels = [-100] * len(prompt_encoding["input_ids"]) + encoding["input_ids"][len(prompt_encoding["input_ids"]):]
        
        # Pad labels to match input_ids length
        if len(labels) < len(encoding["input_ids"]):
            labels = labels + [-100] * (len(encoding["input_ids"]) - len(labels))
        
        sft_data.append({
            "input_ids": encoding["input_ids"],
            "labels": labels,
            "prompt_length": len(prompt_encoding["input_ids"]),
        })
    
    if return_valid_samples:
        return Dataset.from_list(sft_data), samples_to_use
    else:
        return Dataset.from_list(sft_data)


def create_sft_dataset_from_critiques_binary(
    dataset: Dataset,
    critiques: list,
    prompts: list,
    tokenizer,
    max_sft_samples: int = 1000,
    return_valid_samples: bool = False,
):
    """
    Create SFT training dataset from critique prompts and generated critiques (binary format).
    Each sample has one critique (not A/B comparison).
    
    Args:
        dataset: Original dataset with questions and choices
        critiques: List of critique strings (one per sample)
        prompts: List of messages lists (one per sample)
        tokenizer: Tokenizer for formatting
        max_sft_samples: Maximum number of SFT samples
        return_valid_samples: If True, also return samples list for saving JSON
    
    Returns:
        If return_valid_samples=False: Dataset formatted for SFT training
        If return_valid_samples=True: Tuple of (Dataset, samples_list)
    """
    # Step 1: Create samples from critiques
    flattened_samples = []
    for i, critique in enumerate(critiques):
        item = dataset[i]
        messages = prompts[i]
        
        flattened_samples.append({
            "sample_idx": i,
            "item": item,
            "critique": critique,
            "messages": messages,
        })
    
    print(f"  → Created {len(flattened_samples)} samples")
    
    # Step 2: Downsample to max_sft_samples
    samples_to_use = flattened_samples
    if len(samples_to_use) > max_sft_samples:
        random.seed(42)  # Fixed seed for reproducibility
        samples_to_use = random.sample(samples_to_use, max_sft_samples)
        print(f"  → Downsampled to {len(samples_to_use)} SFT samples")
    
    # Step 3: Create SFT dataset from samples
    sft_data = []
    for sample in samples_to_use:
        messages = sample["messages"]
        critique = sample["critique"]
        
        # Create training example: messages (system + user) -> assistant (critique)
        messages_with_response = messages + [
            {"role": "assistant", "content": critique}
        ]
        
        # Apply chat template to get formatted text
        formatted = tokenizer.apply_chat_template(
            messages_with_response,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        # FIX: Remove empty <think> tags that Qwen3's chat template adds unconditionally
        # Use exact string replacement to preserve any leading spaces in labels
        formatted = formatted.replace('<think>\n\n</think>\n\n', '')
        
        # FIX: Replace <|im_end|> with <|endoftext|>
        formatted = formatted.replace('<|im_end|>', '<|endoftext|>')
        
        # Remove trailing newline
        formatted = formatted.rstrip('\n')
        
        # Tokenize
        encoding = tokenizer(
            formatted,
            add_special_tokens=False,
        )
        
        # Get prompt length (everything except the assistant response)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
                
        prompt_encoding = tokenizer(
            prompt,
            add_special_tokens=False,
        )
        
        # Create labels: -100 for prompt tokens, actual tokens for response
        labels = [-100] * len(prompt_encoding["input_ids"]) + encoding["input_ids"][len(prompt_encoding["input_ids"]):]
        
        # Pad labels to match input_ids length
        if len(labels) < len(encoding["input_ids"]):
            labels = labels + [-100] * (len(encoding["input_ids"]) - len(labels))
        
        sft_data.append({
            "input_ids": encoding["input_ids"],
            "labels": labels,
            "prompt_length": len(prompt_encoding["input_ids"]),
        })
    
    if return_valid_samples:
        return Dataset.from_list(sft_data), samples_to_use
    else:
        return Dataset.from_list(sft_data)


def sft_strong_model_on_weak_critiques(
    config: RunConfig,
    train_unlabel_ds: Dataset,
    test_ds: Dataset,  # Added for testing
    strong_tokenizer: AutoTokenizer,
    checkpoint_dir: Path,
    dataset_name: str,
    config_key: str,
    experiment_uid: str = "",
    wandb_group: str = "",  # Separate wandb group name (can include model names)
) -> Tuple[Optional[str], Dict]:
    """
    Step 1.5: SFT strong model on weak model critiques.
    
    Args:
        config: Run configuration
        train_unlabel_ds: Training unlabeled dataset
        strong_tokenizer: Tokenizer for strong model
        checkpoint_dir: Directory for checkpoints
        dataset_name: Dataset name
        config_key: Configuration key for caching
    
    Returns:
        Tuple of (checkpoint_path, metrics_dict)
        checkpoint_path is None if step is disabled or failed
    """
    print("\n[1.5/12] SFT strong model on weak model critiques...")
    
    # Check if step is enabled
    if not config.enable_sft_critic:
        print("  → Step 1.5 disabled (enable_sft_critic=False)")
        return None, {"enabled": False}
    
    step1_5_checkpoint = checkpoint_dir / "step1_5_sft_critic_checkpoint.json"
    
    # Check cache
    cached_data = check_training_checkpoint(step1_5_checkpoint)
    if cached_data:
        # Prefer merged_checkpoint_path (full HF model) over checkpoint_path (LoRA adapter)
        # VERL requires a full HuggingFace model with config.json, not a LoRA checkpoint
        merged_path = cached_data.get("merged_checkpoint_path")
        lora_path = cached_data.get("checkpoint_path")
        
        if merged_path and Path(merged_path).exists():
            print(f"  → Loading from checkpoint: {step1_5_checkpoint}")
            print(f"  → Found merged checkpoint: {merged_path}")
            print(f"  ✓ SFT training already complete (cached)")
            return merged_path, cached_data
        elif lora_path and Path(lora_path).exists():
            # Fallback to LoRA path (may not work with VERL)
            print(f"  → Loading from checkpoint: {step1_5_checkpoint}")
            print(f"  → Found LoRA checkpoint: {lora_path}")
            print(f"  ⚠️  Warning: Only LoRA checkpoint found, VERL may fail")
            print(f"  ✓ SFT training already complete (cached)")
            return lora_path, cached_data
        
    print(f"  → Generating critiques on entire train_unlabeled (total: {len(train_unlabel_ds)} samples)")
    print(f"  → Will downsample to {config.sft_critic_num_samples} SFT samples")
    
    # Detect format
    is_binary = detect_binary_format(train_unlabel_ds)
    if is_binary:
        print("  → Detected binary format (question/choice)")
    else:
        print("  → Detected comparison format (prompt/first/second)")
    
    # Generate critiques using weak instruct model on entire train_unlabeled
    print(f"  → Generating critiques with weak model: {config.weak_model}")
    
    # Load weak tokenizer
    weak_tokenizer = AutoTokenizer.from_pretrained(config.weak_model)
    
    start_time = time.time()
    
    if is_binary:
        # Binary format: use binary critique generation (single critique per sample)
        critiques, prompts = generate_critiques_vllm_binary(
            model_name=config.weak_model,
            lora_checkpoint=None,
            tokenizer=weak_tokenizer,
            dataset=train_unlabel_ds,
            max_new_tokens=config.gen_max_tokens,
            return_prompts=True,
            temperature=1.0,
            top_p=0.9,
            top_k=config.gen_top_k,
            min_p=config.gen_min_p,
            repetition_penalty=config.gen_repetition_penalty,
            presence_penalty=config.gen_presence_penalty,
        )
        critique_gen_time = time.time() - start_time
        print(f"  ✓ Generated {len(critiques)} critiques in {critique_gen_time:.2f}s")
        
        # Downsample critiques for SFT dataset (binary)
        print("  → Downsampling critiques to SFT dataset...")
        sft_dataset, samples = create_sft_dataset_from_critiques_binary(
            dataset=train_unlabel_ds,
            critiques=critiques,
            prompts=prompts,
            tokenizer=strong_tokenizer,
            max_sft_samples=config.sft_critic_num_samples,
            return_valid_samples=True,
        )
    else:
        # Comparison format: use original critique generation (A/B comparison)
        critiques, prompts = generate_critiques_vllm(
            model_name=config.weak_model,
            lora_checkpoint=None,
            tokenizer=weak_tokenizer,
            dataset=train_unlabel_ds,
            max_new_tokens=config.gen_max_tokens,
            return_prompts=True,
            temperature=1.0,
            top_p=0.9,
            top_k=config.gen_top_k,
            min_p=config.gen_min_p,
            repetition_penalty=config.gen_repetition_penalty,
            presence_penalty=config.gen_presence_penalty,
        )
        critique_gen_time = time.time() - start_time
        print(f"  ✓ Generated {len(critiques)} critique pairs in {critique_gen_time:.2f}s")
        
        # Downsample critiques for SFT dataset (comparison)
        print("  → Downsampling critiques to SFT dataset...")
        sft_dataset, samples = create_sft_dataset_from_critiques(
            dataset=train_unlabel_ds,
            critiques=critiques,
            prompts=prompts,
            tokenizer=strong_tokenizer,
            max_sft_samples=config.sft_critic_num_samples,
            return_valid_samples=True,
        )
    
    if len(sft_dataset) == 0:
        print("  ⚠️  Warning: No critiques available. Falling back to base model.")
        return None, {
            "enabled": True,
            "error": "No critiques available",
            "num_samples": config.sft_critic_num_samples,
            "valid_count": 0,
        }
    
    print(f"  ✓ Created SFT dataset with {len(sft_dataset)} examples")
    
    # Save SFT dataset (original data with critiques, using samples from create_sft_dataset_from_critiques)
    sft_dataset_file = checkpoint_dir / "sft_dataset.json"
    sft_dataset_data = []
    
    # Use samples directly (already flattened and downsampled)
    for sample in samples:
        item = sample["item"]
        if is_binary:
            # Binary format: question/choice/label
            sft_dataset_data.append({
                "question": item.get("question", ""),
                "choice": item.get("choice", ""),
                "label": item.get("label", ""),
                "critique": sample["critique"],
                "prompt": sample["messages"],
                "valid": validate_critique(sample["critique"]),
            })
        else:
            # Comparison format: prompt/first/second/label/option
            sft_dataset_data.append({
                "question": item.get("prompt", ""),
                "first": item.get("first", ""),
                "second": item.get("second", ""),
                "label": item.get("label", ""),
                "option": sample["option"],
                "critique": sample["critique"],
                "prompt": sample["messages"],
                "valid": validate_critique(sample["critique"]),
            })
    
    with open(sft_dataset_file, 'w') as f:
        json.dump(sft_dataset_data, f, indent=2)
    print(f"  ✓ Saved SFT dataset to: {sft_dataset_file} ({len(sft_dataset_data)} samples)")
    
    # Cleanup GPU memory before training
    print("  → Cleaning up GPU memory before SFT training...")
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
    log_gpu_memory("[STEP 1.5] Before SFT training: ")
    
    # SFT strong base model on this dataset
    print(f"  → Starting SFT training on strong model: {config.strong_model}")
    print(f"     Epochs: {config.sft_critic_epochs}, LR: {config.sft_critic_lr}, Batch size: {config.batch_size}")
    
    # Include experiment_uid in output_dir name so wandb run name includes it (aligned with other steps)
    uid_suffix = f"_{experiment_uid}" if experiment_uid else ""
    sft_output_dir = checkpoint_dir / f"sft_critic_on_weak_demos{uid_suffix}"
    sft_output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    try:
        # Use direct SFT training for generation task (not classification)
        # train_model is designed for classification and will reformat our dataset
        sft_results = train_model_for_generation(
            model_name=config.strong_model,
            train_dataset=sft_dataset,  # Already formatted with input_ids, labels, prompt_length
            batch_size=config.batch_size,
            lr=config.sft_critic_lr,
            epochs=config.sft_critic_epochs,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            max_ctx=config.max_ctx + config.gen_max_tokens,  # Question+Answer (max_ctx) + critique (gen_max_tokens)
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            load_in_4bit=config.load_in_4bit,
            output_dir=str(sft_output_dir),
            optimizer=config.optimizer,
            lr_scheduler_type=config.lr_schedule,
            bf16=config.bf16,
            fp16=config.fp16,
            seed=config.seed,
            wandb_group=wandb_group or experiment_uid,  # Group related runs together
        )
        
        sft_training_time = time.time() - start_time
        print(f"  ✓ SFT training completed in {sft_training_time:.2f}s")
        
        # Find latest checkpoint
        checkpoint_path = find_latest_checkpoint(str(sft_output_dir))
        if not checkpoint_path:
            print("  ⚠️  Warning: No checkpoint found after training. Falling back to base model.")
            return None, {
                "enabled": True,
                "error": "No checkpoint found after training",
                "num_samples": config.sft_critic_num_samples,
                "sft_dataset_size": len(sft_dataset),
                "training_time": sft_training_time,
            }
        
        print(f"  ✓ Checkpoint saved: {checkpoint_path}")
        
        # Merge LoRA checkpoint to standard HF checkpoint for GRPO compatibility
        print("\n  → Merging LoRA checkpoint to standard HuggingFace checkpoint...")
        merged_checkpoint_path = merge_lora_checkpoint(
            base_model_name=config.strong_model,
            lora_checkpoint=checkpoint_path,
            output_dir=str(sft_output_dir),
            merged_model_name="merged_sft_critic_model",  # Better name than "merged_judge_model"
        )
        print(f"  ✓ Merged checkpoint saved: {merged_checkpoint_path}")
        
        # Save checkpoint metadata
        metrics = {
            "enabled": True,
            "checkpoint_path": checkpoint_path,  # LoRA checkpoint (for vLLM)
            "merged_checkpoint_path": merged_checkpoint_path,  # Disabled for debugging
            "num_samples": config.sft_critic_num_samples,  # Final SFT dataset size (after downsampling)
            "total_critiques_generated": len(critiques),  # Total critiques generated from entire train_unlabeled
            "sft_dataset_size": len(sft_dataset),
            "critique_gen_time": critique_gen_time,
            "training_time": sft_training_time,
            "weak_model": config.weak_model,
            "strong_model": config.strong_model,
            "sft_epochs": config.sft_critic_epochs,
            "sft_lr": config.sft_critic_lr,
        }
        
        with open(step1_5_checkpoint, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"  ✓ Saved checkpoint metadata: {step1_5_checkpoint}")
        
        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()
        log_gpu_memory("[STEP 1.5 END] ")
        
        # Return merged checkpoint path (standard HF format) for use in GRPO
        # The LoRA checkpoint is also saved in metrics for reference
        return merged_checkpoint_path, metrics
        
    except Exception as e:
        print(f"  ✗ Error during SFT training: {e}")
        import traceback
        traceback.print_exc()
        return None, {
            "enabled": True,
            "error": str(e),
            "num_samples": config.sft_critic_num_samples,
            "sft_dataset_size": len(sft_dataset) if 'sft_dataset' in locals() else 0,
        }

