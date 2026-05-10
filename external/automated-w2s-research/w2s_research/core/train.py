"""
Causal LM training with Unsloth for classification via label-as-text.

This approach treats classification as autoregressive generation:
- Input: "Question? (A) opt1 (B) opt2\nAnswer:"
- Target: " B" (only compute loss on label token)
"""
import torch
from pathlib import Path
from transformers import Trainer, TrainingArguments, TrainerCallback, AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSequenceClassification
from trl import SFTTrainer
from typing import Optional, Dict, Any, Tuple
from datasets import Dataset
import json
import os
import wandb

from .data import format_classification_as_causal
from .eval import evaluate_model, print_evaluation_results, generate_predictions
from .seed_utils import set_seed


def is_base_model(model_name: str) -> bool:
    """
    Detect if a model is a base model (not instruction-tuned/chat model).
    
    Base models should use simple text formatting without chat templates,
    as chat templates can cause loss spikes during training.
    
    Args:
        model_name: The model name to check
        
    Returns:
        True if the model appears to be a base model, False otherwise
    """
    model_name_lower = model_name.lower()
    
    # Check for explicit base model indicators
    base_indicators = ['-base', '_base', '/base']
    for indicator in base_indicators:
        if indicator in model_name_lower:
            return True
    
    # Check for chat/instruct model indicators (these are NOT base models)
    chat_indicators = ['-chat', '_chat', '-instruct', '_instruct', '-it', '_it']
    for indicator in chat_indicators:
        if indicator in model_name_lower:
            return False
    
    # Default: assume models without explicit indicators are base models
    # This is safer as chat templates on base models can cause issues
    # Common base models: gpt2, Qwen/Qwen3-4B (without -Chat suffix)
    return True


def normalize_model_name_for_path(model_name: str) -> str:
    """
    Normalize model name for use in file paths.
    Standardizes organization prefixes (e.g., 'Qwen/', 'unsloth/', 'meta-llama/')
    to ensure consistent path construction regardless of which organization's
    version of the model is used.

    For Qwen models specifically, both 'Qwen/' and 'unsloth/' prefixes are
    standardized to 'Qwen/' to match existing checkpoint directories.

    Examples:
        'Qwen/Qwen3-1.7B-Base' -> 'Qwen/Qwen3-1.7B-Base'
        'unsloth/Qwen3-1.7B-Base' -> 'Qwen/Qwen3-1.7B-Base'
        'meta-llama/Llama-3-8B' -> 'meta-llama/Llama-3-8B'
        'gpt2' -> 'gpt2'

    Args:
        model_name: The model name to normalize

    Returns:
        Normalized model name
    """
    if '/' in model_name:
        org, model = model_name.split('/', 1)
        # Standardize unsloth organization to Qwen for Qwen models
        if org == 'unsloth' and model.startswith('Qwen'):
            return f'Qwen/{model}'
    return model_name


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint directory in the output directory.

    Args:
        output_dir: Directory where checkpoints are saved

    Returns:
        Path to latest checkpoint directory, or None if no checkpoints found
    """
    output_path = Path(output_dir)

    if not output_path.exists():
        return None

    # Find all checkpoint directories
    checkpoint_dirs = [d for d in output_path.iterdir()
                      if d.is_dir() and d.name.startswith("checkpoint-")]

    if not checkpoint_dirs:
        return None

    # Sort by checkpoint number
    checkpoint_dirs.sort(key=lambda x: int(x.name.split("-")[1]))

    # Return the latest checkpoint
    latest = checkpoint_dirs[-1]
    print(f"Found checkpoint: {latest}")
    return str(latest)


def load_model_from_checkpoint(
    checkpoint_path: str,
    max_ctx: int = 512,
    load_in_4bit: bool = False,
    bf16: bool = False,
    fp16: bool = False,
) -> Tuple[Any, Any]:
    """
    Load a model and tokenizer from a checkpoint directory.

    Args:
        checkpoint_path: Path to the checkpoint directory
        max_ctx: Maximum context length
        load_in_4bit: Whether to use 4-bit quantization
        bf16: Whether to use bfloat16 precision (if both False, uses fp32)
        fp16: Whether to use float16 precision (if both False, uses fp32)

    Returns:
        Tuple of (model, tokenizer)
    """
    # Check if GPU is available before importing unsloth (which requires GPU)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU not available. Loading models from checkpoints requires a CUDA-capable GPU. "
            "Unsloth library cannot be used without GPU acceleration."
        )

    from unsloth import FastLanguageModel

    print(f"Loading model from checkpoint: {checkpoint_path}")

    # Validate precision mode
    if bf16 and fp16:
        raise ValueError("Cannot use both bf16 and fp16. Choose one or neither (fp32).")

    # Determine dtype based on precision mode
    if bf16:
        dtype = torch.bfloat16
    elif fp16:
        dtype = torch.float16
    else:
        dtype = None  # None = fp32 for most models

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=max_ctx,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    
    adapter_config = json.load(open(os.path.join(checkpoint_path, "adapter_config.json")))

    tokenizer = AutoTokenizer.from_pretrained(adapter_config["base_model_name_or_path"], use_fast=False)


    print("Model loaded successfully from checkpoint")
    return model, tokenizer


class EvaluationCallback(TrainerCallback):
    """Callback to evaluate model with detailed metrics when Trainer runs evaluation."""

    def __init__(self, eval_dataset, label_token_ids, tokenizer, batch_size, original_dataset=None, max_ctx=512):
        self.eval_dataset = eval_dataset
        self.label_token_ids = label_token_ids
        self.tokenizer = tokenizer
        self.original_dataset = original_dataset
        self.batch_size = batch_size
        self.max_ctx = max_ctx
        self.eval_results = []

    def on_evaluate(self, args, state, control, model, metrics=None, **kwargs):
        """Run custom evaluation when Trainer triggers evaluation (epoch end or eval_steps)."""
        print(f"\n\n{'='*80}")
        print(f"Evaluation at Step {state.global_step} (Epoch {state.epoch:.2f})")
        print(f"{'='*80}")

        # Clear GPU cache before evaluation to prevent OOM
        torch.cuda.empty_cache()

        # Find the latest checkpoint (should exist since save_strategy="epoch")
        latest_checkpoint = find_latest_checkpoint(args.output_dir)

        if latest_checkpoint:
            print(f"Using checkpoint for evaluation: {latest_checkpoint}")
            # Extract base model name from the checkpoint's adapter_config.json
            import json
            adapter_config_path = Path(latest_checkpoint) / "adapter_config.json"
            if adapter_config_path.exists():
                with open(adapter_config_path) as f:
                    adapter_config = json.load(f)
                    base_model_name = adapter_config.get("base_model_name_or_path", "")
            else:
                # Fallback: assume model is already specified elsewhere
                base_model_name = model.config._name_or_path

            # Add buffer for special tokens and safety margin (vLLM needs max_model_len >= prompt length)
            # max_ctx is used for Unsloth training, but vLLM evaluation needs extra headroom for long critiques
            max_model_len = max(self.max_ctx + 500, 10240) if self.max_ctx else 10240
            results = evaluate_model(
                test_dataset=self.eval_dataset,
                label_token_ids=self.label_token_ids,
                tokenizer=self.tokenizer,
                model_name=base_model_name,
                lora_checkpoint=latest_checkpoint,
                original_dataset=self.original_dataset,
                max_model_len=max_model_len,
            )
        else:
            # No checkpoint yet (shouldn't happen with save_strategy="epoch")
            print("Warning: No checkpoint found, skipping evaluation")
            return

        # Store results with step and epoch info
        results['global_step'] = state.global_step
        results['epoch'] = state.epoch
        self.eval_results.append(results)

        # Print results
        print_evaluation_results(results, prefix="  ")

        # Log to wandb
        wandb.log({
            "eval/accuracy": results.get('accuracy', 0.0),
            "eval/total_examples": results.get('total', 0),
            "eval/correct_predictions": results.get('correct', 0),
            "global_step": state.global_step,
            "epoch": state.epoch,
        })

        model.train()
        return control


def train_model(
    model_name: str,
    train_dataset: Dataset,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 4,
    lr: float = 2e-4,
    epochs: int = 1,
    warmup_ratio: float = 0.0,
    warmup_steps: int = 0,
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
    trainer_class: Optional[type] = None,
    trainer_kwargs: Optional[Dict[str, Any]] = None,
    bf16: bool = False,
    fp16: bool = False,
    seed: int = 42,
    wandb_group: Optional[str] = None,
    use_chat_template: Optional[bool] = None,
    save_total_limit: Optional[int] = None,
    aar_mode: bool = False,
) -> Dict[str, Any]:
    """
    Train model using Unsloth with causal LM approach for classification.

    Args:
        model_name: Name of model to load
        train_dataset: Training dataset (with 'txt' and 'hard_label')
        test_dataset: Test dataset
        batch_size: Batch size
        lr: Learning rate
        epochs: Number of epochs
        warmup_ratio: Warmup ratio (fraction of total steps, e.g., 0.1 = 10% warmup)
        warmup_steps: Warmup steps (absolute number). If > 0, overrides warmup_ratio
        weight_decay: Weight decay
        max_ctx: Maximum context length
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        load_in_4bit: Whether to use 4-bit quantization
        output_dir: Output directory
        optimizer: Optimizer type (e.g., "adamw", "adamw_8bit")
        lr_scheduler_type: Learning rate scheduler type
        gradient_accumulation_steps: Gradient accumulation steps
        trainer_class: Custom Trainer class (e.g., for custom loss functions). If None, uses standard Trainer.
        trainer_kwargs: Additional keyword arguments to pass to the custom trainer class
        bf16: Whether to use bfloat16 precision (if both False, uses fp32)
        fp16: Whether to use float16 precision (if both False, uses fp32)
        seed: Random seed for reproducibility
        wandb_group: Optional wandb group name for grouping related runs
        use_chat_template: Whether to use chat template formatting. If None (default),
                          auto-detects based on model name: False for base models,
                          True for chat/instruct models.
        aar_mode: If True, returns predictions instead of accuracy (for server-side evaluation).
                 Workers don't have access to ground truth labels in AAR mode.

    Returns:
        Training results dictionary with:
        - train_result: Training metrics
        - final_eval: Final evaluation results with detailed metrics (or predictions if aar_mode)
        - eval_results: Evaluation results at each eval step/epoch (if eval_strategy != "no")
        - model: Trained model (only if return_model=True)
        - tokenizer: Tokenizer
        - predictions: List of predictions (only if aar_mode=True)
    """
    # Auto-detect use_chat_template based on model name if not explicitly set
    if use_chat_template is None:
        use_chat_template = not is_base_model(model_name)
        model_type = "chat/instruct" if use_chat_template else "base"
        print(f"📝 Auto-detected {model_name} as {model_type} model → use_chat_template={use_chat_template}")

    # Check if GPU is available before importing unsloth (which requires GPU)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU not available. Training requires a CUDA-capable GPU. "
            "Unsloth library cannot be used without GPU acceleration."
        )

    # Set random seed for reproducibility
    from unsloth import FastLanguageModel

    # Initialize wandb
    # Extract run name from output directory path
    run_name = Path(output_dir).name

    # Ensure output directory exists for wandb logs
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    wandb.init(
        project="generalization",
        name=run_name,
        group=wandb_group,  # Group related runs together (e.g., judge, grpo, sft)
        dir=output_dir,  # Save wandb logs to results directory for analysis agent access
        config={
            "model_name": model_name,
            "batch_size": batch_size,
            "learning_rate": lr,
            "epochs": epochs,
            "warmup_ratio": warmup_ratio,
            "warmup_steps": warmup_steps,
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

    # Validate precision mode
    if bf16 and fp16:
        raise ValueError("Cannot use both bf16 and fp16. Choose one or neither (fp32).")

    # Determine dtype based on precision mode
    if bf16:
        dtype = torch.bfloat16
    elif fp16:
        dtype = torch.float16
    else:
        dtype = None  # None = fp32 for most models

    # Load model with Unsloth
    model, _ = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_ctx,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)


    # Apply LoRA (only if lora_r >= 1)
    # Unsloth requires r >= 1, so skip PEFT if r is 0 or negative
    if lora_r >= 1:
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
    else:
        # No LoRA - use full model finetuning
        # Just prepare for training without PEFT
        FastLanguageModel.for_training(model)

    # Format datasets for causal LM (handles both binary and comparison formats)
    train_formatted, template, label_tokens = format_classification_as_causal(
        train_dataset,
        tokenizer,
        max_ctx=max_ctx,
        use_judge_template=trainer_kwargs.get("use_judge_template", False) if trainer_kwargs else False,
        dataset_name="train",
        use_chat_template=use_chat_template,
    )

    if test_dataset is not None:
        test_formatted, _, _ = format_classification_as_causal(
            test_dataset,
            tokenizer,
            max_ctx=max_ctx,
            use_judge_template=trainer_kwargs.get("use_judge_template", False) if trainer_kwargs else False,
            dataset_name="test",
            use_chat_template=use_chat_template,
        )
    else:
        test_formatted = None

    # Get label token IDs for evaluation
    label_token_ids = [tokenizer.encode(label_token, add_special_tokens=False)[0] for label_token in label_tokens]

    # Training arguments
    # Note: If warmup_steps > 0, it overrides warmup_ratio
    # If eval_strategy != "no", EvaluationCallback will run at specified intervals
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_accumulation_steps=10,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        warmup_steps=warmup_steps,
        logging_steps=1,  # Log every step for small datasets (already changed earlier)
        eval_strategy="no",
        # eval_steps=0.01,
        save_strategy="epoch",
        adam_beta1=0.9,
        adam_beta2=0.95,
        gradient_checkpointing=True,
        bf16=bf16,
        fp16=fp16,
        report_to="wandb",
        lr_scheduler_type=lr_scheduler_type,
        optim=optimizer,
        seed=seed,
        data_seed=seed,  # Ensure DataLoader workers use the same seed
        remove_unused_columns=False,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_total_limit=save_total_limit,
    )


    # Custom data collator for padding variable-length sequences
    def data_collator(features):
        """Collate and pad variable-length sequences."""
        # Get max length in this batch
        max_length = max(len(f["input_ids"]) for f in features)

        # Pad each feature
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "prompt_length": []
        }

        # Check if soft_labels exist in the features (support both singular and plural)
        has_soft_labels = "soft_labels" in features[0] if features else False
        if has_soft_labels:
            batch["soft_labels"] = []
        
        for f in features:
            input_ids = f["input_ids"]
            labels = f["labels"]

            # Calculate padding
            padding_length = max_length - len(input_ids)

            # Pad input_ids and labels
            padded_input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
            padded_labels = labels + [-100] * padding_length  # -100 for padded positions
            attention_mask = [1] * len(input_ids) + [0] * padding_length

            batch["input_ids"].append(padded_input_ids)
            batch["labels"].append(padded_labels)
            batch["attention_mask"].append(attention_mask)
            batch["prompt_length"].append(f['prompt_length'])
            # Add soft labels if present (support both singular and plural)
            if has_soft_labels:
                batch["soft_labels"].append(f["soft_labels"])
                
        # Convert to tensors
        batch = {k: torch.tensor(v) for k, v in batch.items()}
        return batch

    eval_callback = None
    callbacks = []

    if test_formatted is not None:
        eval_callback = EvaluationCallback(
            eval_dataset=test_formatted,
            label_token_ids=label_token_ids,
            tokenizer=tokenizer,
            batch_size=batch_size,
            original_dataset=test_dataset,
            max_ctx=max_ctx,
        )
        callbacks = [eval_callback]

    # Merge callbacks from trainer_kwargs if provided
    if trainer_kwargs and "callbacks" in trainer_kwargs:
        extra_callbacks = trainer_kwargs.pop("callbacks")
        if extra_callbacks:
            callbacks.extend(extra_callbacks)

    # Allow overriding data_collator from trainer_kwargs if provided
    if trainer_kwargs and "data_collator" in trainer_kwargs:
        custom_data_collator = trainer_kwargs.pop("data_collator")
        data_collator = custom_data_collator

    # Create trainer (use custom trainer_class if provided)
    if trainer_class is not None:
        # Use custom trainer class (e.g., for custom loss functions)
        print(f"Using custom trainer: {trainer_class.__name__}")

        # Prepare trainer kwargs
        custom_kwargs = trainer_kwargs or {}

        trainer = trainer_class(
            model=model,
            args=training_args,
            train_dataset=train_formatted,
            eval_dataset=test_formatted,
            data_collator=data_collator,
            # data_collator=DataCollatorForSeq2Seq(tokenizer = tokenizer, padding=True),
            callbacks=callbacks,
            **custom_kwargs,
        )
    else:
        # Use standard Trainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_formatted,
            eval_dataset=test_formatted,
            data_collator=data_collator,
            daataset_num_proc=None,
            max_seq_length=max_ctx,
            # data_collator=DataCollatorForSeq2Seq(tokenizer = tokenizer, padding=True),
            args=training_args,
            callbacks=callbacks,
        )

    # Train
    print(f"Training {model_name} with LoRA (r={lora_r})")
    print(f"Label tokens: {label_token_ids}")

    train_result = trainer.train()

    del model
    torch.cuda.empty_cache()
    print(f"[Memory] GPU memory freed")

    print("\nGPU memory after finish training:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


    result = {
        "train_result": train_result,
        "tokenizer": tokenizer,
    }

    # Add evaluation results if callback was used
    if eval_callback is not None and len(eval_callback.eval_results) > 0:
        result["eval_results"] = eval_callback.eval_results
        result['final_eval'] = result['eval_results'][-1]
    elif test_formatted is not None:
        # Final evaluation with detailed metrics (when no callback or eval_strategy="no")
        print("\n" + "="*80)
        if aar_mode:
            print("FINAL EVALUATION (AAR Mode - Predictions Only)")
        else:
            print("FINAL EVALUATION")
        print("="*80)

        # Clear GPU cache before final evaluation to prevent OOM
        torch.cuda.empty_cache()

        # Find the latest checkpoint for vLLM evaluation
        latest_checkpoint = find_latest_checkpoint(output_dir)
        print(f"model_name: {model_name}")
        print(f"tokenizer: {tokenizer}")
        if latest_checkpoint:
            print(f"Using checkpoint for final evaluation: {latest_checkpoint}")
            # Add buffer for special tokens and safety margin (vLLM needs max_model_len >= prompt length)
            # max_ctx is used for Unsloth training, but vLLM evaluation needs extra headroom for long critiques
            max_model_len = max(max_ctx + 500, 10240) if max_ctx else 10240
            
            if aar_mode:
                # AAR Mode: Generate predictions only (no ground truth labels available)
                pred_result = generate_predictions(
                    test_dataset=test_formatted,
                    label_token_ids=label_token_ids,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    lora_checkpoint=latest_checkpoint,
                    max_model_len=max_model_len,
                )
                result['predictions'] = pred_result['predictions']
                final_eval = {
                    "num_samples": pred_result['num_samples'],
                    "pred_distribution": pred_result['pred_distribution'],
                    "aar_mode": True,
                }
                print(f"  [AAR Mode] Generated {len(result['predictions'])} predictions")
                print(f"  Prediction distribution: {pred_result['pred_distribution']}")
            else:
                # Normal mode: Evaluate with ground truth labels
                final_eval = evaluate_model(
                    test_dataset=test_formatted,
                    label_token_ids=label_token_ids,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    lora_checkpoint=latest_checkpoint,
                    original_dataset=test_dataset,  # Pass original dataset for efficient label extraction
                    max_model_len=max_model_len,
                )
        else:
            print("Warning: No checkpoint found for final evaluation")
            if aar_mode:
                final_eval = {"num_samples": 0, "aar_mode": True}
                result['predictions'] = []
            else:
                final_eval = {"accuracy": 0.0, "correct": 0, "total": 0}

        result['final_eval'] = final_eval
        if not aar_mode:
            print_evaluation_results(final_eval, prefix="  ")

    # Log final evaluation metrics to wandb
    if 'final_eval' in result:
        if aar_mode:
            # AAR mode: only log prediction count
            wandb.log({
                "final/num_samples": result['final_eval'].get('num_samples', 0),
                "final/aar_mode": True,
            })
        else:
            wandb.log({
                "final/accuracy": result['final_eval'].get('accuracy', 0.0),
                "final/total_examples": result['final_eval'].get('total', 0),
                "final/correct_predictions": result['final_eval'].get('correct', 0),
            })

            # Log per-class metrics if available
            if 'per_class_accuracy' in result['final_eval']:
                for class_name, acc in result['final_eval']['per_class_accuracy'].items():
                    wandb.log({f"final/accuracy_class_{class_name}": acc})

    # Finish wandb run
    wandb.finish()

    # Conditionally remove model from results to save memory

    print("\nGPU memory after finish evaluation:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    return result
