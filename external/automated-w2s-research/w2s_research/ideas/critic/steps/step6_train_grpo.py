"""
Step 6: Train critic with GRPO using VERL.

IMPORTANT: This module MUST NOT import unsloth.
VERL training uses standard PyTorch/transformers, not unsloth.
"""
from pathlib import Path
from typing import Dict, Optional
import json
import tempfile
import time
import pandas as pd
import gc
import torch

    # VERL imports (NO unsloth!)
try:
    import ray
    from omegaconf import OmegaConf
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from verl.trainer.main_ppo import run_ppo
    VERL_AVAILABLE = True
except ImportError as e:
    VERL_AVAILABLE = False
    print(f"Warning: VERL not available: {e}. Please install verl package.")

from datasets import Dataset

from w2s_research.core import RunConfig
from w2s_research.ideas.critic.utils import log_gpu_memory, merge_lora_checkpoint, check_training_checkpoint
from w2s_research.utils.logging_utils import ENABLE_WEAVE_TRACING


def _create_verl_grpo_config(
    model_name: str,
    output_dir: str,
    config: RunConfig,
    train_dataset_size: int,
    reward_function_path: str,
    reward_kwargs: Dict,
    experiment_name: str,  # This is generated from config_key in train_critic_with_grpo
):
    """Create VERL GRPO config programmatically."""
    if not VERL_AVAILABLE:
        raise ImportError("VERL is not available. Please install verl package.")
       
    # Calculate dynamic training parameters
    steps_per_epoch = train_dataset_size // config.grpo_batch_size
    # Only save at the end of training to save disk space
    # Set save_freq to total_training_steps so only final checkpoint is saved
    total_steps = config.grpo_max_steps if config.grpo_max_steps is not None else (steps_per_epoch * config.grpo_epochs)
    save_steps = max(total_steps, 1)  # Save only at the end
    
    # Calculate total_epochs for VERL
    if config.grpo_max_steps is not None and steps_per_epoch > 0:
        calculated_epochs = max(1, (config.grpo_max_steps + steps_per_epoch - 1) // steps_per_epoch)
    else:
        calculated_epochs = config.grpo_epochs
    
    # Prepare serializable reward kwargs
    serializable_reward_kwargs = {
        "prompt_to_info_file": str(reward_kwargs["prompt_to_info_file"]),
        "reward_model_path": str(reward_kwargs["reward_model_path"]),
        "experiment_name": experiment_name,  # For weave trace attributes
    }
    
    # Build VERL config using YAML base config with dynamic overrides
    try:
        config_dir = Path(__file__).parent.parent
        GlobalHydra.instance().clear()
        
        # Dynamic overrides (only values that change per run)
        override_list = [
            # Data configuration
            f"data.train_batch_size={config.grpo_batch_size}",
            f"data.max_response_length={config.gen_max_tokens}",
            
            # Actor model configuration
            f"actor_rollout_ref.model.path={model_name}",
            f"actor_rollout_ref.model.lora_rank={config.lora_r}",
            f"actor_rollout_ref.model.lora_alpha={config.lora_alpha}",
            
            # Actor training configuration
            f"actor_rollout_ref.actor.rollout_n={config.grpo_num_generations}",
            f"actor_rollout_ref.actor.ppo_mini_batch_size={config.grpo_batch_size}",
            f"actor_rollout_ref.actor.use_kl_loss={bool(config.grpo_kl_penalty > 0)}",
            f"actor_rollout_ref.actor.kl_loss_coef={config.grpo_kl_penalty}",
            f"actor_rollout_ref.actor.optim.lr={config.grpo_lr}",
            f"actor_rollout_ref.actor.optim.lr_scheduler_type={getattr(config, 'grpo_lr_scheduler_type', 'constant')}",
            f"actor_rollout_ref.actor.optim.lr_warmup_steps_ratio={getattr(config, 'grpo_warmup_ratio', 0.0)}",
            
            # Rollout generation configuration
            f"actor_rollout_ref.rollout.n={config.grpo_num_generations}",
            f"actor_rollout_ref.rollout.temperature={config.gen_temperature}",
            f"actor_rollout_ref.rollout.top_p={config.gen_top_p}",
            f"actor_rollout_ref.rollout.response_length={config.gen_max_tokens}",
            
            # Trainer configuration
            f"trainer.default_local_dir={Path(output_dir)}",
            f"trainer.experiment_name={experiment_name}",
            f"trainer.total_epochs={calculated_epochs}",
            f"trainer.save_freq={save_steps}",
            
            # Reward model configuration
            f"reward_model.model.path={reward_kwargs['reward_model_path']}",
            f"reward_model.model.input_tokenizer={reward_kwargs['reward_model_path']}",
            f"reward_model.rollout.prompt_length={config.critic_judge_max_ctx}",
            
            # Custom reward function
            f"custom_reward_function.path={reward_function_path}",
        ]
        
        # Conditionally disable VERL trace based on environment variable
        if not ENABLE_WEAVE_TRACING:
            override_list.append("actor_rollout_ref.rollout.trace.backend=null")
        
        # Optional training steps override
        if config.grpo_max_steps is not None:
            override_list.append(f"trainer.total_training_steps={config.grpo_max_steps}")
        else:
            override_list.append("trainer.total_training_steps=null")
        
        # Optional generation parameters (only override if different from defaults)
        # Note: RolloutConfig only supports: temperature, top_k, top_p, repetition_penalty
        # min_p and presence_penalty are not supported by VERL's RolloutConfig
        if config.gen_top_k is not None:
            override_list.append(f"actor_rollout_ref.rollout.top_k={config.gen_top_k}")
        if config.gen_repetition_penalty != 1.0:
            override_list.append(f"actor_rollout_ref.rollout.repetition_penalty={config.gen_repetition_penalty}")
        # Skip min_p and presence_penalty - not supported by RolloutConfig
        
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            verl_config = compose(config_name="critic_grpo_config", overrides=override_list)
        
        OmegaConf.set_struct(verl_config.custom_reward_function, False)
        verl_config.custom_reward_function.reward_kwargs = serializable_reward_kwargs
        OmegaConf.set_struct(verl_config.custom_reward_function, True)
        
        # Ray configuration uses values from YAML (no dynamic overrides needed)
        
    except Exception as e:
        print(f"Error creating VERL config: {e}")
        raise
    
    return verl_config


def train_critic_with_grpo(
    model_name: str,
    grpo_dataset: Dataset,
    original_dataset: Dataset,
    judge_model_name: str,
    judge_checkpoint: str,
    judge_tokenizer,
    config: RunConfig,
    output_dir: str,
    config_key: str,
    experiment_uid: str = "",
    wandb_group: str = "",  # Separate wandb group name (can include model names)
) -> Dict:
    """
    Train critic with GRPO using VERL's RayPPOTrainer.
    
    IMPORTANT: This function MUST NOT import unsloth.
    VERL uses standard PyTorch/transformers, not unsloth.
    """
    if not VERL_AVAILABLE:
        raise ImportError("VERL is not available. Please install verl package.")
    
    # Set wandb group for VERL (must be set before VERL initializes wandb)
    import os
    effective_wandb_group = wandb_group or experiment_uid
    if effective_wandb_group:
        os.environ["WANDB_RUN_GROUP"] = effective_wandb_group
        print(f"Set WANDB_RUN_GROUP={effective_wandb_group} for VERL")
    
    # Use config_key + uid for VERL experiment name (aligned with judge/sft naming)
    uid_suffix = f"_{experiment_uid}" if experiment_uid else ""
    experiment_name = f"grpo{uid_suffix}"
    
    # Merge LoRA checkpoint into base model for GenRM
    # VERL's RewardModelManager expects a full model, not a LoRA adapter
    print(f"\n[Step 6] Preparing merged judge model for GenRM...")
    merged_judge_path = merge_lora_checkpoint(
        base_model_name=judge_model_name,
        lora_checkpoint=judge_checkpoint,
        output_dir=output_dir,
    )
    
    # Disable PyTorch Dynamo compilation
    try:
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.disable = True
        print("✓ Disabled PyTorch Dynamo compilation")
    except (AttributeError, ImportError) as e:
        print(f"Warning: Could not disable Dynamo compilation: {e}")
        
    # Detect binary format
    is_binary = 'choice' in grpo_dataset.column_names
    
    # Build mapping from prompt to question/option info for reward function
    prompt_to_info = {}
    for item in grpo_dataset:
        prompt_messages = item['prompt']
        prompt_key = json.dumps(prompt_messages, sort_keys=True)
        
        if is_binary:
            # Binary format: single answer with label
            prompt_to_info[prompt_key] = {
                'question': item.get('question', ''),
                'choice': item.get('choice', ''),
                'label': item.get('label', None),  # 0=correct (True), 1=incorrect (False)
                'reference_critique': item.get('reference_critique', ''),
            }
        else:
            # Comparison format: A/B options
            prompt_to_info[prompt_key] = {
                'question': item.get('question', ''),
                'option': item.get('option', 'A'),
                'first': item.get('first', ''),
                'second': item.get('second', ''),
                'critique_other_option': item.get('critique_other_option', ''),
                'ground_truth': item.get('ground_truth', None),
            }
    
    # Convert dataset to parquet format for VERL
    grpo_data = []
    for item in grpo_dataset:
        prompt_messages = item['prompt']
        
        if not isinstance(prompt_messages, list):
            raise ValueError(
                f"Expected prompt to be a list of message dicts, got {type(prompt_messages)}. "
                f"This indicates create_grpo_dataset_for_critic is returning templated strings."
            )
        
        prompt_key = json.dumps(prompt_messages, sort_keys=True)
        data_source = 'genrm_judge'  # GenRM mode - judge served via VERL's RewardModelManager
        
        # VERL expects 'ground_truth' key in reward_model (even if our reward function doesn't use it)
        if is_binary:
            data_item = {
                'prompt': prompt_messages,
                'data_source': data_source,
                'reward_model': {
                    'ground_truth': None  # Not used in binary reward, but VERL expects this field
                },
                'extra_info': {
                    'prompt': prompt_key,
                }
            }
        else:
            # Comparison format: include option for reward computation
            data_item = {
                'prompt': prompt_messages,
                'data_source': data_source,
                'reward_model': {
                    'ground_truth': item.get('ground_truth', None)  # VERL expects this field
                },
                'extra_info': {
                    'prompt': prompt_key,
                    'option': item.get('option', None),
                }
            }
        grpo_data.append(data_item)
    
    # Create temporary parquet file
    temp_parquet = tempfile.NamedTemporaryFile(mode='w', suffix='.parquet', delete=False)
    temp_parquet_path = temp_parquet.name
    temp_parquet.close()
    
    df = pd.DataFrame(grpo_data)
    df.to_parquet(temp_parquet_path, index=False)
    
    # Use the reward function module file path
    from w2s_research.ideas.critic import reward_function
    reward_function_file = Path(reward_function.__file__).resolve()
    
    # Write prompt_to_info to file
    prompt_to_info_file = Path(output_dir) / "prompt_to_info.json"
    prompt_to_info_file.parent.mkdir(parents=True, exist_ok=True)
    with open(prompt_to_info_file, 'w') as f:
        json.dump(prompt_to_info, f, indent=2)
    
    # Prepare reward kwargs for GenRM mode
    # The merged_judge_path is passed to VERL config, not reward function
    # The reward function receives reward_router_address from VERL's RewardModelManager
    reward_kwargs = {
        "prompt_to_info_file": str(prompt_to_info_file),
        "merged_judge_path": merged_judge_path,  # Passed to VERL config
        "reward_model_path": merged_judge_path,  # Used in VERL config overrides
        "experiment_name": experiment_name,  # For weave trace experiment name
    }
    
    # Log which model will be used for actor
    # Note: model_name may be the SFT checkpoint from step 1.5 if step 1.5 was enabled and successful
    print(f"\n[Step 6] Using actor model: {model_name}")
    
    verl_config = _create_verl_grpo_config(
        model_name=model_name,
        output_dir=output_dir,
        config=config,
        train_dataset_size=len(grpo_data),
        reward_function_path=str(reward_function_file),
        reward_kwargs=reward_kwargs,
        experiment_name=experiment_name,
    )
    
    # Verify the actor model path in config
    actual_actor_path = verl_config.actor_rollout_ref.model.path
    print(f"[Step 6] VERL actor model path: {actual_actor_path}")
    if actual_actor_path != str(model_name):
        print(f"  ⚠️  WARNING: Actor model path mismatch!")
        print(f"     Expected: {model_name}")
        print(f"     Actual: {actual_actor_path}")
    
    verl_config.data.train_files = temp_parquet_path
    verl_config.data.val_files = temp_parquet_path
    
    # Track training time
    step6_checkpoint = Path(output_dir).parent / "step6_grpo_training.json"
    step6_data = check_training_checkpoint(step6_checkpoint)
    
    if step6_data and step6_data.get('training_time', 0.0) > 0:
        # Training already complete (cached)
        training_time = step6_data.get('training_time', 0.0)
        print(f"  → Using cached training time: {training_time:.2f}s")
    else:
        # Start training and track time
        start_time = time.time()
        
        # VERL's RayPPOTrainer already supports colocate mode via AgentLoopManager
        # No patch needed - AgentLoopManager.__init__ handles colocate mode automatically
        run_ppo(verl_config)
        
        training_time = time.time() - start_time
        
        # Save training time to checkpoint
        step6_data = {
            'training_time': training_time,
            'output_dir': output_dir,
        }
        step6_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with open(step6_checkpoint, 'w') as f:
            json.dump(step6_data, f, indent=2)
        print(f"  ✓ Saved training time: {training_time:.2f}s to {step6_checkpoint}")
    
    # Clean up temporary parquet file
    try:
        Path(temp_parquet_path).unlink()
    except Exception:
        pass
    
    # Cleanup
    try:
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass
    
    try:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception:
        pass
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
    gc.collect()
    torch.cuda.empty_cache()
    
    # Log GPU memory status after cleanup
    log_gpu_memory("[STEP 6 END] ")
    
    return {"output_dir": output_dir, "training_time": training_time}

