"""
Core configuration for weak-to-strong research.
Self-contained, no external dependencies on AI-Scientist-v2 or weak-to-strong.
"""
from dataclasses import dataclass
from typing import Optional


# Fixed epochs for baselines (weak + ceiling) - well-trained
BASELINE_EPOCHS = 5


@dataclass
class RunConfig:
    """
    Simplified unified configuration for run.py scripts.

    This is a flat configuration that can be easily created from
    argparse arguments, avoiding long parameter lists.
    """
    # Data
    data_dir: str = "data/chat"

    # Models
    weak_model: str = "Qwen/Qwen1.5-0.5B-Chat"  # Default: Chat model for better weak supervision
    strong_model: str = "Qwen/Qwen3-4B-Base"

    # Training
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    epochs: int = 5
    judge_epochs: Optional[int] = None  # Judge training epochs (in critic training), defaults to epochs if not set
    seed: int = 42

    # Optimizer
    lr: float = 1e-4  # Learning rate for transfer model training (final SFT in critic)
    lr_schedule: str = "linear"
    warmup_ratio: float = 0.1
    warmup_steps: int = 0
    stable_ratio: float = 0.4
    weight_decay: float = 0.01
    optimizer: str = "adamw_8bit"

    # Loss
    loss: str = "xent"
    logconf_warmup_frac: float = None

    # Optional
    max_ctx: int = 8192  # Default context length for standard training/evaluation
    critic_judge_max_ctx: Optional[int] = None  # Will be computed in __post_init__ based on max_ctx and gen_max_tokens
    train_size: Optional[int] = None  # Size for train_unlabel dataset (-1 means full dataset)
    train_labeled_size: Optional[int] = None  # Size for train_label dataset (-1 means full dataset)
    test_size: Optional[int] = None  # Size for test dataset (-1 means full dataset)
    force_rerun: bool = False  # If True, delete cache and checkpoints to force fresh run

    # LoRA/Unsloth (use_unsloth=True to enable)
    use_unsloth: bool = False
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    load_in_4bit: bool = False

    # Precision modes (if both False, defaults to fp32)
    bf16: bool = True
    fp16: bool = False

    # Critic generation parameters
    gen_temperature: float = 1.0
    gen_top_p: float = 0.99
    gen_top_k: Optional[int] = None
    gen_min_p: float = 0.0
    gen_presence_penalty: float = 0.0
    gen_repetition_penalty: float = 1.0
    gen_max_tokens: int = 1024

    # GRPO training parameters
    grpo_epochs: int = 1
    grpo_max_steps: Optional[int] = 50  # If set, overrides grpo_epochs (useful for HPO sweeps)
    grpo_lr: float = 1e-5  # Lower LR for GRPO to avoid instability (reduced from 5e-5 for more stable training)
    grpo_random_sample_option: bool = True  # If True, randomly sample one option (A or B) per question instead of using both
    grpo_batch_size: int = 64  # Larger batch for more stable GRPO updates
    grpo_num_generations: int = 8  # Number of critique generations per prompt for GRPO (increased from 4 to reduce all-garbage groups)
    grpo_gradient_accumulation_steps: Optional[int] = 1  # If None, uses gradient_accumulation_steps
    grpo_kl_penalty: float = 0.02  # KL penalty coefficient (beta) for GRPO training (increased from 0.01 for more conservative updates)
                                   # Higher values (0.01-0.1) = stronger constraint, prevents reward hacking
                                   # Lower values (0.001) = more flexible, faster learning but riskier
                                   # Set to 0 to disable KL penalty (default: 0.0 for speed, KL computation is expensive)
    grpo_lr_scheduler_type: str = "constant"  # Learning rate scheduler for GRPO training
                                               # "constant" = constant LR (matches Tinker, recommended)
                                               # "linear" = linear decay
                                               # "cosine" = cosine annealing
    grpo_warmup_ratio: float = 0.0  # Warmup ratio for GRPO training (fraction of total steps)
                                    # Set to 0.0 for constant LR (no warmup, matches Tinker)


    # Critic Training: SFT on weak model critiques (Step 1.5) and Judge training (Step 3)
    enable_sft_critic: bool = True  # Enable/disable SFT step on weak model critiques
    sft_critic_num_samples: int = 3000  # Number of samples from train_unlabeled to use for SFT
    sft_critic_epochs: int = 1  # Epochs for SFT training on weak critiques
    sft_critic_lr: float = 2e-4  # Learning rate for Step 1.5 SFT and Step 3 Judge training (higher for faster convergence)

    def __post_init__(self):
        """Validate configuration and compute derived fields."""
        if self.bf16 and self.fp16:
            raise ValueError("Cannot use both bf16 and fp16. Choose one or neither (fp32).")
        
        # Set judge_epochs default to epochs if not explicitly set (backward compatibility)
        if self.judge_epochs is None:
            self.judge_epochs = self.epochs
        
        # Compute critic_judge_max_ctx if not explicitly set
        # Formula: base context + 2 critiques + template overhead (~50 tokens for "Critique of A/B:" labels and longer ending)
        # Ensure minimum of 10240 to handle long judge prompts
        if self.critic_judge_max_ctx is None:
            self.critic_judge_max_ctx = max(self.max_ctx + self.gen_max_tokens * 2 + 50, 10240)

    @classmethod
    def from_args(cls, args):
        """
        Create RunConfig from argparse Namespace.

        Args:
            args: argparse.Namespace object

        Returns:
            RunConfig instance
        """
        config_dict = {}

        # Extract all fields that exist in args
        for field_name in cls.__dataclass_fields__:
            if hasattr(args, field_name):
                value = getattr(args, field_name)
                # Convert -1 to None (means full dataset or no limit)
                if field_name in ('train_size', 'train_labeled_size', 'test_size', 'grpo_max_steps') and value == -1:
                    value = None
                config_dict[field_name] = value
        
        # Handle --no-bf16 flag (overrides default bf16=True)
        if hasattr(args, 'no_bf16') and args.no_bf16:
            config_dict['bf16'] = False

        return cls(**config_dict)


def create_run_arg_parser(description: str = "Run weak-to-strong experiment"):
    """
    Create a standard argument parser with all common training arguments.

    This ensures consistency across all experiment scripts and avoids
    duplicating argument definitions.

    Args:
        description: Description for the argument parser

    Returns:
        argparse.ArgumentParser with all common arguments
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data
    parser.add_argument("--data-dir", type=str, default="data/chat",
                       help="Path to data directory")

    # Models
    parser.add_argument("--weak-model", type=str, default="Qwen/Qwen1.5-0.5B-Chat",
                       help="Weak model name (default: Chat model for better weak supervision)")
    parser.add_argument("--strong-model", type=str, default="Qwen/Qwen3-4B-Base",
                       help="Strong model name")

    # Training
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training (default: 32)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Training epochs (default: 5)")
    parser.add_argument("--judge-epochs", type=int, default=5,
                       help="Judge training epochs (step 3, default: same as --epochs)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    # Optimizer
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate for transfer model training (default: 1e-4)")
    parser.add_argument("--lr-schedule", type=str, default="linear",
                       choices=["linear", "wsd", "cosine_anneal", "constant"],
                       help="Learning rate schedule")
    parser.add_argument("--warmup-ratio", type=float, default=0.1,
                       help="Warmup ratio for WSD/linear scheduler (default: 0.1 = 10%% warmup)")
    parser.add_argument("--warmup-steps", type=int, default=0,
                       help="Warmup steps (absolute number). If > 0, overrides warmup-ratio")
    parser.add_argument("--stable-ratio", type=float, default=0.4,
                       help="Stable ratio for WSD scheduler")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                       help="Weight decay for optimizer")
    parser.add_argument("--optimizer", type=str, default="adamw_8bit",
                       help="Optimizer")

    # Loss
    parser.add_argument("--loss", type=str, default="xent",
                       choices=["xent", "logconf", "product"],
                       help="Loss function")
    parser.add_argument("--logconf-warmup-frac", type=float, default=None)

    # Optional
    parser.add_argument("--max-ctx", type=int, default=8192,
                       help="Maximum context length for standard training/evaluation")
    parser.add_argument("--critic-judge-max-ctx", type=int, default=10240,
                       help="Maximum context length for judge training/evaluation with critiques (longer prompts)")
    parser.add_argument("--train-size", type=int, default=None,
                       help="Limit train_unlabel examples (for testing, -1 means full dataset)")
    parser.add_argument("--train-labeled-size", type=int, default=None,
                       help="Limit train_label examples (for testing, -1 means full dataset, separate from train-size)")
    parser.add_argument("--test-size", type=int, default=None,
                       help="Limit test examples (for testing, -1 means full dataset)")
    parser.add_argument("--force-rerun", action="store_true",
                       help="Force fresh run by deleting cache and checkpoints")

    # LoRA/Unsloth
    parser.add_argument("--use-unsloth", action="store_true",
                       help="Use Unsloth with LoRA for efficient training")
    parser.add_argument("--lora-r", type=int, default=32,
                       help="LoRA rank (higher = more parameters, default: 32)")
    parser.add_argument("--lora-alpha", type=int, default=32,
                       help="LoRA alpha (scaling factor, default: 32)")
    parser.add_argument("--lora-dropout", type=float, default=0.0,
                       help="LoRA dropout")
    parser.add_argument("--load-in-4bit", action="store_true",
                       help="Load model in 4-bit for lower memory usage")

    # Precision modes
    parser.add_argument("--bf16", action="store_true",
                       help="Use bfloat16 precision. Cannot be used with --fp16. (default: True in RunConfig)")
    parser.add_argument("--fp16", action="store_true",
                       help="Use float16 precision. Cannot be used with --bf16. If neither flag is set, fp32 is used.")
    parser.add_argument("--no-bf16", action="store_true",
                       help="Disable bfloat16 precision (use fp32). Overrides default bf16=True in RunConfig.")

    # GRPO-specific parameters (for critic training)
    parser.add_argument("--grpo-lr", type=float, default=1e-5,
                       help="Learning rate for GRPO training (lower than SFT for stability)")
    parser.add_argument("--grpo-max-steps", type=int, default=50,
                       help="Max steps for GRPO training (overrides grpo-epochs if set, default: 1)")
    parser.add_argument("--grpo-epochs", type=int, default=1,
                       help="Number of GRPO training epochs")
    parser.add_argument("--grpo-batch-size", type=int, default=32,
                       help="Batch size for GRPO training")
    parser.add_argument("--grpo-kl-penalty", type=float, default=0.02,
                       help="KL penalty coefficient (beta) for GRPO training. Higher values (0.01-0.1) = stronger constraint, prevents reward hacking. Lower values (0.001) = more flexible. Default 0.0 (disabled for speed, KL computation is expensive).")
    parser.add_argument("--grpo-lr-scheduler-type", type=str, default="constant",
                       choices=["constant", "linear", "cosine"],
                       help="Learning rate scheduler type for GRPO training. 'constant' matches Tinker (recommended). Default: constant")
    parser.add_argument("--grpo-warmup-ratio", type=float, default=0.0,
                       help="Warmup ratio for GRPO training (fraction of total steps). Set to 0.0 for constant LR (no warmup, matches Tinker). Default: 0.0")
    parser.add_argument("--grpo-random-sample-option", action="store_true", default=True,
                       help="Randomly sample one option (A or B) per question instead of using both (default: True)")
    parser.add_argument("--no-grpo-random-sample-option", dest="grpo_random_sample_option", action="store_false",
                       help="Use both options (A and B) for each question (disables random sampling)")

    # SFT Critic parameters (Step 1.5 and Step 3)
    parser.add_argument("--sft-critic-num-samples", type=int, default=3000,
                       help="Number of samples from train_unlabeled to use for SFT on weak critiques. Default: 3000")
    parser.add_argument("--sft-critic-lr", type=float, default=2e-4,
                       help="Learning rate for Step 1.5 SFT and Step 3 Judge training. Default: 2e-4")

    # Note: Generation parameters (gen_temperature, gen_top_p, gen_top_k, etc.)
    # are NOT exposed as CLI arguments. Edit RunConfig dataclass defaults in config.py to change them.

    return parser
