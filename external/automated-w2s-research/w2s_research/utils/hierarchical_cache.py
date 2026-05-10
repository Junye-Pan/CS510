"""
Unified result caching system for weak-to-strong experiments.

This module provides hierarchical caching for hyperparameter sweeps with support
for individual seed results, aggregation, weak artifacts, and backward compatibility
with the old flat cache structure.

Cache Structure:
cache_results/
├── {dataset}_{idea_name}/
│   ├── {hyperparam_config_1}/
│   │   ├── seed_42.json
│   │   ├── seed_43.json
│   │   ├── seed_44.json
│   │   ├── seed_45.json
│   │   ├── seed_46.json
│   │   └── eval.json          # Aggregated metrics across 5 seeds
│   ├── {hyperparam_config_2}/
│   │   ├── seed_42.json
│   │   ├── ...
│   │   └── eval.json
│   └── best_eval.json         # Best hyperparameter config for this idea
├── {dataset}_weak_artifacts/
│   ├── {weak_model_config}/
│   │   ├── seed_42.json       # Weak model outputs (soft labels)
│   │   └── seed_43.json
│   └── ...

Example:
cache_results/
├── math_vanilla_w2s/
│   ├── Qwen_Qwen3_8B_Qwen_Qwen2.5_0.5B_e1_lr5e-05_bs32_schlinear/
│   │   ├── seed_42.json
│   │   ├── seed_43.json
│   │   ├── seed_44.json
│   │   ├── seed_45.json
│   │   ├── seed_46.json
│   │   └── eval.json          # Mean ± SE across 5 seeds
│   ├── Qwen_Qwen3_8B_Qwen_Qwen2.5_0.5B_e3_lr1e-04_bs64_schcosine/
│   │   └── ...
│   └── best_eval.json         # Best config based on highest mean PGR
├── math_train_ceiling/
│   ├── Qwen_Qwen3_8B_e1_lr5e-05_bs32_schlinear/
│   │   ├── seed_42.json
│   │   └── eval.json
│   └── best_eval.json
└── math_weak_artifacts/
    ├── Qwen_Qwen2_5_0_5B_e2_lr2e-04_bs4_schlinear/
    │   ├── seed_42.json        # Soft labels, weak_acc, hard_label_acc
    │   └── seed_43.json
    └── ...
"""
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from w2s_research.core.config import BASELINE_EPOCHS


def compute_hyperparam_config_key(
    strong_model: Optional[str] = None,
    weak_model: Optional[str] = None,
    dataset_name: str = "math",
    epochs: int = 1,
    lr: float = 5e-5,
    batch_size: int = 32,
    scheduler: str = "linear",
    loss: str = "xent",
    **extra_params,
) -> str:
    """
    Compute hyperparameter configuration key for organizing results.

    Args:
        strong_model: Strong model name (e.g., "Qwen/Qwen3-8B-Base") or None
        weak_model: Weak model name (e.g., "Qwen/Qwen2.5-0.5B") or None for ceiling
        dataset_name: Dataset name (e.g., "math", "sciq")
        epochs: Training epochs
        lr: Learning rate
        batch_size: Batch size
        scheduler: Learning rate scheduler
        loss: Loss function
        **extra_params: Additional hyperparameters (e.g., lora_r, lora_alpha)

    Returns:
        Hyperparameter config key (e.g., "Qwen_Qwen3_8B_Qwen_Qwen2.5_0.5B_e1_lr5e-05_bs32_schlinear")
    """
    # Clean model names
    if strong_model:
        strong_clean = strong_model.replace('/', '_').replace('-', '_').replace('.', '_')
    else:
        strong_clean = "no_strong"

    if weak_model:
        weak_clean = weak_model.replace('/', '_').replace('-', '_').replace('.', '_')
        model_part = f"{strong_clean}_{weak_clean}"
    else:
        model_part = strong_clean

    # Format: {models}_e{epochs}_lr{lr}_bs{batch_size}_sch{scheduler}
    config_key = f"{model_part}_e{epochs}_lr{lr}_bs{batch_size}_sch{scheduler}"
    
    if loss != "xent":
        config_key += f"_loss{loss}"

    # Add extra params if provided (e.g., lora_r, lora_alpha)
    if extra_params:
        extra_str = "_".join(f"{k}{v}" for k, v in sorted(extra_params.items()))
        config_key += f"_{extra_str}"

    return config_key


class HierarchicalCache:
    """Manages hierarchical caching for hyperparameter sweeps."""

    def __init__(self, cache_root: str = None):
        """
        Initialize hierarchical cache.

        Args:
            cache_root: Root directory for all cached results
        """
        if cache_root is None:
            import os
            workspace = os.getenv("WORKSPACE_DIR", str(Path(__file__).parent.parent.parent))
            cache_root = os.path.join(workspace, "cache_results")
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def _get_idea_dir(self, idea_name: str, dataset_name: str = "math") -> Path:
        """Get directory for an idea (includes dataset in path)."""
        # Format: {dataset}_{idea_name}
        dir_name = f"{dataset_name}_{idea_name}"
        idea_dir = self.cache_root / dir_name
        idea_dir.mkdir(parents=True, exist_ok=True)
        return idea_dir

    def _get_hyperparam_dir(
        self,
        idea_name: str,
        hyperparam_config_key: str,
        dataset_name: str = "math",
    ) -> Path:
        """Get directory for a specific hyperparameter configuration."""
        idea_dir = self._get_idea_dir(idea_name, dataset_name)
        hyperparam_dir = idea_dir / hyperparam_config_key
        hyperparam_dir.mkdir(parents=True, exist_ok=True)
        return hyperparam_dir

    def get_seed_result(
        self,
        idea_name: str,
        hyperparam_config_key: str,
        seed: int,
        dataset_name: str = "math",
    ) -> Optional[Dict[str, Any]]:
        """
        Get result for a specific seed.

        Args:
            idea_name: Name of the idea (e.g., "vanilla_w2s")
            hyperparam_config_key: Hyperparameter config key
            seed: Random seed
            dataset_name: Dataset name

        Returns:
            Dictionary with experiment metrics or None if not cached

        Note:
            If W2S_OVERWRITE_CACHE=1 environment variable is set, this function
            returns None to force re-running the experiment (for iterative improvement).
        """
        # Check if cache overwrite is requested (for iterative improvement)
        if os.environ.get("W2S_OVERWRITE_CACHE") == "1" and idea_name not in ['weak_artifacts', 'train_ceiling']:
            print(f"  [Cache] Skipping cache lookup (W2S_OVERWRITE_CACHE=1)")
            return None

        hyperparam_dir = self._get_hyperparam_dir(idea_name, hyperparam_config_key, dataset_name)
        seed_file = hyperparam_dir / f"seed_{seed}.json"

        if not seed_file.exists():
            return None

        try:
            print('seed file = ', seed_file)
            with open(seed_file, 'r') as f:
                result = json.load(f)

            print(f"✓ Loaded cached result for seed {seed}")
            if "transfer_acc" in result:
                print(f"  Transfer accuracy: {result['transfer_acc']}")
            if "pgr" in result:
                print(f"  PGR: {result['pgr']}")
            return result

        except Exception as e:
            print(f"Warning: Failed to load seed result from {seed_file}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def delete_seed_result(
        self,
        idea_name: str,
        hyperparam_config_key: str,
        seed: int,
        dataset_name: str = "math",
    ) -> bool:
        """
        Delete cached result for a specific seed.

        Args:
            idea_name: Name of the idea (e.g., "vanilla_w2s")
            hyperparam_config_key: Hyperparameter config key
            seed: Random seed
            dataset_name: Dataset name

        Returns:
            True if file was deleted, False if it didn't exist
        """
        hyperparam_dir = self._get_hyperparam_dir(idea_name, hyperparam_config_key, dataset_name)
        seed_file = hyperparam_dir / f"seed_{seed}.json"

        if seed_file.exists():
            seed_file.unlink()
            print(f"✓ Deleted cached result for seed {seed}")
            return True
        else:
            print(f"  No cached result found for seed {seed}")
            return False

    def set_seed_result(
        self,
        idea_name: str,
        hyperparam_config_key: str,
        seed: int,
        result_data: Dict[str, Any],
        dataset_name: str = "math",
    ):
        """
        Save result for a specific seed.

        Args:
            idea_name: Name of the idea
            hyperparam_config_key: Hyperparameter config key
            seed: Random seed
            result_data: Dictionary with experiment metrics
            dataset_name: Dataset name

        Note:
            If W2S_VALIDATION_MODE=1 environment variable is set, this function
            skips saving to avoid polluting cache during quick validation runs.
        """
        # Skip cache saving during validation mode
        if os.environ.get("W2S_VALIDATION_MODE") == "1":
            print(f"  [Cache] Skipping cache save (W2S_VALIDATION_MODE=1)")
            return

        hyperparam_dir = self._get_hyperparam_dir(idea_name, hyperparam_config_key, dataset_name)
        seed_file = hyperparam_dir / f"seed_{seed}.json"

        try:
            # Add metadata
            result_data = result_data.copy()
            result_data["cached_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            result_data["seed"] = seed
            result_data["idea_name"] = idea_name
            result_data["hyperparam_config_key"] = hyperparam_config_key
            result_data["dataset_name"] = dataset_name

            with open(seed_file, 'w') as f:
                json.dump(result_data, f, indent=2)

            print(f"✓ Cached result for seed {seed} to {seed_file.name}")
            if "transfer_acc" in result_data and result_data['transfer_acc'] is not None:
                print(f"  Transfer accuracy: {result_data['transfer_acc']:.4f}")
            if "pgr" in result_data and result_data['pgr'] is not None:
                print(f"  PGR: {result_data['pgr']:.4f}")

        except Exception as e:
            print(f"Warning: Failed to save seed result to {seed_file}: {e}")

    def get_all_seed_results(
        self,
        idea_name: str,
        hyperparam_config_key: str,
        dataset_name: str = "math",
    ) -> List[Dict[str, Any]]:
        """
        Get all seed results for a hyperparameter configuration.

        Returns:
            List of result dictionaries (may be empty if none exist)
        """
        hyperparam_dir = self._get_hyperparam_dir(idea_name, hyperparam_config_key, dataset_name)

        results = []
        for seed_file in sorted(hyperparam_dir.glob("seed_*.json")):
            try:
                with open(seed_file, 'r') as f:
                    results.append(json.load(f))
            except Exception as e:
                print(f"Warning: Failed to load {seed_file}: {e}")
                continue

        return results

    def aggregate_seed_results(
        self,
        idea_name: str,
        hyperparam_config_key: str,
        dataset_name: str = "math",
        required_seeds: Optional[List[int]] = None,
        fixed_weak_baseline: Optional[Tuple[float, float, int]] = None,
        fixed_ceiling_baseline: Optional[Tuple[float, float, int]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Aggregate results across all seeds for a hyperparameter configuration.

        This computes mean ± standard error for all metrics.

        Args:
            idea_name: Name of the idea
            hyperparam_config_key: Hyperparameter config key
            dataset_name: Dataset name
            required_seeds: List of required seeds (e.g., [42, 43, 44, 45, 46])
                           If provided, only aggregate if all seeds exist
            fixed_weak_baseline: Optional tuple of (mean, se, num_seeds) for fixed weak baseline
                                If provided, uses this for weak_acc and PGR calculation
            fixed_ceiling_baseline: Optional tuple of (mean, se, num_seeds) for fixed ceiling baseline
                                   If provided, uses this for strong_acc and PGR calculation

        Returns:
            Aggregated results dictionary or None if not enough seeds
        """
        seed_results = self.get_all_seed_results(idea_name, hyperparam_config_key, dataset_name)

        if not seed_results:
            return None

        # Check if all required seeds are present
        if required_seeds:
            available_seeds = {r.get("seed") for r in seed_results}
            if not all(s in available_seeds for s in required_seeds):
                missing = set(required_seeds) - available_seeds
                print(f"⚠ Missing seeds {missing} for {hyperparam_config_key}")
                return None

        # Aggregate numeric metrics
        aggregated = {
            "idea_name": idea_name,
            "hyperparam_config_key": hyperparam_config_key,
            "dataset_name": dataset_name,
            "num_seeds": len(seed_results),
            "seeds": [r.get("seed") for r in seed_results],
        }

        # Metrics to aggregate (mean ± SE)
        metrics_to_aggregate = [
            "weak_acc", "strong_acc", "transfer_acc", "pgr",
            "train_time", "eval_time", "total_time"
        ]

        for metric in metrics_to_aggregate:
            values = []
            for result in seed_results:
                val = result.get(metric)
                # Skip non-numeric values (e.g., "N/A" for PGR)
                if val is not None and isinstance(val, (int, float)):
                    values.append(val)

            if values:
                mean_val = np.mean(values)
                se_val = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0

                aggregated[f"{metric}_mean"] = float(mean_val)
                aggregated[f"{metric}_se"] = float(se_val)
                aggregated[f"{metric}_values"] = values

        # Apply fixed baselines if provided
        if fixed_weak_baseline and fixed_weak_baseline[0] is not None:
            weak_mean, weak_se, weak_num_seeds = fixed_weak_baseline
            aggregated["weak_acc_mean"] = weak_mean
            aggregated["weak_acc_se"] = weak_se
            aggregated["weak_acc_fixed_baseline"] = True
            aggregated["weak_acc_fixed_num_seeds"] = weak_num_seeds
            print(f"  Using fixed weak baseline: {weak_mean:.4f} ± {weak_se:.4f} ({weak_num_seeds} seeds)")

        if fixed_ceiling_baseline and fixed_ceiling_baseline[0] is not None:
            strong_mean, strong_se, strong_num_seeds = fixed_ceiling_baseline
            aggregated["strong_acc_mean"] = strong_mean
            aggregated["strong_acc_se"] = strong_se
            aggregated["strong_acc_fixed_baseline"] = True
            aggregated["strong_acc_fixed_num_seeds"] = strong_num_seeds
            print(f"  Using fixed ceiling baseline: {strong_mean:.4f} ± {strong_se:.4f} ({strong_num_seeds} seeds)")

        # Recompute PGR if using fixed baselines
        if (fixed_weak_baseline or fixed_ceiling_baseline) and "transfer_acc_values" in aggregated:
            weak_mean = aggregated.get("weak_acc_mean", 0.0)
            strong_mean = aggregated.get("strong_acc_mean", 0.0)
            transfer_values = aggregated["transfer_acc_values"]

            if strong_mean > weak_mean:
                # Recompute PGR for each seed using fixed baselines
                pgr_values = [(t - weak_mean) / (strong_mean - weak_mean) for t in transfer_values]
                pgr_mean = np.mean(pgr_values)
                pgr_se = np.std(pgr_values, ddof=1) / np.sqrt(len(pgr_values)) if len(pgr_values) > 1 else 0.0

                aggregated["pgr_mean"] = float(pgr_mean)
                aggregated["pgr_se"] = float(pgr_se)
                aggregated["pgr_values"] = pgr_values
                aggregated["pgr_fixed_baseline"] = True
                print(f"  Recomputed PGR with fixed baselines: {pgr_mean:.4f} ± {pgr_se:.4f}")

        # Copy over config metadata from first seed result
        if seed_results:
            first_result = seed_results[0]
            for key in ["strong_model", "weak_model", "epochs", "lr", "batch_size", "scheduler"]:
                if key in first_result:
                    aggregated[key] = first_result[key]

        return aggregated

    def save_aggregated_eval(
        self,
        idea_name: str,
        hyperparam_config_key: str,
        dataset_name: str = "math",
        required_seeds: Optional[List[int]] = None,
        fixed_weak_baseline: Optional[Tuple[float, float, int]] = None,
        fixed_ceiling_baseline: Optional[Tuple[float, float, int]] = None,
    ) -> bool:
        """
        Aggregate seed results and save to eval.json.

        Args:
            idea_name: Name of the idea
            hyperparam_config_key: Hyperparameter config key
            dataset_name: Dataset name
            required_seeds: List of required seeds (only aggregate if all present)
            fixed_weak_baseline: Optional tuple of (mean, se, num_seeds) for fixed weak baseline
            fixed_ceiling_baseline: Optional tuple of (mean, se, num_seeds) for fixed ceiling baseline

        Returns:
            True if aggregation succeeded, False otherwise
        """
        aggregated = self.aggregate_seed_results(
            idea_name, hyperparam_config_key, dataset_name, required_seeds,
            fixed_weak_baseline=fixed_weak_baseline,
            fixed_ceiling_baseline=fixed_ceiling_baseline,
        )

        if aggregated is None:
            return False

        hyperparam_dir = self._get_hyperparam_dir(idea_name, hyperparam_config_key, dataset_name)
        eval_file = hyperparam_dir / "eval.json"

        try:
            aggregated["aggregated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

            with open(eval_file, 'w') as f:
                json.dump(aggregated, f, indent=2)

            print(f"✓ Saved aggregated eval to {eval_file}")
            if "transfer_acc_mean" in aggregated:
                mean = aggregated["transfer_acc_mean"]
                se = aggregated["transfer_acc_se"]
                print(f"  Transfer accuracy: {mean:.4f} ± {se:.4f}")
            if "pgr_mean" in aggregated:
                mean = aggregated["pgr_mean"]
                se = aggregated["pgr_se"]
                print(f"  PGR: {mean:.4f} ± {se:.4f}")

            return True

        except Exception as e:
            print(f"Warning: Failed to save aggregated eval to {eval_file}: {e}")
            return False

    def get_aggregated_eval(
        self,
        idea_name: str,
        hyperparam_config_key: str,
        dataset_name: str = "math",
    ) -> Optional[Dict[str, Any]]:
        """
        Get aggregated eval.json for a hyperparameter configuration.

        Returns:
            Aggregated metrics or None if not available
        """
        hyperparam_dir = self._get_hyperparam_dir(idea_name, hyperparam_config_key, dataset_name)
        eval_file = hyperparam_dir / "eval.json"

        if not eval_file.exists():
            return None

        try:
            with open(eval_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load aggregated eval from {eval_file}: {e}")
            return None

    def find_best_hyperparam_config(
        self,
        idea_name: str,
        dataset_name: str = "math",
        metric: str = "pgr_mean",
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Find the best hyperparameter configuration for an idea.

        Args:
            idea_name: Name of the idea
            dataset_name: Dataset name
            metric: Metric to maximize (default: "pgr_mean")

        Returns:
            Tuple of (hyperparam_config_key, eval_data) or None if no configs found
        """
        idea_dir = self._get_idea_dir(idea_name, dataset_name)

        best_config_key = None
        best_eval = None
        best_metric_value = -float('inf')

        # Iterate over all hyperparameter directories
        for hyperparam_dir in idea_dir.iterdir():
            if not hyperparam_dir.is_dir():
                continue

            eval_file = hyperparam_dir / "eval.json"
            if not eval_file.exists():
                continue

            try:
                with open(eval_file, 'r') as f:
                    eval_data = json.load(f)

                metric_value = eval_data.get(metric)
                if metric_value is None or not isinstance(metric_value, (int, float)):
                    continue

                if metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_config_key = hyperparam_dir.name
                    best_eval = eval_data

            except Exception as e:
                print(f"Warning: Failed to load eval from {eval_file}: {e}")
                continue

        if best_config_key is None:
            return None

        return best_config_key, best_eval

    def save_best_eval(
        self,
        idea_name: str,
        dataset_name: str = "math",
        metric: str = "pgr_mean",
    ) -> bool:
        """
        Find best hyperparameter config and save to best_eval.json.

        Args:
            idea_name: Name of the idea
            dataset_name: Dataset name
            metric: Metric to maximize (default: "pgr_mean")

        Returns:
            True if best eval was saved, False otherwise
        """
        best_result = self.find_best_hyperparam_config(idea_name, dataset_name, metric)

        if best_result is None:
            print(f"⚠ No hyperparameter configs found for {idea_name}")
            return False

        best_config_key, best_eval = best_result

        idea_dir = self._get_idea_dir(idea_name, dataset_name)
        best_eval_file = idea_dir / "best_eval.json"

        try:
            # Add best config metadata
            best_eval_data = best_eval.copy()
            best_eval_data["best_hyperparam_config_key"] = best_config_key
            best_eval_data["selection_metric"] = metric
            best_eval_data["selected_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

            with open(best_eval_file, 'w') as f:
                json.dump(best_eval_data, f, indent=2)

            print(f"✓ Saved best eval to {best_eval_file}")
            print(f"  Best config: {best_config_key}")
            if "transfer_acc_mean" in best_eval_data:
                mean = best_eval_data["transfer_acc_mean"]
                se = best_eval_data["transfer_acc_se"]
                print(f"  Transfer accuracy: {mean:.4f} ± {se:.4f}")
            if "pgr_mean" in best_eval_data:
                mean = best_eval_data["pgr_mean"]
                se = best_eval_data["pgr_se"]
                print(f"  PGR: {mean:.4f} ± {se:.4f}")

            return True

        except Exception as e:
            print(f"Warning: Failed to save best eval to {best_eval_file}: {e}")
            return False

    def get_best_eval(
        self,
        idea_name: str,
        dataset_name: str = "math",
    ) -> Optional[Dict[str, Any]]:
        """
        Get best_eval.json for an idea.

        Returns:
            Best eval data or None if not available
        """
        idea_dir = self._get_idea_dir(idea_name, dataset_name)
        best_eval_file = idea_dir / "best_eval.json"

        if not best_eval_file.exists():
            return None

        try:
            with open(best_eval_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load best eval from {best_eval_file}: {e}")
            return None

    def list_all_ideas(self, dataset_name: Optional[str] = None) -> List[str]:
        """
        List all ideas in the cache.

        Args:
            dataset_name: Filter by dataset name (e.g., "math")

        Returns:
            List of idea names
        """
        ideas = []
        for dir_path in self.cache_root.iterdir():
            if not dir_path.is_dir():
                continue

            # Parse {dataset}_{idea_name}
            dir_name = dir_path.name
            if '_' in dir_name:
                parts = dir_name.split('_', 1)
                if len(parts) == 2:
                    dataset, idea = parts
                    if dataset_name is None or dataset == dataset_name:
                        ideas.append(idea)

        return sorted(ideas)

    def list_hyperparam_configs(
        self,
        idea_name: str,
        dataset_name: str = "math",
    ) -> List[str]:
        """
        List all hyperparameter configurations for an idea.

        Returns:
            List of hyperparam config keys
        """
        idea_dir = self._get_idea_dir(idea_name, dataset_name)

        configs = []
        for hyperparam_dir in idea_dir.iterdir():
            if hyperparam_dir.is_dir():
                configs.append(hyperparam_dir.name)

        return sorted(configs)

    def clear_idea(self, idea_name: str, dataset_name: str = "math"):
        """Clear all results for an idea."""
        idea_dir = self._get_idea_dir(idea_name, dataset_name)

        import shutil
        if idea_dir.exists():
            shutil.rmtree(idea_dir)
            print(f"✓ Cleared all results for {idea_name}")

    def clear_all(self):
        """Clear entire cache."""
        import shutil
        if self.cache_root.exists():
            shutil.rmtree(self.cache_root)
            self.cache_root.mkdir(parents=True, exist_ok=True)
            print(f"✓ Cleared entire cache at {self.cache_root}")




# ============================================================================
# Weak Artifact Support
# ============================================================================

def compute_weak_artifact_cache_key(
    weak_model: str,
    dataset_name: str = "math",
    epochs: int = 2,
    lr: float = 2e-4,
    batch_size: int = 4,
    scheduler: str = "linear",
    **extra_params,
) -> str:
    """
    Compute cache key for weak model artifacts (weak_acc, soft_labels, hard_label_acc).

    This cache is independent of the strong model, so the same weak artifacts
    can be reused across different strong model experiments.

    Args:
        weak_model: Weak model name (e.g., "Qwen/Qwen2.5-0.5B")
        dataset_name: Dataset name (e.g., "math", "sciq")
        epochs: Training epochs for weak model
        lr: Learning rate for weak model training
        batch_size: Batch size for weak model training
        scheduler: Learning rate scheduler
        **extra_params: Additional hyperparameters

    Returns:
        Weak artifact config key (e.g., "Qwen_Qwen2_5_0_5B_e2_lr2e-04_bs4_schlinear")
    """
    # Clean model name
    weak_clean = weak_model.replace('/', '_').replace('-', '_').replace('.', '_')

    # Format: {weak_model}_e{epochs}_lr{lr}_bs{batch_size}_sch{scheduler}
    config_key = f"{weak_clean}_e{epochs}_lr{lr}_bs{batch_size}_sch{scheduler}"

    # Add extra params if provided
    if extra_params:
        extra_str = "_".join(f"{k}{v}" for k, v in sorted(extra_params.items()))
        config_key += f"_{extra_str}"

    return config_key


def get_cached_weak_artifacts(
    weak_model: str,
    dataset_name: str = "math",
    seed: int = 42,
    epochs: int = None,  # Defaults to BASELINE_EPOCHS
    batch_size: int = 32,
    lr: float = 1e-4,
    scheduler: str = "linear",
    **extra_params,
) -> Optional[Dict[str, Any]]:
    """
    Get cached weak model artifacts (weak_acc, soft_labels, hard_label_acc).

    Args:
        weak_model: Weak model name (e.g., "Qwen/Qwen2.5-0.5B")
        dataset_name: Dataset name
        seed: Random seed
        epochs: Training epochs for weak model (default: BASELINE_EPOCHS=5)
        batch_size: Batch size for weak model training
        lr: Learning rate for weak model training
        scheduler: Learning rate scheduler
        **extra_params: Additional hyperparameters

    Returns:
        Dictionary with weak artifacts or None if not cached:
        {
            "weak_acc": float,
            "soft_labels": List[List[float]],
            "hard_label_acc": float,
            "weak_model": str,
            "dataset_name": str,
            "seed": int,
            ...
        }
    """
    if epochs is None:
        epochs = BASELINE_EPOCHS
    cache = HierarchicalCache()

    # Compute config key
    config_key = compute_weak_artifact_cache_key(
        weak_model=weak_model,
        dataset_name=dataset_name,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        scheduler=scheduler,
        **extra_params,
    )

    # Get from cache (stored under special "weak_artifacts" idea)
    cached = cache.get_seed_result(
        idea_name="weak_artifacts",
        hyperparam_config_key=config_key,
        seed=seed,
        dataset_name=dataset_name,
    )

    if cached:
        print(f"✓ Found cached weak artifacts for {weak_model}")
        print(f"  Weak accuracy: {cached.get('weak_acc', 0.0):.4f}")
        print(f"  Hard weak label accuracy: {cached.get('hard_label_acc', 0.0):.4f}")
        print(f"  Soft labels: {len(cached.get('soft_labels', []))} samples")
    return cached


def cache_weak_artifacts(
    weak_model: str,
    weak_acc: float,
    soft_labels: list,
    hard_label_acc: float,
    dataset_name: str = "math",
    seed: int = 42,
    epochs: int = 2,
    batch_size: int = 4,
    lr: float = 2e-4,
    scheduler: str = "linear",
    training_time: float = 0.0,
    config: Optional[Dict[str, Any]] = None,
    **extra_params,
):
    """
    Cache weak model artifacts for reuse across different strong models.

    Args:
        weak_model: Weak model name
        weak_acc: Weak model test accuracy
        soft_labels: Soft label probabilities for train_unlabel (List[List[float]])
        hard_label_acc: Hard label accuracy on train_unlabel
        dataset_name: Dataset name
        seed: Random seed
        epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        scheduler: Learning rate scheduler
        training_time: Time spent training weak model
        config: Additional configuration details
        **extra_params: Additional hyperparameters

    Note:
        If W2S_VALIDATION_MODE=1 environment variable is set, this function
        skips saving to avoid polluting cache during quick validation runs.
    """
    # Skip cache saving during validation mode
    if os.environ.get("W2S_VALIDATION_MODE") == "1":
        print(f"  [Cache] Skipping weak artifacts cache (W2S_VALIDATION_MODE=1)")
        return

    cache = HierarchicalCache()

    # Compute config key
    config_key = compute_weak_artifact_cache_key(
        weak_model=weak_model,
        dataset_name=dataset_name,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        scheduler=scheduler,
        **extra_params,
    )

    # Build artifacts data
    artifacts = {
        "weak_model": weak_model,
        "weak_acc": weak_acc,
        "soft_labels": soft_labels,
        "hard_label_acc": hard_label_acc,
        "dataset_name": dataset_name,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "scheduler": scheduler,
        "training_time": training_time,
        "config": config or {},
    }

    # Add extra params
    artifacts.update(extra_params)

    # Save to cache (under special "weak_artifacts" idea)
    cache.set_seed_result(
        idea_name="weak_artifacts",
        hyperparam_config_key=config_key,
        seed=seed,
        result_data=artifacts,
        dataset_name=dataset_name,
    )

    print(f"✓ Cached weak artifacts")
    print(f"  Weak accuracy: {weak_acc:.4f}")
    print(f"  Hard weak label accuracy: {hard_label_acc:.4f}")
    print(f"  Soft labels: {len(soft_labels)} samples")


def get_fixed_weak_baseline(
    weak_model: str,
    dataset_name: str = "math",
    epochs: int = None,  # Defaults to BASELINE_EPOCHS
    batch_size: int = 32,
    lr: float = 1e-4,
    scheduler: str = "linear",
    cache_root: str = None,
    **extra_params,
) -> Tuple[Optional[float], Optional[float], int]:
    """
    Get fixed weak baseline by averaging weak_acc across ALL available seeds.

    This is useful for computing PGR with a stable baseline that doesn't vary
    across different experiment seeds.

    Args:
        weak_model: Weak model name (e.g., "Qwen/Qwen2.5-0.5B")
        dataset_name: Dataset name
        epochs: Training epochs for weak model (default: BASELINE_EPOCHS=5)
        batch_size: Batch size for weak model training
        lr: Learning rate for weak model training
        scheduler: Learning rate scheduler
        cache_root: Optional custom cache root directory
        **extra_params: Additional hyperparameters

    Returns:
        Tuple of (mean_weak_acc, se_weak_acc, num_seeds)
        Returns (None, None, 0) if no seeds found
    """
    if epochs is None:
        epochs = BASELINE_EPOCHS
    cache = HierarchicalCache(cache_root=cache_root) if cache_root else HierarchicalCache()

    # Compute config key
    config_key = compute_weak_artifact_cache_key(
        weak_model=weak_model,
        dataset_name=dataset_name,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        scheduler=scheduler,
        **extra_params,
    )

    # Get all seed results
    print(f"[get_fixed_weak_baseline] Looking for weak artifacts:")
    print(f"  - Cache root: {cache.cache_root}")
    print(f"  - Dataset: {dataset_name}")
    print(f"  - Config key: {config_key}")
    print(f"  - Idea name: weak_artifacts")
    
    all_results = cache.get_all_seed_results(
        idea_name="weak_artifacts",
        hyperparam_config_key=config_key,
        dataset_name=dataset_name,
    )

    print(f"[get_fixed_weak_baseline] Found {len(all_results)} seed results")
    
    if not all_results:
        # Check if directory exists
        hyperparam_dir = cache._get_hyperparam_dir("weak_artifacts", config_key, dataset_name)
        print(f"[get_fixed_weak_baseline] No results found. Checking directory: {hyperparam_dir}")
        print(f"[get_fixed_weak_baseline] Directory exists: {hyperparam_dir.exists()}")
        if hyperparam_dir.exists():
            seed_files = list(hyperparam_dir.glob("seed_*.json"))
            print(f"[get_fixed_weak_baseline] Found {len(seed_files)} seed files in directory")
            if seed_files:
                print(f"[get_fixed_weak_baseline] Example files: {[f.name for f in seed_files[:3]]}")
        return None, None, 0

    # Extract weak_acc values
    weak_accs = [r.get("weak_acc", 0.0) for r in all_results if r.get("weak_acc") is not None]

    if not weak_accs:
        print(f"[get_fixed_weak_baseline] WARNING: Found {len(all_results)} results but none have weak_acc field")
        print(f"[get_fixed_weak_baseline] Available keys in first result: {list(all_results[0].keys()) if all_results else 'N/A'}")
        return None, None, 0

    mean_acc = np.mean(weak_accs)
    se_acc = np.std(weak_accs, ddof=1) / np.sqrt(len(weak_accs)) if len(weak_accs) > 1 else 0.0

    print(f"[get_fixed_weak_baseline] Successfully computed baseline: {mean_acc:.4f} ± {se_acc:.4f} (from {len(weak_accs)} seeds)")
    return float(mean_acc), float(se_acc), len(weak_accs)


def get_fixed_ceiling_baseline(
    strong_model: str,
    dataset_name: str = "math",
    epochs: int = None,  # Defaults to BASELINE_EPOCHS
    batch_size: int = 32,
    lr: float = 1e-4,
    scheduler: str = "linear",
    cache_root: str = None,
    **extra_params,
) -> Tuple[Optional[float], Optional[float], int]:
    """
    Get fixed ceiling baseline by averaging strong_acc across ALL available seeds.

    This is useful for computing PGR with a stable baseline that doesn't vary
    across different experiment seeds.

    Args:
        strong_model: Strong model name (e.g., "Qwen/Qwen3-4B-Base")
        dataset_name: Dataset name
        epochs: Training epochs (default: BASELINE_EPOCHS=5)
        batch_size: Batch size
        lr: Learning rate
        scheduler: Learning rate scheduler
        cache_root: Optional custom cache root directory
        **extra_params: Additional hyperparameters

    Returns:
        Tuple of (mean_strong_acc, se_strong_acc, num_seeds)
        Returns (None, None, 0) if no seeds found
    """
    if epochs is None:
        epochs = BASELINE_EPOCHS
    cache = HierarchicalCache(cache_root=cache_root) if cache_root else HierarchicalCache()

    # Compute config key (ceiling training doesn't use weak model)
    config_key = compute_hyperparam_config_key(
        strong_model=strong_model,
        weak_model=None,  # No weak model for ceiling
        dataset_name=dataset_name,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        scheduler=scheduler,
        **extra_params,
    )

    # Get all seed results from "train_ceiling"
    print(f"[get_fixed_ceiling_baseline] Looking for ceiling results:")
    print(f"  - Cache root: {cache.cache_root}")
    print(f"  - Dataset: {dataset_name}")
    print(f"  - Config key: {config_key}")
    print(f"  - Idea name: train_ceiling")
    
    all_results = cache.get_all_seed_results(
        idea_name="train_ceiling",
        hyperparam_config_key=config_key,
        dataset_name=dataset_name,
    )

    print(f"[get_fixed_ceiling_baseline] Found {len(all_results)} seed results")
    
    if not all_results:
        # Check if directory exists
        hyperparam_dir = cache._get_hyperparam_dir("train_ceiling", config_key, dataset_name)
        print(f"[get_fixed_ceiling_baseline] No results found. Checking directory: {hyperparam_dir}")
        print(f"[get_fixed_ceiling_baseline] Directory exists: {hyperparam_dir.exists()}")
        if hyperparam_dir.exists():
            seed_files = list(hyperparam_dir.glob("seed_*.json"))
            print(f"[get_fixed_ceiling_baseline] Found {len(seed_files)} seed files in directory")
            if seed_files:
                print(f"[get_fixed_ceiling_baseline] Example files: {[f.name for f in seed_files[:3]]}")
        return None, None, 0

    # Extract strong_acc values
    strong_accs = []
    for r in all_results:
        acc = r.get("strong_gt_acc", r.get("strong_acc", None))
        if acc is not None:
            strong_accs.append(acc)

    if not strong_accs:
        print(f"[get_fixed_ceiling_baseline] WARNING: Found {len(all_results)} results but none have strong_acc/strong_gt_acc field")
        print(f"[get_fixed_ceiling_baseline] Available keys in first result: {list(all_results[0].keys()) if all_results else 'N/A'}")
        return None, None, 0

    mean_acc = np.mean(strong_accs)
    se_acc = np.std(strong_accs, ddof=1) / np.sqrt(len(strong_accs)) if len(strong_accs) > 1 else 0.0

    print(f"[get_fixed_ceiling_baseline] Successfully computed baseline: {mean_acc:.4f} ± {se_acc:.4f} (from {len(strong_accs)} seeds)")
    return float(mean_acc), float(se_acc), len(strong_accs)


def get_cached_ceiling_result(
    strong_model: str,
    dataset_name: str = "math",
    seed: int = 42,
    epochs: int = None,  # Defaults to BASELINE_EPOCHS
    batch_size: int = 32,
    lr: float = 1e-4,
    scheduler: str = "linear",
    **extra_params,
) -> Optional[Dict[str, Any]]:
    """
    Get cached ceiling result (strong model trained on ground truth).

    Args:
        strong_model: Strong model name (e.g., "Qwen/Qwen3-4B-Base")
        dataset_name: Dataset name
        seed: Random seed
        epochs: Training epochs (default: BASELINE_EPOCHS=5)
        batch_size: Batch size
        lr: Learning rate
        scheduler: Learning rate scheduler
        **extra_params: Additional hyperparameters

    Returns:
        Dictionary with ceiling result or None if not cached:
        {
            "strong_gt_acc": float,  # Strong model accuracy on ground truth
            "strong_model": str,
            "dataset_name": str,
            "seed": int,
            ...
        }
    """
    if epochs is None:
        epochs = BASELINE_EPOCHS
    cache = HierarchicalCache()

    # Compute config key (ceiling training doesn't use weak model)
    config_key = compute_hyperparam_config_key(
        strong_model=strong_model,
        weak_model=None,  # No weak model for ceiling
        dataset_name=dataset_name,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        scheduler=scheduler,
        **extra_params,
    )

    # Get from cache (stored under "train_ceiling")
    cached = cache.get_seed_result(
        idea_name="train_ceiling",
        hyperparam_config_key=config_key,
        seed=seed,
        dataset_name=dataset_name,
    )

    if cached:
        strong_gt_acc = cached.get('strong_gt_acc', cached.get('strong_acc', 0.0))
        print(f"✓ Found cached ceiling result for {strong_model}")
        print(f"  Strong GT accuracy: {strong_gt_acc:.4f}")

    return cached


# ============================================================================
# Command-Line Interface
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manage hierarchical cache")
    parser.add_argument("--action", choices=["list", "clear", "aggregate", "find-best"], required=True)
    parser.add_argument("--idea-name", required=True, help="Idea name")
    parser.add_argument("--dataset-name", default="math", help="Dataset name")
    parser.add_argument("--metric", default="pgr_mean", help="Metric for finding best config")

    args = parser.parse_args()

    cache = HierarchicalCache()

    if args.action == "list":
        configs = cache.list_hyperparam_configs(args.idea_name, args.dataset_name)
        print(f"\nHyperparameter configs for {args.idea_name} ({len(configs)}):")
        for config in configs:
            print(f"  - {config}")
            eval_data = cache.get_aggregated_eval(args.idea_name, config, args.dataset_name)
            if eval_data:
                if "pgr_mean" in eval_data:
                    print(f"    PGR: {eval_data['pgr_mean']:.4f} ± {eval_data['pgr_se']:.4f}")
            else:
                print(f"    (No eval.json yet)")

    elif args.action == "clear":
        cache.clear_idea(args.idea_name, args.dataset_name)

    elif args.action == "aggregate":
        configs = cache.list_hyperparam_configs(args.idea_name, args.dataset_name)
        for config in configs:
            print(f"\nAggregating {config}...")
            cache.save_aggregated_eval(args.idea_name, config, args.dataset_name)

    elif args.action == "find-best":
        cache.save_best_eval(args.idea_name, args.dataset_name, args.metric)
