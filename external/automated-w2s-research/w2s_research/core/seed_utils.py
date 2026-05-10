"""
Utilities for setting random seeds to ensure reproducible training.

This module provides functions to set random seeds for:
- Python's random module
- NumPy
- PyTorch (CPU and CUDA)
- CUDA deterministic operations
- Transformers library
"""
import random
import numpy as np
import torch
from transformers import set_seed as transformers_set_seed

def set_seed(seed: int):
    """
    Set random seeds for all libraries to ensure reproducible results.

    This function sets seeds for:
    - Python's random module
    - NumPy's random number generator
    - PyTorch's random number generator (CPU)
    - PyTorch's CUDA random number generator (if available)
    - Transformers library (if available)

    Args:
        seed: Random seed value (integer)
    """
    # Set Python, NumPy, and PyTorch seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers_set_seed(seed)


    print(f"✓ Random seed set to {seed}")
