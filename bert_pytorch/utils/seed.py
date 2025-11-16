"""Reproducibility utilities for BERT-pytorch"""

import random
import numpy as np
import torch
from typing import Optional

from .logger import get_logger

logger = get_logger(__name__)


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)

    Args:
        seed: Random seed value
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be an integer, got {type(seed)}")

    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    # Set Python random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set PyTorch random seed
    torch.manual_seed(seed)

    # Set CUDA random seed if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # For full reproducibility with CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Random seed set to {seed}")
