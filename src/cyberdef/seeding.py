"""Utilities for reproducible experiments."""

from __future__ import annotations

import os
import random
from numbers import Integral

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch random number generators.

    Args:
        seed: Non-negative integer used as the random seed.
        deterministic: Configure cuDNN for reproducible execution when possible.

    Notes:
        PYTHONHASHSEED is set for child processes. Changing it at runtime
        does not alter hash randomization in the current Python process.
        Strict deterministic algorithms are intentionally not enforced.
    """
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")

    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative")

    if not isinstance(deterministic, bool):
        raise TypeError("deterministic must be a boolean")

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
