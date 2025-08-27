"""
Pipeline utilities: seeding and device helpers.
"""

from __future__ import annotations
import os, random
import numpy as np
import torch


def set_global_seed(seed: int = 1337, deterministic: bool = True) -> None:
    """
    Set seeds across python/random, numpy, and torch.

    Args:
        seed: integer seed.
        deterministic: if True, configure cuDNN for deterministic ops.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        # Torch might not be installed in minimal environments.
        pass
