from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # warn_only: a few CUDA ops have no deterministic kernel; warn instead of crashing the run
    torch.use_deterministic_algorithms(True, warn_only=True)
