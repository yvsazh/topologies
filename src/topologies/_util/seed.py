"""Global seeding for reproducible runs."""

from __future__ import annotations

import random
import sys

import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy (when already imported), torch and every CUDA device."""
    random.seed(seed)
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        numpy.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
