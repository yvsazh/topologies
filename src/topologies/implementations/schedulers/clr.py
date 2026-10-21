"""Cyclical learning-rate scheduler factory."""

import torch


def clr(optimizer, total_steps, **kwargs):
    """Create a triangular cyclical learning-rate scheduler.

    ``total_steps`` defines one complete rise-and-fall cycle. Options are
    forwarded to PyTorch's ``CyclicLR`` (for example ``base_lr`` and ``max_lr``).
    """
    if total_steps < 2:
        raise ValueError("total_steps must be at least 2")
    step_size_up = (total_steps + 1) // 2
    step_size_down = total_steps - step_size_up
    return torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        step_size_up=step_size_up,
        step_size_down=step_size_down,
        **kwargs,
    )
