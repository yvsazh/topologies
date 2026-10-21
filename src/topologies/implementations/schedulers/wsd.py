"""Warmup-stable-decay learning-rate schedule."""

import math


def wsd(optimizer, total_steps, **kwargs):
    """Create a WSD scheduler for an optimizer.

    ``warmup_steps`` and ``stable_steps`` define the initial linear warmup and
    constant-rate phases. The remaining steps decay the rate to ``min_lr_ratio``
    of the initial rate, using cosine decay by default (or linear decay).
    """
    import torch

    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    warmup_steps = kwargs.pop("warmup_steps", max(1, int(total_steps * 0.1)))
    stable_steps = kwargs.pop("stable_steps", max(0, int(total_steps * 0.4)))
    min_lr_ratio = kwargs.pop("min_lr_ratio", 0.0)
    decay = kwargs.pop("decay", "cosine")
    if kwargs:
        raise TypeError(f"unexpected scheduler options: {', '.join(kwargs)}")
    if warmup_steps < 0 or stable_steps < 0:
        raise ValueError("warmup_steps and stable_steps must be non-negative")
    if warmup_steps + stable_steps > total_steps:
        raise ValueError("warmup_steps + stable_steps cannot exceed total_steps")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be between 0 and 1")
    if decay not in ("cosine", "linear"):
        raise ValueError("decay must be 'cosine' or 'linear'")

    def multiplier(step):
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        decay_start = warmup_steps + stable_steps
        if step < decay_start:
            return 1.0
        decay_steps = total_steps - decay_start
        if decay_steps <= 1:
            progress = 1.0
        else:
            progress = min(1.0, max(0.0, (step - decay_start) / (decay_steps - 1)))
        shaped = (1.0 - math.cos(math.pi * progress)) / 2 if decay == "cosine" else progress
        return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - shaped)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
