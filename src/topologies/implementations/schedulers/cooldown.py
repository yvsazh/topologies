"""Linear learning-rate decay after an optional cooldown period."""


def cooldown(optimizer, total_steps, **kwargs):
    """Return a scheduler that holds the base rate, then decays it linearly.

    ``cooldown_steps`` defaults to zero. ``min_lr_ratio`` is the final learning
    rate as a fraction of each parameter group's initial rate.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    cooldown_steps = kwargs.pop("cooldown_steps", 0)
    min_lr_ratio = kwargs.pop("min_lr_ratio", 0.0)
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported cooldown scheduler options: {unsupported}")
    if not 0 <= cooldown_steps < total_steps:
        raise ValueError("cooldown_steps must be non-negative and less than total_steps")
    if not 0 <= min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio must be between 0 and 1")

    initial_lrs = [group["lr"] for group in optimizer.param_groups]
    decay_steps = total_steps - cooldown_steps

    class _CooldownScheduler:
        def __init__(self):
            self.step_number = 0

        def step(self):
            self.step_number += 1
            progress = min(max(self.step_number - cooldown_steps, 0) / decay_steps, 1.0)
            scale = 1.0 - progress * (1.0 - min_lr_ratio)
            for group, base_lr in zip(optimizer.param_groups, initial_lrs):
                group["lr"] = base_lr * scale

    return _CooldownScheduler()
