"""Exponential learning-rate decay scheduler factory."""


def exponential(optimizer, total_steps, **kwargs):
    """Return a scheduler that decays each learning rate exponentially.

    ``gamma`` is the per-step multiplier and defaults to 0.9. The learning
    rate after step ``n`` is the initial rate multiplied by ``gamma ** n``.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    gamma = kwargs.pop("gamma", 0.9)
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported exponential scheduler options: {unsupported}")
    if gamma <= 0:
        raise ValueError("gamma must be positive")

    initial_lrs = [group["lr"] for group in optimizer.param_groups]

    class _ExponentialScheduler:
        def __init__(self):
            self.step_number = 0

        def step(self):
            self.step_number += 1
            scale = gamma ** self.step_number
            for group, base_lr in zip(optimizer.param_groups, initial_lrs):
                group["lr"] = base_lr * scale

    return _ExponentialScheduler()
