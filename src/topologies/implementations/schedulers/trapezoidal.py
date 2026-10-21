"""Linear warmup, constant plateau, and linear decay learning-rate schedule."""


def trapezoidal(optimizer, total_steps, **kwargs):
    """Create a trapezoidal schedule, updating learning rates on ``step()``.

    Warmup and decay lengths default to one quarter of the run (at least one
    step each); the remaining steps form a plateau. ``min_lr`` is the final
    learning rate (default zero).
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    warmup_steps = kwargs.pop("warmup_steps", max(1, total_steps // 4))
    decay_steps = kwargs.pop("decay_steps", max(1, total_steps // 4))
    min_lr = kwargs.pop("min_lr", 0.0)
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise TypeError(f"Trapezoidal scheduler does not support options: {unsupported}")
    if warmup_steps < 0 or decay_steps < 0 or warmup_steps + decay_steps > total_steps:
        raise ValueError("warmup_steps and decay_steps must be non-negative and fit total_steps")
    if min_lr < 0:
        raise ValueError("min_lr must be non-negative")

    base_lrs = [group["lr"] for group in optimizer.param_groups]

    class _TrapezoidalScheduler:
        def __init__(self):
            self.step_number = 0

        def step(self):
            self.step_number += 1
            step = min(self.step_number, total_steps)
            if warmup_steps and step <= warmup_steps:
                factor = step / warmup_steps
            elif step > total_steps - decay_steps and decay_steps:
                factor = (total_steps - step) / decay_steps
            else:
                factor = 1.0
            for group, base_lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = min_lr + (base_lr - min_lr) * factor

    return _TrapezoidalScheduler()
