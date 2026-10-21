"""Linear learning-rate warmup scheduler factory."""


def warmup_linear(optimizer, total_steps, **kwargs):
    """Return a scheduler that linearly warms each learning rate to its base value.

    ``warmup_steps`` defaults to ``total_steps``. After warmup, the learning
    rate remains at its original value.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    warmup_steps = kwargs.pop("warmup_steps", total_steps)
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported linear warmup scheduler options: {unsupported}")
    if warmup_steps <= 0:
        raise ValueError("warmup_steps must be positive")

    initial_lrs = [group["lr"] for group in optimizer.param_groups]

    class _LinearWarmupScheduler:
        def __init__(self):
            self.step_number = 0

        def step(self):
            self.step_number += 1
            scale = min(self.step_number / warmup_steps, 1.0)
            for group, base_lr in zip(optimizer.param_groups, initial_lrs):
                group["lr"] = base_lr * scale

    return _LinearWarmupScheduler()
