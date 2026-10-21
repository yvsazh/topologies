"""Linear schedule with a final re-warmup grafted onto its decay."""


def grafted(optimizer, total_steps, **kwargs):
    """Return a scheduler that warms up, decays, then re-warms the LR.

    ``warmup_steps`` and ``rewarmup_steps`` default to one step each. The
    intervening steps linearly decay from the base learning rate to zero.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    warmup_steps = kwargs.pop("warmup_steps", 1)
    rewarmup_steps = kwargs.pop("rewarmup_steps", 1)
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported grafted scheduler options: {unsupported}")
    if warmup_steps <= 0 or rewarmup_steps <= 0:
        raise ValueError("warmup_steps and rewarmup_steps must be positive")
    if warmup_steps + rewarmup_steps > total_steps:
        raise ValueError("warmup_steps and rewarmup_steps cannot exceed total_steps")

    initial_lrs = [group["lr"] for group in optimizer.param_groups]

    class _GraftedScheduler:
        def __init__(self):
            self.step_number = 0

        def step(self):
            self.step_number += 1
            step = self.step_number
            decay_steps = total_steps - warmup_steps - rewarmup_steps
            if step <= warmup_steps:
                scale = step / warmup_steps
            elif step <= warmup_steps + decay_steps:
                scale = 1.0 - (step - warmup_steps) / decay_steps
            else:
                scale = (step - warmup_steps - decay_steps) / rewarmup_steps
            scale = min(max(scale, 0.0), 1.0)
            for group, base_lr in zip(optimizer.param_groups, initial_lrs):
                group["lr"] = base_lr * scale

    return _GraftedScheduler()
