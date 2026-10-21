"""Noam learning-rate schedule used for Transformer training."""



def noam(optimizer, total_steps, **kwargs):
    """Return an object that applies the Noam learning-rate schedule.

    ``warmup_steps`` defaults to ``total_steps`` so a finite training run has
    a rising warmup throughout the run. ``model_size`` and ``factor`` control
    the scale, as in the original schedule.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    model_size = kwargs.pop("model_size", kwargs.pop("d_model", 512))
    warmup_steps = kwargs.pop("warmup_steps", total_steps)
    factor = kwargs.pop("factor", 1.0)
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported Noam scheduler options: {unsupported}")
    if model_size <= 0 or warmup_steps <= 0 or factor <= 0:
        raise ValueError("model_size, warmup_steps, and factor must be positive")

    initial_lrs = [group["lr"] for group in optimizer.param_groups]

    class _NoamScheduler:
        def __init__(self):
            self.step_number = 0

        def step(self):
            self.step_number += 1
            step = self.step_number
            scale = factor * model_size ** -0.5 * min(
                step ** -0.5, step * warmup_steps ** -1.5
            )
            for group, base_lr in zip(optimizer.param_groups, initial_lrs):
                group["lr"] = base_lr * scale

    return _NoamScheduler()
