import math

import torch
from torch import nn

_value_error = ValueError


class OneCycle(nn.Module):
    component_id = "onecycle"

    def __init__(self, max_lr, start_lr, end_lr, steps, warmup):
        super().__init__()
        for name, value in (("max_lr", max_lr), ("start_lr", start_lr), ("end_lr", end_lr)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise _value_error(f"{name} must be a finite number")
        if isinstance(steps, bool) or not isinstance(steps, int):
            raise _value_error("steps must be an integer")
        if isinstance(warmup, bool) or not isinstance(warmup, int):
            raise _value_error("warmup must be an integer")
        if not 0 < warmup < steps:
            raise _value_error("warmup must satisfy 0 < warmup < steps")
        self.max_lr = float(max_lr)
        self.start_lr = float(start_lr)
        self.end_lr = float(end_lr)
        self.steps = steps
        self.warmup = warmup

    def forward(self, step):
        if isinstance(step, torch.Tensor):
            if step.numel() == 0:
                raise _value_error("step tensor must be nonempty")
            if step.is_floating_point():
                query = step
            else:
                if step.ndim != 0:
                    raise _value_error("integer step tensor must be a scalar")
                try:
                    torch.iinfo(step.dtype)
                except TypeError as exc:
                    raise _value_error("step tensor must have an integer or floating-point dtype") from exc
                query = step.to(dtype=torch.float64)
            if not bool(torch.isfinite(query).all()):
                raise _value_error("step tensor must contain finite values")
            if bool(((query < 0) | (query > self.steps)).any()):
                raise _value_error("step must satisfy 0 <= step <= steps")
            rising = self.start_lr + (self.max_lr - self.start_lr) * query / self.warmup
            falling = self.max_lr + (self.end_lr - self.max_lr) * (query - self.warmup) / (
                self.steps - self.warmup
            )
            return torch.where(query <= self.warmup, rising, falling)
        if isinstance(step, bool) or not isinstance(step, int):
            raise _value_error("step must be an integer or a scalar integer tensor")
        if not 0 <= step <= self.steps:
            raise _value_error("step must satisfy 0 <= step <= steps")
        if step == 0:
            return self.start_lr
        if step == self.warmup:
            return self.max_lr
        if step == self.steps:
            return self.end_lr
        if step < self.warmup:
            return self.start_lr + (self.max_lr - self.start_lr) * step / self.warmup
        return self.max_lr + (self.end_lr - self.max_lr) * (step - self.warmup) / (
            self.steps - self.warmup
        )
