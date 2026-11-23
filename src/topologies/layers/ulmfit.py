import math

import torch
from torch import nn

_value_error = ValueError


class SlantedTriangular(nn.Module):
    component_id = "ulmfit"
    PRESETS = {
        "tiny": {"max_lr": 0.01, "steps": 10, "cut_frac": 0.1, "ratio": 32},
        "small": {"max_lr": 0.01, "steps": 100, "cut_frac": 0.1, "ratio": 32},
        "base": {"max_lr": 0.01, "steps": 1000, "cut_frac": 0.1, "ratio": 32},
    }

    def __init__(self, steps, max_lr=0.01, cut_frac=0.1, ratio=32):
        super().__init__()
        for name, value in (("max_lr", max_lr), ("cut_frac", cut_frac), ("ratio", ratio)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise _value_error(f"{name} must be a finite number")
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError(f"{name} must be a finite number")
        if max_lr <= 0:
            raise ValueError("max_lr must be positive")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("steps must be a positive integer")
        if not 0 < cut_frac < 1:
            raise ValueError("cut_frac must be between 0 and 1")
        if isinstance(ratio, bool) or ratio < 1:
            raise ValueError("ratio must be at least 1")
        try:
            cut = math.floor(steps * cut_frac)
        except OverflowError as exc:
            raise ValueError("steps * cut_frac must have a positive floor") from exc
        if cut == 0:
            raise ValueError("steps * cut_frac must have a positive floor")
        self.max_lr = float(max_lr)
        self.steps = steps
        self.cut_frac = float(cut_frac)
        self.ratio = float(ratio)
        self.cut = cut

    def forward(self, step):
        if isinstance(step, torch.Tensor):
            if step.ndim > 2 or step.numel() == 0 or (step.ndim == 2 and step.shape[-1] != 1):
                raise ValueError("step tensor must be scalar or a nonempty batch of scalar steps")
            if step.is_floating_point():
                query = step
            else:
                try:
                    torch.iinfo(step.dtype)
                except (TypeError, RuntimeError) as exc:
                    raise ValueError("step tensor must have an integer or floating-point dtype") from exc
                query = step.to(dtype=torch.get_default_dtype())
        elif isinstance(step, int) and not isinstance(step, bool):
            query = torch.tensor(step, dtype=torch.get_default_dtype())
        else:
            raise _value_error("step must be an integer or a scalar integer or floating-point tensor")
        if not bool(torch.isfinite(query).all()):
            raise ValueError("step must be finite")
        if bool((query < 0).any()):
            raise ValueError("step must be nonnegative")
        cut = query.new_tensor(self.cut)
        rising_position = query / self.cut
        decay_denominator = self.cut * (1 / self.cut_frac - 1)
        falling_position = 1 - (query - cut) / decay_denominator
        position = torch.where(query < cut, rising_position, falling_position)
        return self.max_lr * (1 + position * (self.ratio - 1)) / self.ratio
