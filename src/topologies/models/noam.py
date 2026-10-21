import torch
from torch import nn


class NoamSchedule(nn.Module):
    def __init__(self, d_model, warmup):
        super().__init__()
        if isinstance(d_model, bool) or not isinstance(d_model, (int, float)) or d_model <= 0:
            raise ValueError("d_model must be positive")
        if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup <= 0:
            raise ValueError("warmup must be a positive integer")
        self.d_model = float(d_model)
        self.warmup = warmup

    def forward(self, step):
        if isinstance(step, bool):
            raise ValueError("step must be a positive integer or scalar tensor")
        if isinstance(step, int):
            step = torch.tensor(step, dtype=torch.get_default_dtype())
        elif isinstance(step, torch.Tensor):
            if step.ndim != 0 or step.dtype == torch.bool or step.is_complex():
                raise ValueError("step must be a positive integer or scalar tensor")
            step = step.to(dtype=torch.get_default_dtype())
            if not torch.isfinite(step).item() or step.item() != int(step.item()):
                raise ValueError("step must be a positive integer or scalar tensor")
        else:
            raise ValueError("step must be a positive integer or scalar tensor")
        if step.item() <= 0:
            raise ValueError("step must be positive")
        warmup = step.new_tensor(self.warmup)
        scale = step.new_tensor(self.d_model).rsqrt()
        return scale * torch.minimum(step.rsqrt(), step * warmup.pow(-1.5))
