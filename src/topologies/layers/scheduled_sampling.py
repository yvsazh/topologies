import math
from numbers import Real

import torch
from torch import nn


class ScheduledSampler(nn.Module):
    component_id = "scheduled-sampling"

    def __init__(self):
        super().__init__()

    def forward(self, teacher_tokens, sampled_tokens, probability, generator=None):
        if not isinstance(teacher_tokens, torch.Tensor) or not isinstance(sampled_tokens, torch.Tensor):
            raise TypeError("teacher_tokens and sampled_tokens must be tensors")
        if teacher_tokens.ndim != 2 or sampled_tokens.ndim != 2:
            raise ValueError("teacher_tokens and sampled_tokens must have shape [batch, time]")
        if teacher_tokens.shape != sampled_tokens.shape:
            raise ValueError("teacher_tokens and sampled_tokens must have the same shape")
        if teacher_tokens.dtype != sampled_tokens.dtype:
            raise ValueError("teacher_tokens and sampled_tokens must have the same dtype")
        if teacher_tokens.device != sampled_tokens.device:
            raise ValueError("teacher_tokens and sampled_tokens must be on the same device")

        if isinstance(probability, torch.Tensor):
            if probability.ndim != 0 or probability.dtype == torch.bool or probability.is_complex():
                raise ValueError("probability tensor must be a real scalar")
            probability_value = float(probability.item())
        elif isinstance(probability, Real) and not isinstance(probability, bool):
            probability_value = float(probability)
        else:
            raise TypeError("probability must be a scalar number or zero-dimensional tensor")
        if not math.isfinite(probability_value) or not 0.0 <= probability_value <= 1.0:
            raise ValueError("probability must be finite and between 0 and 1 inclusive")

        if generator is not None and not isinstance(generator, torch.Generator):
            raise ValueError("generator must be a torch.Generator or None")
        if generator is not None and generator.device.type != teacher_tokens.device.type:
            raise ValueError("generator device must match the token tensor device")

        mask = torch.rand(teacher_tokens.shape, device=teacher_tokens.device, generator=generator)
        return torch.where(mask < probability_value, teacher_tokens, sampled_tokens)
