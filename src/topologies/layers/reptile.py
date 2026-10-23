import math
from numbers import Real

import torch
from torch import nn


class Reptile(nn.Module):
    component_id = "reptile"

    def __init__(self, epsilon):
        super().__init__()
        epsilon_is_real = isinstance(epsilon, Real) and not isinstance(epsilon, bool)
        try:
            epsilon = float(epsilon) if epsilon_is_real else float("nan")
        except OverflowError as error:
            raise ValueError("epsilon must be a finite real scalar") from error
        if not math.isfinite(epsilon):
            raise ValueError("epsilon must be a finite real scalar")
        self.epsilon = epsilon

    def forward(self, initial, adapted):
        if not isinstance(initial, torch.Tensor) or not isinstance(adapted, torch.Tensor):
            raise TypeError("initial and adapted parameters must be tensors")
        if initial.dtype != adapted.dtype or initial.device != adapted.device:
            raise ValueError("initial and adapted parameters must have matching dtype and device")
        if not (torch.is_floating_point(initial) or torch.is_complex(initial)):
            raise ValueError("parameter tensors must have a floating-point or complex dtype")
        if adapted.shape == initial.shape:
            task_mean = adapted
        elif adapted.ndim == initial.ndim + 1 and adapted.shape[1:] == initial.shape:
            if adapted.shape[0] == 0:
                raise ValueError("the adapted task axis must not be empty")
            task_mean = adapted.mean(dim=0)
        else:
            raise ValueError(
                "adapted parameters must have the same shape or have a leading task axis"
            )
        return initial + self.epsilon * (task_mean - initial)
