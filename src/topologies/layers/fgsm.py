import math

import torch
from torch import nn


class FastGradientSignMethod(nn.Module):
    component_id = "fgsm"

    def __init__(self, epsilon=0.25):
        super().__init__()
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
            error = ValueError("epsilon must be a finite nonnegative number")
            raise error
        try:
            epsilon = float(epsilon)
        except OverflowError as error:
            raise ValueError("epsilon must be a finite nonnegative number") from error
        if not math.isfinite(epsilon) or epsilon < 0:
            raise ValueError("epsilon must be a finite nonnegative number")
        self.epsilon = epsilon

    def forward(self, x, gradient):
        if not isinstance(x, torch.Tensor) or not isinstance(gradient, torch.Tensor):
            error = ValueError("x and gradient must be torch tensors")
            raise error
        if x.shape != gradient.shape:
            raise ValueError("x and gradient must have the same shape")
        if not x.is_floating_point() or not gradient.is_floating_point():
            raise ValueError("x and gradient must be floating-point tensors")
        if x.device != gradient.device:
            raise ValueError("x and gradient must be on the same device")
        return x + self.epsilon * torch.sign(gradient.detach())
