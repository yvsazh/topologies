import math
import numbers

import torch
from torch import nn


class FastAdversarialStep(nn.Module):
    component_id = "fast-at"

    def __init__(self, epsilon=8 / 255, alpha=None):
        super().__init__()
        if isinstance(epsilon, bool) or not isinstance(epsilon, numbers.Real):
            error = ValueError("epsilon must be a finite positive number")
            raise error
        try:
            epsilon = float(epsilon)
        except OverflowError as error:
            raise ValueError("epsilon must be a finite positive number") from error
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be a finite positive number")
        if alpha is None:
            alpha = 1.25 * epsilon
        if isinstance(alpha, bool) or not isinstance(alpha, numbers.Real):
            error = ValueError("alpha must be a finite positive number")
            raise error
        try:
            alpha = float(alpha)
        except OverflowError as error:
            raise ValueError("alpha must be a finite positive number") from error
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be a finite positive number")
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, x, delta_0, gradient):
        if not all(isinstance(value, torch.Tensor) for value in (x, delta_0, gradient)):
            raise ValueError("x, delta_0, and gradient must be torch tensors")
        if x.shape != delta_0.shape or x.shape != gradient.shape:
            raise ValueError("x, delta_0, and gradient must have the same shape")
        if not all(value.is_floating_point() for value in (x, delta_0, gradient)):
            raise ValueError("x, delta_0, and gradient must be floating-point tensors")
        if x.dtype != delta_0.dtype or x.dtype != gradient.dtype:
            raise ValueError("x, delta_0, and gradient must have the same dtype")
        if x.device != delta_0.device or x.device != gradient.device:
            raise ValueError("x, delta_0, and gradient must be on the same device")
        perturbation = delta_0.detach().clamp(min=-self.epsilon, max=self.epsilon)
        perturbation = (perturbation + self.alpha * gradient.detach().sign()).clamp(
            min=-self.epsilon, max=self.epsilon
        )
        return x + perturbation
