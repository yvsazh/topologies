import math
import numbers

import torch
from torch import nn


class AdamMiniStep(nn.Module):
    component_id = "adam-mini"

    @staticmethod
    def _raise_value_error(message):
        raise ValueError(message)

    def __init__(self, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__()
        if isinstance(beta1, bool) or not isinstance(beta1, numbers.Real):
            self._raise_value_error("beta1 must be a finite real number in [0, 1)")
        if isinstance(beta2, bool) or not isinstance(beta2, numbers.Real):
            self._raise_value_error("beta2 must be a finite real number in [0, 1)")
        if not math.isfinite(float(beta1)) or not 0.0 <= float(beta1) < 1.0:
            raise ValueError("beta1 must be a finite real number in [0, 1)")
        if not math.isfinite(float(beta2)) or not 0.0 <= float(beta2) < 1.0:
            raise ValueError("beta2 must be a finite real number in [0, 1)")
        if isinstance(eps, bool) or not isinstance(eps, numbers.Real):
            self._raise_value_error("eps must be a positive finite real number")
        if not math.isfinite(float(eps)) or float(eps) <= 0.0:
            raise ValueError("eps must be a positive finite real number")
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.register_buffer("m", None)
        self.register_buffer("v", None)
        self.step = 0

    def forward(self, parameter, gradient, lr):
        if not isinstance(parameter, torch.Tensor) or not isinstance(gradient, torch.Tensor):
            self._raise_value_error("parameter and gradient must be tensors")
        if parameter.numel() == 0 or gradient.numel() == 0:
            raise ValueError("parameter and gradient must be nonempty")
        if not parameter.is_floating_point() or not gradient.is_floating_point():
            raise ValueError("parameter and gradient must have floating-point dtype")
        if parameter.shape != gradient.shape:
            raise ValueError("parameter and gradient must have the same shape")
        if parameter.dtype != gradient.dtype:
            raise ValueError("parameter and gradient must have the same dtype")
        if parameter.device != gradient.device:
            raise ValueError("parameter and gradient must be on the same device")
        if isinstance(lr, torch.Tensor):
            if lr.numel() != 1:
                raise ValueError("lr must be a finite nonnegative scalar")
            lr = lr.item()
        if isinstance(lr, bool) or not isinstance(lr, numbers.Real):
            self._raise_value_error("lr must be a finite nonnegative scalar")
        lr = float(lr)
        if not math.isfinite(lr) or lr < 0.0:
            raise ValueError("lr must be a finite nonnegative scalar")
        if self.m is not None and (
            self.m.shape != parameter.shape
            or self.m.dtype != parameter.dtype
            or self.m.device != parameter.device
        ):
            self.m = None
            self.v = None
            self.step = 0
        if self.m is None:
            previous_m = torch.zeros_like(parameter)
            previous_v = torch.zeros((), dtype=parameter.dtype, device=parameter.device)
        else:
            previous_m = self.m
            previous_v = self.v
        detached_gradient = gradient.detach()
        self.m = (
            self.beta1 * previous_m + (1.0 - self.beta1) * detached_gradient
        ).detach()
        self.v = (
            self.beta2 * previous_v
            + (1.0 - self.beta2) * detached_gradient.square().mean()
        ).detach()
        self.step += 1
        corrected_m = self.m / (1.0 - self.beta1**self.step)
        corrected_v = self.v / (1.0 - self.beta2**self.step)
        return parameter - lr * corrected_m / (corrected_v.sqrt() + self.eps)
