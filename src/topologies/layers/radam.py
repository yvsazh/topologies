import math
import numbers

import torch
from torch import nn


class RAdamStep(nn.Module):
    component_id = "radam"

    @staticmethod
    def _raise_value_error(message):
        raise ValueError(message)

    @staticmethod
    def _finite_float(value, message):
        try:
            converted = float(value)
        except (OverflowError, TypeError, ValueError):
            raise ValueError(message) from None
        if not math.isfinite(converted):
            raise ValueError(message)
        return converted

    def __init__(self, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__()
        for name, value in (("beta1", beta1), ("beta2", beta2)):
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                self._raise_value_error(f"{name} must be a finite real number in [0, 1)")
            converted = self._finite_float(value, f"{name} must be a finite real number in [0, 1)")
            if not 0.0 <= converted < 1.0:
                raise ValueError(f"{name} must be a finite real number in [0, 1)")
            if name == "beta1":
                beta1 = converted
            else:
                beta2 = converted
        if isinstance(eps, bool) or not isinstance(eps, numbers.Real):
            self._raise_value_error("eps must be a finite nonnegative real number")
        eps = self._finite_float(eps, "eps must be a finite nonnegative real number")
        if eps < 0.0:
            raise ValueError("eps must be a finite nonnegative real number")
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
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
            try:
                lr = lr.item()
            except (RuntimeError, ValueError):
                self._raise_value_error("lr must be a finite nonnegative scalar")
        if isinstance(lr, bool) or not isinstance(lr, numbers.Real):
            self._raise_value_error("lr must be a finite nonnegative scalar")
        lr = self._finite_float(lr, "lr must be a finite nonnegative scalar")
        if lr < 0.0:
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
            previous_v = torch.zeros_like(parameter)
        else:
            previous_m = self.m
            previous_v = self.v
        detached_gradient = gradient.detach()
        self.m = (
            self.beta1 * previous_m + (1.0 - self.beta1) * detached_gradient
        ).detach()
        self.v = (
            self.beta2 * previous_v
            + (1.0 - self.beta2) * detached_gradient.square()
        ).detach()
        self.step += 1
        beta1_power = self.beta1**self.step
        beta2_power = self.beta2**self.step
        corrected_m = self.m / (1.0 - beta1_power)
        rho_inf = 2.0 / (1.0 - self.beta2) - 1.0
        rho_t = rho_inf - 2.0 * self.step * beta2_power / (1.0 - beta2_power)
        if rho_t > 4.0:
            rectification = math.sqrt(
                ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf)
                / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)
            )
            adaptive_rate = math.sqrt(1.0 - beta2_power) / (
                self.eps + self.v.sqrt()
            )
            return parameter - lr * rectification * corrected_m * adaptive_rate
        return parameter - lr * corrected_m
