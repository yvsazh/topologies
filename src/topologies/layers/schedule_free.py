import math

import torch
from torch import nn


class ScheduleFreeStep(nn.Module):
    component_id = "schedule-free"
    PRESETS = {"tiny": {"beta": 0.9}, "small": {"beta": 0.9}}

    def __init__(self, beta=0.9):
        super().__init__()
        if isinstance(beta, bool) or not isinstance(beta, (int, float)):
            error = ValueError("beta must be a finite number in [0, 1]")
            raise error
        try:
            beta = float(beta)
        except OverflowError as error:
            raise ValueError("beta must be a finite number in [0, 1]") from error
        if not math.isfinite(beta) or beta < 0.0 or beta > 1.0:
            raise ValueError("beta must be a finite number in [0, 1]")
        self.beta = beta
        self.register_buffer("_x", None)
        self.register_buffer("_z", None)
        self.register_buffer("_updates", torch.zeros((), dtype=torch.long))

    def _validate_lr(self, lr, parameter):
        if isinstance(lr, bool):
            error = ValueError("lr must be a positive finite scalar")
            raise error
        if isinstance(lr, torch.Tensor):
            if lr.ndim != 0 or lr.is_complex() or lr.dtype == torch.bool:
                raise ValueError("lr must be a positive finite scalar")
            if lr.device != parameter.device:
                raise ValueError("lr must be on the same device as parameter")
            try:
                finite = bool(torch.isfinite(lr).item())
                positive = bool((lr > 0).item())
            except (RuntimeError, TypeError, ValueError) as error:
                raise ValueError("lr must be a positive finite scalar") from error
            if not finite or not positive:
                raise ValueError("lr must be a positive finite scalar")
            return lr
        if not isinstance(lr, (int, float)):
            error = ValueError("lr must be a positive finite scalar")
            raise error
        try:
            lr = float(lr)
        except OverflowError as error:
            raise ValueError("lr must be a positive finite scalar") from error
        if not math.isfinite(lr) or lr <= 0.0:
            raise ValueError("lr must be a positive finite scalar")
        return lr

    def forward(self, parameter, gradient, lr, *, return_iterates=True):
        if not isinstance(parameter, torch.Tensor) or not isinstance(gradient, torch.Tensor):
            error = ValueError("parameter and gradient must be torch tensors")
            raise error
        if parameter.shape != gradient.shape:
            raise ValueError("parameter and gradient must have the same shape")
        if not parameter.is_floating_point() or not gradient.is_floating_point():
            raise ValueError("parameter and gradient must be floating-point tensors")
        if parameter.dtype != gradient.dtype:
            raise ValueError("parameter and gradient must have the same dtype")
        if parameter.device != gradient.device:
            raise ValueError("parameter and gradient must be on the same device")
        if not isinstance(return_iterates, bool):
            error = ValueError("return_iterates must be a boolean")
            raise error
        lr = self._validate_lr(lr, parameter)
        state_matches = (
            self._x is not None
            and self._x.shape == parameter.shape
            and self._x.dtype == parameter.dtype
            and self._x.device == parameter.device
        )
        x = self._x if state_matches else parameter
        z = self._z if state_matches else parameter
        z_next = z - lr * gradient
        updates = int(self._updates.item()) if state_matches else 0
        averaging_weight = 1.0 / (updates + 2)
        x_next = (1.0 - averaging_weight) * x + averaging_weight * z_next
        y_next = (1.0 - self.beta) * z_next + self.beta * x_next
        if self.training:
            self._x = x_next.detach()
            self._z = z_next.detach()
            self._updates.fill_(updates + 1)
        if return_iterates:
            return x_next, y_next
        return y_next
