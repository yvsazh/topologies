import math
from collections.abc import Mapping
from numbers import Real

import torch
from torch import nn


class MAML(nn.Module):
    component_id = "maml"
    PRESETS = {"tiny": {"inner_lr": 0.01}, "small": {"inner_lr": 0.01}}

    def __init__(self, inner_lr=0.01):
        super().__init__()
        try:
            valid_inner_lr = (
                isinstance(inner_lr, Real)
                and not isinstance(inner_lr, bool)
                and math.isfinite(inner_lr)
                and inner_lr > 0
            )
        except (OverflowError, TypeError, ValueError):
            valid_inner_lr = False
        if not valid_inner_lr:
            raise ValueError("inner_lr must be a finite positive number")
        self.inner_lr = float(inner_lr)

    def forward(self, params, support_grads):
        if isinstance(params, torch.Tensor) and isinstance(support_grads, torch.Tensor):
            if params.shape != support_grads.shape:
                raise ValueError("parameter and support gradient shapes must match")
            if params.device != support_grads.device or params.dtype != support_grads.dtype:
                raise ValueError("parameter and support gradient dtype and device must match")
            return params - self.inner_lr * support_grads
        if not isinstance(params, Mapping) or not isinstance(support_grads, Mapping):
            raise TypeError("params and support_grads must be mappings")
        if not params or not support_grads:
            raise ValueError("params and support_grads must be non-empty")
        if params.keys() != support_grads.keys():
            raise ValueError("params and support_grads must have matching keys")
        for name, parameter in params.items():
            gradient = support_grads[name]
            if not isinstance(parameter, torch.Tensor) or not isinstance(gradient, torch.Tensor):
                raise TypeError("each parameter and support gradient must be a tensor")
            if parameter.shape != gradient.shape:
                raise ValueError(f"parameter and support gradient shapes must match for {name!r}")
            if parameter.device != gradient.device or parameter.dtype != gradient.dtype:
                raise ValueError(f"parameter and support gradient dtype and device must match for {name!r}")
        return {
            name: parameter - self.inner_lr * support_grads[name]
            for name, parameter in params.items()
        }
