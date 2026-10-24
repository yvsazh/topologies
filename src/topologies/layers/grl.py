import math
import numbers

import torch
from torch import nn


class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, scale):
        ctx.scale = scale
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg().mul(ctx.scale), None


class GradientReversalLayer(nn.Module):
    component_id = "grl"

    def __init__(self, scale=1.0):
        super().__init__()
        valid_real_scale = isinstance(scale, numbers.Real) and not isinstance(scale, bool)
        if not valid_real_scale:
            raise ValueError("scale must be a finite non-negative number")
        try:
            numeric_scale = float(scale)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("scale must be a finite non-negative number") from error
        if not math.isfinite(numeric_scale) or numeric_scale < 0:
            raise ValueError("scale must be a finite non-negative number")
        self.scale = numeric_scale

    def forward(self, inputs):
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("inputs must be a torch.Tensor")
        return _GradientReversalFunction.apply(inputs, self.scale)
