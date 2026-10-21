

import math

import torch
from torch import nn


class _SurrogateStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, slope, kind):
        ctx.save_for_backward(input)
        ctx.slope = slope
        ctx.kind = kind
        return (input >= 0).to(input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        slope = ctx.slope
        if ctx.kind == "fast_sigmoid":
            derivative = 1.0 / (1.0 + slope * x.abs()).square()
        else:
            derivative = 0.5 * slope / (1.0 + (math.pi * slope * x / 2.0).square())
        return grad_output * derivative, None, None


class SurrogateGradient(nn.Module):
    component_id = "surrogate-gradient"
    """Binary Heaviside activation with fast-sigmoid or ATan backward."""

    def __init__(self, slope=1.0, surrogate="fast_sigmoid"):
        super().__init__()
        if slope <= 0:
            raise ValueError("slope must be positive")
        if surrogate not in {"fast_sigmoid", "atan"}:
            raise ValueError("surrogate must be 'fast_sigmoid' or 'atan'")
        self.slope = float(slope)
        self.surrogate = surrogate

    def forward(self, input):
        return _SurrogateStep.apply(input, self.slope, self.surrogate)


def fast_sigmoid_grad(x, slope=1.0):

    if slope <= 0:
        raise ValueError("slope must be positive")
    return 1.0 / (1.0 + slope * torch.as_tensor(x).abs()).square()


def atan_grad(x, slope=1.0):

    if slope <= 0:
        raise ValueError("slope must be positive")
    x = torch.as_tensor(x)
    return 0.5 * slope / (1.0 + (math.pi * slope * x / 2.0).square())
