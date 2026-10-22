import math

from torch import nn


class ClassifierFreeGuidance(nn.Module):
    component_id = "classifier-free-guidance"

    def __init__(self, w):
        super().__init__()
        try:
            w = float(w) if type(w) in (int, float) else float("nan")
        except OverflowError:
            w = float("nan")
        if not math.isfinite(w):
            raise ValueError("w must be a finite real scalar")
        self.w = w

    def forward(self, eps_cond, eps_uncond):
        if eps_cond.shape != eps_uncond.shape:
            raise ValueError("eps_cond and eps_uncond must have the same shape")
        return (1.0 + self.w) * eps_cond - self.w * eps_uncond
