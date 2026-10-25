import math
from numbers import Real

import torch
from torch import nn


class TargetedReg(nn.Module):
    component_id = "targeted-reg"

    def __init__(self, beta=1.0, epsilon_init=0.0):
        super().__init__()
        if isinstance(beta, bool) or not isinstance(beta, Real) or not math.isfinite(beta) or beta < 0:
            raise ValueError("beta must be a finite nonnegative number")
        if isinstance(epsilon_init, bool) or not isinstance(epsilon_init, Real) or not math.isfinite(epsilon_init):
            raise ValueError("epsilon_init must be a finite number")
        self.beta = float(beta)
        self.epsilon = nn.Parameter(torch.tensor(float(epsilon_init)))

    def forward(self, q0, q1, propensity, treatment, outcome):
        inputs = (q0, q1, propensity, treatment, outcome)
        if any(not isinstance(value, torch.Tensor) for value in inputs):
            raise ValueError("all inputs must be torch tensors")
        if any(value.shape != q0.shape for value in inputs[1:]):
            raise ValueError("input tensors must all have the same shape")
        if q0.ndim == 1:
            valid_shape = q0.shape[0] > 0
        else:
            valid_shape = q0.ndim == 2 and q0.shape[0] > 0 and q0.shape[1] == 1
        if not valid_shape:
            raise ValueError("input tensors must have shape [B] or [B, 1]")
        if any(value.device != q0.device for value in inputs[1:]):
            raise ValueError("all input tensors must be on the same device")
        if self.epsilon.device != q0.device:
            raise ValueError("module parameters and inputs must be on the same device")
        if any(not value.is_floating_point() for value in (q0, q1, propensity, outcome)):
            raise ValueError("q0, q1, propensity, and outcome must be floating-point tensors")
        if treatment.is_complex():
            raise ValueError("treatment must contain binary numeric values")
        if any(not torch.isfinite(value).all() for value in inputs):
            raise ValueError("all inputs must contain finite values")
        treatment = treatment.to(dtype=q0.dtype)
        if not torch.all((treatment == 0) | (treatment == 1)):
            raise ValueError("treatment values must be 0 or 1")
        if not torch.all((propensity > 0) & (propensity < 1)):
            raise ValueError("propensity values must be strictly between 0 and 1")
        factual = (1 - treatment) * q0 + treatment * q1
        clever_covariate = treatment / propensity - (1 - treatment) / (1 - propensity)
        corrected = factual + self.epsilon * clever_covariate
        return self.beta * (outcome - corrected).square()
