import math

import torch
from torch import nn


class DeltaModulation(nn.Module):
    component_id = "delta-modulation"

    def __init__(self, step_size=1.0):
        super().__init__()
        if type(step_size) in (bool, str, bytes):
            raise ValueError("step_size must be a positive finite real scalar")
        try:
            step_size = float(step_size)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("step_size must be a positive finite real scalar") from error
        if not math.isfinite(step_size) or step_size <= 0:
            raise ValueError("step_size must be a positive finite real scalar")
        self.step_size = step_size

    def forward(self, x):
        if not torch.is_tensor(x):
            raise ValueError("signal must be a floating-point tensor")
        if not torch.is_floating_point(x):
            raise ValueError("signal must have a floating-point dtype")
        if x.ndim not in (1, 2):
            raise ValueError("signal must have rank 1 or 2 with time in the last dimension")
        if not torch.isfinite(x).all():
            raise ValueError("signal must contain only finite values")
        if x.shape[-1] <= 1:
            return x[..., 1:]

        state = x[..., 0]
        decisions = []
        for index in range(1, x.shape[-1]):
            residual = x[..., index] - state
            if not torch.isfinite(residual).all():
                raise ValueError("signal recurrence must remain finite")
            hard = torch.where(residual >= 0, torch.ones_like(residual), -torch.ones_like(residual))
            decision = hard + (residual - residual.detach())
            decisions.append(decision)
            state = state + self.step_size * decision
        return torch.stack(decisions, dim=-1)

    def reconstruct(self, initial_value, decisions):
        if not torch.is_tensor(decisions):
            raise ValueError("decisions must be a floating-point tensor")
        if not torch.is_floating_point(decisions):
            raise ValueError("decisions must have a floating-point dtype")
        if decisions.ndim not in (1, 2):
            raise ValueError("decisions must have rank 1 or 2")
        if not torch.isfinite(decisions).all():
            raise ValueError("decisions must contain only finite values")
        valid_codes = (decisions == 1) | (decisions == -1)
        if not valid_codes.all():
            raise ValueError("decisions must contain only -1 or 1")

        try:
            initial = torch.as_tensor(initial_value, device=decisions.device)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError("initial_value must be a finite real scalar or one value per batch") from error
        if initial.is_complex() or initial.dtype == torch.bool:
            raise ValueError("initial_value must be a finite real scalar or one value per batch")
        if not torch.isfinite(initial).all():
            raise ValueError("initial_value must contain only finite values")
        initial = initial.to(dtype=decisions.dtype)
        if not torch.isfinite(initial).all():
            raise ValueError("initial_value must be representable in the decisions dtype")

        if initial.ndim == 0:
            state = initial.expand(decisions.shape[0]) if decisions.ndim == 2 else initial
        elif initial.ndim == 1 and decisions.ndim == 2 and initial.shape[0] == decisions.shape[0]:
            state = initial
        elif initial.ndim == 1 and decisions.ndim == 1 and initial.shape[0] == 1:
            state = initial[0]
        else:
            raise ValueError("initial_value batch dimensions must match decisions")

        reconstructed = [state]
        for index in range(decisions.shape[-1]):
            state = state + self.step_size * decisions[..., index]
            reconstructed.append(state)
        return torch.stack(reconstructed, dim=-1)
