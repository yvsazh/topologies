import math
import numbers

import torch
from torch import nn


class GAE(nn.Module):
    component_id = "gae"

    def __init__(self, gamma=0.99, lam=1.0):
        super().__init__()
        for name, value in (("gamma", gamma), ("lam", lam)):
            valid_number = isinstance(value, numbers.Real) and not isinstance(value, bool)
            try:
                numeric_value = float(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"{name} must be a finite number in [0, 1]") from error
            if not valid_number or not math.isfinite(numeric_value) or not 0.0 <= numeric_value <= 1.0:
                raise ValueError(f"{name} must be a finite number in [0, 1]")
            setattr(self, name, numeric_value)

    def forward(self, rewards, values, next_values, continuation_mask):
        inputs = (rewards, values, next_values, continuation_mask)
        if any(not isinstance(value, torch.Tensor) for value in inputs):
            raise ValueError("rewards, values, next_values, and continuation_mask must be tensors")
        if any(value.ndim == 0 for value in inputs):
            raise ValueError("inputs must have a time dimension")
        if rewards.shape[0] == 0:
            raise ValueError("inputs must have a non-empty time dimension")
        if any(value.shape != rewards.shape for value in inputs[1:]):
            raise ValueError("rewards, values, next_values, and continuation_mask must have matching shapes")
        if any(value.device != rewards.device for value in inputs[1:]):
            raise ValueError("rewards, values, next_values, and continuation_mask must be on the same device")
        deltas = rewards + self.gamma * continuation_mask * next_values - values
        advantages = []
        next_advantage = torch.zeros_like(deltas[0])
        for time_index in range(deltas.shape[0] - 1, -1, -1):
            next_advantage = deltas[time_index] + (
                self.gamma * self.lam * continuation_mask[time_index] * next_advantage
            )
            advantages.append(next_advantage)
        return torch.stack(advantages[::-1], dim=0)
