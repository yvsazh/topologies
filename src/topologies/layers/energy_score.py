import math
import numbers

import torch
from torch import nn


def _is_valid_temperature_type(temperature):
    return not isinstance(temperature, bool) and isinstance(temperature, numbers.Real)


class EnergyScore(nn.Module):
    component_id = "energy-score"

    def __init__(self, temperature=1.0):
        super().__init__()
        if not _is_valid_temperature_type(temperature):
            raise ValueError("temperature must be a finite positive scalar")
        try:
            temperature_value = float(temperature)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError(
                "temperature must be a finite positive scalar"
            ) from error
        if not math.isfinite(temperature_value) or temperature_value <= 0:
            raise ValueError("temperature must be a finite positive scalar")
        self.temperature = temperature_value

    def forward(self, logits):
        if not isinstance(logits, torch.Tensor):
            raise TypeError("logits must be a torch.Tensor")
        if logits.ndim == 0:
            raise ValueError("logits must have at least one dimension")
        if logits.shape[-1] == 0:
            raise ValueError("logits must have a non-empty class dimension")
        if not torch.isfinite(logits).all():
            raise ValueError("logits must contain only finite values")
        return -self.temperature * torch.logsumexp(
            logits / self.temperature, dim=-1
        )
