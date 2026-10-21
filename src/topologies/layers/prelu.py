

import torch
from torch import nn


def prelu(x, slope):

    return torch.where(x > 0, x, slope * x)


class PReLU(nn.Module):
    component_id = "prelu"
    """PReLU activation with one learned, shared negative slope."""

    def __init__(self, initial_slope: float = 0.25) -> None:
        super().__init__()
        self.slope = nn.Parameter(torch.tensor(float(initial_slope)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return prelu(x, self.slope)
