import math

import torch
from torch import nn


class GPBO(nn.Module):
    component_id = "gp-bo"
    def __init__(self, xi=0.0):
        super().__init__()
        self.xi = float(xi)
        if self.xi < 0.0:
            raise ValueError("xi must be non-negative")

    def forward(self, mean, variance, best):
        if not torch.is_tensor(mean):
            mean = torch.as_tensor(mean, dtype=torch.get_default_dtype())
        if not torch.is_tensor(variance):
            variance = torch.as_tensor(variance, dtype=mean.dtype, device=mean.device)
        if not torch.is_tensor(best):
            best = torch.as_tensor(best, dtype=mean.dtype, device=mean.device)
        if torch.any(variance < 0):
            raise ValueError("variance must be non-negative")
        improvement = mean - best - self.xi
        positive_variance = variance > 0
        safe_variance = torch.where(positive_variance, variance, torch.ones_like(variance))
        sigma = torch.sqrt(safe_variance)
        z = improvement / sigma
        cdf = 0.5 * torch.erfc(-z / math.sqrt(2.0))
        density = torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        expected = torch.relu(improvement * cdf + sigma * density)
        return torch.where(positive_variance, expected, torch.relu(improvement))

    def expected_improvement(self, mean, variance, best):
        return self(mean, variance, best)


def expected_improvement(mean, variance, best, xi=0.0):
    return GPBO(xi)(mean, variance, best)
