import torch
from torch import nn


class StochasticDepth(nn.Module):
    component_id = "stochastic-depth"
    PRESETS = {
        "tiny": {"survival_prob": 0.5},
        "small": {"survival_prob": 0.5},
    }

    def __init__(self, survival_prob=0.5):
        super().__init__()
        if (isinstance(survival_prob, bool)
                or not isinstance(survival_prob, (int, float))
                or not 0.0 <= survival_prob <= 1.0):
            raise ValueError("survival_prob must be a finite number in [0, 1]")
        self.survival_prob = float(survival_prob)

    def forward(self, x):
        if not torch.is_tensor(x) or x.ndim == 0:
            raise ValueError("x must be a batched tensor with at least one dimension")
        if not self.training:
            return x * self.survival_prob
        if self.survival_prob == 0.0:
            return x * 0.0
        if self.survival_prob == 1.0:
            return x
        mask = torch.empty((1,) * x.ndim, device=x.device).bernoulli_(self.survival_prob)
        return x * mask.to(dtype=x.dtype)
