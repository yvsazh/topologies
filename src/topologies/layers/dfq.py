import torch
from torch import nn


class DFQWeightEqualization(nn.Module):
    component_id = "dfq"
    PRESETS = {
        "tiny": {"in_features": 2, "hidden_features": 2, "out_features": 2},
        "small": {"in_features": 8, "hidden_features": 16, "out_features": 4},
        "base": {"in_features": 16, "hidden_features": 32, "out_features": 8},
    }

    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        dimensions = {
            "in_features": in_features,
            "hidden_features": hidden_features,
            "out_features": out_features,
        }
        for name, value in dimensions.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.first = nn.Linear(in_features, hidden_features)
        self.second = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.second(torch.relu(self.first(x)))

    @torch.no_grad()
    def equalize_(self):
        first_ranges = 2 * self.first.weight.abs().amax(dim=1)
        second_ranges = 2 * self.second.weight.abs().amax(dim=0)
        if (
            not torch.isfinite(first_ranges).all()
            or not torch.isfinite(second_ranges).all()
            or torch.any(first_ranges <= 0)
            or torch.any(second_ranges <= 0)
        ):
            raise ValueError("all hidden-channel weight ranges must be positive and finite")
        scales = torch.sqrt(first_ranges) / torch.sqrt(second_ranges)
        if not torch.isfinite(scales).all() or torch.any(scales <= 0):
            raise ValueError("equalization scales must be positive and finite")
        first_weight = self.first.weight / scales.unsqueeze(1)
        first_bias = self.first.bias / scales
        second_weight = self.second.weight * scales.unsqueeze(0)
        if (
            not torch.isfinite(first_weight).all()
            or not torch.isfinite(first_bias).all()
            or not torch.isfinite(second_weight).all()
        ):
            raise ValueError("equalized parameters must be finite")
        self.first.weight.copy_(first_weight)
        self.first.bias.copy_(first_bias)
        self.second.weight.copy_(second_weight)
        return scales
