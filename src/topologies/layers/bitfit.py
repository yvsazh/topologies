import torch
from torch import nn


class BitFit(nn.Module):
    component_id = "bitfit"
    PRESETS = {
        "tiny": {"in_features": 3, "out_features": 2},
        "small": {"in_features": 16, "out_features": 8},
    }

    def __init__(self, in_features: int = 3, out_features: int = 2) -> None:
        super().__init__()
        if not isinstance(in_features, int) or isinstance(in_features, bool) or in_features <= 0:
            raise ValueError("in_features must be a positive integer")
        if not isinstance(out_features, int) or isinstance(out_features, bool) or out_features <= 0:
            raise ValueError("out_features must be a positive integer")
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        bound = in_features ** -0.5
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)
