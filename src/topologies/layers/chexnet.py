import torch
from torch import nn


class CheXNet(nn.Module):
    component_id = "chexnet"
    PRESETS = {
        "tiny": {"in_features": 16, "num_classes": 14},
        "small": {"in_features": 64, "num_classes": 14},
        "base": {"in_features": 64, "num_classes": 14},
    }

    def __init__(self, in_features, num_classes=14, bias=True):
        super().__init__()
        if not isinstance(in_features, int) or isinstance(in_features, bool) or in_features <= 0:
            raise ValueError("in_features must be a positive integer")
        if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")
        if not isinstance(bias, bool):
            error = ValueError("bias must be a boolean")
            raise error
        self.linear = nn.Linear(in_features, num_classes, bias=bias)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim != 2:
            raise ValueError("x must have rank 2 with shape [batch, in_features]")
        if x.shape[1] != self.linear.in_features:
            raise ValueError(
                f"x feature width must be {self.linear.in_features}, got {x.shape[1]}"
            )
        return torch.sigmoid(self.linear(x))
