import torch
from torch import nn


class DeepSets(nn.Module):
    architecture_id = "deepsets"
    PRESETS = {
        "tiny": {"in_features": 2, "hidden_dim": 8, "out_features": 1},
        "small": {"in_features": 2, "hidden_dim": 64, "out_features": 1},
        "base": {"in_features": 2, "hidden_dim": 128, "out_features": 1},
    }

    def __init__(self, in_features=2, hidden_dim=32, out_features=1):
        super().__init__()
        for name, value in (
            ("in_features", in_features),
            ("hidden_dim", hidden_dim),
            ("out_features", out_features),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        self.phi = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor with rank 3 [B, N, D]")
        if x.ndim != 3:
            raise ValueError(f"x must have rank 3 [B, N, D], got rank {x.ndim}")
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"x last dimension must equal in_features={self.in_features}, "
                f"got {x.shape[-1]}"
            )
        if not x.is_floating_point():
            raise ValueError("x must have a floating-point dtype")
        return self.rho(self.phi(x).sum(dim=1))
