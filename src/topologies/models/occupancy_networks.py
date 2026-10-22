import math

import torch
from torch import nn
from torch.nn import functional as F


class OccupancyNetwork(nn.Module):
    architecture_id = "occupancy-networks"
    PRESETS = {
        "tiny": {
            "point_dim": 3,
            "condition_dim": 8,
            "hidden_dim": 16,
            "num_blocks": 2,
            "cbn_eps": 1e-5,
            "batch_norm_momentum": 0.1,
        },
        "small": {
            "point_dim": 3,
            "condition_dim": 32,
            "hidden_dim": 64,
            "num_blocks": 5,
            "cbn_eps": 1e-5,
            "batch_norm_momentum": 0.1,
        },
        "base": {
            "point_dim": 3,
            "condition_dim": 128,
            "hidden_dim": 256,
            "num_blocks": 5,
            "cbn_eps": 1e-5,
            "batch_norm_momentum": 0.1,
        },
    }

    class ConditionalBatchNorm(nn.Module):
        def __init__(self, condition_dim, hidden_dim, eps=1e-5, momentum=0.1):
            super().__init__()
            self.norm = nn.BatchNorm1d(
                hidden_dim, eps=eps, momentum=momentum, affine=False
            )
            self.gamma = nn.Linear(condition_dim, hidden_dim)
            self.beta = nn.Linear(condition_dim, hidden_dim)
            nn.init.zeros_(self.gamma.weight)
            nn.init.ones_(self.gamma.bias)
            nn.init.zeros_(self.beta.weight)
            nn.init.zeros_(self.beta.bias)

        def forward(self, features, condition):
            normalized = self.norm(features.transpose(1, 2)).transpose(1, 2)
            gamma = self.gamma(condition).unsqueeze(1)
            beta = self.beta(condition).unsqueeze(1)
            return gamma * normalized + beta

    class ResidualBlock(nn.Module):
        def __init__(self, condition_dim, hidden_dim, eps=1e-5, momentum=0.1):
            super().__init__()
            self.norm0 = OccupancyNetwork.ConditionalBatchNorm(
                condition_dim, hidden_dim, eps=eps, momentum=momentum
            )
            self.fc0 = nn.Linear(hidden_dim, hidden_dim)
            self.norm1 = OccupancyNetwork.ConditionalBatchNorm(
                condition_dim, hidden_dim, eps=eps, momentum=momentum
            )
            self.fc1 = nn.Linear(hidden_dim, hidden_dim)
            nn.init.zeros_(self.fc1.weight)
            nn.init.zeros_(self.fc1.bias)

        def forward(self, features, condition):
            residual = self.fc0(F.relu(self.norm0(features, condition)))
            residual = self.fc1(F.relu(self.norm1(residual, condition)))
            return features + residual

    def __init__(
        self,
        point_dim=3,
        condition_dim=128,
        hidden_dim=256,
        num_blocks=5,
        cbn_eps=1e-5,
        batch_norm_momentum=0.1,
    ):
        super().__init__()
        for name, value in (
            ("point_dim", point_dim),
            ("condition_dim", condition_dim),
            ("hidden_dim", hidden_dim),
            ("num_blocks", num_blocks),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(cbn_eps, (int, float))
            or isinstance(cbn_eps, bool)
            or not math.isfinite(cbn_eps)
            or cbn_eps <= 0
        ):
            raise ValueError("cbn_eps must be a positive finite number")
        if (
            not isinstance(batch_norm_momentum, (int, float))
            or isinstance(batch_norm_momentum, bool)
            or not math.isfinite(batch_norm_momentum)
            or not 0 <= batch_norm_momentum <= 1
        ):
            raise ValueError("batch_norm_momentum must be a finite number between 0 and 1")

        self.point_dim = point_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.point_projection = nn.Linear(point_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            self.ResidualBlock(
                condition_dim,
                hidden_dim,
                eps=cbn_eps,
                momentum=batch_norm_momentum,
            )
            for _ in range(num_blocks)
        )
        self.final_norm = self.ConditionalBatchNorm(
            condition_dim, hidden_dim, eps=cbn_eps, momentum=batch_norm_momentum
        )
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, points, condition):
        if not isinstance(points, torch.Tensor) or points.ndim != 3:
            raise ValueError("points must have shape (batch, queries, point_dim)")
        if points.shape[-1] != self.point_dim:
            raise ValueError(f"points last dimension must equal point_dim={self.point_dim}")
        if not isinstance(condition, torch.Tensor) or condition.ndim != 2:
            raise ValueError("conditioning must have shape (batch, condition_dim)")
        if condition.shape[-1] != self.condition_dim:
            raise ValueError(
                f"conditioning last dimension must equal condition_dim={self.condition_dim}"
            )
        if condition.shape[0] != points.shape[0]:
            raise ValueError("points and conditioning must have the same batch size")
        if points.shape[1] < 1:
            raise ValueError("points must contain at least one query point")

        hidden = self.point_projection(points)
        for block in self.blocks:
            hidden = block(hidden, condition)
        hidden = F.relu(self.final_norm(hidden, condition))
        return self.output(hidden).squeeze(-1)
