import torch
from torch import nn


class PointLLM(nn.Module):
    component_id = "pointllm"

    def __init__(self, point_dim=384, hidden_dim=256, text_dim=5120):
        super().__init__()
        if min(point_dim, hidden_dim, text_dim) <= 0:
            raise ValueError("feature dimensions must be positive")
        self.projector = nn.Sequential(
            nn.Linear(point_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, text_dim),
        )

    def forward(self, point_features):
        if point_features.ndim != 3 or point_features.shape[1] == 0:
            raise ValueError("point_features must have shape [batch, non-empty points, point_dim]")
        if point_features.shape[-1] != self.projector[0].in_features:
            raise ValueError("point feature dimension does not match point_dim")
        return self.projector(point_features)
