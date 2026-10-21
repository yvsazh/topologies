import torch
from torch import nn


class I3DInflation(nn.Module):
    component_id = "i3d"

    def __init__(self, temporal_size=3):
        super().__init__()
        if temporal_size <= 0:
            raise ValueError("temporal_size must be positive")
        self.temporal_size = temporal_size

    def forward(self, weight_2d):
        if not torch.is_tensor(weight_2d) or weight_2d.ndim != 4:
            raise ValueError("weight_2d must have shape (out_channels, in_channels, height, width)")
        return weight_2d.unsqueeze(2).expand(-1, -1, self.temporal_size, -1, -1) / self.temporal_size
