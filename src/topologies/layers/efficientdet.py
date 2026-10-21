import torch
from torch import nn
from torch.nn import functional as F


class EfficientDet(nn.Module):
    component_id = "efficientdet"
    def __init__(self, in_channels=4, eps=1e-4):
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.in_channels = in_channels
        self.eps = eps
        self.fusion_weights = nn.Parameter(torch.ones(2))

    def normalized_weights(self):
        positive = F.relu(self.fusion_weights)
        return positive / (positive.sum() + self.eps)

    def forward(self, level_one, level_two):
        if level_one.ndim != 4 or level_two.ndim != 4:
            raise ValueError("feature maps must be four-dimensional NCHW tensors")
        if level_one.shape[:2] != level_two.shape[:2]:
            raise ValueError("feature maps must have matching batch and channel dimensions")
        if level_one.shape[1] != self.in_channels:
            raise ValueError("feature channels do not match in_channels")
        if level_one.shape[-2:] != level_two.shape[-2:]:
            level_two = F.interpolate(level_two, size=level_one.shape[-2:], mode="nearest")
        weights = self.normalized_weights()
        return weights[0] * level_one + weights[1] * level_two
