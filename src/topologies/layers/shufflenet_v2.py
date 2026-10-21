import torch
from torch import nn


class ShuffleNetV2Unit(nn.Module):
    component_id = "shufflenet-v2"

    def __init__(self, channels):
        super().__init__()
        if channels % 2:
            raise ValueError("channels must be even")
        half = channels // 2
        self.branch = nn.Sequential(
            nn.Conv2d(half, half, 1, bias=False),
            nn.BatchNorm2d(half),
            nn.ReLU(inplace=True),
            nn.Conv2d(half, half, 3, padding=1, groups=half, bias=False),
            nn.BatchNorm2d(half),
            nn.Conv2d(half, half, 1, bias=False),
            nn.BatchNorm2d(half),
        )

    def forward(self, x):
        unchanged, transformed = x.chunk(2, dim=1)
        combined = torch.cat((unchanged, self.branch(transformed)), dim=1)
        batch, channels, height, width = combined.shape
        return combined.reshape(batch, 2, channels // 2, height, width).transpose(1, 2).reshape(batch, channels, height, width)
