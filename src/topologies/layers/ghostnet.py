import torch
from torch import nn


class GhostModule(nn.Module):
    component_id = "ghostnet"

    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2):
        super().__init__()
        if in_channels < 1 or out_channels < 1 or kernel_size < 1 or ratio < 2:
            raise ValueError("channel counts and kernel_size must be positive; ratio must be at least two")
        primary_channels = (out_channels + ratio - 1) // ratio
        ghost_channels = primary_channels * (ratio - 1)
        padding = kernel_size // 2
        self.primary = nn.Conv2d(in_channels, primary_channels, kernel_size, padding=padding, bias=False)
        self.cheap = nn.Conv2d(primary_channels, ghost_channels, 3, padding=1, groups=primary_channels, bias=False) if ghost_channels else None
        self.out_channels = out_channels

    def forward(self, x):
        primary = self.primary(x)
        if self.cheap is None:
            return primary
        return torch.cat((primary, self.cheap(primary)), dim=1)[:, :self.out_channels]


class GhostNet(nn.Module):
    architecture_id = "ghostnet"
    def __init__(self, in_channels=3, num_classes=1000, stages=3, channels=16, input_size=224, **kwargs):
        super().__init__()
        if stages < 1 or channels < 1:
            raise ValueError("stages and channels must be positive")
        blocks = []
        current_channels = in_channels
        for _ in range(stages):
            blocks.extend((GhostModule(current_channels, channels), nn.BatchNorm2d(channels), nn.ReLU(inplace=True)))
            current_channels = channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
