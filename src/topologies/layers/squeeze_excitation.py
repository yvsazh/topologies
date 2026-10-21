import torch
from torch import nn

class SqueezeExcitation(nn.Module):
    component_id = 'squeeze-excitation'

    def __init__(self, channels, reduction=4):
        super().__init__()
        if min(channels, reduction) <= 0:
            raise ValueError('channels/reduction must be positive')
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, channels, 1), nn.Sigmoid())

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.fc[0].in_channels:
            raise ValueError('expected [B,C,H,W]')
        return x * self.fc(self.pool(x))
