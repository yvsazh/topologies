import torch
from torch import nn
import torch.nn.functional as F

class FeaturePyramid(nn.Module):
    component_id = 'feature-pyramid'

    def __init__(self, in_channels, out_channels):
        super().__init__()
        if not in_channels or min(*in_channels, out_channels) <= 0:
            raise ValueError('positive input/output channels required')
        self.lateral = nn.ModuleList((nn.Conv2d(c, out_channels, 1) for c in in_channels))
        self.output = nn.ModuleList((nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels))

    def forward(self, features):
        if len(features) != len(self.lateral):
            raise ValueError('feature level count mismatch')
        y = [None] * len(features)
        for i in range(len(features) - 1, -1, -1):
            z = self.lateral[i](features[i])
            y[i] = z if i == len(features) - 1 else z + F.interpolate(y[i + 1], size=z.shape[-2:], mode='nearest')
        return [layer(z) for layer, z in zip(self.output, y)]
