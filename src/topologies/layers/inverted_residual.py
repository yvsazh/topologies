import torch
from torch import nn

class InvertedResidual(nn.Module):
    component_id = 'inverted-residual-block'

    def __init__(self, in_channels, out_channels, stride=1, expansion=6):
        super().__init__()
        if min(in_channels, out_channels, stride, expansion) <= 0 or stride not in (1, 2):
            raise ValueError('invalid geometry')
        hidden = in_channels * expansion
        layers = []
        if expansion != 1:
            layers += [nn.Conv2d(in_channels, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True)]
        layers += [nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False), nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True), nn.Conv2d(hidden, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels)]
        self.body = nn.Sequential(*layers)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x):
        y = self.body(x)
        return x + y if self.use_residual else y
