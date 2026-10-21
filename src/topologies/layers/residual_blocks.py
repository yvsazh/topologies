import torch
from torch import nn

class ResidualBlock(nn.Module):
    component_id = 'residual-conv-block'

    def __init__(self, in_channels, out_channels, stride=1, kind='basic', dimension=2):
        super().__init__()
        if min(in_channels, out_channels, stride) <= 0 or kind not in ('basic', 'bottleneck') or dimension not in (1, 2):
            raise ValueError('invalid block configuration')
        conv = nn.Conv1d if dimension == 1 else nn.Conv2d
        norm = nn.BatchNorm1d if dimension == 1 else nn.BatchNorm2d
        mid = out_channels if kind == 'basic' else max(1, out_channels // 4)
        self.body = nn.Sequential(conv(in_channels, mid, 3, stride, padding=1, bias=False), norm(mid), nn.ReLU(inplace=True), conv(mid, out_channels, 3 if kind == 'basic' else 1, padding=1 if kind == 'basic' else 0, bias=False), norm(out_channels))
        self.skip = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(conv(in_channels, out_channels, 1, stride, bias=False), norm(out_channels))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))
