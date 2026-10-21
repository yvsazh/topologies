import math

import torch
from torch import nn


def generate_stage_configuration(w0, wa, wm, depth):
    if w0 <= 0 or wa < 0 or wm <= 1 or depth <= 0:
        raise ValueError("invalid RegNet design parameters")
    stages = []
    for j in range(depth):
        linear_width = w0 + wa * j
        exponent = round(math.log(linear_width / w0) / math.log(wm))
        stage_width = int(round(w0 * (wm ** exponent)))
        if stages and stages[-1][0] == stage_width:
            previous_width, stage_depth = stages[-1]
            stages[-1] = (previous_width, stage_depth + 1)
        else:
            stages.append((stage_width, 1))
    return stages


def generate_stage_widths(w0, wa, wm, depth):
    return [width for width, _ in generate_stage_configuration(w0, wa, wm, depth)]


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, group_width, stride):
        super().__init__()
        if group_width <= 0:
            raise ValueError("group_width must be positive")
        groups = max(1, out_channels // group_width)
        bottleneck_channels = groups * group_width
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.projection = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        residual = self.projection(x)
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return self.activation(x + residual)


class RegNet(nn.Module):
    architecture_id = "regnet"
    def __init__(self, w0=32, wa=32, wm=2.5, depth=4, group_width=8, in_channels=3, num_classes=1000):
        super().__init__()
        if in_channels <= 0 or num_classes <= 0 or group_width <= 0:
            raise ValueError("channel counts and group_width must be positive")
        stages = generate_stage_configuration(w0, wa, wm, depth)
        self.stage_widths = [width for width, _ in stages]
        self.stage_depths = [stage_depth for _, stage_depth in stages]
        first_width = self.stage_widths[0]
        self.stem = nn.Sequential(nn.Conv2d(in_channels, first_width, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(first_width), nn.ReLU(inplace=True))
        blocks = []
        current = first_width
        for stage_index, (width, stage_depth) in enumerate(stages):
            for block_index in range(stage_depth):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                blocks.append(Bottleneck(current, width, group_width, stride))
                current = width
        self.stages = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(current, num_classes)

    def forward(self, x):
        x = self.stages(self.stem(x))
        return self.classifier(self.pool(x).flatten(1))
