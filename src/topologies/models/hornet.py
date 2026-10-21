import torch
from torch import nn


class HorNet(nn.Module):
    architecture_id = "hornet"
    def __init__(self, in_channels=3, num_classes=10, channels=32, order=2, kernel_size=7):
        super().__init__()
        if order < 1:
            raise ValueError("order must be at least one")
        self.stem = nn.Conv2d(in_channels, channels, kernel_size=1)
        self.project = nn.Conv2d(channels, channels * (order + 1), kernel_size=1)
        self.depthwise = nn.ModuleList(
            nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels)
            for _ in range(order)
        )
        self.output = nn.Conv2d(channels, channels, kernel_size=1)
        self.classifier = nn.Linear(channels, num_classes)
        self.order = order

    def forward(self, x):
        x = self.stem(x)
        projected = self.project(x)
        chunks = projected.chunk(self.order + 1, dim=1)
        features, gates = chunks[0], chunks[1:]
        for spatial, gate in zip(self.depthwise, gates):
            features = spatial(gate) * features
        features = self.output(features)
        return self.classifier(features.mean(dim=(-2, -1)))
