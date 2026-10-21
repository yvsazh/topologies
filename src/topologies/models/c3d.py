import torch
from torch import nn


class C3D(nn.Module):
    architecture_id = "c3d"

    def __init__(self, in_channels=3, hidden_channels=64, num_classes=10):
        super().__init__()
        if in_channels <= 0 or hidden_channels <= 0 or num_classes <= 0:
            raise ValueError("channel and class counts must be positive")
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool3d((1, 1, 1)),
        )
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, clips):
        if clips.ndim != 5:
            raise ValueError("clips must have shape [batch, channels, time, height, width]")
        return self.classifier(self.features(clips).flatten(1))
