import torch
from torch import nn


class LeNet(nn.Module):
    architecture_id = "lenet"
    PRESETS = {
        "tiny": {"in_channels": 1, "num_classes": 10, "width": 1},
        "base": {"in_channels": 1, "num_classes": 10, "width": 2},
    }

    def __init__(self, in_channels=1, num_classes=10, width=1, dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if min(in_channels, num_classes, width) <= 0:
            raise ValueError("invalid LeNet dimensions")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.width = width
        self.vocab_size = num_classes
        self.context_length = 32
        planes = (6 * width, 16 * width, 120 * width, 84 * width)
        self.conv1 = nn.Conv2d(in_channels, planes[0], 5)
        self.pool = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(planes[0], planes[1], 5)
        self.conv3 = nn.Conv2d(planes[1], planes[2], 5)
        self.fc1 = nn.Linear(planes[2], planes[3])
        self.fc2 = nn.Linear(planes[3], num_classes)

    def features(self, pixels):
        pixels = torch.as_tensor(pixels)
        if pixels.ndim != 4 or pixels.shape[1] != self.in_channels or pixels.shape[-2:] != (32, 32):
            raise ValueError("pixels must have shape (batch, in_channels, 32, 32)")
        first = self.pool(torch.tanh(self.conv1(pixels)))
        second = self.pool(torch.tanh(self.conv2(first)))
        third = torch.tanh(self.conv3(second))
        return first, second, third.flatten(1)

    def forward(self, pixels):
        _, _, hidden = self.features(pixels)
        return self.fc2(torch.tanh(self.fc1(hidden)))
