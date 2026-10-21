import torch
from torch import nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    architecture_id = "alexnet"
    PRESETS = {
        "tiny": {"in_channels": 3, "num_classes": 10, "width": 1},
        "base": {"in_channels": 3, "num_classes": 10, "width": 2},
    }

    def __init__(self, in_channels=3, num_classes=10, width=1, dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if min(in_channels, num_classes, width) <= 0:
            raise ValueError("invalid AlexNet dimensions")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.width = width
        self.vocab_size = num_classes
        self.context_length = 227
        channels = [3 * width, 8 * width, 12 * width, 12 * width, 8 * width]
        hidden = 128 * width
        self.conv1 = nn.Conv2d(in_channels, channels[0], 11, stride=4)
        self.conv2 = nn.Conv2d(channels[0], channels[1], 5, padding=2)
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, padding=1)
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, padding=1)
        self.conv5 = nn.Conv2d(channels[3], channels[4], 3, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2)
        self.fc1 = nn.Linear(channels[4] * 6 * 6, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, num_classes)

    def local_response_norm(self, values):
        squares = values.pow(2)
        padded = F.pad(squares, (0, 0, 0, 0, 2, 2))
        window = padded.unfold(1, 5, 1).sum(dim=-1)
        return values / (2.0 + 1e-4 * window).pow(0.75)

    def features(self, pixels):
        pixels = torch.as_tensor(pixels)
        if pixels.ndim != 4 or pixels.shape[1] != self.in_channels or pixels.shape[-2:] != (227, 227):
            raise ValueError("pixels must have shape (batch, in_channels, 227, 227)")
        first = self.pool(self.local_response_norm(F.relu(self.conv1(pixels))))
        second = self.pool(self.local_response_norm(F.relu(self.conv2(first))))
        third = F.relu(self.conv3(second))
        fourth = F.relu(self.conv4(third))
        fifth = self.pool(F.relu(self.conv5(fourth)))
        return first, second, fifth

    def forward(self, pixels):
        _, _, fifth = self.features(pixels)
        hidden = F.relu(self.fc1(fifth.flatten(1)))
        hidden = F.relu(self.fc2(hidden))
        return self.fc3(hidden)
