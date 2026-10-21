import torch
from torch import nn


_CONFIGS = {
    11: (8, "M", 16, "M", 32, 32, "M", 64, 64, "M", 64, 64, "M"),
    13: (8, 8, "M", 16, 16, "M", 32, 32, "M", 64, 64, "M", 64, 64, "M"),
    16: (8, 8, "M", 16, 16, "M", 32, 32, 32, "M", 64, 64, 64, "M", 64, 64, 64, "M"),
    19: (8, 8, "M", 16, 16, "M", 32, 32, 32, 32, "M", 64, 64, 64, 64, "M", 64, 64, 64, 64, "M"),
}


class VGG(nn.Module):
    architecture_id = "vgg"
    PRESETS = {
        "tiny": {"in_channels": 3, "num_classes": 10, "depth": 11, "width": 1, "batch_norm": False},
        "base": {"in_channels": 3, "num_classes": 10, "depth": 16, "width": 1, "batch_norm": True},
    }

    def __init__(self, in_channels=3, num_classes=10, depth=16, width=1, batch_norm=False, dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if depth not in _CONFIGS or min(in_channels, num_classes, width) <= 0:
            raise ValueError("invalid VGG dimensions")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.depth = depth
        self.width = width
        self.batch_norm = batch_norm
        self.vocab_size = num_classes
        self.context_length = 224
        layers = []
        channels = in_channels
        for spec in _CONFIGS[depth]:
            if spec == "M":
                layers.append(nn.MaxPool2d(2, stride=2))
                continue
            out = spec * width
            layers.append(nn.Conv2d(channels, out, 3, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out))
            layers.append(nn.ReLU())
            channels = out
        self.features_net = nn.Sequential(*layers)
        hidden = 64 * width
        self.fc1 = nn.Linear(channels * 7 * 7, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, num_classes)

    def features(self, pixels):
        pixels = torch.as_tensor(pixels)
        if pixels.ndim != 4 or pixels.shape[1] != self.in_channels or pixels.shape[-2:] != (224, 224):
            raise ValueError("pixels must have shape (batch, in_channels, 224, 224)")
        return self.features_net(pixels)

    def forward(self, pixels):
        hidden = torch.relu(self.fc1(self.features(pixels).flatten(1)))
        hidden = torch.relu(self.fc2(hidden))
        return self.fc3(hidden)
