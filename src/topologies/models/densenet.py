import torch
from torch import nn
import torch.nn.functional as F


_LAYOUTS = {121: (6, 12, 24, 16), 169: (6, 12, 32, 32), 201: (6, 12, 48, 32)}


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.bottleneck = nn.Conv2d(in_channels, 4 * growth, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(4 * growth)
        self.conv = nn.Conv2d(4 * growth, growth, 3, padding=1, bias=False)

    def forward(self, values):
        hidden = self.bottleneck(F.relu(self.norm1(values)))
        new = self.conv(F.relu(self.norm2(hidden)))
        return torch.cat((values, new), dim=1)


class Transition(nn.Module):
    def __init__(self, in_channels, compression=0.5):
        super().__init__()
        self.out_channels = max(1, int(in_channels * compression))
        self.norm = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, self.out_channels, 1, bias=False)
        self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, values):
        return self.pool(self.conv(F.relu(self.norm(values))))


class DenseNet(nn.Module):
    architecture_id = "densenet"
    PRESETS = {
        "tiny": {"in_channels": 3, "num_classes": 10, "depth": 121, "growth": 2},
        "base": {"in_channels": 3, "num_classes": 10, "depth": 169, "growth": 4},
    }

    def __init__(self, in_channels=3, num_classes=10, depth=121, growth=4, dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if depth not in _LAYOUTS or min(in_channels, num_classes, growth) <= 0:
            raise ValueError("invalid DenseNet dimensions")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.depth = depth
        self.growth = growth
        self.vocab_size = num_classes
        self.context_length = 32
        channels = 2 * growth
        self.stem = nn.Conv2d(in_channels, channels, 3, stride=2, padding=1, bias=False)
        blocks = []
        transitions = []
        for index, layers in enumerate(_LAYOUTS[depth]):
            for _ in range(layers):
                blocks.append(DenseLayer(channels, growth))
                channels += growth
            if index < len(_LAYOUTS[depth]) - 1:
                transition = Transition(channels)
                transitions.append(transition)
                channels = transition.out_channels
        self.blocks = nn.ModuleList(blocks)
        self.transitions = nn.ModuleList(transitions)
        self.fc = nn.Linear(channels, num_classes)

    def features(self, pixels):
        pixels = torch.as_tensor(pixels)
        if pixels.ndim != 4 or pixels.shape[1] != self.in_channels or pixels.shape[-2:] != (32, 32):
            raise ValueError("pixels must have shape (batch, in_channels, 32, 32)")
        hidden = self.stem(pixels)
        cursor = 0
        for index, layers in enumerate(_LAYOUTS[self.depth]):
            for _ in range(layers):
                hidden = self.blocks[cursor](hidden)
                cursor += 1
            if index < len(self.transitions):
                hidden = self.transitions[index](hidden)
        return hidden

    def forward(self, pixels):
        return self.fc(self.features(pixels).mean(dim=(2, 3)))
