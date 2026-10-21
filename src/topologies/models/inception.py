import torch
from torch import nn
import torch.nn.functional as F


class Inception(nn.Module):
    def __init__(self, in_channels, width, mode):
        super().__init__()
        if mode not in {"v1", "v3", "v4", "inception-resnet"}:
            raise ValueError("mode must be v1, v3, v4, or inception-resnet")
        narrow = 4 * width
        wide = 8 * width
        self.mode = mode
        self.branch1 = nn.Conv2d(in_channels, narrow, 1)
        self.reduce3 = nn.Conv2d(in_channels, narrow, 1)
        self.branch3 = nn.Conv2d(narrow, wide, 3, padding=1)
        self.reduce5 = nn.Conv2d(in_channels, narrow, 1)
        if mode == "v3":
            self.factor_a = nn.Conv2d(narrow, wide, (1, 3), padding=(0, 1))
            self.factor_b = nn.Conv2d(wide, wide, (3, 1), padding=(1, 0))
        elif mode == "v4":
            self.stack_a = nn.Conv2d(narrow, wide, 3, padding=1)
            self.stack_b = nn.Conv2d(wide, wide, 3, padding=1)
        else:
            self.branch5 = nn.Conv2d(narrow, wide, 5, padding=2)
        self.pool = nn.MaxPool2d(3, stride=1, padding=1)
        self.branch_pool = nn.Conv2d(in_channels, narrow, 1)
        mixed = narrow + wide + wide + narrow
        self.project = nn.Conv2d(mixed, in_channels, 1) if mode == "inception-resnet" else None
        self.out_channels = in_channels if mode == "inception-resnet" else mixed

    def branches(self, values):
        first = self.branch1(values)
        third = self.branch3(self.reduce3(values))
        reduced = self.reduce5(values)
        if self.mode == "v3":
            fifth = self.factor_b(self.factor_a(reduced))
        elif self.mode == "v4":
            fifth = self.stack_b(self.stack_a(reduced))
        else:
            fifth = self.branch5(reduced)
        pooled = self.branch_pool(self.pool(values))
        return torch.cat((first, third, fifth, pooled), dim=1)

    def forward(self, values):
        mixed = self.branches(values)
        if self.project is None:
            return mixed
        return values + self.project(mixed)


class InceptionNet(nn.Module):
    architecture_id = "inception"
    PRESETS = {
        "tiny": {"in_channels": 3, "num_classes": 10, "width": 1, "mode": "v1"},
        "base": {"in_channels": 3, "num_classes": 10, "width": 2, "mode": "v3"},
    }

    def __init__(self, in_channels=3, num_classes=10, width=1, mode="v1", dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if mode not in {"v1", "v3", "v4", "inception-resnet"} or min(in_channels, num_classes, width) <= 0:
            raise ValueError("invalid Inception dimensions")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.width = width
        self.mode = mode
        self.vocab_size = num_classes
        self.context_length = 32
        stem_channels = 8 * width
        self.stem = nn.Conv2d(in_channels, stem_channels, 3, stride=2, padding=1)
        self.block1 = Inception(stem_channels, width, mode)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.block2 = Inception(self.block1.out_channels, width, mode)
        self.aux = nn.Linear(stem_channels, num_classes)
        self.fc = nn.Linear(self.block2.out_channels, num_classes)

    def aux_logits(self, pixels):
        pixels = self._pixels(pixels)
        return self.aux(self.stem(pixels).mean(dim=(2, 3)))

    def _pixels(self, pixels):
        pixels = torch.as_tensor(pixels)
        if pixels.ndim != 4 or pixels.shape[1] != self.in_channels or pixels.shape[-2:] != (32, 32):
            raise ValueError("pixels must have shape (batch, in_channels, 32, 32)")
        return pixels

    def forward(self, pixels):
        pixels = self._pixels(pixels)
        hidden = F.relu(self.stem(pixels))
        hidden = self.pool(self.block1(hidden))
        hidden = self.block2(hidden).mean(dim=(2, 3))
        return self.fc(hidden)
