import torch
from torch import nn
import torch.nn.functional as F


def path_augment(low, high):
    return low + F.avg_pool2d(high, kernel_size=2)


def weighted_fuse(values, weights, eps=1e-4):
    weights = F.relu(weights)
    return (values * weights).sum() / (weights.sum() + eps)


def global_pool_cell(values):
    return values.mean()


def queen_fuse(levels):
    fused = levels[0]
    for level in levels[1:]:
        fused = fused + level
    return fused


def elan_concat(parts):
    return torch.cat(parts, dim=-1)


class VisionNeck(nn.Module):
    architecture_id = "vision-neck"
    PRESETS = {
        "tiny": {"in_channels": 3, "num_classes": 10, "width": 8, "mode": "panet"},
        "base": {"in_channels": 3, "num_classes": 10, "width": 16, "mode": "bifpn"},
    }

    def __init__(self, in_channels=3, num_classes=10, width=8, mode="panet", dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if mode not in {"panet", "bifpn", "nas-fpn", "giraffe", "elan"}:
            raise ValueError("unknown neck mode")
        if min(in_channels, num_classes, width) <= 0:
            raise ValueError("invalid neck dimensions")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.width = width
        self.mode = mode
        self.vocab_size = num_classes
        self.context_length = 32
        self.stem = nn.Conv2d(in_channels, width, 3, stride=2, padding=1, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.fc = nn.Linear(width, num_classes)

    def forward(self, pixels):
        pixels = torch.as_tensor(pixels)
        if pixels.ndim != 4 or pixels.shape[1] != self.in_channels or pixels.shape[-2:] != (32, 32):
            raise ValueError("pixels must have shape (batch, in_channels, 32, 32)")
        hidden = F.relu(self.stem(pixels))
        if self.mode == "panet":
            low = F.avg_pool2d(hidden, 2)
            hidden = F.interpolate(path_augment(low, hidden), size=hidden.shape[-2:], mode="nearest")
        pooled = self.proj(hidden.mean(dim=(2, 3)))
        return self.fc(pooled)
