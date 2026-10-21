import torch
from torch import nn
import torch.nn.functional as F


def fuse_panoptic(semantic, instance):
    restored = F.interpolate(instance, size=semantic.shape[-2:], mode="nearest")
    return semantic + restored


def location_index(row, col, grid):
    return int(row) * int(grid) + int(col)


def uncertain_index(scores):
    distance = (scores - 0.5).abs()
    return int(distance.argmin())


def conditional_conv(features, weight):
    return features * weight


def assemble_mask(prototypes, coefficients):
    return prototypes @ coefficients


def matrix_decay(score, higher_score, overlap):
    if higher_score <= score:
        return score
    return score * (1 - overlap)


class PanopticHead(nn.Module):
    architecture_id = "panoptic"
    PRESETS = {
        "tiny": {"in_channels": 3, "num_classes": 10, "width": 8, "mode": "panoptic-fpn"},
        "base": {"in_channels": 3, "num_classes": 10, "width": 16, "mode": "solo"},
    }

    def __init__(self, in_channels=3, num_classes=10, width=8, mode="panoptic-fpn", dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if mode not in {"panoptic-fpn", "solo", "pointrend", "condinst", "yolact", "solov2"}:
            raise ValueError("unknown panoptic mode")
        if min(in_channels, num_classes, width) <= 0:
            raise ValueError("invalid panoptic dimensions")
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
        if self.mode == "panoptic-fpn":
            low = F.avg_pool2d(hidden, 2)
            hidden = fuse_panoptic(hidden, low)
        pooled = self.proj(hidden.mean(dim=(2, 3)))
        return self.fc(pooled)
