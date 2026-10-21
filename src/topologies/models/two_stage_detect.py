import torch
from torch import nn
import torch.nn.functional as F


def bilinear(image, y, x):
    height, width = image.shape[-2:]
    y0 = int(torch.floor(y).item())
    x0 = int(torch.floor(x).item())
    y1 = min(y0 + 1, height - 1)
    x1 = min(x0 + 1, width - 1)
    wy = y - y0
    wx = x - x0
    top = image[y0, x0] * (1 - wx) + image[y0, x1] * wx
    bottom = image[y1, x0] * (1 - wx) + image[y1, x1] * wx
    return top * (1 - wy) + bottom * wy


def fpn_merge(deep, lateral):
    return F.interpolate(deep, size=lateral.shape[-2:], mode="nearest") + lateral


def decode_box(anchor, delta):
    cx = anchor[0] + delta[0] * anchor[2]
    cy = anchor[1] + delta[1] * anchor[3]
    width = anchor[2] * torch.exp(delta[2])
    height = anchor[3] * torch.exp(delta[3])
    return torch.stack((cx, cy, width, height))


def box_iou(first, second):
    left = torch.maximum(first[0], second[0])
    top = torch.maximum(first[1], second[1])
    right = torch.minimum(first[2], second[2])
    bottom = torch.minimum(first[3], second[3])
    intersection = torch.clamp(right - left, min=0) * torch.clamp(bottom - top, min=0)
    area_first = (first[2] - first[0]) * (first[3] - first[1])
    area_second = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (area_first + area_second - intersection)


def mask_bce(logits, target, label):
    return F.binary_cross_entropy_with_logits(logits[label], target)


def cascade_positive(iou, thresholds):
    return iou >= thresholds


class TwoStageDetector(nn.Module):
    architecture_id = "two-stage-detect"
    PRESETS = {
        "tiny": {"in_channels": 3, "num_classes": 10, "width": 8, "mode": "faster"},
        "base": {"in_channels": 3, "num_classes": 10, "width": 16, "mode": "mask"},
    }

    def __init__(self, in_channels=3, num_classes=10, width=8, mode="faster", dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if mode not in {"faster", "fpn", "mask", "cascade"}:
            raise ValueError("unknown two-stage mode")
        if min(in_channels, num_classes, width) <= 0:
            raise ValueError("invalid two-stage dimensions")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.width = width
        self.mode = mode
        self.vocab_size = num_classes
        self.context_length = 32
        self.stem = nn.Conv2d(in_channels, width, 3, stride=2, padding=1, bias=False)
        self.lateral = nn.Conv2d(width, width, 1, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.fc = nn.Linear(width, num_classes)

    def forward(self, pixels):
        pixels = torch.as_tensor(pixels)
        if pixels.ndim != 4 or pixels.shape[1] != self.in_channels or pixels.shape[-2:] != (32, 32):
            raise ValueError("pixels must have shape (batch, in_channels, 32, 32)")
        deep = F.relu(self.stem(pixels))
        if self.mode == "fpn":
            deep = fpn_merge(deep[:, :, 8:9, 8:9], self.lateral(deep))
        pooled = self.proj(deep.mean(dim=(2, 3)))
        return self.fc(pooled)
