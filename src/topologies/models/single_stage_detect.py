import torch
from torch import nn
import torch.nn.functional as F


def yolo_confidence(objectness, class_prob):
    return objectness * class_prob


def ssd_scale(index, count, scale_min, scale_max):
    if count == 1:
        return scale_min
    step = (scale_max - scale_min) / (count - 1)
    return scale_min + step * index


def centerness(left, top, right, bottom):
    horizontal = torch.minimum(left, right) / torch.maximum(left, right)
    vertical = torch.minimum(top, bottom) / torch.maximum(top, bottom)
    return torch.sqrt(horizontal * vertical)


def heatmap_peak(heatmap):
    flat = heatmap.reshape(-1)
    index = int(flat.argmax().item())
    width = heatmap.shape[-1]
    return index // width, index % width


def corner_pull(top_left, bottom_right):
    return (top_left - bottom_right).pow(2).mean()


def bifpn_fuse(first, second, weight_first, weight_second):
    first_weight = F.relu(weight_first)
    second_weight = F.relu(weight_second)
    return (first_weight * first + second_weight * second) / (first_weight + second_weight + 1e-4)


class SingleStageDetector(nn.Module):
    architecture_id = "single-stage-detect"
    PRESETS = {
        "tiny": {"in_channels": 3, "num_classes": 10, "width": 8, "mode": "yolo"},
        "base": {"in_channels": 3, "num_classes": 10, "width": 16, "mode": "fcos"},
    }

    def __init__(self, in_channels=3, num_classes=10, width=8, mode="yolo", dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if mode not in {"yolo", "ssd", "fcos", "centernet", "cornernet", "efficientdet", "yolov8"}:
            raise ValueError("unknown single-stage mode")
        if min(in_channels, num_classes, width) <= 0:
            raise ValueError("invalid single-stage dimensions")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.width = width
        self.mode = mode
        self.vocab_size = num_classes
        self.context_length = 32
        self.stem = nn.Conv2d(in_channels, width, 3, stride=2, padding=1, bias=False)
        self.box_head = nn.Conv2d(width, 4, 1, bias=False)
        self.cls_head = nn.Conv2d(width, num_classes, 1, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.fc = nn.Linear(width, num_classes)
        self.fuse_first = nn.Parameter(torch.ones(1))
        self.fuse_second = nn.Parameter(torch.ones(1))

    def forward(self, pixels):
        pixels = torch.as_tensor(pixels)
        if pixels.ndim != 4 or pixels.shape[1] != self.in_channels or pixels.shape[-2:] != (32, 32):
            raise ValueError("pixels must have shape (batch, in_channels, 32, 32)")
        hidden = F.relu(self.stem(pixels))
        if self.mode == "efficientdet":
            hidden = bifpn_fuse(hidden, self.proj(hidden.mean(dim=(2, 3))).reshape(hidden.shape[0], hidden.shape[1], 1, 1), self.fuse_first, self.fuse_second)
        elif self.mode == "yolov8":
            hidden = hidden + self.box_head(hidden).mean(dim=1, keepdim=True)
        pooled = self.proj(hidden.mean(dim=(2, 3)))
        return self.fc(pooled)
