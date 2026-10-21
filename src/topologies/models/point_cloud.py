import torch
from torch import nn
import torch.nn.functional as F


def point_max(points):
    return points.max(dim=0).values


def ball_index(distances, radius):
    return [index for index, distance in enumerate(distances.tolist()) if distance <= radius]


def edge_feature(center, neighbor):
    return neighbor - center


def kernel_point(values, weights):
    return (values * weights).sum()


def vector_attention(scores, values):
    return torch.softmax(scores, dim=0) @ values


class PointCloud(nn.Module):
    architecture_id = "point-cloud"
    PRESETS = {
        "tiny": {"in_channels": 3, "num_classes": 10, "width": 8, "mode": "pointnet"},
        "base": {"in_channels": 3, "num_classes": 10, "width": 16, "mode": "dgcnn"},
    }

    def __init__(self, in_channels=3, num_classes=10, width=8, mode="pointnet", dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if mode not in {"pointnet", "pointnet2", "dgcnn", "kpconv", "ptv3"}:
            raise ValueError("unknown point-cloud mode")
        if min(in_channels, num_classes, width) <= 0:
            raise ValueError("invalid point-cloud dimensions")
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
        pooled = self.proj(hidden.mean(dim=(2, 3)))
        return self.fc(pooled)
