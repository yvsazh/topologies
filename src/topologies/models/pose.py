import torch
from torch import nn
import torch.nn.functional as F


def limb_vector(start, end):
    delta = end - start
    return delta / delta.norm()


def heatmap_peak(heatmap):
    index = int(heatmap.argmax())
    width = heatmap.shape[-1]
    return index // width, index % width


def soft_coordinate(logits):
    weights = torch.softmax(logits, dim=-1)
    index = torch.arange(logits.shape[-1], dtype=logits.dtype, device=logits.device)
    return (weights * index).sum()


def simcc_coord(x_logits, y_logits):
    return int(x_logits.argmax()), int(y_logits.argmax())


def object_keypoint_similarity(distance, scale, kappa):
    return torch.exp(-(distance ** 2) / (2 * scale ** 2 * kappa ** 2))


class PoseEstimator(nn.Module):
    architecture_id = "pose"
    PRESETS = {
        "tiny": {"in_channels": 3, "num_classes": 10, "width": 8, "mode": "openpose"},
        "base": {"in_channels": 3, "num_classes": 10, "width": 16, "mode": "vitpose"},
    }

    def __init__(self, in_channels=3, num_classes=10, width=8, mode="openpose", dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if mode not in {"openpose", "simple", "vitpose", "rtmpose"}:
            raise ValueError("unknown pose mode")
        if min(in_channels, num_classes, width) <= 0:
            raise ValueError("invalid pose dimensions")
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
