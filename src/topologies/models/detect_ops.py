import torch
from torch import nn
import torch.nn.functional as F


def box_iou(first, second):
    left = torch.maximum(first[0], second[0])
    top = torch.maximum(first[1], second[1])
    right = torch.minimum(first[2], second[2])
    bottom = torch.minimum(first[3], second[3])
    intersection = torch.clamp(right - left, min=0) * torch.clamp(bottom - top, min=0)
    area_first = (first[2] - first[0]) * (first[3] - first[1])
    area_second = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (area_first + area_second - intersection)


def anchor_center(row, col, stride):
    return torch.tensor([(col + 0.5) * stride, (row + 0.5) * stride])


def encode_delta(anchor, box):
    dx = (box[0] - anchor[0]) / anchor[2]
    dy = (box[1] - anchor[1]) / anchor[3]
    dw = torch.log(box[2] / anchor[2])
    dh = torch.log(box[3] / anchor[3])
    return torch.stack((dx, dy, dw, dh))


def nms(boxes, scores, threshold):
    order = scores.argsort(descending=True)
    keep = []
    while order.numel():
        current = int(order[0].item())
        keep.append(current)
        if order.numel() == 1:
            break
        overlaps = torch.stack([box_iou(boxes[current], boxes[int(index)]) for index in order[1:]])
        order = order[1:][overlaps <= threshold]
    return keep


def soft_nms_scores(boxes, scores):
    scores = scores.clone()
    order = scores.argsort(descending=True)
    current = int(order[0].item())
    for index in order[1:]:
        overlap = box_iou(boxes[current], boxes[int(index)])
        scores[int(index)] = scores[int(index)] * (1 - overlap)
    return scores


def diou(first, second):
    overlap = box_iou(first, second)
    center_first = torch.stack(((first[0] + first[2]) / 2, (first[1] + first[3]) / 2))
    center_second = torch.stack(((second[0] + second[2]) / 2, (second[1] + second[3]) / 2))
    distance = (center_first - center_second).pow(2).sum()
    left = torch.minimum(first[0], second[0])
    top = torch.minimum(first[1], second[1])
    right = torch.maximum(first[2], second[2])
    bottom = torch.maximum(first[3], second[3])
    diagonal = (right - left).pow(2) + (bottom - top).pow(2)
    return overlap - distance / diagonal


def simota_k(overlaps):
    return max(1, int(torch.round(overlaps.sum()).item()))


def nearest_anchors(ground_truth, anchors, count):
    distances = (anchors - ground_truth).pow(2).sum(dim=-1)
    return torch.topk(distances, count, largest=False).indices


class DetectOps(nn.Module):
    architecture_id = "detect-ops"
    PRESETS = {
        "tiny": {"in_channels": 3, "num_classes": 10, "width": 8},
        "base": {"in_channels": 3, "num_classes": 10, "width": 16},
    }

    def __init__(self, in_channels=3, num_classes=10, width=8, dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if min(in_channels, num_classes, width) <= 0:
            raise ValueError("invalid detection-op dimensions")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.width = width
        self.vocab_size = num_classes
        self.context_length = 32
        self.stem = nn.Conv2d(in_channels, width, 3, stride=2, padding=1, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.fc = nn.Linear(width, num_classes)

    def forward(self, pixels):
        pixels = torch.as_tensor(pixels)
        if pixels.ndim != 4 or pixels.shape[1] != self.in_channels or pixels.shape[-2:] != (32, 32):
            raise ValueError("pixels must have shape (batch, in_channels, 32, 32)")
        hidden = F.relu(self.stem(pixels)).mean(dim=(2, 3))
        return self.fc(self.proj(hidden))
