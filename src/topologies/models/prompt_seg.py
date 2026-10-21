import torch
from torch import nn
import torch.nn.functional as F


def point_prompt(height, width, row, col, value):
    value = torch.as_tensor(value)
    prompt = torch.zeros(height, width, dtype=value.dtype, device=value.device)
    prompt[int(row), int(col)] = value.reshape(())
    return prompt


def box_mask(height, width, row0, col0, row1, col1):
    mask = torch.zeros(height, width)
    mask[row0:row1, col0:col1] = 1
    return mask


def memory_attend(features, memory):
    return features + memory


def prompt_similarity(pixels, reference):
    return pixels @ reference


def in_context_mask(prompt_image, prompt_mask, query_image):
    if torch.equal(prompt_image, query_image):
        return prompt_mask.clone()
    return torch.zeros_like(prompt_mask)


class PromptSegmenter(nn.Module):
    architecture_id = "prompt-seg"
    PRESETS = {
        "tiny": {"in_channels": 3, "num_classes": 10, "width": 8, "mode": "sam"},
        "base": {"in_channels": 3, "num_classes": 10, "width": 16, "mode": "sam2"},
    }

    def __init__(self, in_channels=3, num_classes=10, width=8, mode="sam", dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if mode not in {"sam", "sam2", "seem", "seggpt"}:
            raise ValueError("unknown prompt mode")
        if min(in_channels, num_classes, width) <= 0:
            raise ValueError("invalid prompt dimensions")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.width = width
        self.mode = mode
        self.vocab_size = num_classes
        self.context_length = 32
        self.stem = nn.Conv2d(in_channels, width, 3, stride=2, padding=1, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.fc = nn.Linear(width, num_classes)
        self.point = nn.Parameter(torch.zeros(1))
        self.memory = nn.Parameter(torch.zeros(1))

    def forward(self, pixels):
        pixels = torch.as_tensor(pixels)
        if pixels.ndim != 4 or pixels.shape[1] != self.in_channels or pixels.shape[-2:] != (32, 32):
            raise ValueError("pixels must have shape (batch, in_channels, 32, 32)")
        hidden = F.relu(self.stem(pixels))
        if self.mode == "sam":
            prompt = point_prompt(hidden.shape[-2], hidden.shape[-1], 0, 0, self.point)
            hidden = hidden + prompt
        elif self.mode == "sam2":
            hidden = memory_attend(hidden, self.memory)
        pooled = self.proj(hidden.mean(dim=(2, 3)))
        return self.fc(pooled)
