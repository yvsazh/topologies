import torch
from torch import nn
import torch.nn.functional as F


def global_response(values, gamma, beta):
    norm = torch.norm(values, p=2, dim=(2, 3), keepdim=True)
    scale = norm / (norm.mean(dim=1, keepdim=True) + 1e-6)
    return gamma * (values * scale) + beta + values


def recursive_gate(values, depthwise):
    first, second = values.chunk(2, dim=1)
    return first * depthwise(second)


def large_kernel_attention(values, mixer):
    return values * mixer(values)


def selective_scan(values, decay):
    state = torch.zeros_like(values[:, :, :, :1])
    outputs = []
    for index in range(values.shape[-1]):
        state = decay * state + values[:, :, :, index:index + 1]
        outputs.append(state)
    return torch.cat(outputs, dim=-1)


def cross_scan(values):
    left = torch.cumsum(values, dim=-1)
    right = torch.cumsum(values.flip(-1), dim=-1).flip(-1)
    down = torch.cumsum(values, dim=-2)
    up = torch.cumsum(values.flip(-2), dim=-2).flip(-2)
    return left + right + down + up


class ModernBlock(nn.Module):
    def __init__(self, channels, mode):
        super().__init__()
        if mode not in {"convnext-v2", "hornet", "van", "convformer", "mambavision", "vmamba"}:
            raise ValueError("unknown modern convolution mode")
        if channels % 2:
            raise ValueError("channels must be even")
        self.mode = mode
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.half_depthwise = nn.Conv2d(channels // 2, channels // 2, 3, padding=1, groups=channels // 2, bias=False)
        self.mixer = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.decay = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, values):
        if self.mode == "convnext-v2":
            return global_response(self.depthwise(values), self.gamma, self.beta)
        if self.mode == "hornet":
            return recursive_gate(values, self.half_depthwise)
        if self.mode == "van":
            return large_kernel_attention(values, self.mixer)
        if self.mode == "mambavision":
            return selective_scan(values, torch.sigmoid(self.decay))
        if self.mode == "vmamba":
            return cross_scan(values)
        return values + self.depthwise(values)


class ModernConv(nn.Module):
    architecture_id = "modern-conv"
    PRESETS = {
        "tiny": {"in_channels": 3, "num_classes": 10, "width": 4, "mode": "convnext-v2"},
        "base": {"in_channels": 3, "num_classes": 10, "width": 8, "mode": "van"},
    }

    def __init__(self, in_channels=3, num_classes=10, width=4, mode="convnext-v2", dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if min(in_channels, num_classes, width) <= 0 or width % 2:
            raise ValueError("invalid modern convolution dimensions")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.width = width
        self.mode = mode
        self.vocab_size = num_classes
        self.context_length = 32
        self.stem = nn.Conv2d(in_channels, width, 3, stride=2, padding=1, bias=False)
        self.proj = nn.Conv2d(width, width, 1, bias=False)
        self.block = ModernBlock(width, mode)
        self.fc = nn.Linear(width, num_classes)

    def mask_patches(self, pixels, mask):
        return pixels * mask

    def forward(self, pixels):
        pixels = torch.as_tensor(pixels)
        if pixels.ndim != 4 or pixels.shape[1] != self.in_channels or pixels.shape[-2:] != (32, 32):
            raise ValueError("pixels must have shape (batch, in_channels, 32, 32)")
        hidden = self.block(F.relu(self.proj(self.stem(pixels))))
        return self.fc(hidden.mean(dim=(2, 3)))
