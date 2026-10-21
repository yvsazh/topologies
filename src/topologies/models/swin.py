import torch
from torch import nn
import torch.nn.functional as F


def partition(values, window):
    batch, height, width, channels = values.shape
    values = values.reshape(batch, height // window, window, width // window, window, channels)
    return values.permute(0, 1, 3, 2, 4, 5).reshape(-1, window * window, channels)


def merge(windows, window, height, width):
    channels = windows.shape[-1]
    batch = windows.shape[0] // ((height // window) * (width // window))
    values = windows.reshape(batch, height // window, width // window, window, window, channels)
    return values.permute(0, 1, 3, 2, 4, 5).reshape(batch, height, width, channels)


def scaled_cosine(query, key, scale):
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)
    return scale * (query @ key.transpose(-1, -2))


def log_position_bias(relative, weight):
    signed = torch.sign(relative) * torch.log1p(relative.abs())
    return signed @ weight


class SwinBlock(nn.Module):
    def __init__(self, channels, mode, window=4):
        super().__init__()
        if mode not in {"swin", "swin-v2"}:
            raise ValueError("unknown Swin mode")
        self.mode = mode
        self.window = window
        self.qkv = nn.Linear(channels, 3 * channels, bias=False)
        self.proj = nn.Linear(channels, channels, bias=False)
        self.cosine_scale = nn.Parameter(torch.zeros(1))
        self.bias_weight = nn.Parameter(torch.zeros(2, 1))

    def pair_bias(self, tokens, device):
        side = int(tokens ** 0.5)
        coords = torch.arange(side, device=device)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        points = torch.stack((grid_y.reshape(-1), grid_x.reshape(-1)), dim=-1).float()
        relative = points[:, None, :] - points[None, :, :]
        return log_position_bias(relative, self.bias_weight).reshape(1, tokens, tokens)

    def attend(self, windows):
        query, key, value = self.qkv(windows).chunk(3, dim=-1)
        if self.mode == "swin-v2":
            logits = scaled_cosine(query, key, self.cosine_scale.exp())
            logits = logits + self.pair_bias(windows.shape[1], windows.device)
        else:
            logits = query @ key.transpose(-1, -2) / (query.shape[-1] ** 0.5)
        return self.proj(torch.softmax(logits, dim=-1) @ value)

    def forward(self, values):
        shortcut = values
        values = values.permute(0, 2, 3, 1)
        shift = self.window // 2
        values = torch.roll(values, shifts=(-shift, -shift), dims=(1, 2))
        windows = partition(values, self.window)
        windows = self.attend(windows)
        values = merge(windows, self.window, shortcut.shape[-2], shortcut.shape[-1])
        values = torch.roll(values, shifts=(shift, shift), dims=(1, 2))
        return shortcut + values.permute(0, 3, 1, 2)


class Swin(nn.Module):
    architecture_id = "swin"
    PRESETS = {
        "tiny": {"in_channels": 3, "num_classes": 10, "width": 8, "mode": "swin"},
        "base": {"in_channels": 3, "num_classes": 10, "width": 16, "mode": "swin-v2"},
    }

    def __init__(self, in_channels=3, num_classes=10, width=8, mode="swin", dropout=0.0):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must be 0.0")
        if min(in_channels, num_classes, width) <= 0:
            raise ValueError("invalid Swin dimensions")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.width = width
        self.mode = mode
        self.vocab_size = num_classes
        self.context_length = 32
        self.stem = nn.Conv2d(in_channels, width, 3, stride=2, padding=1, bias=False)
        self.block = SwinBlock(width, mode)
        self.fc = nn.Linear(width, num_classes)

    def forward(self, pixels):
        pixels = torch.as_tensor(pixels)
        if pixels.ndim != 4 or pixels.shape[1] != self.in_channels or pixels.shape[-2:] != (32, 32):
            raise ValueError("pixels must have shape (batch, in_channels, 32, 32)")
        hidden = self.block(F.relu(self.stem(pixels)))
        return self.fc(hidden.mean(dim=(2, 3)))
