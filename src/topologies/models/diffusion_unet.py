from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

def _groups(channels: int) -> int:
    return next((g for g in range(min(8, channels), 0, -1) if channels % g == 0))

class _ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time = nn.Linear(time_dim, out_channels)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = F.silu(self.norm2(h) + self.time(F.silu(time))[:, :, None, None])
        h = self.conv2(h)
        return h + self.skip(x)

class DiffusionUNet(nn.Module):
    architecture_id = 'diffusion-unet'
    PRESETS = {'tiny': (16, 32), 'small': (32, 64, 128), 'base': (64, 128, 256, 512)}

    def __init__(self, in_channels: int=3, out_channels: int | None=None, features: tuple[int, ...]=(32, 64, 128), time_dim: int=128, num_classes: int | None=None):
        super().__init__()
        if in_channels < 1 or time_dim < 4 or (not features) or any((c < 1 for c in features)):
            raise ValueError('channels, time_dim, and features must be positive (time_dim >= 4)')
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.num_classes = num_classes
        self.time_dim = time_dim
        self.input = nn.Conv2d(in_channels, features[0], 3, padding=1)
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim * 4), nn.SiLU(), nn.Linear(time_dim * 4, time_dim))
        self.class_embedding = nn.Embedding(num_classes, time_dim) if num_classes is not None else None
        self.down_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        channels = features[0]
        for i, width in enumerate(features):
            self.down_blocks.append(_ResidualBlock(channels, width, time_dim))
            channels = width
            if i < len(features) - 1:
                self.downsample.append(nn.Conv2d(width, features[i + 1], 3, stride=2, padding=1))
                channels = features[i + 1]
        self.middle1 = _ResidualBlock(features[-1], features[-1], time_dim)
        self.middle2 = _ResidualBlock(features[-1], features[-1], time_dim)
        self.up_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()
        channels = features[-1]
        for i in reversed(range(len(features) - 1)):
            self.upsample.append(nn.Conv2d(channels, features[i], 3, padding=1))
            self.up_blocks.append(_ResidualBlock(features[i] * 2, features[i], time_dim))
            channels = features[i]
        self.output = nn.Sequential(nn.GroupNorm(_groups(features[0]), features[0]), nn.SiLU(), nn.Conv2d(features[0], self.out_channels, 3, padding=1))

    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half = self.time_dim // 2
        scale = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * (-math.log(10000.0) / max(half - 1, 1)))
        phase = t.float().reshape(-1, 1) * scale.reshape(1, -1)
        emb = torch.cat((phase.sin(), phase.cos()), dim=-1)
        return F.pad(emb, (0, self.time_dim - emb.shape[-1]))

    def forward(self, x: torch.Tensor, t: torch.Tensor, class_labels: torch.Tensor | None=None) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f'x must have shape [B,{self.in_channels},H,W]')
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        if t.shape != (x.shape[0],):
            raise ValueError('t must be scalar or shape [B]')
        time = self.time_mlp(self._time_embedding(t))
        if self.class_embedding is not None:
            if class_labels is None or class_labels.shape != (x.shape[0],):
                raise ValueError('class_labels with shape [B] required for class-conditional model')
            time = time + self.class_embedding(class_labels.long())
        elif class_labels is not None:
            raise ValueError('class_labels supplied to an unconditional model')
        h = self.input(x)
        skips = []
        for i, block in enumerate(self.down_blocks):
            h = block(h, time)
            skips.append(h)
            if i < len(self.downsample):
                h = self.downsample[i](h)
        h = self.middle2(self.middle1(h, time), time)
        for up, block, skip in zip(self.upsample, self.up_blocks, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode='nearest')
            h = block(torch.cat((up(h), skip), dim=1), time)
        return self.output(h)
