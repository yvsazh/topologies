"""Compact clean-room Swin-UNet-inspired segmentation model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def window_partition(x: torch.Tensor, window: int) -> torch.Tensor:
    """Split NHWC feature map into nonoverlapping windows.

    The input must have spatial dimensions divisible by ``window``.
    """
    if window < 1:
        raise ValueError("window must be at least 1")
    if x.ndim != 4:
        raise ValueError("x must have shape (batch, height, width, channels)")
    batch, height, width, channels = x.shape
    if height % window or width % window:
        raise ValueError("feature map dimensions must be divisible by window")
    return (
        x.reshape(batch, height // window, window, width // window, window, channels)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(-1, window * window, channels)
    )


def _window_reverse(windows: torch.Tensor, window: int, batch: int, height: int, width: int):
    channels = windows.shape[-1]
    return (
        windows.reshape(batch, height // window, width // window, window, window, channels)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch, height, width, channels)
    )


def _shifted_window_attention_mask(
    height: int,
    width: int,
    window: int,
    shift: int,
    batch: int,
    valid_height: int,
    valid_width: int,
    device: torch.device,
):
    """Build wraparound and padding masks for cyclically shifted windows."""
    valid = torch.zeros((batch, height, width, 1), dtype=torch.bool, device=device)
    valid[:, :valid_height, :valid_width] = True
    if shift:
        valid = torch.roll(valid, shifts=(-shift, -shift), dims=(1, 2))

    valid_windows = window_partition(valid, window).squeeze(-1)
    key_padding_mask = ~valid_windows
    if not shift:
        return None, key_padding_mask

    region_ids = torch.zeros((1, height, width, 1), dtype=torch.int64, device=device)
    h_slices = (slice(0, -window), slice(-window, -shift), slice(-shift, None))
    w_slices = (slice(0, -window), slice(-window, -shift), slice(-shift, None))
    region = 0
    for h_slice in h_slices:
        for w_slice in w_slices:
            region_ids[:, h_slice, w_slice] = region
            region += 1
    region_windows = window_partition(region_ids, window).squeeze(-1)
    one_batch_mask = region_windows[:, :, None] != region_windows[:, None, :]
    attention_mask = one_batch_mask.repeat(batch, 1, 1)
    return attention_mask, key_padding_mask


class _WindowBlock(nn.Module):
    def __init__(self, channels: int, window: int, shifted: bool):
        super().__init__()
        self.window = window
        self.shifted = shifted
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, num_heads=1, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(nn.Linear(channels, channels * 2), nn.GELU(), nn.Linear(channels * 2, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = x.shape
        window = min(self.window, height, width)
        padded_h = ((height + window - 1) // window) * window
        padded_w = ((width + window - 1) // window) * window
        value = x.permute(0, 3, 1, 2)
        value = F.pad(value, (0, padded_w - width, 0, padded_h - height))
        value = value.permute(0, 2, 3, 1)
        shift = window // 2 if self.shifted and window > 1 else 0
        if shift:
            value = torch.roll(value, shifts=(-shift, -shift), dims=(1, 2))
        attention_mask, key_padding_mask = _shifted_window_attention_mask(
            padded_h,
            padded_w,
            window,
            shift,
            batch,
            height,
            width,
            value.device,
        )
        parts = window_partition(value, window)
        normalized = self.norm1(parts)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        parts = parts + attended
        parts = parts + self.mlp(self.norm2(parts))
        value = _window_reverse(parts, window, batch, padded_h, padded_w)
        if shift:
            value = torch.roll(value, shifts=(shift, shift), dims=(1, 2))
        return value[:, :height, :width, :]


class SwinUNet(nn.Module):
    """Small segmentation U-Net using window-contained self-attention."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1, dim: int = 32, window: int = 4):
        super().__init__()
        if window < 1:
            raise ValueError("window must be at least 1")
        self.window = window
        self.stem = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)
        self.encoder = _WindowBlock(dim, window, shifted=False)
        self.shifted = _WindowBlock(dim * 2, window, shifted=True)
        self.down = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1)
        self.up = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.decoder = _WindowBlock(dim, window, shifted=False)
        self.head = nn.Conv2d(dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        skip = self.encoder(self.stem(x).permute(0, 2, 3, 1))
        low = self.down(skip.permute(0, 3, 1, 2))
        low = self.shifted(low.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        decoded = F.interpolate(self.up(low), size=input_size, mode="bilinear", align_corners=False)
        decoded = decoded + skip.permute(0, 3, 1, 2)
        decoded = self.decoder(decoded.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.head(decoded)
