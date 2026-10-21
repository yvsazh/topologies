import torch
from torch import nn


class TSM(nn.Module):

    def __init__(self, in_channels=3, num_classes=2, num_frames=4, **kwargs):
        super().__init__()
        if in_channels < 1 or num_classes < 1:
            raise ValueError("in_channels and num_classes must be positive")
        if num_frames < 1:
            raise ValueError("num_frames must be positive")
        self.in_channels = in_channels
        self.num_frames = num_frames
        self.fold = in_channels // 8
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_channels, num_classes)

    def shift(self, clip):
        if clip.ndim != 5:
            raise ValueError("clip must have shape (batch, time, channels, height, width)")
        batch, frames, channels, height, width = clip.shape
        if channels != self.in_channels:
            raise ValueError("clip channel count does not match in_channels")
        if frames != self.num_frames:
            raise ValueError("clip frame count does not match num_frames")
        shifted = torch.zeros_like(clip)
        shifted[:, 1:, :self.fold] = clip[:, :-1, :self.fold]
        shifted[:, :-1, self.fold:2 * self.fold] = clip[:, 1:, self.fold:2 * self.fold]
        shifted[:, :, 2 * self.fold:] = clip[:, :, 2 * self.fold:]
        return shifted

    def forward(self, clip):
        batch, frames, channels, height, width = clip.shape
        shifted = self.shift(clip).reshape(batch * frames, channels, height, width)
        features = self.conv(shifted)
        features = self.pool(features).flatten(1).reshape(batch, frames, channels)
        return self.classifier(features.mean(dim=1))
