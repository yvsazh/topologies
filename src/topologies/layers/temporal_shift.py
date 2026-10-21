import torch
from torch import nn


class TemporalShift(nn.Module):
    component_id = "tsm"

    def __init__(self, channels=8, num_frames=4):
        super().__init__()
        if channels < 8:
            raise ValueError("channels must be at least 8 to form both shift folds")
        if num_frames < 1:
            raise ValueError("num_frames must be positive")
        self.channels = channels
        self.num_frames = num_frames
        self.fold = channels // 8

    def forward(self, clip):
        if clip.ndim != 5:
            raise ValueError("clip must have shape (batch, time, channels, height, width)")
        _, frames, channels, _, _ = clip.shape
        if channels != self.channels:
            raise ValueError("clip channel count does not match channels")
        if frames != self.num_frames:
            raise ValueError("clip frame count does not match num_frames")

        shifted = torch.zeros_like(clip)
        shifted[:, 1:, :self.fold] = clip[:, :-1, :self.fold]
        shifted[:, :-1, self.fold:2 * self.fold] = clip[:, 1:, self.fold:2 * self.fold]
        shifted[:, :, 2 * self.fold:] = clip[:, :, 2 * self.fold:]
        return shifted
