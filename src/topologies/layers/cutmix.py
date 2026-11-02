import math
import numbers

import torch
from torch import nn


class CutMix(nn.Module):
    component_id = "cutmix"

    def __init__(self, alpha=1.0):
        super().__init__()
        try:
            alpha_value = float(alpha)
        except (TypeError, OverflowError, ValueError):
            alpha_value = math.nan
        if isinstance(alpha, bool) or not isinstance(alpha, numbers.Real):
            alpha_value = math.nan
        if not math.isfinite(alpha_value) or alpha_value <= 0.0:
            raise ValueError("alpha must be a finite positive real number")
        self.alpha = alpha_value

    def _sample_lambda(self, device):
        concentration = torch.full((), self.alpha, dtype=torch.float64, device=device)
        return torch.distributions.Beta(concentration, concentration).sample()

    def _sample_center(self, height, width, device):
        coordinates = torch.rand(2, dtype=torch.float64, device=device)
        return float(coordinates[0]) * width, float(coordinates[1]) * height

    def _sample_permutation(self, batch_size, device):
        return torch.randperm(batch_size, device=device)

    def forward(self, images, targets=None):
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("images must be a tensor with shape [B, C, H, W]")
        if any(dimension < 1 for dimension in images.shape):
            raise ValueError("images must have positive batch, channel, height, and width sizes")
        if not images.is_floating_point():
            raise ValueError("images must have a floating-point dtype")
        if targets is not None:
            if not isinstance(targets, torch.Tensor) or targets.ndim != 2:
                raise ValueError("targets must be a tensor with shape [B, K]")
            if targets.shape[0] != images.shape[0] or targets.shape[1] < 1:
                raise ValueError("targets must have shape [B, K] with matching batch size and K >= 1")
            if not targets.is_floating_point():
                raise ValueError("targets must have a floating-point dtype")
            if targets.device != images.device:
                raise ValueError("images and targets must be on the same device")
        if not self.training:
            return images if targets is None else (images, targets)

        batch_size, _, height, width = images.shape
        lam = min(1.0, max(0.0, float(self._sample_lambda(images.device))))
        center_x, center_y = self._sample_center(height, width, images.device)
        permutation = self._sample_permutation(batch_size, images.device)
        box_width = width * math.sqrt(1.0 - lam)
        box_height = height * math.sqrt(1.0 - lam)
        x1 = int(max(0.0, min(float(width), center_x - box_width / 2.0)))
        x2 = int(max(0.0, min(float(width), center_x + box_width / 2.0)))
        y1 = int(max(0.0, min(float(height), center_y - box_height / 2.0)))
        y2 = int(max(0.0, min(float(height), center_y + box_height / 2.0)))
        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
        if targets is None:
            return mixed_images
        lam_actual = 1.0 - ((x2 - x1) * (y2 - y1)) / float(width * height)
        mixed_targets = lam_actual * targets + (1.0 - lam_actual) * targets[permutation]
        return mixed_images, mixed_targets
