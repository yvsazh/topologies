import torch
from torch import nn
from torch.nn import functional as F


class ProGAN(nn.Module):
    architecture_id = "progan"

    def __init__(self, channels: int = 4, image_channels: int = 3, alpha: float = 0.5):
        super().__init__()
        if channels < 1 or image_channels < 1:
            raise ValueError("feature and image channel counts must be positive")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = float(alpha)
        self.channels = int(channels)
        self.image_channels = int(image_channels)
        self.to_rgb = nn.Conv2d(self.channels, self.image_channels, kernel_size=1)

    def forward(
        self,
        previous_image: torch.Tensor,
        new_features: torch.Tensor,
        alpha: float | None = None,
    ) -> torch.Tensor:
        blend = self.alpha if alpha is None else float(alpha)
        if not 0.0 <= blend <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if previous_image.ndim != 4 or new_features.ndim != 4:
            raise ValueError("both inputs must have shape (batch, channels, height, width)")
        if previous_image.shape[1] != self.image_channels:
            raise ValueError("previous_image has the wrong number of image channels")
        if new_features.shape[1] != self.channels:
            raise ValueError("new_features has the wrong number of feature channels")
        if new_features.shape[0] != previous_image.shape[0] or new_features.shape[-2:] != tuple(2 * d for d in previous_image.shape[-2:]):
            raise ValueError("new_features must match the batch and have twice the spatial resolution")
        low_path = F.interpolate(previous_image, scale_factor=2, mode="nearest")
        high_path = self.to_rgb(new_features)
        return low_path.mul(1.0 - blend).add(high_path, alpha=blend)
