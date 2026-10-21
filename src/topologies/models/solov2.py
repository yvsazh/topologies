import torch
from torch import nn
from torch.nn import functional as F


class SOLOv2(nn.Module):
    architecture_id = "solov2"
    def __init__(self, in_channels=4, grid_size=2, hidden_channels=None):
        super().__init__()
        if in_channels <= 0 or grid_size <= 0:
            raise ValueError("channel count and grid size must be positive")
        hidden_channels = hidden_channels or in_channels
        if hidden_channels <= 0:
            raise ValueError("hidden channel count must be positive")
        self.grid_size = grid_size
        self.mask_feature = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
        )
        self.kernel_generator = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
        )

    def forward(self, features):
        if features.ndim != 4:
            raise ValueError("features must have shape (batch, channels, height, width)")
        batch, _, height, width = features.shape
        mask_feature = self.mask_feature(features)
        grid_features = F.adaptive_avg_pool2d(features, (self.grid_size, self.grid_size))
        kernels = self.kernel_generator(grid_features).flatten(2).transpose(1, 2)
        masks = torch.einsum("bkc,bchw->bkhw", kernels, mask_feature)
        masks = masks / (mask_feature.shape[1] ** 0.5)
        if masks.shape[-2:] != (height, width):
            masks = F.interpolate(masks, size=(height, width), mode="bilinear", align_corners=False)
        return {"masks": masks}
