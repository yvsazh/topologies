import math

import torch
from torch import nn


class EDSR(nn.Module):
    architecture_id = "edsr"
    PRESETS = {
        "tiny": {"scale": 2, "num_blocks": 4, "num_features": 32, "res_scale": 0.1,
                 "rgb_range": 255},
        "small": {"scale": 4, "num_blocks": 32, "num_features": 256, "res_scale": 0.1,
                  "rgb_range": 255},
        "base": {"scale": 4, "num_blocks": 16, "num_features": 64, "res_scale": 1.0,
                 "rgb_range": 255},
    }

    def __init__(self, scale=4, num_blocks=32, num_features=256, res_scale=0.1,
                 rgb_range=255):
        super().__init__()
        if isinstance(scale, bool) or not isinstance(scale, int) or scale not in (2, 3, 4):
            raise ValueError("scale must be one of 2, 3, or 4")
        if isinstance(num_blocks, bool) or not isinstance(num_blocks, int) or num_blocks < 1:
            raise ValueError("num_blocks must be a positive integer")
        if isinstance(num_features, bool) or not isinstance(num_features, int) or num_features < 1:
            raise ValueError("num_features must be a positive integer")
        if (isinstance(res_scale, bool) or not isinstance(res_scale, (int, float))
                or not math.isfinite(res_scale) or res_scale <= 0):
            raise ValueError("res_scale must be a positive finite number")
        if (isinstance(rgb_range, bool) or not isinstance(rgb_range, (int, float))
                or not math.isfinite(rgb_range) or rgb_range <= 0):
            raise ValueError("rgb_range must be a positive finite number")
        self.scale = scale
        self.num_blocks = num_blocks
        self.num_features = num_features
        self.res_scale = float(res_scale)
        self.rgb_range = float(rgb_range)
        self.register_buffer(
            "rgb_mean",
            self.rgb_range * torch.tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1),
        )
        self.head = nn.Conv2d(3, num_features, 3, padding=1)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(num_features, num_features, 3, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(num_features, num_features, 3, padding=1),
            )
            for _ in range(num_blocks)
        )
        self.body_conv = nn.Conv2d(num_features, num_features, 3, padding=1)
        upsampler = []
        stages = 2 if scale == 4 else 1
        stage_scale = 2 if scale == 4 else scale
        for _ in range(stages):
            upsampler.extend((
                nn.Conv2d(num_features, num_features * stage_scale ** 2, 3, padding=1),
                nn.PixelShuffle(stage_scale),
            ))
        self.upsampler = nn.Sequential(*upsampler)
        self.tail = nn.Conv2d(num_features, 3, 3, padding=1)

    def _residual_block(self, block, features):
        return features + self.res_scale * block(features)

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            raise ValueError("input must have shape [N, 3, H, W]")
        if x.shape[0] < 1 or x.shape[1] != 3:
            raise ValueError("input channel dimension must be 3 and batch size must be positive")
        if x.shape[2] < 1 or x.shape[3] < 1:
            raise ValueError("input spatial dimensions must be positive")
        if not x.is_floating_point():
            raise ValueError("input must use a floating point dtype")
        mean = self.rgb_mean.to(dtype=x.dtype)
        features = self.head(x - mean)
        residual = features
        for block in self.blocks:
            residual = self._residual_block(block, residual)
        features = features + self.body_conv(residual)
        return self.tail(self.upsampler(features)) + mean
