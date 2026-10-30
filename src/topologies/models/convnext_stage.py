import math

import torch
from torch import nn


class ConvNeXtBlock(nn.Module):
    architecture_id = "convnext-stage"
    PRESETS = {
        "tiny": {
            "dim": 32,
            "drop_path": 0.0,
            "layer_scale_init_value": 1e-6,
            "eps": 1e-6,
        },
        "small": {
            "dim": 96,
            "drop_path": 0.0,
            "layer_scale_init_value": 1e-6,
            "eps": 1e-6,
        },
        "base": {
            "dim": 128,
            "drop_path": 0.0,
            "layer_scale_init_value": 1e-6,
            "eps": 1e-6,
        },
    }

    def __init__(
        self,
        dim=96,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        eps=1e-6,
    ):
        super().__init__()
        if type(dim) is not int or dim < 1:
            raise ValueError("dim must be a positive integer")
        if (
            type(drop_path) not in (int, float)
            or not 0 <= drop_path < 1
            or (type(drop_path) is float and not math.isfinite(drop_path))
        ):
            raise ValueError("drop_path must be a finite number in [0, 1)")
        if (
            type(layer_scale_init_value) not in (int, float)
            or layer_scale_init_value < 0
            or (
                type(layer_scale_init_value) is float
                and not math.isfinite(layer_scale_init_value)
            )
        ):
            raise ValueError("layer_scale_init_value must be a finite nonnegative number")
        if (
            type(eps) not in (int, float)
            or eps <= 0
            or (type(eps) is float and not math.isfinite(eps))
        ):
            raise ValueError("eps must be a finite positive number")
        self.dim = dim
        self.drop_path = float(drop_path)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(torch.full((dim,), float(layer_scale_init_value)))
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x):
        if not torch.is_tensor(x):
            raise ValueError("input must be a torch.Tensor")
        if x.ndim != 4:
            raise ValueError("input must have NCHW shape [B, C, H, W]")
        if x.shape[0] < 1 or x.shape[2] < 1 or x.shape[3] < 1:
            raise ValueError("batch and spatial dimensions must be positive")
        if x.shape[1] != self.dim:
            raise ValueError(f"input channel count must equal dim ({self.dim})")
        if not x.is_floating_point():
            raise ValueError("input must have a floating-point dtype")
        residual = self.dwconv(x).permute(0, 2, 3, 1)
        residual = self.pwconv2(self.act(self.pwconv1(self.norm(residual))))
        if self.gamma is not None:
            residual = residual * self.gamma
        residual = residual.permute(0, 3, 1, 2)
        if self.training and self.drop_path > 0:
            keep_probability = 1.0 - self.drop_path
            mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            mask = x.new_empty(mask_shape).bernoulli_(keep_probability)
            residual = residual * mask / keep_probability
        return x + residual
