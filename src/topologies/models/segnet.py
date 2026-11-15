import math

import torch
from torch import nn


class SegNet(nn.Module):
    architecture_id = "segnet"
    PRESETS = {"tiny": {"width_mult": 0.25}, "small": {"width_mult": 0.5}}

    def __init__(self, in_channels=3, num_classes=11, width_mult=1.0):
        super().__init__()
        if type(in_channels) is not int or in_channels < 1:
            raise ValueError("in_channels must be a positive integer")
        if type(num_classes) is not int or num_classes < 2:
            raise ValueError("num_classes must be an integer of at least 2")
        if (
            isinstance(width_mult, bool)
            or not isinstance(width_mult, (int, float))
            or not math.isfinite(width_mult)
            or width_mult <= 0
        ):
            raise ValueError("width_mult must be a finite positive number")

        channels = [max(8, round(value * width_mult / 8) * 8) for value in (64, 128, 256, 512, 512)]
        encoder_depths = (2, 2, 3, 3, 3)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.encoder_blocks = nn.ModuleList()
        self.encoder_pools = nn.ModuleList()
        current_channels = in_channels
        for output_channels, depth in zip(channels, encoder_depths, strict=True):
            layers = []
            for _ in range(depth):
                layers.extend(
                    (
                        nn.Conv2d(current_channels, output_channels, 3, padding=1),
                        nn.BatchNorm2d(output_channels),
                        nn.ReLU(inplace=False),
                    )
                )
                current_channels = output_channels
            self.encoder_blocks.append(nn.Sequential(*layers))
            self.encoder_pools.append(nn.MaxPool2d(2, stride=2, return_indices=True))

        decoder_outputs = (
            (channels[4], channels[4], channels[3]),
            (channels[3], channels[3], channels[2]),
            (channels[2], channels[2], channels[1]),
            (channels[1], channels[0]),
            (channels[0],),
        )
        self.decoder_unpools = nn.ModuleList(nn.MaxUnpool2d(2, stride=2) for _ in channels)
        self.decoder_blocks = nn.ModuleList()
        current_channels = channels[-1]
        for stage_channels in decoder_outputs:
            layers = []
            for output_channels in stage_channels:
                layers.extend(
                    (
                        nn.Conv2d(current_channels, output_channels, 3, padding=1),
                        nn.BatchNorm2d(output_channels),
                        nn.ReLU(inplace=False),
                    )
                )
                current_channels = output_channels
            self.decoder_blocks.append(nn.Sequential(*layers))
        self.classifier = nn.Conv2d(channels[0], num_classes, 3, padding=1)

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            raise ValueError("input must be a [B, C, H, W] tensor")
        if x.shape[0] < 1 or x.shape[1] != self.in_channels:
            raise ValueError(f"input must have a positive batch and {self.in_channels} channels")
        if min(x.shape[-2:]) < 32:
            raise ValueError("input height and width must both be at least 32")
        if not x.is_floating_point():
            raise ValueError("input must have a floating point dtype")

        pool_indices = []
        output_sizes = []
        for block, pool in zip(self.encoder_blocks, self.encoder_pools, strict=True):
            x = block(x)
            output_sizes.append(x.size())
            x, indices = pool(x)
            pool_indices.append(indices)

        for block, unpool, indices, output_size in zip(
            self.decoder_blocks,
            self.decoder_unpools,
            reversed(pool_indices),
            reversed(output_sizes),
            strict=True,
        ):
            x = unpool(x, indices, output_size=output_size)
            x = block(x)
        return self.classifier(x)
