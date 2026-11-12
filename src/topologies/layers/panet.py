from itertools import pairwise
from numbers import Integral

import torch
from torch import nn


class PANet(nn.Module):
    component_id = "panet"
    PRESETS = {
        "tiny": {"channels": 8},
        "small": {"channels": 32},
        "base": {"channels": 256},
    }

    def __init__(self, channels=256, output_mode="pyramid"):
        super().__init__()
        if isinstance(channels, bool) or not isinstance(channels, Integral) or channels <= 0:
            raise ValueError("channels must be a positive integer")
        if not isinstance(output_mode, str) or output_mode not in {"pyramid", "pooled"}:
            raise ValueError("output_mode must be 'pyramid' or 'pooled'")
        self.channels = int(channels)
        self.output_mode = output_mode
        self.downsample_convs = nn.ModuleList(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=2, padding=1)
            for _ in range(3)
        )
        self.fusion_convs = nn.ModuleList(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1)
            for _ in range(3)
        )
        self.relu = nn.ReLU()

    def _validate_features(self, features):
        if not isinstance(features, (tuple, list)) or len(features) != 4:
            raise ValueError("features must be a sequence of four P2-P5 tensors")
        if any(not isinstance(feature, torch.Tensor) for feature in features):
            raise ValueError("each feature level must be a tensor")
        if any(feature.ndim != 4 for feature in features):
            raise ValueError("each feature level must have shape [batch, channels, height, width]")
        if any(any(size <= 0 for size in feature.shape) for feature in features):
            raise ValueError("feature dimensions must be positive")
        first = features[0]
        if any(feature.shape[0] != first.shape[0] for feature in features):
            raise ValueError("all feature levels must have the same batch size")
        if any(feature.shape[1] != self.channels for feature in features):
            raise ValueError(f"all feature levels must have {self.channels} channels")
        if any(feature.dtype != first.dtype for feature in features):
            raise ValueError("all feature levels must have the same dtype")
        if not first.is_floating_point():
            raise ValueError("feature levels must use a floating-point dtype")
        if any(feature.device != first.device for feature in features):
            raise ValueError("all feature levels must be on the same device")
        if any(feature.layout != torch.strided for feature in features):
            raise ValueError("feature levels must use strided tensor layout")
        parameter = self.downsample_convs[0].weight
        if first.dtype != parameter.dtype or first.device != parameter.device:
            raise ValueError("feature levels must match the module parameter dtype and device")
        for current, following in pairwise(features):
            height, width = current.shape[-2:]
            next_height, next_width = following.shape[-2:]
            if height % 2 or width % 2 or (next_height * 2, next_width * 2) != (height, width):
                raise ValueError("each feature level must be half the spatial size of the previous level")
        return tuple(features)

    def forward(self, features):
        pyramid = self._validate_features(features)
        outputs = [pyramid[0]]
        for index, (downsample, fusion) in enumerate(
            zip(self.downsample_convs, self.fusion_convs)
        ):
            reduced = self.relu(downsample(outputs[-1]))
            outputs.append(self.relu(fusion(reduced + pyramid[index + 1])))
        if self.output_mode == "pooled":
            return torch.cat([level.flatten(start_dim=2).mean(dim=-1) for level in outputs], dim=1)
        return tuple(outputs)
