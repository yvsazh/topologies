import math

import torch
from torch import nn


class Deformable3DGS(nn.Module):
    component_id = "deformable-3dgs"
    PRESETS = {
        "tiny": {"hidden_dim": 32, "depth": 8, "position_levels": 2, "time_levels": 2},
        "small": {"hidden_dim": 64, "depth": 8, "position_levels": 6, "time_levels": 6},
        "base": {"hidden_dim": 256, "depth": 8, "position_levels": 10, "time_levels": 6},
    }

    def __init__(self, hidden_dim=256, depth=8, position_levels=10, time_levels=6):
        super().__init__()
        for name, value in (
            ("hidden_dim", hidden_dim),
            ("depth", depth),
            ("position_levels", position_levels),
            ("time_levels", time_levels),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.position_levels = position_levels
        self.time_levels = time_levels
        self.encoded_dim = 6 * position_levels + 2 * time_levels
        self.layers = nn.ModuleList()
        for index in range(depth):
            if index == 0:
                in_features = self.encoded_dim
            elif index == 4:
                in_features = hidden_dim + self.encoded_dim
            else:
                in_features = hidden_dim
            self.layers.append(nn.Linear(in_features, hidden_dim))
        self.mean_offset = nn.Linear(hidden_dim, 3)

    @staticmethod
    def positional_encoding(value, levels):
        if not isinstance(value, torch.Tensor) or value.ndim < 1:
            raise ValueError("value must be a tensor with at least one dimension")
        if not value.is_floating_point():
            raise ValueError("value must have a floating-point dtype")
        if not isinstance(levels, int) or isinstance(levels, bool) or levels <= 0:
            raise ValueError("levels must be a positive integer")
        frequencies = torch.pow(
            value.new_tensor(2.0), torch.arange(levels, device=value.device, dtype=value.dtype)
        ) * math.pi
        angles = value.unsqueeze(-1) * frequencies
        encoded = torch.stack((angles.sin(), angles.cos()), dim=-1)
        return encoded.reshape(*value.shape[:-1], value.shape[-1] * levels * 2)

    def forward(self, x, time):
        if not isinstance(x, torch.Tensor) or x.ndim < 1 or x.shape[-1] != 3:
            raise ValueError("x must be a tensor whose final dimension is 3")
        if not x.is_floating_point():
            raise ValueError("x must have a floating-point dtype")
        if not isinstance(time, torch.Tensor) or not time.is_floating_point():
            raise ValueError("time must be a floating-point tensor")
        if time.device != x.device or time.dtype != x.dtype:
            raise ValueError("time and x must have the same device and dtype")
        try:
            broadcast_time = torch.broadcast_to(time, x.shape[:-1])
        except (RuntimeError, TypeError) as error:
            raise ValueError("time must be broadcastable to x.shape[:-1]") from error
        encoded_input = torch.cat(
            (
                self.positional_encoding(x.detach(), self.position_levels),
                self.positional_encoding(broadcast_time.unsqueeze(-1), self.time_levels),
            ),
            dim=-1,
        )
        hidden = encoded_input
        for index, layer in enumerate(self.layers):
            if index == 4:
                hidden = torch.cat((hidden, encoded_input), dim=-1)
            hidden = torch.relu(layer(hidden))
        return x + self.mean_offset(hidden)
