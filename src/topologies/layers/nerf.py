import math

import torch
from torch import nn


class NeRFField(nn.Module):
    component_id = "nerf"

    def __init__(self, position_bands=10, direction_bands=4, depth=8, hidden_dim=256,
                 skip_after_layer=5, view_hidden_dim=128, density_noise_std=0.0):
        super().__init__()
        integer_values = {
            "position_bands": position_bands,
            "direction_bands": direction_bands,
            "depth": depth,
            "hidden_dim": hidden_dim,
            "skip_after_layer": skip_after_layer,
            "view_hidden_dim": view_hidden_dim,
        }
        for name, value in integer_values.items():
            if not isinstance(value, int) or isinstance(value, bool):
                self._invalid_argument(f"{name} must be an integer.")
        if position_bands < 1:
            raise ValueError("position_bands must be at least 1.")
        if direction_bands < 1:
            raise ValueError("direction_bands must be at least 1.")
        if depth < 2:
            raise ValueError("depth must be at least 2 to include a position skip connection.")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be at least 1.")
        if not 1 <= skip_after_layer < depth:
            raise ValueError("skip_after_layer must be between 1 and depth - 1.")
        if view_hidden_dim < 1:
            raise ValueError("view_hidden_dim must be at least 1.")
        if (not isinstance(density_noise_std, (int, float))
                or isinstance(density_noise_std, bool)
                or not math.isfinite(density_noise_std)
                or density_noise_std < 0):
            raise ValueError("density_noise_std must be a finite nonnegative number.")
        self.position_bands = position_bands
        self.direction_bands = direction_bands
        self.depth = depth
        self.hidden_dim = hidden_dim
        self.skip_after_layer = skip_after_layer
        self.view_hidden_dim = view_hidden_dim
        self.density_noise_std = float(density_noise_std)
        self.position_dim = 3 * 2 * position_bands
        direction_dim = 3 * 2 * direction_bands
        self.position_layers = nn.ModuleList()
        for index in range(depth):
            if index == 0:
                in_features = self.position_dim
            elif index == skip_after_layer:
                in_features = hidden_dim + self.position_dim
            else:
                in_features = hidden_dim
            self.position_layers.append(nn.Linear(in_features, hidden_dim))
        self.density_feature = nn.Linear(hidden_dim, hidden_dim + 1)
        self.view_layer = nn.Linear(hidden_dim + direction_dim, view_hidden_dim)
        self.rgb_layer = nn.Linear(view_hidden_dim, 3)

    @staticmethod
    def _invalid_argument(message):
        raise ValueError(message)

    @staticmethod
    def positional_encoding(values, bands):
        if not isinstance(values, torch.Tensor) or values.ndim < 1:
            raise ValueError("values must be a tensor with at least one dimension.")
        if not values.is_floating_point():
            raise ValueError("values must have a floating-point dtype.")
        if not isinstance(bands, int) or isinstance(bands, bool) or bands < 1:
            raise ValueError("bands must be a positive integer.")
        frequencies = math.pi * (2.0 ** torch.arange(bands, device=values.device,
                                                    dtype=values.dtype))
        angles = values.unsqueeze(-1) * frequencies
        encoded = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1)
        return encoded.flatten(start_dim=-3)

    def forward(self, positions, directions):
        if not isinstance(positions, torch.Tensor) or positions.ndim < 1:
            raise ValueError("positions must be a tensor with a last dimension of 3.")
        if positions.shape[-1] != 3:
            raise ValueError("positions must have a last dimension of 3.")
        if not isinstance(directions, torch.Tensor) or directions.ndim < 1:
            raise ValueError("directions must be a tensor with a last dimension of 3.")
        if directions.shape[-1] != 3:
            raise ValueError("directions must have a last dimension of 3.")
        if positions.shape[:-1] != directions.shape[:-1]:
            raise ValueError("positions and directions must have matching leading dimensions.")
        if not positions.is_floating_point() or not directions.is_floating_point():
            raise ValueError("positions and directions must have floating-point dtypes.")
        if positions.device != directions.device or positions.dtype != directions.dtype:
            raise ValueError("positions and directions must have matching devices and dtypes.")
        encoded_positions = self.positional_encoding(positions, self.position_bands)
        hidden = encoded_positions
        for index, layer in enumerate(self.position_layers):
            hidden = torch.relu(layer(hidden))
            if index + 1 == self.skip_after_layer:
                hidden = torch.cat((hidden, encoded_positions), dim=-1)
        field_values = self.density_feature(hidden)
        raw_density = field_values[..., :1]
        features = field_values[..., 1:]
        if self.training and self.density_noise_std > 0:
            raw_density = raw_density + torch.randn_like(raw_density) * self.density_noise_std
        density = torch.relu(raw_density)
        encoded_directions = self.positional_encoding(directions, self.direction_bands)
        view_features = torch.relu(self.view_layer(torch.cat((features, encoded_directions), dim=-1)))
        rgb = torch.sigmoid(self.rgb_layer(view_features))
        return torch.cat((rgb, density), dim=-1)
