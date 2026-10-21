import itertools
import math

import torch
from torch import nn


class InstantNGP(nn.Module):
    component_id = "instant-ngp"
    def __init__(self, levels=16, features_per_level=2, log2_hashmap_size=15, min_resolution=16, max_resolution=512, input_dim=2):
        super().__init__()
        if levels < 1 or features_per_level < 1 or log2_hashmap_size < 1 or min_resolution < 2 or max_resolution < min_resolution or input_dim < 1:
            raise ValueError("invalid multiresolution hash encoding configuration")
        self.levels = levels
        self.features_per_level = features_per_level
        self.input_dim = input_dim
        self.table_size = 1 << log2_hashmap_size
        if levels == 1:
            resolutions = [min_resolution]
        else:
            scale = math.exp(math.log(max_resolution / min_resolution) / (levels - 1))
            resolutions = [int(math.floor(min_resolution * scale ** i)) for i in range(levels)]
        self.resolutions = tuple(resolutions)
        self.tables = nn.ParameterList([nn.Parameter(torch.empty(self.table_size, features_per_level).uniform_(-1e-4, 1e-4)) for _ in resolutions])
        self._corner_offsets = tuple(itertools.product((0, 1), repeat=input_dim))
        self._hash_primes = (1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437)
        if input_dim > len(self._hash_primes):
            raise ValueError("input_dim must not exceed 6")

    @property
    def output_dim(self):
        return self.levels * self.features_per_level

    def forward(self, coordinates):
        if coordinates.shape[-1] != self.input_dim:
            raise ValueError(f"expected coordinates with final dimension {self.input_dim}")
        outputs = []
        for resolution, table in zip(self.resolutions, self.tables):
            position = coordinates.clamp(0, 1) * resolution
            lower = torch.floor(position).to(torch.long)
            fraction = position - lower.to(position.dtype)
            values = coordinates.new_zeros(*coordinates.shape[:-1], self.features_per_level)
            for offset in self._corner_offsets:
                corner = lower + torch.tensor(offset, device=coordinates.device)
                weight = coordinates.new_ones(coordinates.shape[:-1])
                for axis, bit in enumerate(offset):
                    weight = weight * (fraction[..., axis] if bit else 1 - fraction[..., axis])
                hashed = torch.zeros_like(corner[..., 0])
                for axis, prime in enumerate(self._hash_primes[:self.input_dim]):
                    hashed = torch.bitwise_xor(hashed, corner[..., axis] * prime)
                index = torch.remainder(hashed, self.table_size)
                values = values + table[index] * weight.unsqueeze(-1)
            outputs.append(values)
        return torch.cat(outputs, dim=-1)
