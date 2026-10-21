import torch
from torch import nn
from torch.nn import functional as F


class KPlanes(nn.Module):
    component_id = "k-planes"
    def __init__(self, axes, feature_dim=8, resolution=16, coordinate_dim=None):
        super().__init__()
        axes = tuple(tuple(pair) for pair in axes)
        if not axes:
            raise ValueError("axes must contain at least one plane")
        if any(len(pair) != 2 or pair[0] == pair[1] or min(pair) < 0 for pair in axes):
            raise ValueError("each plane must specify two distinct nonnegative axis indices")
        inferred_dim = max(max(pair) for pair in axes) + 1
        coordinate_dim = inferred_dim if coordinate_dim is None else coordinate_dim
        if coordinate_dim < inferred_dim:
            raise ValueError("coordinate_dim does not cover all plane axes")
        if feature_dim < 1 or resolution < 1:
            raise ValueError("feature_dim and resolution must be positive")
        self.axes = axes
        self.coordinate_dim = coordinate_dim
        self.feature_dim = feature_dim
        self.planes = nn.ParameterList(
            nn.Parameter(torch.ones(1, feature_dim, resolution, resolution))
            for _ in axes
        )

    def sample_planes(self, coordinates):
        if coordinates.ndim != 2 or coordinates.shape[1] != self.coordinate_dim:
            raise ValueError("coordinates must have shape (batch, coordinate_dim)")
        samples = []
        for axes, plane in zip(self.axes, self.planes):
            grid = coordinates[:, list(axes)].mul(2).sub(1).reshape(-1, 1, 1, 2)
            sampled = F.grid_sample(plane.expand(coordinates.shape[0], -1, -1, -1), grid,
                                    mode="bilinear", padding_mode="border", align_corners=True)
            samples.append(sampled[:, :, 0, 0])
        return samples

    def forward(self, coordinates):
        samples = self.sample_planes(coordinates)
        result = samples[0]
        for sample in samples[1:]:
            result = result * sample
        return result
