import torch
from torch import nn


class KPConv(nn.Module):
    component_id = "kpconv"
    def __init__(self, input_channels=4, output_channels=4, num_kernel_points=3,
                 influence_distance=1.0, convolution_radius=None):
        super().__init__()
        if input_channels <= 0 or output_channels <= 0 or num_kernel_points <= 0:
            raise ValueError("channel counts and kernel-point count must be positive")
        if influence_distance <= 0:
            raise ValueError("influence_distance must be positive")
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_kernel_points = num_kernel_points
        self.influence_distance = float(influence_distance)
        self.convolution_radius = float(
            2.5 * influence_distance if convolution_radius is None else convolution_radius
        )
        if self.convolution_radius <= 0:
            raise ValueError("convolution_radius must be positive")
        kernel_points = torch.zeros(num_kernel_points, 3, dtype=torch.float32)
        if num_kernel_points > 1:
            count = num_kernel_points - 1
            indices = torch.arange(count, dtype=torch.float32)
            z = 1.0 - 2.0 * (indices + 0.5) / count
            angle = indices * (torch.pi * (3.0 - 5.0 ** 0.5))
            radial = (1.0 - z.square()).clamp_min(0).sqrt()
            kernel_points[1:] = 1.5 * influence_distance * torch.stack(
                (radial * angle.cos(), radial * angle.sin(), z), dim=-1
            )
        self.register_buffer("kernel_points", kernel_points)
        self.kernel_weights = nn.Parameter(torch.empty(num_kernel_points, input_channels, output_channels))
        nn.init.xavier_uniform_(self.kernel_weights)

    def influence(self, relative_points):
        kernel_points = self.kernel_points.to(device=relative_points.device, dtype=relative_points.dtype)
        distances = torch.linalg.vector_norm(
            relative_points.unsqueeze(-2) - kernel_points, dim=-1
        )
        return torch.relu(1.0 - distances / self.influence_distance)

    def forward(self, points, features):
        if points.ndim != 3 or points.shape[-1] != 3 or not points.is_floating_point():
            raise ValueError("points must have shape [batch, points, 3]")
        if features.ndim != 3 or features.shape[:2] != points.shape[:2] or features.shape[-1] != self.input_channels:
            raise ValueError("features must have shape [batch, points, input_channels]")
        if points.shape[1] == 0:
            raise ValueError("point clouds must contain at least one point")
        if points.device != features.device or points.device != self.kernel_weights.device:
            raise ValueError("points, features, and module parameters must share a device")
        if points.dtype != features.dtype or points.dtype != self.kernel_weights.dtype:
            raise ValueError("points, features, and module parameters must share a dtype")
        distances = torch.cdist(points, points)
        neighborhood = distances <= self.convolution_radius
        relative = points.unsqueeze(1) - points.unsqueeze(2)
        influence = self.influence(relative)
        influence = influence * neighborhood.unsqueeze(-1)
        transformed = torch.einsum("bjc,kco->bjko", features, self.kernel_weights).unsqueeze(1)
        return (transformed * influence.unsqueeze(-1)).sum(dim=(2, 3))
