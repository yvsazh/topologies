import math
from numbers import Real

import torch
from torch import nn


class DIBR(nn.Module):
    component_id = "dib-r"

    def __init__(self, height=64, width=64, sigma_inv=7000):
        super().__init__()
        if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
            raise ValueError("height must be a positive integer")
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise ValueError("width must be a positive integer")
        if (
            isinstance(sigma_inv, bool)
            or not isinstance(sigma_inv, Real)
            or not math.isfinite(float(sigma_inv))
            or sigma_inv <= 0
        ):
            raise ValueError("sigma_inv must be a positive finite number")
        self.height = height
        self.width = width
        self.sigma_inv = float(sigma_inv)

    def forward(self, vertices, attributes, return_alpha=False):
        if not isinstance(vertices, torch.Tensor) or vertices.ndim != 3 or vertices.shape[1:] != (3, 2):
            raise ValueError("vertices must have shape (B, 3, 2)")
        if (
            not isinstance(attributes, torch.Tensor)
            or attributes.ndim != 3
            or attributes.shape[1] != 3
            or attributes.shape[2] <= 0
        ):
            raise ValueError("attributes must have shape (B, 3, C) with C positive")
        if vertices.shape[0] <= 0 or attributes.shape[0] != vertices.shape[0]:
            raise ValueError("vertices and attributes must have matching batch dimensions and positive batch size")
        if not vertices.is_floating_point() or not attributes.is_floating_point():
            raise ValueError("vertices and attributes must use floating-point tensors")
        if vertices.dtype != attributes.dtype or vertices.device != attributes.device:
            raise ValueError("vertices and attributes must have matching dtype and device")
        if not torch.isfinite(vertices).all() or not torch.isfinite(attributes).all():
            raise ValueError("vertices and attributes must contain only finite values")

        first = vertices[:, 0]
        second = vertices[:, 1]
        third = vertices[:, 2]
        first_edge = second - first
        second_edge = third - first
        twice_area = (
            first_edge[:, 0] * second_edge[:, 1]
            - second_edge[:, 0] * first_edge[:, 1]
        )
        if (twice_area == 0).any():
            raise ValueError("vertices must not describe a zero-area triangle")

        columns = torch.arange(self.width, dtype=vertices.dtype, device=vertices.device)
        columns = (2 * columns + 1 - self.width) / self.width
        rows = torch.arange(self.height, dtype=vertices.dtype, device=vertices.device)
        rows = (self.height - 2 * rows - 1) / self.height
        y, x = torch.meshgrid(rows, columns, indexing="ij")

        x0 = first[:, 0, None, None]
        y0 = first[:, 1, None, None]
        m = first_edge[:, 0, None, None]
        p = first_edge[:, 1, None, None]
        n = second_edge[:, 0, None, None]
        q = second_edge[:, 1, None, None]
        s = x[None, :, :] - x0
        t = y[None, :, :] - y0
        denominator = m * q - n * p
        epsilon = torch.where(denominator >= 0, 1e-15, -1e-15)
        denominator = denominator + epsilon
        weight1 = (s * q - n * t) / denominator
        weight2 = (m * t - s * p) / denominator
        weight0 = 1 - weight1 - weight2
        weights = torch.stack((weight0, weight1, weight2), dim=-1)
        covered = (weights >= 0).all(dim=-1)

        image = (
            weight0[..., None] * attributes[:, None, None, 0, :]
            + weight1[..., None] * attributes[:, None, None, 1, :]
            + weight2[..., None] * attributes[:, None, None, 2, :]
        )
        image = torch.where(covered[..., None], image, torch.zeros_like(image))

        points = torch.stack(
            (
                x[None, :, :].expand(vertices.shape[0], -1, -1),
                y[None, :, :].expand(vertices.shape[0], -1, -1),
            ),
            dim=-1,
        )
        edge_starts = vertices
        edge_vectors = vertices.roll(-1, dims=1) - edge_starts
        relative = points[:, None, :, :, :] - edge_starts[:, :, None, None, :]
        edge_length_squared = edge_vectors.square().sum(dim=-1)
        projection = (relative * edge_vectors[:, :, None, None, :]).sum(dim=-1)
        projection = projection / edge_length_squared[:, :, None, None]
        projection = projection.clamp(0, 1)
        residual = relative - projection[..., None] * edge_vectors[:, :, None, None, :]
        distance_squared = residual.square().sum(dim=-1).min(dim=1).values
        soft_alpha = torch.exp(-self.sigma_inv * distance_squared)
        alpha = torch.where(covered, torch.ones_like(soft_alpha), soft_alpha)
        if return_alpha:
            return image, alpha[..., None]
        return image
