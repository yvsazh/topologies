import math

import torch
from torch import nn


class KorniaGeometry(nn.Module):
    component_id = "kornia-geometry"

    @staticmethod
    def _invalid_value(message):
        raise ValueError(message)

    def __init__(self, eps=1e-8, *args, **kwargs):
        super().__init__()
        if args or kwargs:
            self._invalid_value("only eps is a supported constructor argument")
        if isinstance(eps, bool) or not isinstance(eps, (int, float)):
            self._invalid_value("eps must be a finite positive number")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be a finite positive number")
        self.eps = float(eps)

    def _coordinates(self, value, name):
        if not isinstance(value, torch.Tensor):
            self._invalid_value(f"{name} must be a torch.Tensor")
        if value.ndim not in (2, 3) or value.shape[-1] != 2:
            raise ValueError(f"{name} must have shape [N, 2] or [B, N, 2]")
        if value.shape[-2] == 0 or (value.ndim == 3 and value.shape[0] == 0):
            raise ValueError(f"{name} must not have an empty batch or point dimension")
        if value.dtype not in (torch.float32, torch.float64):
            raise ValueError(f"{name} must use float32 or float64 coordinates")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite coordinates")
        return value.unsqueeze(0) if value.ndim == 2 else value

    def _normalization(self, points, weights):
        weight_sum = weights.sum(dim=-1, keepdim=True)
        if (weight_sum <= 0).any():
            raise ValueError("each batch must have positive total correspondence weight")
        center = (points * weights.unsqueeze(-1)).sum(dim=-2) / weight_sum
        radius = (
            torch.linalg.vector_norm(points - center.unsqueeze(-2), dim=-1) * weights
        ).sum(dim=-1, keepdim=True) / weight_sum
        if (radius <= self.eps).any():
            raise ValueError("correspondences are rank-deficient")
        scale = math.sqrt(2.0) / (radius + self.eps)
        zeros = torch.zeros_like(scale)
        ones = torch.ones_like(scale)
        transform = torch.stack(
            (
                torch.cat((scale, zeros, -scale * center[..., 0:1]), dim=-1),
                torch.cat((zeros, scale, -scale * center[..., 1:2]), dim=-1),
                torch.cat((zeros, zeros, ones), dim=-1),
            ),
            dim=-2,
        )
        normalized = (points - center.unsqueeze(-2)) * scale.unsqueeze(-1)
        return normalized, transform

    def _design_matrix(self, source, destination):
        x = source[..., 0]
        y = source[..., 1]
        u = destination[..., 0]
        v = destination[..., 1]
        zero = torch.zeros_like(x)
        one = torch.ones_like(x)
        row_v = torch.stack(
            (zero, zero, zero, -x, -y, -one, v * x, v * y, v), dim=-1
        )
        row_u = torch.stack(
            (x, y, one, zero, zero, zero, -u * x, -u * y, -u), dim=-1
        )
        return torch.stack((row_v, row_u), dim=-2).reshape(source.shape[0], -1, 9)

    def estimate_homography(self, source, destination, weights=None):
        source_batch = self._coordinates(source, "source")
        destination_batch = self._coordinates(destination, "destination")
        if source.shape != destination.shape:
            raise ValueError("source and destination must have matching shapes")
        if source.device != destination.device or source.dtype != destination.dtype:
            raise ValueError("source and destination must share a device and dtype")
        if source.shape[-2] < 4:
            raise ValueError("at least four correspondence pairs are required")

        batch_size, point_count = source_batch.shape[:2]
        if weights is None:
            weight_batch = torch.ones(
                (batch_size, point_count), dtype=source.dtype, device=source.device
            )
        else:
            weight_batch = torch.as_tensor(weights, dtype=source.dtype, device=source.device)
            if weight_batch.ndim == 1 and weight_batch.shape[0] == point_count:
                weight_batch = weight_batch.unsqueeze(0).expand(batch_size, -1)
            elif weight_batch.ndim == 2 and weight_batch.shape == (batch_size, point_count):
                pass
            else:
                raise ValueError("weights must have shape [N] or [B, N]")
            if not torch.isfinite(weight_batch).all():
                raise ValueError("weights must contain only finite values")
            if (weight_batch < 0).any():
                raise ValueError("weights must be nonnegative")

        normalized_source, source_transform = self._normalization(
            source_batch, weight_batch
        )
        normalized_destination, destination_transform = self._normalization(
            destination_batch, weight_batch
        )
        design = self._design_matrix(normalized_source, normalized_destination)
        row_weights = weight_batch.repeat_interleave(2, dim=-1).sqrt().unsqueeze(-1)
        weighted_design = design * row_weights
        ranks = torch.linalg.matrix_rank(weighted_design)
        if (ranks < 8).any():
            raise ValueError("correspondences are rank-deficient")

        normal_matrix = design.transpose(-1, -2) @ (design * row_weights.square())
        normal_matrix = (normal_matrix + normal_matrix.transpose(-1, -2)) * 0.5
        _, eigenvectors = torch.linalg.eigh(normal_matrix)
        normalized_homography = eigenvectors[..., :, 0].reshape(-1, 3, 3)
        homography = (
            torch.linalg.inv(destination_transform)
            @ normalized_homography
            @ source_transform
        )
        gauge = homography[..., 2, 2]
        divisor = torch.where(gauge.abs() > self.eps, gauge, torch.ones_like(gauge))
        homography = homography / divisor[..., None, None]
        if not torch.isfinite(homography).all():
            raise ValueError("correspondences do not produce a finite homography")
        if source.ndim == 2:
            homography = homography.squeeze(0)
        return homography

    def forward(self, source, destination, weights=None, points=None):
        homography = self.estimate_homography(source, destination, weights=weights)
        if points is None:
            return homography
        return homography, self.apply_homography(homography, points)

    def apply_homography(self, points, homography):
        if (
            isinstance(points, torch.Tensor)
            and points.shape[-2:] == (3, 3)
            and isinstance(homography, torch.Tensor)
            and homography.ndim in (2, 3)
            and homography.shape[-1:] == (2,)
        ):
            homography, points = points, homography
        if not isinstance(homography, torch.Tensor):
            self._invalid_value("homography must be a torch.Tensor")
        if homography.ndim not in (2, 3) or homography.shape[-2:] != (3, 3):
            raise ValueError("homography must have shape [3, 3] or [B, 3, 3]")
        if not torch.is_floating_point(homography) or not torch.isfinite(homography).all():
            raise ValueError("homography must contain finite floating-point values")
        point_batch = self._coordinates(points, "points")
        if homography.device != points.device or homography.dtype != points.dtype:
            raise ValueError("homography and points must share a device and dtype")
        homography_batch = homography.unsqueeze(0) if homography.ndim == 2 else homography
        if homography_batch.shape[0] != point_batch.shape[0]:
            if homography_batch.shape[0] == 1:
                homography_batch = homography_batch.expand(point_batch.shape[0], -1, -1)
            elif point_batch.shape[0] == 1:
                point_batch = point_batch.expand(homography_batch.shape[0], -1, -1)
            else:
                raise ValueError("homography and points batch dimensions must match")
        homogeneous_points = torch.cat(
            (point_batch, torch.ones_like(point_batch[..., :1])), dim=-1
        )
        projected = homography_batch.unsqueeze(1) @ homogeneous_points.unsqueeze(-1)
        denominator = projected[..., 2, 0]
        if not torch.isfinite(projected).all() or (denominator.abs() <= self.eps).any():
            raise ValueError("homography produces an infinite projection")
        mapped = projected[..., :2, 0] / denominator.unsqueeze(-1)
        if homography.ndim == 2 and points.ndim == 2:
            return mapped.squeeze(0)
        return mapped
