import math
from numbers import Real

import torch
from torch import nn


def _value_error(message: str) -> ValueError:
    return ValueError(message)


class SoftRas(nn.Module):
    component_id = "softras"

    def __init__(self, sigma: float = 1e-4):
        super().__init__()
        if not isinstance(sigma, Real) or isinstance(sigma, bool):
            raise _value_error("sigma must be a finite positive scalar")
        try:
            sigma = float(sigma)
        except (OverflowError, TypeError, ValueError) as error:
            raise _value_error("sigma must be a finite positive scalar") from error
        if not math.isfinite(sigma) or sigma <= 0:
            raise _value_error("sigma must be a finite positive scalar")
        self.sigma = sigma

    def forward(self, point: torch.Tensor, triangle: torch.Tensor) -> torch.Tensor:
        if not isinstance(point, torch.Tensor):
            raise _value_error("point must be a torch.Tensor with shape (2,)")
        if not isinstance(triangle, torch.Tensor):
            raise _value_error("triangle must be a torch.Tensor with shape (3, 2)")
        if point.ndim == 2 or triangle.ndim == 3:
            if (point.ndim != 2 or triangle.ndim != 3 or point.shape[1:] != (2,)
                    or triangle.shape[1:] != (3, 2) or point.shape[0] != triangle.shape[0]
                    or point.shape[0] == 0):
                raise _value_error("batched point and triangle must have shapes (batch, 2) and (batch, 3, 2)")
            return torch.stack([self.forward(item, face) for item, face in zip(point, triangle)])
        if point.shape != (2,):
            raise _value_error("point must have shape (2,)")
        if triangle.shape != (3, 2):
            raise _value_error("triangle must have shape (3, 2)")
        if not point.is_floating_point() or not triangle.is_floating_point():
            raise _value_error("point and triangle must use floating-point tensors")
        if point.dtype != triangle.dtype:
            raise _value_error("point and triangle must use the same floating-point dtype")
        if point.device != triangle.device:
            raise _value_error("point and triangle must be on the same device")
        if not torch.isfinite(point).all().item() or not torch.isfinite(triangle).all().item():
            raise _value_error("point and triangle values must be finite")

        first, second, third = triangle.unbind(dim=0)
        orientation = (second[0] - first[0]) * (third[1] - first[1]) - (
            second[1] - first[1]
        ) * (third[0] - first[0])
        if orientation.item() == 0:
            raise _value_error("triangle must be non-degenerate")

        starts = torch.stack((first, second, third))
        ends = torch.stack((second, third, first))
        edges = ends - starts
        point_offsets = point.unsqueeze(0) - starts
        edge_lengths_squared = edges.square().sum(dim=-1)
        projections = (point_offsets * edges).sum(dim=-1) / edge_lengths_squared
        closest = starts + projections.clamp(0, 1).unsqueeze(-1) * edges
        distances_squared = (point.unsqueeze(0) - closest).square().sum(dim=-1)
        distance_squared = distances_squared.min()

        crosses = edges[:, 0] * point_offsets[:, 1] - edges[:, 1] * point_offsets[:, 0]
        inside = (crosses >= 0).all() if orientation.item() > 0 else (crosses <= 0).all()
        sign = 1.0 if inside.item() else -1.0
        return torch.sigmoid(sign * distance_squared / self.sigma)
