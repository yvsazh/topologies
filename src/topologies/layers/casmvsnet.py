import math

import torch
from torch import nn


def _invalid(message):
    raise ValueError(message)


class CascadeDepthHypotheses(nn.Module):
    component_id = "casmvsnet"

    def __init__(self, ndepths=(48, 32, 8), depth_interval_ratios=(4, 2, 1),
                 detach_previous=True):
        super().__init__()
        if not isinstance(ndepths, (tuple, list)) or not ndepths:
            _invalid("ndepths must be a non-empty tuple or list")
        if any(type(count) is not int or count < 2 for count in ndepths):
            raise ValueError("every ndepths value must be an integer of at least 2")
        if not isinstance(depth_interval_ratios, (tuple, list)):
            _invalid("depth_interval_ratios must be a tuple or list")
        if len(depth_interval_ratios) != len(ndepths):
            raise ValueError("depth_interval_ratios must match the length of ndepths")
        if any(
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not math.isfinite(ratio)
            or ratio <= 0
            for ratio in depth_interval_ratios
        ):
            raise ValueError("every depth_interval_ratios value must be finite and positive")
        if not isinstance(detach_previous, bool):
            _invalid("detach_previous must be a boolean")
        self.ndepths = tuple(ndepths)
        self.depth_interval_ratios = tuple(float(ratio) for ratio in depth_interval_ratios)
        self.detach_previous = detach_previous

    def forward(self, depth_values, stage_idx, spatial_size, previous_depth=None):
        if not isinstance(depth_values, torch.Tensor) or depth_values.ndim != 2:
            raise ValueError("depth_values must be a rank-2 tensor with shape [B,D0]")
        if depth_values.shape[0] < 1 or depth_values.shape[1] < 2:
            raise ValueError("depth_values must contain a batch and at least two depth samples")
        if not depth_values.dtype.is_floating_point:
            raise ValueError("depth_values must have a floating-point dtype")
        if not torch.isfinite(depth_values).all():
            raise ValueError("depth_values must contain only finite values")
        if torch.any(depth_values[:, 1:] <= depth_values[:, :-1]):
            raise ValueError("depth_values must be strictly ascending")
        if type(stage_idx) is not int or not 0 <= stage_idx < len(self.ndepths):
            raise ValueError("stage_idx must select a configured cascade stage")
        if not isinstance(spatial_size, (tuple, list)) or len(spatial_size) != 2:
            raise ValueError("spatial_size must contain positive height and width")
        if any(type(size) is not int or size < 1 for size in spatial_size):
            raise ValueError("spatial_size must contain positive integer height and width")

        batch_size, input_planes = depth_values.shape
        height, width = spatial_size
        output_planes = self.ndepths[stage_idx]
        minimum = depth_values[:, 0]
        maximum = depth_values[:, -1]
        plane_indices = torch.arange(output_planes, dtype=depth_values.dtype,
                                     device=depth_values.device)

        if stage_idx == 0:
            positions = plane_indices / (output_planes - 1)
            hypotheses = minimum[:, None] + (maximum - minimum)[:, None] * positions[None, :]
            return hypotheses[:, :, None, None].expand(batch_size, output_planes, height, width)

        if not isinstance(previous_depth, torch.Tensor) or previous_depth.ndim != 3:
            raise ValueError("previous_depth must be a rank-3 tensor for later stages")
        if previous_depth.shape != (batch_size, height, width):
            raise ValueError("previous_depth shape must match [B,H,W] from depth_values and spatial_size")
        if not previous_depth.dtype.is_floating_point:
            raise ValueError("previous_depth must have a floating-point dtype")
        if previous_depth.device != depth_values.device:
            raise ValueError("previous_depth and depth_values must be on the same device")
        if not torch.isfinite(previous_depth).all():
            raise ValueError("previous_depth must contain only finite values")

        center = previous_depth.detach() if self.detach_previous else previous_depth
        interval = (maximum - minimum) / input_planes
        step = interval * self.depth_interval_ratios[stage_idx]
        offset = plane_indices * (output_planes / (output_planes - 1))
        start = center[:, None] - step[:, None, None, None] * (output_planes / 2)
        return start + step[:, None, None, None] * offset[None, :, None, None]
