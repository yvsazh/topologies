import torch
from torch import nn
from torch.nn import functional as F


class CREStereoAGCL(nn.Module):
    component_id = "crestereo"

    def __init__(self, search_mode="1d"):
        super().__init__()
        if not isinstance(search_mode, str) or search_mode not in ("1d", "2d"):
            raise ValueError("search_mode must be '1d' or '2d'")
        self.search_mode = search_mode

    def forward(self, left_feature, right_feature, flow, adaptive_offsets=None):
        if not isinstance(left_feature, torch.Tensor) or not isinstance(right_feature, torch.Tensor):
            raise TypeError("left_feature and right_feature must be tensors")
        if left_feature.ndim != 4 or right_feature.ndim != 4:
            raise ValueError("left_feature and right_feature must have shape [B, C, H, W]")
        if left_feature.shape != right_feature.shape:
            raise ValueError("left_feature and right_feature must have matching shapes")
        batch, channels, height, width = left_feature.shape
        if channels == 0 or channels % 4 != 0:
            raise ValueError("feature channels must be positive and divisible by 4")
        if height == 0 or width == 0:
            raise ValueError("feature height and width must be positive")
        if not left_feature.is_floating_point() or not right_feature.is_floating_point():
            raise ValueError("feature maps must use a floating point dtype")
        if left_feature.dtype != right_feature.dtype or left_feature.device != right_feature.device:
            raise ValueError("feature maps must have matching dtype and device")
        if not isinstance(flow, torch.Tensor) or flow.shape != (batch, 2, height, width):
            raise ValueError("flow must have shape [B, 2, H, W] matching the feature maps")
        if not flow.is_floating_point() or flow.dtype != left_feature.dtype or flow.device != left_feature.device:
            raise ValueError("flow must match the feature maps' floating point dtype and device")
        if adaptive_offsets is not None:
            if not isinstance(adaptive_offsets, torch.Tensor) or adaptive_offsets.shape != (
                batch,
                18,
                height,
                width,
            ):
                raise ValueError("adaptive_offsets must have shape [B, 18, H, W] matching the feature maps")
            if (
                not adaptive_offsets.is_floating_point()
                or adaptive_offsets.dtype != left_feature.dtype
                or adaptive_offsets.device != left_feature.device
            ):
                raise ValueError("adaptive_offsets must match the feature maps' floating point dtype and device")

        if self.search_mode == "1d":
            horizontal = torch.arange(-4, 5, dtype=left_feature.dtype, device=left_feature.device)
            candidates = torch.stack((horizontal, torch.zeros_like(horizontal)), dim=-1)
        else:
            vertical = torch.arange(-1, 2, dtype=left_feature.dtype, device=left_feature.device)
            horizontal = torch.arange(-1, 2, dtype=left_feature.dtype, device=left_feature.device)
            y_offsets, x_offsets = torch.meshgrid(vertical, horizontal, indexing="ij")
            candidates = torch.stack((x_offsets.reshape(-1), y_offsets.reshape(-1)), dim=-1)

        if adaptive_offsets is None:
            adaptive_offsets = left_feature.new_zeros((batch, 18, height, width))
        per_candidate_offsets = adaptive_offsets.reshape(batch, 9, 2, height, width).permute(0, 1, 3, 4, 2)
        base_grid_y, base_grid_x = torch.meshgrid(
            torch.arange(height, dtype=left_feature.dtype, device=left_feature.device),
            torch.arange(width, dtype=left_feature.dtype, device=left_feature.device),
            indexing="ij",
        )
        base_grid = torch.stack((base_grid_x, base_grid_y), dim=-1)
        flow_grid = flow.permute(0, 2, 3, 1)
        sample_grid = base_grid.unsqueeze(0).unsqueeze(0) + flow_grid.unsqueeze(1)
        sample_grid = sample_grid + candidates.reshape(1, 9, 1, 1, 2) + per_candidate_offsets
        normalized_x = (sample_grid[..., 0] + 0.5) * (2.0 / width) - 1.0
        normalized_y = (sample_grid[..., 1] + 0.5) * (2.0 / height) - 1.0
        normalized_grid = torch.stack((normalized_x, normalized_y), dim=-1).reshape(
            batch * 9, height, width, 2
        )
        expanded_right = right_feature.unsqueeze(1).expand(-1, 9, -1, -1, -1).reshape(
            batch * 9, channels, height, width
        )
        sampled_right = F.grid_sample(
            expanded_right,
            normalized_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).reshape(batch, 9, 4, channels // 4, height, width)
        grouped_left = left_feature.reshape(batch, 4, channels // 4, height, width).unsqueeze(1)
        correlations = (grouped_left * sampled_right).mean(dim=3).permute(0, 2, 1, 3, 4)
        return correlations.reshape(batch, 36, height, width)
