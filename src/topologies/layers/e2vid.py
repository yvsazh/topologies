import torch
from torch import nn


class E2VID(nn.Module):
    component_id = "e2vid"

    def __init__(self, height, width, bins=10):
        super().__init__()
        if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
            raise ValueError("height must be a positive integer")
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            raise ValueError("width must be a positive integer")
        if not isinstance(bins, int) or isinstance(bins, bool) or bins <= 0:
            raise ValueError("bins must be a positive integer")
        self.height = height
        self.width = width
        self.bins = bins

    def forward(self, events):
        if not isinstance(events, torch.Tensor):
            raise TypeError("events must be a floating point tensor")
        if not events.is_floating_point():
            raise ValueError("events must be a floating point tensor")
        if events.ndim == 2:
            if events.shape[-1] != 4:
                raise ValueError("events must have shape [N, 4] or [M, N, 4]")
            events = events.unsqueeze(0)
            unbatched = True
        elif events.ndim == 3 and events.shape[-1] == 4:
            unbatched = False
        else:
            raise ValueError("events must have shape [N, 4] or [M, N, 4]")
        if events.shape[0] == 0:
            raise ValueError("events batch must contain at least one window")
        if events.shape[1] < 2:
            raise ValueError("each window must contain at least two events")
        if not torch.isfinite(events).all().item():
            raise ValueError("events must contain only finite values")
        x = events[..., 0]
        y = events[..., 1]
        if not torch.equal(x, torch.floor(x)) or not torch.equal(y, torch.floor(y)):
            raise ValueError("events must use integer pixel coordinates")
        if (x < 0).any().item() or (x >= self.width).any().item():
            raise ValueError("event x coordinates must be within sensor bounds")
        if (y < 0).any().item() or (y >= self.height).any().item():
            raise ValueError("event y coordinates must be within sensor bounds")
        timestamps = events[..., 2]
        if (timestamps[..., 1:] < timestamps[..., :-1]).any().item():
            raise ValueError("event timestamps must be nondecreasing")
        duration = timestamps[..., -1:] - timestamps[..., :1]
        if (duration <= 0).any().item():
            raise ValueError("each event window must have positive duration")
        normalized_times = (self.bins - 1) * (timestamps - timestamps[..., :1]) / duration
        lower_bins = torch.floor(normalized_times).to(torch.long)
        upper_bins = (lower_bins + 1).clamp(max=self.bins - 1)
        upper_weights = normalized_times - lower_bins.to(normalized_times.dtype)
        lower_weights = 1.0 - upper_weights
        spatial_indices = y.to(torch.long) * self.width + x.to(torch.long)
        plane_size = self.height * self.width
        batch_offsets = (
            torch.arange(events.shape[0], device=events.device).unsqueeze(1)
            * (self.bins * plane_size)
        )
        lower_indices = batch_offsets + lower_bins * plane_size + spatial_indices
        upper_indices = batch_offsets + upper_bins * plane_size + spatial_indices
        polarities = events[..., 3]
        voxel_grid = events.new_zeros(events.shape[0] * self.bins * plane_size)
        voxel_grid = voxel_grid.scatter_add(
            0, lower_indices.reshape(-1), (polarities * lower_weights).reshape(-1)
        )
        voxel_grid = voxel_grid.scatter_add(
            0, upper_indices.reshape(-1), (polarities * upper_weights).reshape(-1)
        )
        voxel_grid = voxel_grid.reshape(events.shape[0], self.bins, self.height, self.width)
        if unbatched:
            return voxel_grid[0]
        return voxel_grid
