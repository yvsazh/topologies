import torch
from torch import nn


class PointPillars(nn.Module):
    architecture_id = "pointpillars"

    def __init__(
        self, pillar_size=1.0, hidden_channels=64, input_channels=4, x_range=None, y_range=None
    ):
        super().__init__()
        if pillar_size <= 0 or hidden_channels <= 0 or input_channels < 3:
            raise ValueError("pillar_size and hidden_channels must be positive; input_channels must be at least 3")
        for bounds in (x_range, y_range):
            if bounds is not None and (len(bounds) != 2 or bounds[1] <= bounds[0]):
                raise ValueError("coordinate ranges must be increasing (minimum, maximum) pairs")
        self.pillar_size = float(pillar_size)
        self.hidden_channels = int(hidden_channels)
        self.input_channels = int(input_channels)
        self.x_range = x_range
        self.y_range = y_range
        self.point_mlp = nn.Sequential(
            nn.Conv1d(self.input_channels + 5, self.hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.hidden_channels),
            nn.ReLU(),
        )

    def forward(self, points):
        if points.ndim != 2 or points.shape[1] != self.input_channels:
            raise ValueError("points must have shape (num_points, input_channels)")
        if points.shape[0] == 0:
            raise ValueError("points must contain at least one point")
        xy = points[:, :2]
        if self.x_range is not None:
            valid_x = (xy[:, 0] >= self.x_range[0]) & (xy[:, 0] < self.x_range[1])
        else:
            valid_x = torch.ones(points.shape[0], dtype=torch.bool, device=points.device)
        if self.y_range is not None:
            valid_y = (xy[:, 1] >= self.y_range[0]) & (xy[:, 1] < self.y_range[1])
        else:
            valid_y = torch.ones(points.shape[0], dtype=torch.bool, device=points.device)
        valid = valid_x & valid_y
        if not bool(valid.any()):
            raise ValueError("at least one point must fall within the configured ranges")
        selected = points[valid]
        origin_x = self.x_range[0] if self.x_range is not None else 0.0
        origin_y = self.y_range[0] if self.y_range is not None else 0.0
        origin = selected.new_tensor([origin_x, origin_y])
        cells = torch.floor((selected[:, :2] - origin) / self.pillar_size).to(torch.long)
        unique_cells, inverse = torch.unique(cells, dim=0, sorted=True, return_inverse=True)
        xyz = selected[:, :3]
        xyz_sum = xyz.new_zeros((unique_cells.shape[0], 3)).index_add(0, inverse, xyz)
        counts = torch.bincount(inverse, minlength=unique_cells.shape[0]).to(xyz.dtype).unsqueeze(-1)
        cluster_offsets = xyz - (xyz_sum / counts).index_select(0, inverse)
        center_xy = origin + (cells.to(selected.dtype) + 0.5) * self.pillar_size
        center_offsets = selected[:, :2] - center_xy
        decorated = torch.cat((selected, cluster_offsets, center_offsets), dim=-1)
        features = self.point_mlp(decorated.transpose(0, 1).unsqueeze(0)).squeeze(0).transpose(0, 1)
        pooled = features.new_full((unique_cells.shape[0], self.hidden_channels), -torch.inf)
        pooled = pooled.scatter_reduce(0, inverse[:, None].expand_as(features), features, reduce="amax", include_self=True)
        if self.x_range is not None:
            width = int(torch.ceil(torch.tensor((self.x_range[1] - self.x_range[0]) / self.pillar_size)).item())
        else:
            min_x = unique_cells[:, 0].min()
            width = int((unique_cells[:, 0].max() - min_x).item()) + 1
        if self.y_range is not None:
            height = int(torch.ceil(torch.tensor((self.y_range[1] - self.y_range[0]) / self.pillar_size)).item())
        else:
            min_y = unique_cells[:, 1].min()
            height = int((unique_cells[:, 1].max() - min_y).item()) + 1
        scatter_cells = unique_cells.clone()
        if self.x_range is None:
            scatter_cells[:, 0] -= unique_cells[:, 0].min()
        if self.y_range is None:
            scatter_cells[:, 1] -= unique_cells[:, 1].min()
        bev = pooled.new_zeros((self.hidden_channels, height, width))
        bev[:, scatter_cells[:, 1], scatter_cells[:, 0]] = pooled.transpose(0, 1)
        return {"pillar_features": pooled, "pillar_indices": unique_cells, "bev_map": bev}
