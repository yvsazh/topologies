import torch
from torch import nn


class PointTransformerV2(nn.Module):
    architecture_id = "point-transformer-v2"
    def __init__(self, input_dim=3, channels=64, groups=4, k=16, **kwargs):
        super().__init__()
        if input_dim < 1 or channels < 1 or groups < 1 or k < 1:
            raise ValueError("input_dim, channels, groups, and k must be positive")
        if channels % groups:
            raise ValueError("channels must be divisible by groups")
        self.channels = channels
        self.groups = groups
        self.group_channels = channels // groups
        self.k = k
        self.input_projection = nn.Linear(input_dim, channels)
        self.query = nn.Linear(channels, channels, bias=False)
        self.key = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.position_bias = nn.Sequential(
            nn.Linear(input_dim, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
        )
        self.position_multiplier = nn.Sequential(
            nn.Linear(input_dim, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
        )
        self.group_linear = nn.ModuleList(nn.Linear(self.group_channels, 1) for _ in range(groups))
        self.group_norm = nn.LayerNorm(groups)
        self.group_activation = nn.ReLU()
        self.group_output = nn.Linear(groups, groups)
        self.output_projection = nn.Linear(channels, channels)

    def forward(self, points):
        if points.ndim != 3:
            raise ValueError("points must have shape (batch, points, coordinates)")
        batch, count, _ = points.shape
        if count < 1:
            raise ValueError("point clouds must contain at least one point")
        features = self.input_projection(points)
        distances = torch.cdist(points, points)
        indices = distances.topk(min(self.k, count), dim=-1, largest=False).indices
        batch_indices = torch.arange(batch, device=points.device).view(batch, 1, 1)
        neighbors = features[batch_indices, indices]
        neighbor_positions = points[batch_indices, indices]
        relative_positions = points.unsqueeze(2) - neighbor_positions
        position_bias = self.position_bias(relative_positions)
        position_multiplier = self.position_multiplier(relative_positions)
        query = self.query(features).unsqueeze(2)
        difference = position_multiplier * (query - self.key(neighbors)) + position_bias
        value = self.value(neighbors)
        query_groups = difference.split(self.group_channels, dim=-1)
        value_groups = value.split(self.group_channels, dim=-1)
        grouped_scores = torch.stack(
            [layer(part).squeeze(-1) for layer, part in zip(self.group_linear, query_groups)],
            dim=-1,
        )
        grouped_scores = self.group_output(self.group_activation(self.group_norm(grouped_scores)))
        weights = torch.softmax(grouped_scores, dim=2)
        outputs = [
            (weights[..., group].unsqueeze(-1) * value_groups[group]).sum(dim=2)
            for group in range(self.groups)
        ]
        return self.output_projection(torch.cat(outputs, dim=-1))
