import torch
from torch import nn


class PointTransformer(nn.Module):
    architecture_id = "point-transformer"
    def __init__(self, input_dim=3, channels=64, num_classes=2, k=16, **kwargs):
        super().__init__()
        if input_dim < 1 or channels < 1 or num_classes < 1 or k < 1:
            raise ValueError("input_dim, channels, num_classes, and k must be positive")
        self.k = k
        self.input_projection = nn.Linear(input_dim, channels)
        self.query = nn.Linear(channels, channels, bias=False)
        self.key = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.position_encoding = nn.Sequential(
            nn.Linear(input_dim, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
        )
        self.attention = nn.Linear(channels, channels)
        self.classifier = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(), nn.Linear(channels, num_classes))

    def forward(self, points):
        if points.ndim != 3:
            raise ValueError("points must have shape (batch, points, coordinates)")
        batch, count, _ = points.shape
        if count < 1:
            raise ValueError("point clouds must contain at least one point")
        features = self.input_projection(points)
        distances = torch.cdist(points, points)
        neighbor_count = min(self.k, count)
        indices = distances.topk(neighbor_count, dim=-1, largest=False).indices
        batch_indices = torch.arange(batch, device=points.device).view(batch, 1, 1)
        neighbors = features[batch_indices, indices]
        neighbor_positions = points[batch_indices, indices]
        relative_positions = points.unsqueeze(2) - neighbor_positions
        position = self.position_encoding(relative_positions)
        query = self.query(features).unsqueeze(2)
        differences = query - self.key(neighbors) + position
        scores = self.attention(torch.tanh(differences))
        weights = torch.softmax(scores, dim=2)
        values = self.value(neighbors) + position
        aggregated = (weights * values).sum(dim=2)
        return self.classifier(aggregated.mean(dim=1))
