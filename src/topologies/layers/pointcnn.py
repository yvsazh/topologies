import torch
from torch import nn
from torch.nn import functional as F


class PointCNN(nn.Module):
    architecture_id = "pointcnn"
    component_id = "pointcnn"

    def __init__(self, input_channels=0, hidden_channels=64, num_classes=40, k=8, delta_channels=None):
        super().__init__()
        if input_channels < 0 or min(hidden_channels, num_classes, k) <= 0:
            raise ValueError("input channels must be nonnegative; other dimensions must be positive")
        self.input_channels = input_channels
        self.k = k
        self.delta_channels = hidden_channels if delta_channels is None else delta_channels
        if self.delta_channels <= 0:
            raise ValueError("delta_channels must be positive")
        self.coordinate_lift = nn.Sequential(
            nn.Linear(3, self.delta_channels),
            nn.ReLU(),
            nn.Linear(self.delta_channels, self.delta_channels),
        )
        self.transform_mlp = nn.Sequential(
            nn.Linear(k * 3, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, k * k),
        )
        self.xconv_weight = nn.Parameter(torch.empty(k, self.delta_channels + input_channels, hidden_channels))
        self.xconv_bias = nn.Parameter(torch.zeros(hidden_channels))
        nn.init.kaiming_uniform_(self.xconv_weight, a=5 ** 0.5)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def encode(self, coordinates, features=None):
        if coordinates.ndim != 3 or coordinates.shape[1] == 0 or coordinates.shape[2] != 3:
            raise ValueError("coordinates must have shape (batch, non-empty points, 3)")
        if features is None:
            features = coordinates.new_zeros((*coordinates.shape[:2], self.input_channels))
        if (features.ndim != 3 or features.shape[:2] != coordinates.shape[:2]
                or features.shape[2] != self.input_channels):
            raise ValueError("features must match coordinates and input_channels")
        count = coordinates.shape[1]
        distances = torch.cdist(coordinates, coordinates)
        indices = distances.topk(min(self.k, count), dim=-1, largest=False).indices
        if indices.shape[-1] < self.k:
            indices = torch.cat((indices, indices[..., -1:].expand(-1, -1, self.k - indices.shape[-1])), dim=-1)
        neighbor_coordinates = coordinates.unsqueeze(1).expand(-1, count, -1, -1).gather(
            2, indices.unsqueeze(-1).expand(-1, -1, -1, 3)
        )
        neighbor_features = features.unsqueeze(1).expand(-1, count, -1, -1).gather(
            2, indices.unsqueeze(-1).expand(-1, -1, -1, self.input_channels)
        )
        relative_coordinates = neighbor_coordinates - coordinates.unsqueeze(2)
        lifted_coordinates = self.coordinate_lift(relative_coordinates)
        lifted_features = torch.cat((lifted_coordinates, neighbor_features), dim=-1)
        transform = self.transform_mlp(relative_coordinates.flatten(start_dim=2))
        transform = transform.reshape(coordinates.shape[0], count, self.k, self.k)
        transformed = torch.matmul(transform, lifted_features)
        return F.relu(torch.einsum("bnkc,kco->bno", transformed, self.xconv_weight) + self.xconv_bias)

    def forward(self, coordinates, features=None):
        return self.classifier(self.encode(coordinates, features).max(dim=1).values)
