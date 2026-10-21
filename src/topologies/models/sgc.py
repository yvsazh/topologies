import torch
from torch import nn


class SGC(nn.Module):
    architecture_id = "sgc"
    def __init__(self, in_features, out_features, K=2, bias=True):
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("feature dimensions must be positive")
        if not isinstance(K, int) or K < 0:
            raise ValueError("K must be a nonnegative integer")
        self.K = K
        self.classifier = nn.Linear(in_features, out_features, bias=bias)

    @staticmethod
    def propagate(features, adjacency, steps=2):
        features = torch.as_tensor(features)
        adjacency = torch.as_tensor(adjacency, dtype=features.dtype, device=features.device)
        if features.ndim != 2 or adjacency.ndim != 2:
            raise ValueError("features and adjacency must be matrices")
        if adjacency.shape[0] != adjacency.shape[1] or adjacency.shape[0] != features.shape[0]:
            raise ValueError("adjacency must be square and match the node count")
        if not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        augmented = adjacency + torch.eye(adjacency.shape[0], dtype=adjacency.dtype, device=adjacency.device)
        degree = augmented.sum(dim=1)
        if torch.any(degree <= 0):
            raise ValueError("renormalized adjacency has nonpositive degree")
        inv_sqrt_degree = degree.rsqrt()
        normalized = inv_sqrt_degree.unsqueeze(1) * augmented * inv_sqrt_degree.unsqueeze(0)
        output = features
        for _ in range(steps):
            output = normalized @ output
        return output

    def forward(self, features, adjacency):
        propagated = self.propagate(features, adjacency, self.K)
        return self.classifier(propagated)
