import torch
from torch import nn


class SIGN(nn.Module):
    architecture_id = "sign"
    def __init__(self, in_features, hidden_features, out_features, num_hops=2):
        super().__init__()
        if min(in_features, hidden_features, out_features) <= 0 or num_hops < 0:
            raise ValueError("feature sizes must be positive and num_hops non-negative")
        self.in_features = in_features
        self.num_hops = num_hops
        self.hop_mlps = nn.ModuleList(
            nn.Sequential(nn.Linear(in_features, hidden_features), nn.ReLU())
            for _ in range(num_hops + 1)
        )
        self.classifier = nn.Linear(hidden_features * (num_hops + 1), out_features)

    def forward(self, features, normalized_adjacency):
        if features.ndim != 2 or features.shape[-1] != self.in_features:
            raise ValueError("features must have shape (nodes, in_features)")
        if normalized_adjacency.ndim != 2 or normalized_adjacency.shape != (features.shape[0], features.shape[0]):
            raise ValueError("normalized_adjacency must have shape (nodes, nodes)")
        powers = [features]
        for _ in range(self.num_hops):
            powers.append(normalized_adjacency @ powers[-1])
        scale_features = [mlp(power) for mlp, power in zip(self.hop_mlps, powers)]
        return self.classifier(torch.cat(scale_features, dim=-1))
