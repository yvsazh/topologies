import math

import torch
from torch import nn


class PerformerAttention(nn.Module):
    component_id = "performer"
    def __init__(self, embed_dim=4, num_features=4):
        super().__init__()
        if embed_dim < 1 or num_features < 1:
            raise ValueError("embed_dim and num_features must be positive")
        self.embed_dim = embed_dim
        self.num_features = num_features
        self.head_dim = embed_dim
        projection = self._orthogonal_projection(self.head_dim, num_features)
        self.register_buffer("projection", projection)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    @staticmethod
    def _orthogonal_projection(head_dim, num_features):
        rows = []
        remaining = num_features
        while remaining:
            gaussian = torch.randn(head_dim, head_dim)
            orthogonal, _ = torch.linalg.qr(gaussian)
            block_size = min(head_dim, remaining)
            radii = torch.linalg.vector_norm(torch.randn(block_size, head_dim), dim=-1)
            rows.append(orthogonal.T[:block_size] * radii.unsqueeze(-1))
            remaining -= block_size
        return torch.cat(rows, dim=0).T

    def _features(self, x):
        projected = x @ self.projection
        norm = x.square().sum(dim=-1, keepdim=True) / 2
        return torch.exp(projected - norm) / math.sqrt(self.num_features)

    def forward(self, x):
        q = self._features(self.query(x))
        k = self._features(self.key(x))
        v = self.value(x)
        context = k.transpose(-2, -1) @ v
        normalizer = (q @ k.sum(dim=-2, keepdim=True).transpose(-2, -1)).clamp_min(1e-8)
        return (q @ context) / normalizer
