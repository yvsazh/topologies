import torch
from torch import nn
from torch.nn import functional as F


class WMSE(nn.Module):
    architecture_id = "w-mse"
    PRESETS = {
        "tiny": {"input_dim": 512, "hidden_dim": 1024, "embedding_dim": 64},
        "small": {"input_dim": 2048, "hidden_dim": 1024, "embedding_dim": 128},
        "base": {"input_dim": 2048, "hidden_dim": 2048, "embedding_dim": 128},
    }

    def __init__(self, input_dim=512, hidden_dim=1024, embedding_dim=64):
        super().__init__()
        dimensions = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "embedding_dim": embedding_dim,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def project_features(self, features):
        if not isinstance(features, torch.Tensor) or features.ndim != 3:
            raise ValueError("features must be a tensor with shape [batch, views, input_dim]")
        if features.shape[0] < 1 or features.shape[1] < 2:
            raise ValueError("features must contain at least one item and two views per item")
        if features.shape[2] != self.input_dim:
            raise ValueError(f"features must have input_dim={self.input_dim} in the last dimension")
        projected = self.projector(features.flatten(0, 1))
        return projected.reshape(features.shape[0], features.shape[1], self.embedding_dim)

    def whiten(self, projected_features):
        if not isinstance(projected_features, torch.Tensor) or projected_features.ndim != 3:
            raise ValueError("projected_features must be a tensor with shape [batch, views, embedding_dim]")
        if projected_features.shape[0] < 1 or projected_features.shape[1] < 2:
            raise ValueError("projected_features must contain at least one item and two views per item")
        if projected_features.shape[2] != self.embedding_dim:
            raise ValueError(
                f"projected_features must have embedding_dim={self.embedding_dim} in the last dimension"
            )
        flattened = projected_features.flatten(0, 1)
        centered = flattened - flattened.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / (flattened.shape[0] - 1)
        try:
            lower = torch.linalg.cholesky(covariance)
        except RuntimeError as error:
            raise ValueError("projected feature covariance must be positive definite") from error
        whitened = torch.linalg.solve_triangular(lower, centered.T, upper=False).T
        return whitened.reshape_as(projected_features)

    def loss_from_projected_features(self, projected_features):
        whitened = self.whiten(projected_features)
        normalized = F.normalize(whitened, p=2, dim=-1)
        views = normalized.shape[1]
        pair_losses = [
            (normalized[:, first] - normalized[:, second]).square().sum(dim=-1).mean()
            for first in range(views)
            for second in range(first + 1, views)
        ]
        return torch.stack(pair_losses).mean()

    def forward(self, features):
        return self.loss_from_projected_features(self.project_features(features))
