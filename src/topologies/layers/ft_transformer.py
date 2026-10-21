import torch
from torch import nn


class FTTransformerFeatureTokenizer(nn.Module):
    component_id = "ft-transformer"

    def __init__(self, n_features: int, d_token: int = 32) -> None:
        super().__init__()
        if n_features < 1 or d_token < 1:
            raise ValueError("n_features and d_token must be positive")
        self.n_features = n_features
        self.d_token = d_token
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 1 or x.shape[-1] != self.n_features:
            raise ValueError(f"x must end in {self.n_features} numeric features")
        return x.unsqueeze(-1) * self.weight + self.bias
