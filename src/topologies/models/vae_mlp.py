

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F


class VAEMLP(nn.Module):


    architecture_id = "vae-mlp"

    def __init__(self, dim: int = 32, latent: int = 8, hidden: int = 64) -> None:
        super().__init__()
        if min(dim, latent, hidden) <= 0:
            raise ValueError("dim, latent, and hidden must be positive")
        self.e1 = nn.Linear(dim, hidden)
        self.e2 = nn.Linear(hidden, hidden)
        self.mu = nn.Linear(hidden, latent)
        self.lv = nn.Linear(hidden, latent)
        self.d1 = nn.Linear(latent, hidden)
        self.d2 = nn.Linear(hidden, dim)

    def encode(self, x: torch.Tensor):
        h = torch.relu(self.e2(torch.relu(self.e1(x))))
        return self.mu(h), self.lv(h)

    def reparam(self, mu: torch.Tensor, lv: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * (0.5 * lv).exp()

    def forward(self, x: torch.Tensor):
        mu, lv = self.encode(x)
        z = self.reparam(mu, lv)
        return torch.sigmoid(self.d2(torch.relu(self.d1(z)))), mu, lv

    def loss(self, x: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        reconstruction, mean, log_variance = self(x)
        return self.negative_elbo(x, reconstruction, mean, log_variance, reduction)

    @staticmethod
    def negative_elbo(
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        mean: torch.Tensor,
        log_variance: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        if x.shape != reconstruction.shape or mean.shape != log_variance.shape:
            raise ValueError("input/reconstruction and posterior parameter shapes must match")
        if x.shape[:-1] != mean.shape[:-1]:
            raise ValueError("input and posterior batch dimensions must match")
        if torch.any((x < 0) | (x > 1)):
            raise ValueError("Bernoulli targets must be in [0, 1]")
        if reduction not in {"none", "sum", "mean"}:
            raise ValueError("reduction must be 'none', 'sum', or 'mean'")
        reconstruction_nll = F.binary_cross_entropy(
            reconstruction, x, reduction="none"
        ).sum(dim=-1)
        kl = -0.5 * (1 + log_variance - mean.square() - log_variance.exp()).sum(dim=-1)
        per_example = reconstruction_nll + kl
        if reduction == "sum":
            return per_example.sum()
        if reduction == "mean":
            return per_example.mean()
        return per_example
