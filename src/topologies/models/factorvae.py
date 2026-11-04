import math

import torch
from torch import nn
from torch.nn import functional as F


class FactorVAE(nn.Module):
    architecture_id = "factorvae"
    PRESETS = {
        "tiny": {
            "input_dim": 8,
            "latent_dim": 2,
            "hidden_dim": 16,
            "discriminator_hidden_dim": 32,
            "discriminator_layers": 2,
            "gamma": 35.0,
        },
        "small": {
            "input_dim": 32,
            "latent_dim": 10,
            "hidden_dim": 64,
            "discriminator_hidden_dim": 128,
            "discriminator_layers": 3,
            "gamma": 35.0,
        },
    }

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 10,
        hidden_dim: int = 64,
        discriminator_hidden_dim: int = 64,
        discriminator_layers: int = 2,
        gamma: float | None = None,
    ):
        super().__init__()
        dimensions = {
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "discriminator_hidden_dim": discriminator_hidden_dim,
            "discriminator_layers": discriminator_layers,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if gamma is None or isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
            raise ValueError("gamma must be a finite nonnegative number")
        if not math.isfinite(float(gamma)) or gamma < 0:
            raise ValueError("gamma must be a finite nonnegative number")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.gamma = float(gamma)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, latent_dim)
        self.log_variance_head = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        discriminator = []
        width = latent_dim
        for _ in range(discriminator_layers):
            discriminator.extend(
                [nn.Linear(width, discriminator_hidden_dim), nn.LeakyReLU(0.2)]
            )
            width = discriminator_hidden_dim
        discriminator.append(nn.Linear(width, 2))
        self.discriminator = nn.Sequential(*discriminator)

    def _flatten_observations(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or x.ndim < 2:
            raise ValueError("input must have a batch dimension and observation dimensions")
        if x.shape[0] < 1:
            raise ValueError("input batch must be nonempty")
        if not x.is_floating_point():
            raise ValueError("input observations must be floating point")
        flattened = x.reshape(x.shape[0], -1)
        if flattened.shape[1] != self.input_dim:
            raise ValueError("input feature size must equal input_dim")
        if not torch.isfinite(flattened).all():
            raise ValueError("input observations must be finite")
        if torch.any((flattened < 0) | (flattened > 1)):
            raise ValueError("Bernoulli targets must be in [0, 1]")
        return flattened

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(self._flatten_observations(x))
        return self.mean_head(hidden), self.log_variance_head(hidden)

    @staticmethod
    def reparameterize(
        mean: torch.Tensor, log_variance: torch.Tensor
    ) -> torch.Tensor:
        if mean.shape != log_variance.shape:
            raise ValueError("mean and log_variance must have the same shape")
        return mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError("latent must have shape [batch, latent_dim]")
        return self.decoder(latent)

    @staticmethod
    def permute_dims(latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[0] < 1 or latent.shape[1] < 1:
            raise ValueError("latent must have shape [batch, latent_dim] with positive dimensions")
        columns = [
            latent[torch.randperm(latent.shape[0], device=latent.device), column]
            for column in range(latent.shape[1])
        ]
        return torch.stack(columns, dim=1)

    @staticmethod
    def estimate_total_correlation(
        joint_logits: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        if joint_logits.ndim != 2 or joint_logits.shape[0] < 1 or joint_logits.shape[1] != 2:
            raise ValueError("joint_logits must have shape [batch, 2]")
        if reduction not in {"none", "sum", "mean"}:
            raise ValueError("reduction must be 'none', 'sum', or 'mean'")
        log_odds = joint_logits[:, 0] - joint_logits[:, 1]
        if reduction == "none":
            return log_odds
        if reduction == "sum":
            return log_odds.sum()
        return log_odds.mean()

    @staticmethod
    def discriminator_loss(
        joint_logits: torch.Tensor, product_logits: torch.Tensor
    ) -> torch.Tensor:
        if (
            joint_logits.ndim != 2
            or joint_logits.shape[0] < 1
            or joint_logits.shape[1] != 2
            or product_logits.shape != joint_logits.shape
        ):
            raise ValueError("joint_logits and product_logits must both have shape [batch, 2]")
        logits = torch.cat((joint_logits, product_logits), dim=0)
        labels = torch.cat(
            (
                torch.zeros(joint_logits.shape[0], dtype=torch.long, device=joint_logits.device),
                torch.ones(product_logits.shape[0], dtype=torch.long, device=product_logits.device),
            )
        )
        return F.cross_entropy(logits, labels)

    @staticmethod
    def negative_elbo(
        observations: torch.Tensor,
        decoder_logits: torch.Tensor,
        mean: torch.Tensor,
        log_variance: torch.Tensor,
        tc_estimate: torch.Tensor | float,
        gamma: float,
        reduction: str = "mean",
    ) -> torch.Tensor:
        if not isinstance(observations, torch.Tensor) or observations.ndim < 2:
            raise ValueError("observations must have a batch dimension and observation dimensions")
        if observations.shape != decoder_logits.shape:
            raise ValueError("observations and decoder_logits must have the same shape")
        if mean.ndim != 2 or mean.shape != log_variance.shape:
            raise ValueError("mean and log_variance must have the same shape [batch, latent_dim]")
        batch_size = observations.shape[0]
        if mean.shape[0] != batch_size:
            raise ValueError("observation and posterior batch dimensions must match")
        if isinstance(gamma, bool) or not isinstance(gamma, (int, float)):
            raise TypeError("gamma must be a finite nonnegative number")
        if not math.isfinite(float(gamma)) or gamma < 0:
            raise ValueError("gamma must be a finite nonnegative number")
        if reduction not in {"none", "sum", "mean"}:
            raise ValueError("reduction must be 'none', 'sum', or 'mean'")
        flattened_observations = observations.reshape(batch_size, -1)
        flattened_logits = decoder_logits.reshape(batch_size, -1)
        if not observations.is_floating_point() or not torch.isfinite(flattened_observations).all():
            raise ValueError("Bernoulli targets must be finite floating point values in [0, 1]")
        if torch.any((flattened_observations < 0) | (flattened_observations > 1)):
            raise ValueError("Bernoulli targets must be in [0, 1]")
        reconstruction = F.binary_cross_entropy_with_logits(
            flattened_logits, flattened_observations, reduction="none"
        ).sum(dim=1)
        conditional_kl = 0.5 * (
            mean.square() + log_variance.exp() - 1 - log_variance
        ).sum(dim=1)
        if isinstance(tc_estimate, torch.Tensor):
            tc = tc_estimate.reshape(-1)
            if tc.numel() == 1:
                tc = tc.expand(batch_size)
            if tc.shape[0] != batch_size:
                raise ValueError("tc_estimate must be scalar or have one value per observation")
        else:
            tc = mean.new_full((batch_size,), float(tc_estimate))
        per_example = reconstruction + conditional_kl + float(gamma) * tc
        if reduction == "none":
            return per_example
        if reduction == "sum":
            return per_example.sum()
        return per_example.mean()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mean, log_variance = self.encode(x)
        latent = self.reparameterize(mean, log_variance)
        decoder_logits = self.decode(latent).reshape_as(x)
        permuted_latent = self.permute_dims(latent)
        return {
            "reconstruction": torch.sigmoid(decoder_logits),
            "decoder_logits": decoder_logits,
            "mean": mean,
            "log_variance": log_variance,
            "latent": latent,
            "joint_logits": self.discriminator(latent),
            "permuted_latent": permuted_latent,
            "permuted_logits": self.discriminator(permuted_latent),
        }
