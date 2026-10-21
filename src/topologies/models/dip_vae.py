import torch
from torch import nn


class DIPVAE(nn.Module):
    architecture_id = "dip-vae"
    def __init__(self, input_dim=8, latent_dim=4, hidden_dim=8, lambda_diag=10.0, lambda_offdiag=5.0):
        super().__init__()
        if min(input_dim, latent_dim, hidden_dim) <= 0:
            raise ValueError("dimensions must be positive")
        if lambda_diag < 0 or lambda_offdiag < 0:
            raise ValueError("penalty weights must be nonnegative")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lambda_diag = float(lambda_diag)
        self.lambda_offdiag = float(lambda_offdiag)
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_variance = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))

    def encode(self, inputs):
        hidden = self.encoder(inputs)
        return self.mean(hidden), self.log_variance(hidden).clamp(-20.0, 20.0)

    def aggregated_covariance(self, mean, log_variance):
        if mean.ndim != 2 or log_variance.shape != mean.shape:
            raise ValueError("mean and log_variance must have shape (batch, latent_dim)")
        if mean.shape[0] < 2:
            raise ValueError("aggregated covariance requires at least two examples")
        centered = mean - mean.mean(dim=0, keepdim=True)
        mean_covariance = centered.transpose(0, 1) @ centered / (mean.shape[0] - 1)
        posterior_variance = log_variance.exp().mean(dim=0)
        return mean_covariance + torch.diag(posterior_variance)

    def covariance_penalty(self, mean, log_variance):
        covariance = self.aggregated_covariance(mean, log_variance)
        diagonal = torch.diagonal(covariance) - 1.0
        off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
        return self.lambda_diag * diagonal.square().sum() + self.lambda_offdiag * off_diagonal.square().sum()

    def forward(self, inputs, sample=None):
        if inputs.ndim != 2 or inputs.shape[-1] != self.input_dim:
            raise ValueError("inputs must have shape (batch, input_dim)")
        if inputs.shape[0] < 2:
            raise ValueError("DIP-VAE covariance requires at least two examples")
        mean, log_variance = self.encode(inputs)
        if sample is None:
            sample = self.training
        latent = mean + torch.randn_like(mean) * (0.5 * log_variance).exp() if sample else mean
        reconstruction = self.decoder(latent)
        reconstruction_loss = 0.5 * (reconstruction - inputs).square().sum(dim=-1).mean()
        kl_loss = 0.5 * (
            mean.square() + log_variance.exp() - 1.0 - log_variance
        ).sum(dim=-1).mean()
        penalty = self.covariance_penalty(mean, log_variance)
        return {
            "reconstruction": reconstruction,
            "mean": mean,
            "log_variance": log_variance,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "covariance_penalty": penalty,
            "loss": reconstruction_loss + kl_loss + penalty,
        }
