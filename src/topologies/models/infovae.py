import torch
from torch import nn
from torch.nn import functional as F


class InfoVAE(nn.Module):
    architecture_id = "infovae"
    def __init__(self, input_dim=8, latent_dim=4, hidden_dim=16, mmd_weight=10.0):
        super().__init__()
        if input_dim <= 0 or latent_dim <= 0 or hidden_dim <= 0:
            raise ValueError("dimensions must be positive")
        if mmd_weight < 0:
            raise ValueError("mmd_weight must be nonnegative")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.mmd_weight = mmd_weight
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_variance = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))
        self.last_reconstruction_loss = None
        self.last_mmd_loss = None

    @staticmethod
    def mmd(posterior, prior):
        if posterior.ndim != 2 or prior.ndim != 2:
            raise ValueError("samples must have shape (batch, latent_dim)")
        if posterior.shape[0] == 0 or prior.shape[0] == 0:
            raise ValueError("posterior and prior must be non-empty")
        if posterior.shape[1] != prior.shape[1]:
            raise ValueError("posterior and prior must have matching latent dimensions")
        scale = posterior.shape[-1]
        xx = torch.cdist(posterior, posterior).square() / scale
        yy = torch.cdist(prior, prior).square() / scale
        xy = torch.cdist(posterior, prior).square() / scale
        kernels_xx = torch.exp(-0.5 * xx) + torch.exp(-xx) + torch.exp(-2.0 * xx)
        kernels_yy = torch.exp(-0.5 * yy) + torch.exp(-yy) + torch.exp(-2.0 * yy)
        kernels_xy = torch.exp(-0.5 * xy) + torch.exp(-xy) + torch.exp(-2.0 * xy)
        return kernels_xx.mean() + kernels_yy.mean() - 2.0 * kernels_xy.mean()

    def forward(self, inputs, prior_samples=None):
        if inputs.ndim < 2 or inputs.shape[-1] != self.input_dim:
            raise ValueError("inputs must end with input_dim")
        original_shape = inputs.shape
        flattened = inputs.reshape(-1, self.input_dim)
        hidden = self.encoder(flattened)
        mean = self.mean(hidden)
        log_variance = self.log_variance(hidden).clamp(-20.0, 20.0)
        posterior = mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)
        reconstruction = self.decoder(posterior).reshape(original_shape)
        if prior_samples is None:
            prior_samples = torch.randn_like(posterior)
        else:
            prior_samples = torch.as_tensor(
                prior_samples, device=posterior.device, dtype=posterior.dtype
            )
            if prior_samples.ndim < 2 or prior_samples.shape[-1] != self.latent_dim:
                raise ValueError("prior_samples must end with latent_dim")
            prior_samples = prior_samples.reshape(-1, self.latent_dim)
        reconstruction_loss = F.mse_loss(reconstruction, inputs)
        mmd_loss = self.mmd(posterior, prior_samples)
        self.last_reconstruction_loss = reconstruction_loss
        self.last_mmd_loss = mmd_loss
        self.loss = reconstruction_loss + self.mmd_weight * mmd_loss
        return reconstruction
