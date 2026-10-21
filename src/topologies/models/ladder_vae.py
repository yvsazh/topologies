import torch
from torch import nn
from torch.nn import functional as F


class LadderVAE(nn.Module):
    architecture_id = "ladder-vae"
    def __init__(self, input_dim=8, hidden_dim=4, latent_dim=4, levels=2):
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or latent_dim < 1 or levels < 1:
            raise ValueError("input_dim, hidden_dim, latent_dim, and levels must be positive")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.levels = levels
        self.bottom_up = nn.ModuleList()
        width = input_dim
        for _ in range(levels):
            self.bottom_up.append(nn.Sequential(nn.Linear(width, hidden_dim), nn.ELU()))
            width = hidden_dim
        self.evidence = nn.ModuleList(nn.Linear(hidden_dim, 2 * latent_dim) for _ in range(levels))
        self.top_priors = nn.ModuleList(nn.Linear(latent_dim, 2 * latent_dim) for _ in range(levels - 1))
        self.decoder = nn.Sequential(nn.Linear(levels * latent_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, input_dim))

    def prior_parameters(self, upper_latent, lower_level):
        if lower_level < 0 or lower_level >= self.levels - 1:
            raise ValueError("lower_level must identify a non-top level")
        mean, raw_scale = self.top_priors[lower_level](upper_latent).chunk(2, dim=-1)
        return mean, F.softplus(raw_scale) + 1e-4

    @staticmethod
    def combine_gaussians(evidence_mean, evidence_scale, prior_mean, prior_scale):
        evidence_precision = evidence_scale.reciprocal().square()
        prior_precision = prior_scale.reciprocal().square()
        posterior_scale = (evidence_precision + prior_precision).rsqrt()
        posterior_mean = posterior_scale.square() * (
            evidence_mean * evidence_precision + prior_mean * prior_precision
        )
        return posterior_mean, posterior_scale

    def forward(self, x):
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"expected final input dimension {self.input_dim}")
        original_shape = x.shape
        value = x.reshape(-1, self.input_dim)
        features = []
        for layer in self.bottom_up:
            value = layer(value)
            features.append(value)
        latents = [None] * self.levels
        upper = None
        for level in reversed(range(self.levels)):
            evidence_mean, evidence_raw_scale = self.evidence[level](features[level]).chunk(2, dim=-1)
            evidence_scale = F.softplus(evidence_raw_scale) + 1e-4
            if upper is None:
                prior_mean = torch.zeros_like(evidence_mean)
                prior_scale = torch.ones_like(evidence_scale)
            else:
                prior_mean, prior_scale = self.prior_parameters(upper, level)
            posterior_mean, posterior_scale = self.combine_gaussians(
                evidence_mean, evidence_scale, prior_mean, prior_scale
            )
            upper = posterior_mean + posterior_scale * torch.randn_like(posterior_mean)
            latents[level] = upper
        reconstruction = self.decoder(torch.cat(latents, dim=-1)).reshape(original_shape)
        return reconstruction
