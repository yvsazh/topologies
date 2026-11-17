import torch
from torch import nn


class VITS(nn.Module):
    architecture_id = "vits"
    PRESETS = {
        "tiny": {"text_channels": 2, "posterior_channels": 1, "latent_channels": 1},
        "small": {"text_channels": 8, "posterior_channels": 4, "latent_channels": 4},
    }

    def __init__(self, text_channels=8, posterior_channels=4, latent_channels=4):
        super().__init__()
        channels = (text_channels, posterior_channels, latent_channels)
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in channels):
            raise ValueError("text_channels, posterior_channels, and latent_channels must be positive integers")
        self.text_channels = text_channels
        self.posterior_channels = posterior_channels
        self.latent_channels = latent_channels
        self.prior_projection = nn.Conv1d(text_channels, 2 * latent_channels, kernel_size=1)
        self.posterior_projection = nn.Conv1d(posterior_channels, 2 * latent_channels, kernel_size=1)

    def forward(self, text_features, posterior_features, sample=True):
        if not isinstance(sample, bool):
            raise TypeError("sample must be a bool")
        if not isinstance(text_features, torch.Tensor) or text_features.ndim != 3:
            raise ValueError("text_features must have shape [batch, text_channels, time]")
        if not isinstance(posterior_features, torch.Tensor) or posterior_features.ndim != 3:
            raise ValueError("posterior_features must have shape [batch, posterior_channels, time]")
        if text_features.shape[1] != self.text_channels:
            raise ValueError(f"text_features must have shape [batch, {self.text_channels}, time]")
        if posterior_features.shape[1] != self.posterior_channels:
            raise ValueError(f"posterior_features must have shape [batch, {self.posterior_channels}, time]")
        if text_features.shape[0] != posterior_features.shape[0] or text_features.shape[2] != posterior_features.shape[2]:
            raise ValueError("text_features and posterior_features must have matching batch and sequence dimensions")
        if text_features.shape[0] < 1 or text_features.shape[2] < 1:
            raise ValueError("text_features and posterior_features must have non-empty batch and sequence dimensions")
        if not torch.is_floating_point(text_features) or not torch.is_floating_point(posterior_features):
            raise ValueError("text_features and posterior_features must be floating-point tensors")
        if text_features.device != posterior_features.device or text_features.dtype != posterior_features.dtype:
            raise ValueError("text_features and posterior_features must have matching device and dtype")
        for features, projection in (
            (text_features, self.prior_projection),
            (posterior_features, self.posterior_projection),
        ):
            if features.device != projection.weight.device or features.dtype != projection.weight.dtype:
                raise ValueError("feature device and dtype must match the projection parameters")

        prior_mean, prior_log_variance = self.prior_projection(text_features).chunk(2, dim=1)
        posterior_mean, posterior_log_variance = self.posterior_projection(posterior_features).chunk(2, dim=1)
        if sample:
            latent = posterior_mean + torch.exp(0.5 * posterior_log_variance) * torch.randn_like(posterior_mean)
        else:
            latent = posterior_mean
        kl_per_position = 0.5 * (
            prior_log_variance
            - posterior_log_variance
            + torch.exp(posterior_log_variance - prior_log_variance)
            + (posterior_mean - prior_mean).square() * torch.exp(-prior_log_variance)
            - 1
        ).sum(dim=1)
        sequence_kl = kl_per_position.sum(dim=1)
        return {
            "prior_mean": prior_mean,
            "prior_log_variance": prior_log_variance,
            "posterior_mean": posterior_mean,
            "posterior_log_variance": posterior_log_variance,
            "latent": latent,
            "kl_per_position": kl_per_position,
            "sequence_kl": sequence_kl,
            "kl": sequence_kl.mean(),
        }
