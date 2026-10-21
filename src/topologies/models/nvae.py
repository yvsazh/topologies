import torch
from torch import nn
from torch.nn import functional as F


class NVAE(nn.Module):
    architecture_id = "nvae"
    def __init__(self, in_channels=3, coarse_channels=4, fine_channels=4, hidden_channels=16):
        super().__init__()
        self.coarse_channels = coarse_channels
        self.fine_channels = fine_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.coarse_stats = nn.Conv2d(hidden_channels, 2 * coarse_channels, 3, padding=1)
        self.fine_context = nn.Conv2d(hidden_channels + coarse_channels, hidden_channels, 3, padding=1)
        self.fine_stats = nn.Conv2d(hidden_channels, 2 * fine_channels, 3, padding=1)
        self.decoder = nn.Sequential(
            nn.Conv2d(coarse_channels + fine_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, in_channels, 3, padding=1),
        )

    @staticmethod
    def _sample(stats):
        mean, log_variance = stats.chunk(2, dim=1)
        log_variance = log_variance.clamp(-12.0, 8.0)
        sample = mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)
        return sample, mean, log_variance

    @staticmethod
    def _kl(mean, log_variance):
        return 0.5 * (mean.square() + log_variance.exp() - 1.0 - log_variance).sum(dim=(1, 2, 3))

    def forward(self, image):
        features = self.encoder(image)
        coarse, coarse_mean, coarse_log_variance = self._sample(self.coarse_stats(features))
        coarse_up = F.interpolate(coarse, size=features.shape[-2:], mode="nearest")
        fine_features = F.silu(self.fine_context(torch.cat((features, coarse_up), dim=1)))
        fine, fine_mean, fine_log_variance = self._sample(self.fine_stats(fine_features))
        latent = torch.cat((coarse, fine), dim=1)
        latent = F.interpolate(latent, size=image.shape[-2:], mode="nearest")
        reconstruction = self.decoder(latent)
        return reconstruction, {
            "coarse": self._kl(coarse_mean, coarse_log_variance),
            "fine": self._kl(fine_mean, fine_log_variance),
        }
