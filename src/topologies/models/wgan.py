import torch
from torch import nn


class WGAN(nn.Module):
    architecture_id = "wgan"
    def __init__(self, sample_dim=4, features=4, clip_value=0.1):
        super().__init__()
        if sample_dim <= 0 or features <= 0 or clip_value <= 0:
            raise ValueError("sample_dim, features, and clip_value must be positive")
        self.sample_dim = sample_dim
        self.clip_value = float(clip_value)
        self.critic = nn.Sequential(
            nn.Linear(sample_dim, features),
            nn.LeakyReLU(0.2),
            nn.Linear(features, features),
            nn.LeakyReLU(0.2),
            nn.Linear(features, 1),
        )

    def forward(self, samples):
        if samples.ndim != 2 or samples.shape[-1] != self.sample_dim:
            raise ValueError(f"samples must have shape (batch, {self.sample_dim})")
        return self.critic(samples).squeeze(-1)

    def critic_loss(self, real_samples, fake_samples):
        return self(fake_samples).mean() - self(real_samples).mean()

    def clip_weights(self):
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.clamp_(-self.clip_value, self.clip_value)
