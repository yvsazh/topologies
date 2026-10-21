import torch
from torch import nn
from torch.nn import functional as functional


class SimpleNet(nn.Module):
    component_id = "simplenet"
    def __init__(self, feature_dim, hidden_dim=32, noise_scale=0.015,
                 positive_margin=0.5, negative_margin=-0.5):
        super().__init__()
        if feature_dim < 1 or hidden_dim < 1:
            raise ValueError("feature_dim and hidden_dim must be positive")
        if noise_scale < 0:
            raise ValueError("noise_scale must be non-negative")
        if positive_margin <= negative_margin:
            raise ValueError("positive_margin must exceed negative_margin")
        self.feature_dim = feature_dim
        self.noise_scale = float(noise_scale)
        self.positive_margin = float(positive_margin)
        self.negative_margin = float(negative_margin)
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        if features.shape[-1] != self.feature_dim:
            raise ValueError("last feature dimension does not match feature_dim")
        return self.discriminator(features).squeeze(-1)

    def synthesize_anomaly(self, features, noise_scale=None):
        scale = self.noise_scale if noise_scale is None else float(noise_scale)
        if scale < 0:
            raise ValueError("noise_scale must be non-negative")
        return features + torch.randn_like(features) * scale

    def loss(self, clean_features, noisy_features=None):
        if noisy_features is None:
            noisy_features = self.synthesize_anomaly(clean_features)
        if clean_features.shape != noisy_features.shape:
            raise ValueError("clean and noisy feature shapes must match")
        clean_logits = self(clean_features)
        noisy_logits = self(noisy_features)
        clean_loss = functional.relu(self.positive_margin - clean_logits)
        noisy_loss = functional.relu(noisy_logits - self.negative_margin)
        return (clean_loss + noisy_loss).mean()
