import torch
from torch import nn


class SPADE(nn.Module):
    architecture_id = "spade"
    def __init__(self, channels, label_channels, hidden_channels=128, eps=1e-5):
        super().__init__()
        if channels <= 0 or label_channels <= 0 or hidden_channels <= 0:
            raise ValueError("channel counts must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.channels = channels
        self.label_channels = label_channels
        self.param_free_norm = nn.InstanceNorm2d(channels, affine=False, eps=eps)
        self.shared = nn.Sequential(
            nn.Conv2d(label_channels, hidden_channels, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.ReLU(inplace=False),
        )
        self.gamma = nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1, padding_mode="replicate")
        self.beta = nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1, padding_mode="replicate")

    def forward(self, features, semantic_map):
        if features.ndim != 4 or features.shape[1] != self.channels:
            raise ValueError("features must have shape (batch, channels, height, width)")
        if semantic_map.ndim != 4 or semantic_map.shape[1] != self.label_channels:
            raise ValueError("semantic_map must have shape (batch, label_channels, height, width)")
        if features.shape[0] != semantic_map.shape[0]:
            raise ValueError("features and semantic_map batch sizes must match")
        if features.shape[-2:] != semantic_map.shape[-2:]:
            semantic_map = torch.nn.functional.interpolate(
                semantic_map, size=features.shape[-2:], mode="nearest"
            )
        conditioned = self.shared(semantic_map)
        normalized = self.param_free_norm(features)
        return normalized * (1 + self.gamma(conditioned)) + self.beta(conditioned)
