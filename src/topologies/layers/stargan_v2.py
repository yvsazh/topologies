import torch
from torch import nn
from torch.nn import functional as F


class StarGANv2AdaIN(nn.Module):
    component_id = "stargan-v2"

    def __init__(self, channels=4, style_dim=4, eps=1e-5):
        super().__init__()
        if channels < 1 or style_dim < 1 or eps <= 0:
            raise ValueError("channels, style_dim, and eps must be positive")
        self.channels = channels
        self.style_dim = style_dim
        self.eps = eps
        self.affine = nn.Linear(style_dim, channels * 2)

    def forward(self, features, style_code):
        if features.ndim != 4 or features.shape[1] != self.channels:
            raise ValueError("features must have shape (batch, channels, height, width)")
        style_code = torch.as_tensor(
            style_code, dtype=features.dtype, device=features.device
        )
        if style_code.ndim == 1:
            style_code = style_code.unsqueeze(0)
        if style_code.shape != (features.shape[0], self.style_dim):
            raise ValueError("style_code must have shape (batch, style_dim)")
        scale, bias = self.affine(style_code).chunk(2, dim=-1)
        normalized = F.instance_norm(features, eps=self.eps)
        return (
            normalized * (1.0 + scale[:, :, None, None])
            + bias[:, :, None, None]
        )


class StarGANv2(nn.Module):
    architecture_id = "stargan-v2"
    def __init__(self, channels=4, style_dim=4, num_domains=2):
        super().__init__()
        if channels < 1 or style_dim < 1 or num_domains < 1:
            raise ValueError("channels, style_dim, and num_domains must be positive")
        self.channels = channels
        self.style_dim = style_dim
        self.num_domains = num_domains
        self.content = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.style = nn.Embedding(num_domains, style_dim)
        self.adain = StarGANv2AdaIN(channels, style_dim)
        self.output = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, content, domain):
        if content.ndim != 4 or content.shape[1] != self.channels:
            raise ValueError("content must have shape (batch, channels, height, width)")
        if isinstance(domain, int):
            domain = torch.full((content.shape[0],), domain, device=content.device, dtype=torch.long)
        else:
            domain = torch.as_tensor(domain, device=content.device, dtype=torch.long)
        if domain.ndim == 0:
            domain = domain.expand(content.shape[0])
        if domain.shape != (content.shape[0],):
            raise ValueError("domain must be a scalar or one index per content item")
        if torch.any(domain < 0) or torch.any(domain >= self.num_domains):
            raise ValueError("domain index is outside the configured domain range")
        features = F.relu(self.content(content))
        modulated = self.adain(features, self.style(domain))
        return self.output(F.relu(modulated))
