import torch
from torch import nn


class CausalVAELatentMixer(nn.Module):
    component_id = "causalvae"
    PRESETS = {"tiny": {"latent_dim": 3}, "small": {"latent_dim": 8}}

    def __init__(self, latent_dim=4):
        super().__init__()
        if not isinstance(latent_dim, int) or isinstance(latent_dim, bool) or latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer")
        self.latent_dim = latent_dim
        self.adjacency = nn.Parameter(torch.zeros(latent_dim, latent_dim))
        self.register_buffer(
            "order_mask",
            torch.triu(torch.ones(latent_dim, latent_dim, dtype=torch.bool), diagonal=1),
        )

    def forward(self, epsilon):
        if not torch.is_tensor(epsilon):
            raise ValueError("epsilon must be a torch.Tensor")
        if epsilon.ndim != 2:
            raise ValueError("epsilon must have rank 2 with shape [batch, latent_dim]")
        if epsilon.shape[-1] != self.latent_dim:
            raise ValueError("epsilon last dimension must match latent_dim")
        if not epsilon.is_floating_point():
            raise ValueError("epsilon must have a floating-point dtype")
        adjacency = self.adjacency.to(device=epsilon.device, dtype=epsilon.dtype)
        mask = self.order_mask.to(device=epsilon.device)
        effective_adjacency = adjacency * mask
        identity = torch.eye(self.latent_dim, device=epsilon.device, dtype=epsilon.dtype)
        system = identity - effective_adjacency.transpose(0, 1)
        return torch.linalg.solve(system, epsilon.transpose(0, 1)).transpose(0, 1)
