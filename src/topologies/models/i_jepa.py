import torch
from torch import nn


class IJEPA(nn.Module):
    """Small predictor/loss component for masked target embeddings."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.embedding_dim = embedding_dim
        self.predictor = nn.Linear(embedding_dim * 2, embedding_dim)

    def predict(self, context: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        context = torch.as_tensor(context)
        mask = torch.as_tensor(mask, device=context.device, dtype=context.dtype)
        if context.ndim < 2 or context.numel() == 0 or context.shape[-2] == 0:
            raise ValueError("context must not be empty")
        if context.shape[-1] != self.embedding_dim or mask.shape[-1] != self.embedding_dim:
            raise ValueError("context and mask embedding dimensions must match embedding_dim")
        context_embedding = context.mean(dim=-2)
        if mask.ndim < 1 or mask.numel() == 0:
            raise ValueError("mask must not be empty")
        mask_embedding = mask.mean(dim=-2) if mask.ndim >= 2 else mask
        if context_embedding.shape[:-1] != mask_embedding.shape[:-1]:
            raise ValueError("context and mask batch dimensions must match")
        return self.predictor(torch.cat((context_embedding, mask_embedding), dim=-1))

    def loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean squared error to a detached target embedding."""
        return torch.mean((prediction - target.detach()) ** 2)
