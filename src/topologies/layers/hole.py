import torch
from torch import nn


class HolE(nn.Module):
    component_id = "hole"
    def __init__(self, embedding_dim=None):
        super().__init__()
        if embedding_dim is not None and embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        self.embedding_dim = embedding_dim

    @staticmethod
    def circular_correlation(head, tail):
        head = torch.as_tensor(head)
        tail = torch.as_tensor(tail, device=head.device, dtype=head.dtype)
        if head.shape != tail.shape:
            raise ValueError("head and tail must have identical shapes")
        if head.ndim == 0:
            raise ValueError("head and tail must have a feature dimension")
        return torch.fft.irfft(torch.conj(torch.fft.rfft(head, dim=-1)) * torch.fft.rfft(tail, dim=-1), n=head.shape[-1], dim=-1)

    def forward(self, head, relation, tail):
        head = torch.as_tensor(head)
        relation = torch.as_tensor(relation, device=head.device, dtype=head.dtype)
        tail = torch.as_tensor(tail, device=head.device, dtype=head.dtype)
        if head.shape != relation.shape or head.shape != tail.shape:
            raise ValueError("head, relation, and tail must have identical shapes")
        if head.ndim == 0:
            raise ValueError("head, relation, and tail must have a feature dimension")
        if self.embedding_dim is not None and head.shape[-1] != self.embedding_dim:
            raise ValueError("input feature dimension does not match embedding_dim")
        return torch.sum(relation * self.circular_correlation(head, tail), dim=-1)

    @staticmethod
    def score(head, relation, tail):
        return HolE()(head, relation, tail)
