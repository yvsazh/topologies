import torch
from torch import nn


def multiset(items, seeds):
    if items.ndim == 2:
        items = items.unsqueeze(0)
        squeeze = True
    elif items.ndim == 3:
        squeeze = False
    else:
        raise ValueError("items must have shape (nodes, features) or (batch, nodes, features)")
    if seeds.ndim == 2:
        seeds = seeds.unsqueeze(0).expand(items.shape[0], -1, -1)
    elif seeds.ndim != 3:
        raise ValueError("seeds must have shape (seeds, features) or (batch, seeds, features)")
    if seeds.shape[0] != items.shape[0] or seeds.shape[-1] != items.shape[-1]:
        raise ValueError("seeds and items must agree on batch and feature dimensions")
    if items.shape[-2] == 0:
        raise ValueError("items must contain at least one node")
    attention = torch.softmax(torch.matmul(seeds, items.transpose(-1, -2)), dim=-1)
    pooled = torch.matmul(attention, items)
    return pooled.squeeze(0) if squeeze else pooled


class GMT(nn.Module):
    component_id = "gmt"

    def __init__(self, input_dim, num_seeds=1):
        super().__init__()
        if input_dim <= 0 or num_seeds <= 0:
            raise ValueError("input_dim and num_seeds must be positive")
        self.input_dim = input_dim
        self.num_seeds = num_seeds
        self.seed_vectors = nn.Parameter(torch.empty(num_seeds, input_dim))
        nn.init.xavier_uniform_(self.seed_vectors)

    def forward(self, nodes):
        if not isinstance(nodes, torch.Tensor):
            nodes = torch.as_tensor(nodes, dtype=self.seed_vectors.dtype, device=self.seed_vectors.device)
        if nodes.ndim not in (2, 3) or nodes.shape[-1] != self.input_dim:
            raise ValueError("nodes must have shape (nodes, input_dim) or (batch, nodes, input_dim)")
        seeds = self.seed_vectors.to(dtype=nodes.dtype, device=nodes.device)
        return multiset(nodes, seeds)
