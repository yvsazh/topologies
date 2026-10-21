import torch
from torch import nn
from torch.nn import functional as F


class LinearAttention(nn.Module):
    component_id = "linear-attention"
    def __init__(self, embed_dim=4, causal=False, eps=1e-6):
        super().__init__()
        if not isinstance(embed_dim, int) or isinstance(embed_dim, bool) or embed_dim < 1:
            raise ValueError("embed_dim must be a positive integer")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.embed_dim = embed_dim
        self.causal = bool(causal)
        self.eps = float(eps)

    def forward(self, query, key, value):
        if query.shape[-1] != self.embed_dim or key.shape[-1] != self.embed_dim or value.shape[-1] != self.embed_dim:
            raise ValueError("query, key, and value feature dimensions must match embed_dim")
        if query.ndim < 2 or key.ndim != query.ndim or value.ndim != key.ndim:
            raise ValueError("query, key, and value must have the same rank of at least two")
        if key.shape[:-2] != query.shape[:-2] or value.shape[:-2] != key.shape[:-2]:
            raise ValueError("query, key, and value leading dimensions must match")
        if key.shape[-2] != value.shape[-2]:
            raise ValueError("key and value sequence lengths must match")
        if self.causal and query.shape[-2] != key.shape[-2]:
            raise ValueError("causal attention requires equal query and key sequence lengths")
        query_features = F.elu(query) + 1
        key_features = F.elu(key) + 1
        if self.causal:
            key_value = key_features.unsqueeze(-1) * value.unsqueeze(-2)
            key_value = key_value.cumsum(dim=-3)
            key_sum = key_features.cumsum(dim=-2)
            numerator = torch.einsum("...nd,...nde->...ne", query_features, key_value)
            denominator = torch.einsum("...nd,...nd->...n", query_features, key_sum).unsqueeze(-1)
            return numerator / denominator.clamp_min(self.eps)
        key_value = torch.einsum("...nd,...ne->...de", key_features, value)
        key_sum = key_features.sum(dim=-2)
        numerator = torch.einsum("...nd,...de->...ne", query_features, key_value)
        denominator = torch.einsum("...nd,...d->...n", query_features, key_sum).unsqueeze(-1)
        return numerator / denominator.clamp_min(self.eps)
