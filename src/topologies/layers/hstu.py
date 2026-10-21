import torch
from torch import nn
from torch.nn import functional as F


class HSTUPointwiseAttention(nn.Module):
    component_id = "hstu"

    def __init__(self, hidden_dim=16, num_heads=4, eps=1e-5):
        super().__init__()
        if hidden_dim <= 0 or num_heads <= 0 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.project = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim, eps=eps)
        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, relative_bias=None, causal=True):
        if x.ndim != 3 or x.shape[-1] != self.hidden_dim:
            raise ValueError("x must have shape (batch, sequence, hidden_dim)")
        batch, length, _ = x.shape
        u, v, q, k = self.project(x).chunk(4, dim=-1)
        u, v, q, k = (F.silu(t) for t in (u, v, q, k))

        def split_heads(tensor):
            return tensor.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

        u, v, q, k = (split_heads(t) for t in (u, v, q, k))
        scores = torch.matmul(q, k.transpose(-1, -2))
        if relative_bias is not None:
            scores = scores + torch.as_tensor(relative_bias, dtype=scores.dtype, device=scores.device)
        weights = F.silu(scores)
        if causal:
            allowed = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
            weights = weights.masked_fill(~allowed[None, None, :, :], 0.0)
        pooled = torch.matmul(weights, v).transpose(1, 2).contiguous().reshape(batch, length, self.hidden_dim)
        gate = u.transpose(1, 2).contiguous().reshape(batch, length, self.hidden_dim)
        return self.output(self.norm(pooled) * gate)
