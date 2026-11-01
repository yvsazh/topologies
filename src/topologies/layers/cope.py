import torch
from torch import nn


class CoPE(nn.Module):
    component_id = "cope"

    def __init__(self, npos_max, head_dim):
        super().__init__()
        if isinstance(npos_max, bool) or not isinstance(npos_max, int) or npos_max <= 0:
            raise ValueError("npos_max must be a positive integer")
        if isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim <= 0:
            raise ValueError("head_dim must be a positive integer")
        self.npos_max = npos_max
        self.head_dim = head_dim
        self.pos_emb = nn.Parameter(torch.zeros(1, head_dim, npos_max))

    def forward(self, query, attn_logits):
        if not isinstance(query, torch.Tensor) or not isinstance(attn_logits, torch.Tensor):
            raise TypeError("query and attn_logits must be tensors")
        if query.ndim < 2 or attn_logits.ndim < 2:
            raise ValueError("query and attn_logits must have at least two dimensions")
        if not query.is_floating_point() or not attn_logits.is_floating_point():
            raise ValueError("query and attn_logits must have floating point dtypes")
        if query.shape[-1] != self.head_dim:
            raise ValueError("query feature width must equal head_dim")
        if attn_logits.shape[-2] != attn_logits.shape[-1]:
            raise ValueError("attn_logits must be square")
        if query.shape[:-2] != attn_logits.shape[:-2]:
            raise ValueError("query and attn_logits leading dimensions must match")
        if query.shape[-2] <= 0:
            raise ValueError("query and attn_logits sequence dimensions must be positive")
        if query.shape[-2] != attn_logits.shape[-2]:
            raise ValueError("query and attn_logits sequence dimensions must match")
        gates = torch.sigmoid(attn_logits)
        positions = torch.flip(torch.cumsum(torch.flip(gates, dims=(-1,)), dim=-1), dims=(-1,))
        positions = positions.clamp(max=self.npos_max - 1)
        positions_ceil = positions.ceil().long()
        positions_floor = positions.floor().long()
        position_logits = torch.matmul(query, self.pos_emb.squeeze(0))
        logits_ceil = position_logits.gather(-1, positions_ceil)
        logits_floor = position_logits.gather(-1, positions_floor)
        weights = positions - positions_floor
        return logits_ceil * weights + logits_floor * (1 - weights)
