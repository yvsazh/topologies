import torch
from torch import nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    component_id = 'multi-head-attention'

    def __init__(self, d_model, num_heads, kv_heads=None, dropout=0.0, causal=False):
        super().__init__()
        kv_heads = num_heads if kv_heads is None else kv_heads
        dimensions = (d_model, num_heads, kv_heads)
        if any((isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in dimensions)) or d_model % num_heads or num_heads % kv_heads or isinstance(dropout, bool) or (not isinstance(dropout, (int, float))) or (not 0 <= dropout < 1) or (not isinstance(causal, bool)):
            raise ValueError('invalid attention dimensions')
        self.dim = d_model // num_heads
        self.heads = num_heads
        self.kv_heads = kv_heads
        self.causal = causal
        self.dropout = dropout
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, kv_heads * self.dim)
        self.v = nn.Linear(d_model, kv_heads * self.dim)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, context=None, mask=None, causal=None):
        context = x if context is None else context
        if x.ndim != 3 or context.ndim != 3 or x.shape[0] != context.shape[0]:
            raise ValueError('query and context must be [B,T,D] with matching batches')
        b, t, _ = x.shape
        s = context.shape[1]
        if x.shape[-1] != self.q.in_features or context.shape[-1] != self.k.in_features:
            raise ValueError('query/context feature width mismatch')
        q = self.q(x).view(b, t, self.heads, self.dim).transpose(1, 2)
        k = self.k(context).view(b, s, self.kv_heads, self.dim).transpose(1, 2)
        v = self.v(context).view(b, s, self.kv_heads, self.dim).transpose(1, 2)
        if self.kv_heads != self.heads:
            k = k.repeat_interleave(self.heads // self.kv_heads, 1)
            v = v.repeat_interleave(self.heads // self.kv_heads, 1)
        allowed = None
        if mask is not None:
            mask = torch.as_tensor(mask, device=x.device)
            if mask.shape != (b, s):
                raise ValueError('mask must have shape [B,S]')
            if mask.dtype == torch.bool:
                valid = mask
            elif mask.is_floating_point() or mask.is_complex() or bool(((mask != 0) & (mask != 1)).any()):
                raise ValueError('mask values must be exactly 0/1 or bool (True means valid)')
            else:
                valid = mask == 1
            if bool((~valid.any(-1)).any()):
                raise ValueError('each attention row must have a valid key')
            allowed = valid[:, None, None, :]
        use_causal = self.causal if causal is None else causal
        if use_causal:
            cm = torch.ones((t, s), device=x.device, dtype=torch.bool).tril(diagonal=s - t)
            allowed = cm[None, None] if allowed is None else allowed & cm[None, None]
        if allowed is not None and bool((~allowed.any(dim=-1)).any()):
            raise ValueError('attention mask leaves at least one query without a valid key')
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(b, t, -1))
