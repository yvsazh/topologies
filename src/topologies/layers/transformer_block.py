import torch
from torch import nn
import torch.nn.functional as F

class _Attention(nn.Module):

    def __init__(self, dim, heads, causal=False):
        super().__init__()
        if dim % heads:
            raise ValueError('heads must divide d_model')
        self.heads = heads
        self.width = dim // heads
        self.causal = causal
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.heads, self.width).permute(2, 0, 3, 1, 4).unbind(0)
        allowed = None
        if mask is not None:
            allowed = mask.bool()[:, None, None, :]
        if self.causal:
            cm = torch.ones(t, t, device=x.device, dtype=torch.bool).tril()
            allowed = cm if allowed is None else allowed & cm
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        return self.out(y.transpose(1, 2).reshape(b, t, d))

class _FeedForward(nn.Module):

    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    component_id = 'transformer-block'

    def __init__(self, d_model, num_heads, hidden_dim=None, causal=False, norm='pre', cross_attention=False, dropout=0.0):
        super().__init__()
        if norm not in ('pre', 'post'):
            raise ValueError('norm must be pre or post')
        self.norm = norm
        self.attn = _Attention(d_model, num_heads, causal)
        self.cross = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=dropout) if cross_attention else None
        self.ff = _FeedForward(d_model, hidden_dim or 4 * d_model)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.n3 = nn.LayerNorm(d_model) if cross_attention else None

    def forward(self, x, mask=None, context=None):
        if self.norm == 'pre':
            x = x + self.attn(self.n1(x), mask=mask)
            if self.cross is not None:
                if context is None:
                    raise ValueError('context required for cross attention')
                x = x + self.cross(self.n3(x), context, context, need_weights=False)[0]
            return x + self.ff(self.n2(x))
        x = self.n1(x + self.attn(x, mask=mask))
        if self.cross is not None:
            if context is None:
                raise ValueError('context required for cross attention')
            x = self.n3(x + self.cross(x, context, context, need_weights=False)[0])
        return self.n2(x + self.ff(x))
