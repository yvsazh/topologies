import math
import torch
from torch import nn

def _positive(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')

def _dropout(value):
    if not isinstance(value, (int, float)) or not 0 <= value < 1:
        raise ValueError('dropout must be in [0, 1)')

def _tokens(tokens, mask, vocab, context):
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.dtype != torch.long:
        raise ValueError('tokens must be a LongTensor shaped [batch, time]')
    if not 0 < tokens.shape[1] <= context:
        raise ValueError('token sequence length must be in [1, context_length]')
    if tokens.device.type != 'meta' and tokens.numel() and (tokens.min() < 0 or tokens.max() >= vocab):
        raise ValueError('token IDs must be within the configured vocabulary')
    if mask is None:
        return torch.ones_like(tokens, dtype=torch.bool)
    if not isinstance(mask, torch.Tensor) or mask.shape != tokens.shape or mask.device != tokens.device:
        raise ValueError('mask must match tokens shape and device')
    if mask.dtype != torch.bool and (not mask.is_floating_point()) and (mask.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)):
        raise ValueError('mask must contain boolean or numeric 0/1 values')
    if mask.device.type != 'meta' and (not torch.all((mask == 0) | (mask == 1))):
        raise ValueError('mask values must be 0/1 (1 means valid)')
    return mask.bool()

class _Attention(nn.Module):

    def __init__(self, width, heads, dropout, rotary=False):
        super().__init__()
        self.heads, self.dim = (heads, width // heads)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.dropout = dropout
        self.rotary = rotary

    def forward(self, x, valid, causal=False):
        b, t, width = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.heads, self.dim).permute(2, 0, 3, 1, 4)
        if self.rotary:
            pairs = self.dim // 2
            if pairs:
                inv = torch.exp(torch.arange(pairs, device=x.device, dtype=x.dtype) * (-math.log(10000.0) / max(pairs, 1)))
                phase = torch.arange(t, device=x.device, dtype=x.dtype)[:, None] * inv[None]
                cos, sin = (phase.cos()[None, None], phase.sin()[None, None])

                def rotate(a):
                    even, odd = (a[..., :2 * pairs:2], a[..., 1:2 * pairs:2])
                    rotary = torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)
                    return torch.cat((rotary, a[..., 2 * pairs:]), -1)
                q, k = (rotate(q), rotate(k))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dim)
        allow = valid[:, None, None, :].expand(b, 1, t, t)
        if causal:
            allow = allow & torch.ones((t, t), dtype=torch.bool, device=x.device).tril()[None, None]
        allow = allow | ~valid[:, None, :, None] & torch.eye(t, dtype=torch.bool, device=x.device)[None, None]
        weights = scores.masked_fill(~allow, torch.finfo(scores.dtype).min).softmax(-1)
        weights = nn.functional.dropout(weights, self.dropout, self.training)
        out = (weights @ v).transpose(1, 2).reshape(b, t, width)
        return self.out(out) * valid.unsqueeze(-1)

class _Block(nn.Module):

    def __init__(self, width, heads, dropout, norm_first):
        super().__init__()
        self.attn = _Attention(width, heads, dropout, rotary=False)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * width, width))
        self.n1, self.n2 = (nn.LayerNorm(width), nn.LayerNorm(width))
        self.drop, self.norm_first = (nn.Dropout(dropout), norm_first)

    def forward(self, x, valid, causal=False):
        if self.norm_first:
            x = x + self.drop(self.attn(self.n1(x), valid, causal))
            x = x + self.drop(self.ff(self.n2(x)))
        else:
            x = self.n1(x + self.drop(self.attn(x, valid, causal)))
            x = self.n2(x + self.drop(self.ff(x)))
        return x * valid.unsqueeze(-1)

class _Encoder(nn.Module):

    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, dropout, position, norm_first):
        super().__init__()
        for key, value in (('vocab_size', vocab_size), ('context_length', context_length), ('d_model', d_model), ('num_layers', num_layers), ('num_heads', num_heads)):
            _positive(key, value)
        _dropout(dropout)
        if d_model % num_heads:
            raise ValueError('num_heads must divide d_model')
        if position not in ('learned', 'sinusoidal', 'rope'):
            raise ValueError('position must be learned, sinusoidal, or rope')
        self.vocab_size, self.context_length, self.position = (vocab_size, context_length, position)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(context_length, d_model) if position == 'learned' else None
        self.blocks = nn.ModuleList([_Block(d_model, num_heads, dropout, norm_first) for _ in range(num_layers)])
        if position == 'rope':
            for block in self.blocks:
                block.attn.rotary = True
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, mask=None, causal=False):
        valid = _tokens(tokens, mask, self.vocab_size, self.context_length)
        t = tokens.shape[1]
        x = self.embedding(tokens)
        if self.position == 'learned':
            x = x + self.pos(torch.arange(t, device=tokens.device))[None]
        elif self.position == 'sinusoidal':
            ix = torch.arange(t, device=tokens.device, dtype=x.dtype)[:, None]
            scale = torch.exp(torch.arange(0, x.shape[-1], 2, device=x.device, dtype=x.dtype) * (-math.log(10000.0) / x.shape[-1]))
            pe = torch.zeros((t, x.shape[-1]), device=x.device, dtype=x.dtype)
            pe[:, 0::2] = torch.sin(ix * scale)
            pe[:, 1::2] = torch.cos(ix * scale[:pe[:, 1::2].shape[-1]])
            x = x + pe[None]
        x = self.dropout(x) * valid.unsqueeze(-1)
        for block in self.blocks:
            x = block(x, valid, causal)
        return (self.norm(x) * valid.unsqueeze(-1), valid)

class TransformerLM(nn.Module):
    architecture_id = 'transformer-lm'

    def __init__(self, vocab_size, context_length, d_model=128, num_layers=2, num_heads=4, dropout=0.0, position='learned', norm_first=True, preset=None):
        super().__init__()
        if preset is not None:
            if preset not in ('tiny', 'small', 'base'):
                raise ValueError('preset must be tiny, small, or base')
            d_model, num_layers, num_heads = {'tiny': (32, 1, 2), 'small': (128, 2, 4), 'base': (256, 4, 8)}[preset]
        self.encoder = _Encoder(vocab_size, context_length, d_model, num_layers, num_heads, dropout, position, norm_first)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.vocab_size, self.context_length = (vocab_size, context_length)

    def forward(self, tokens, mask=None):
        x, _ = self.encoder(tokens, mask, causal=True)
        return self.lm_head(x)

class TransformerClassifier(nn.Module):
    architecture_id = 'transformer-classifier'

    def __init__(self, vocab_size, num_classes, context_length, d_model=128, num_layers=2, num_heads=4, dropout=0.0, position='learned', norm_first=True, preset=None):
        super().__init__()
        _positive('num_classes', num_classes)
        if preset is not None:
            if preset not in ('tiny', 'small', 'base'):
                raise ValueError('preset must be tiny, small, or base')
            d_model, num_layers, num_heads = {'tiny': (32, 1, 2), 'small': (128, 2, 4), 'base': (256, 4, 8)}[preset]
        self.encoder = _Encoder(vocab_size, context_length, d_model, num_layers, num_heads, dropout, position, norm_first)
        self.head = nn.Linear(d_model, num_classes)
        self.vocab_size, self.context_length = (vocab_size, context_length)

    def forward(self, tokens, mask=None):
        x, valid = self.encoder(tokens, mask)
        pooled = x.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        return self.head(pooled)

class TransformerTagger(nn.Module):
    architecture_id = 'transformer-tagger'

    def __init__(self, vocab_size, num_labels, context_length, d_model=128, num_layers=2, num_heads=4, dropout=0.0, position='learned', norm_first=True, preset=None):
        super().__init__()
        _positive('num_labels', num_labels)
        if preset is not None:
            if preset not in ('tiny', 'small', 'base'):
                raise ValueError('preset must be tiny, small, or base')
            d_model, num_layers, num_heads = {'tiny': (32, 1, 2), 'small': (128, 2, 4), 'base': (256, 4, 8)}[preset]
        self.encoder = _Encoder(vocab_size, context_length, d_model, num_layers, num_heads, dropout, position, norm_first)
        self.head = nn.Linear(d_model, num_labels)
        self.vocab_size, self.context_length = (vocab_size, context_length)

    def forward(self, tokens, mask=None):
        x, valid = self.encoder(tokens, mask)
        return self.head(x) * valid.unsqueeze(-1)
