import math
import torch
from torch import nn

class _Attention(nn.Module):

    def __init__(self, width, heads, dropout, rotary=False):
        super().__init__()
        self.h, self.d = (heads, width // heads)
        self.q, self.k, self.v, self.o = (nn.Linear(width, width) for _ in range(4))
        self.dropout = dropout
        self.rotary = rotary

    def forward(self, q, kv, key_valid, causal=False):
        b, tq, w = q.shape
        tk = kv.shape[1]
        q = self.q(q).view(b, tq, self.h, self.d).transpose(1, 2)
        k = self.k(kv).view(b, tk, self.h, self.d).transpose(1, 2)
        v = self.v(kv).view(b, tk, self.h, self.d).transpose(1, 2)
        if self.rotary:
            pairs = self.d // 2
            if pairs:
                inv = torch.exp(torch.arange(pairs, device=q.device, dtype=q.dtype) * (-math.log(10000.0) / pairs))

                def rotate(a):
                    pos = torch.arange(a.shape[-2], device=a.device, dtype=a.dtype)[:, None]
                    phase = pos * inv[None]
                    c, s = (phase.cos()[None, None], phase.sin()[None, None])
                    e, o = (a[..., :2 * pairs:2], a[..., 1:2 * pairs:2])
                    return torch.cat((torch.stack((e * c - o * s, e * s + o * c), -1).flatten(-2), a[..., 2 * pairs:]), -1)
                q, k = (rotate(q), rotate(k))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d)
        allowed = key_valid[:, None, None, :].expand(b, 1, tq, tk)
        if causal:
            allowed = allowed & torch.ones((tq, tk), dtype=torch.bool, device=q.device).tril()[None, None]
        if tk:
            fallback = torch.zeros_like(allowed)
            fallback[..., 0] = ~key_valid.any(-1)[:, None, None]
            allowed = allowed | fallback
        p = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min).softmax(-1)
        return self.o((nn.functional.dropout(p, self.dropout, self.training) @ v).transpose(1, 2).reshape(b, tq, w))

class _Layer(nn.Module):

    def __init__(self, w, h, dropout, norm_first):
        super().__init__()
        self.sa = _Attention(w, h, dropout)
        self.ca = _Attention(w, h, dropout)
        self.ff = nn.Sequential(nn.Linear(w, 4 * w), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * w, w))
        self.n = nn.ModuleList([nn.LayerNorm(w) for _ in range(3)])
        self.drop = nn.Dropout(dropout)
        self.pre = norm_first

    def forward(self, x, valid, memory, mvalid, causal=True):
        for i, fn in enumerate((lambda z: self.sa(z, z, valid, causal), lambda z: self.ca(z, memory, mvalid), self.ff)):
            if self.pre:
                x = x + self.drop(fn(self.n[i](x)))
            else:
                x = self.n[i](x + self.drop(fn(x)))
        return x * valid.unsqueeze(-1)

class Seq2Seq(nn.Module):
    architecture_id = 'seq2seq'

    def __init__(self, source_vocab_size, target_vocab_size, source_context_length, target_context_length, bos_id, eos_id, pad_id, d_model=128, num_layers=2, num_heads=4, dropout=0.0, position='learned', norm_first=True, preset=None):
        super().__init__()
        if preset is not None:
            if preset not in ('tiny', 'small', 'base'):
                raise ValueError('invalid preset')
            d_model, num_layers, num_heads = {'tiny': (32, 1, 2), 'small': (128, 2, 4), 'base': (256, 4, 8)}[preset]
        for name, v in (('source_vocab_size', source_vocab_size), ('target_vocab_size', target_vocab_size), ('source_context_length', source_context_length), ('target_context_length', target_context_length), ('d_model', d_model), ('num_layers', num_layers), ('num_heads', num_heads)):
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise ValueError(f'{name} must be positive')
        if d_model % num_heads:
            raise ValueError('num_heads must divide d_model')
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0,1)')
        if position not in ('learned', 'sinusoidal', 'rope'):
            raise ValueError('unsupported position mode')
        if any((not isinstance(i, int) or not 0 <= i < target_vocab_size for i in (bos_id, eos_id, pad_id))):
            raise ValueError('BOS/EOS/PAD IDs must be valid target vocabulary IDs')
        self.source_vocab_size, self.target_vocab_size = (source_vocab_size, target_vocab_size)
        self.source_context_length, self.target_context_length = (source_context_length, target_context_length)
        self.bos_id, self.eos_id, self.pad_id = (bos_id, eos_id, pad_id)
        self.position = position
        self.semb = nn.Embedding(source_vocab_size, d_model)
        self.temb = nn.Embedding(target_vocab_size, d_model)
        self.spos = nn.Embedding(source_context_length, d_model) if position == 'learned' else None
        self.tpos = nn.Embedding(target_context_length, d_model) if position == 'learned' else None
        self.enc = nn.ModuleList([_Layer(d_model, num_heads, dropout, norm_first) for _ in range(num_layers)])
        self.dec = nn.ModuleList([_Layer(d_model, num_heads, dropout, norm_first) for _ in range(num_layers)])
        if position == 'rope':
            for layer in self.enc:
                layer.sa.rotary = True
            for layer in self.dec:
                layer.sa.rotary = True
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _valid(tokens, mask, vocab, context, name):
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.dtype != torch.long:
            raise ValueError(f'{name} must be LongTensor[B,T]')
        if not 0 < tokens.shape[1] <= context:
            raise ValueError(f'{name} exceeds context bounds')
        if tokens.device.type != 'meta' and tokens.numel() and (tokens.min() < 0 or tokens.max() >= vocab):
            raise ValueError(f'{name} IDs outside vocabulary')
        if mask is None:
            return torch.ones_like(tokens, dtype=torch.bool)
        if mask.shape != tokens.shape or mask.device != tokens.device:
            raise ValueError(f'{name} mask must match tokens')
        if mask.device.type != 'meta' and (not torch.all((mask == 0) | (mask == 1))):
            raise ValueError('mask must contain 0/1')
        return mask.bool()

    def _embed(self, tokens, embedding, pos):
        x = embedding(tokens)
        if pos is not None:
            x = x + pos(torch.arange(tokens.shape[1], device=tokens.device))[None]
        elif self.position == 'sinusoidal':
            t, w = x.shape[1:]
            ix = torch.arange(t, device=x.device, dtype=x.dtype)[:, None]
            scale = torch.exp(torch.arange(0, w, 2, device=x.device, dtype=x.dtype) * (-math.log(10000.0) / w))
            pe = torch.zeros((t, w), device=x.device, dtype=x.dtype)
            pe[:, 0::2] = torch.sin(ix * scale)
            pe[:, 1::2] = torch.cos(ix * scale[:pe[:, 1::2].shape[-1]])
            x = x + pe[None]
        return self.dropout(x)

    def encode(self, src, src_mask=None):
        valid = self._valid(src, src_mask, self.source_vocab_size, self.source_context_length, 'src')
        x = self._embed(src, self.semb, self.spos) * valid.unsqueeze(-1)
        for layer in self.enc:
            x = layer(x, valid, x, valid, causal=False)
        return (self.norm(x) * valid.unsqueeze(-1), valid)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory, mvalid = self.encode(src, src_mask)
        target_valid = self._valid(tgt, tgt_mask, self.target_vocab_size, self.target_context_length, 'tgt')
        decoder = torch.full_like(tgt, self.pad_id)
        decoder[:, 0] = self.bos_id
        if tgt.shape[1] > 1:
            decoder[:, 1:] = tgt[:, :-1]
        x = self._embed(decoder, self.temb, self.tpos) * target_valid.unsqueeze(-1)
        for layer in self.dec:
            x = layer(x, target_valid, memory, mvalid)
        return self.head(self.norm(x)) * target_valid.unsqueeze(-1)

    @torch.no_grad()
    def generate(self, src, src_mask=None, max_new_tokens=None):
        memory, mvalid = self.encode(src, src_mask)
        limit = max_new_tokens or self.target_context_length - 1
        if not 1 <= limit < self.target_context_length:
            raise ValueError('max_new_tokens out of range')
        out = torch.full((src.shape[0], 1), self.bos_id, dtype=torch.long, device=src.device)
        done = torch.zeros(src.shape[0], dtype=torch.bool, device=src.device)
        for _ in range(limit):
            valid = torch.ones_like(out, dtype=torch.bool)
            x = self._embed(out, self.temb, self.tpos)
            for layer in self.dec:
                x = layer(x, valid, memory, mvalid)
            nxt = self.head(self.norm(x[:, -1])).argmax(-1)
            nxt = torch.where(done, self.pad_id, nxt)
            out = torch.cat((out, nxt[:, None]), 1)
            done |= nxt.eq(self.eos_id)
            if done.all():
                break
        return out
