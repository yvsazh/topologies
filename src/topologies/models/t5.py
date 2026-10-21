from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def relative_position_bucket(relative_position: torch.Tensor, num_buckets: int = 32,
                             max_distance: int = 128, bidirectional: bool = True) -> torch.Tensor:
    buckets = 0
    if bidirectional:
        num_buckets //= 2
        buckets = (relative_position > 0).long() * num_buckets
        distance = relative_position.abs()
    else:
        distance = (-relative_position).clamp_min(0)
    exact = num_buckets // 2
    large = exact + (torch.log(distance.float().clamp_min(exact) / exact) /
                     math.log(max_distance / exact) * (num_buckets - exact)).long()
    return buckets + torch.where(distance < exact, distance, large.clamp_max(num_buckets - 1))


class _Norm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class _Attention(nn.Module):
    def __init__(self, dim, heads, d_kv, dropout):
        super().__init__()
        self.heads, self.d_kv, self.dropout = heads, d_kv, dropout
        self.q = nn.Linear(dim, heads * d_kv, bias=False)
        self.k = nn.Linear(dim, heads * d_kv, bias=False)
        self.v = nn.Linear(dim, heads * d_kv, bias=False)
        self.o = nn.Linear(heads * d_kv, dim, bias=False)

    def forward(self, qx, kx, key_mask, causal=False, bias=None, bidirectional=True):
        b, qlen, _ = qx.shape
        klen = kx.shape[1]
        split = lambda x: x.view(b, -1, self.heads, self.d_kv).transpose(1, 2)
        q, k, v = split(self.q(qx)), split(self.k(kx)), split(self.v(kx))
        scores = q @ k.transpose(-2, -1)
        qi = torch.arange(qlen, device=qx.device)[:, None]
        ki = torch.arange(klen, device=qx.device)[None, :]
        if bias is not None:
            buckets = relative_position_bucket(ki - qi, bias.num_embeddings, bidirectional=bidirectional)
            scores = scores + bias(buckets).permute(2, 0, 1).unsqueeze(0)
        valid = key_mask[:, None, None, :]
        if causal:
            valid = valid & (ki <= qi)[None, None]
        probs = torch.softmax(scores.masked_fill(~valid, torch.finfo(scores.dtype).min), -1)
        return self.o((F.dropout(probs, self.dropout, self.training) @ v).transpose(1, 2).contiguous().view(b, qlen, -1))

    def init_t5(self, dim, factor):
        for name, layer in (("q", self.q), ("k", self.k), ("v", self.v), ("o", self.o)):
            denom = (dim * self.d_kv) ** .5 if name == "q" else dim ** .5 if name in ("k", "v") else (self.heads * self.d_kv) ** .5
            nn.init.normal_(layer.weight, std=factor / denom)


class _FF(nn.Module):
    def __init__(self, dim, hidden, gated, dropout):
        super().__init__()
        self.gated, self.dropout = gated, dropout
        self.wi = nn.Linear(dim, hidden, bias=False)
        self.wi_gate = nn.Linear(dim, hidden, bias=False) if gated else None
        self.wo = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        y = self.wi(x)
        y = F.gelu(self.wi_gate(x), approximate="tanh") * y if self.gated else F.relu(y)
        return self.wo(F.dropout(y, self.dropout, self.training))


class _Enc(nn.Module):
    def __init__(self, d, h, kv, ff, gated, p, eps):
        super().__init__(); self.n1, self.n2 = _Norm(d, eps), _Norm(d, eps); self.a = _Attention(d,h,kv,p); self.f = _FF(d,ff,gated,p); self.p=p
    def forward(self,x,mask,bias):
        x=x+F.dropout(self.a(self.n1(x),self.n1(x),mask,bias=bias),self.p,self.training)
        return x+F.dropout(self.f(self.n2(x)),self.p,self.training)


class _Dec(nn.Module):
    def __init__(self,d,h,kv,ff,gated,p,eps):
        super().__init__(); self.n1,self.n2,self.n3=_Norm(d,eps),_Norm(d,eps),_Norm(d,eps); self.sa=_Attention(d,h,kv,p); self.ca=_Attention(d,h,kv,p); self.f=_FF(d,ff,gated,p); self.p=p
    def forward(self,x,mem,mask,memmask,bias):
        n=self.n1(x); x=x+F.dropout(self.sa(n,n,mask,True,bias,False),self.p,self.training)
        x=x+F.dropout(self.ca(self.n2(x),mem,memmask),self.p,self.training)
        return x+F.dropout(self.f(self.n3(x)),self.p,self.training)


class T5(nn.Module):
    architecture_id = "t5"
    PRESETS = {
        "tiny": dict(vocab_size=128, d_model=32, num_heads=4, d_kv=8, d_ff=64, num_layers=2, num_decoder_layers=2, relative_attention_num_buckets=16, num_sentinels=8),
        "small": dict(vocab_size=32128, context_length=512, d_model=512, num_heads=8, d_kv=64, d_ff=2048, num_layers=6, num_decoder_layers=6, dropout=0.1, relative_attention_num_buckets=32, relative_attention_max_distance=128, layer_norm_epsilon=1e-6, tie_word_embeddings=True, gated_ff=False, num_sentinels=100),
        "base": dict(vocab_size=32128, context_length=512, d_model=768, num_heads=12, d_kv=64, d_ff=3072, num_layers=12, num_decoder_layers=12, dropout=0.1, relative_attention_num_buckets=32, relative_attention_max_distance=128, layer_norm_epsilon=1e-6, tie_word_embeddings=True, gated_ff=False, num_sentinels=100),
        "flan-small": dict(vocab_size=32128, context_length=512, d_model=512, num_heads=6, d_kv=64, d_ff=1024, num_layers=8, num_decoder_layers=8, dropout=0.1, relative_attention_num_buckets=32, relative_attention_max_distance=128, layer_norm_epsilon=1e-6, gated_ff=True, tie_word_embeddings=False, num_sentinels=100),
        "flan-base": dict(vocab_size=32128, context_length=512, d_model=768, num_heads=12, d_kv=64, d_ff=2048, num_layers=12, num_decoder_layers=12, dropout=0.1, relative_attention_num_buckets=32, relative_attention_max_distance=128, layer_norm_epsilon=1e-6, gated_ff=True, tie_word_embeddings=False, num_sentinels=100),
    }
    BUDGET_KNOBS = {"d_model": range(32, 257, 16)}

    def __init__(self, vocab_size=128, context_length=2048, d_model=64, num_layers=2, num_decoder_layers=2,
                 num_heads=4, d_kv=16, d_ff=128, dropout=0.0, pad_token_id=0, eos_token_id=1,
                 decoder_start_token_id=0, relative_attention_num_buckets=32, relative_attention_max_distance=128,
                 layer_norm_epsilon=1e-6, tie_word_embeddings=True, gated_ff=False, num_sentinels=8,
                 initializer_factor=1.0, **unused):
        super().__init__()
        if min(vocab_size, context_length, d_model, num_layers, num_decoder_layers, num_heads, d_kv, d_ff) <= 0: raise ValueError("model dimensions must be positive")
        if not 0 <= pad_token_id < vocab_size or not 0 <= eos_token_id < vocab_size or decoder_start_token_id != pad_token_id: raise ValueError("valid PAD/EOS and PAD decoder start required")
        if relative_attention_num_buckets < 4 or relative_attention_num_buckets % 2 or relative_attention_max_distance <= relative_attention_num_buckets // 4: raise ValueError("invalid relative bucket geometry")
        if not 0 <= dropout < 1 or layer_norm_epsilon <= 0 or initializer_factor <= 0: raise ValueError("invalid dropout/epsilon/initializer")
        if not 1 <= num_sentinels or vocab_size < num_sentinels + 2: raise ValueError("sentinels must fit vocabulary; configure explicit count (HF padded vocab may not end at sentinels)")
        self.vocab_size,self.context_length,self.d_model=vocab_size,context_length,d_model
        self.pad_token_id,self.eos_token_id,self.decoder_start_token_id=pad_token_id,eos_token_id,decoder_start_token_id
        self.num_sentinels=num_sentinels; self.tie_word_embeddings=tie_word_embeddings; self.dropout=dropout; self.initializer_factor=initializer_factor
        self.shared=nn.Embedding(vocab_size,d_model); self.encoder_bias=nn.Embedding(relative_attention_num_buckets,num_heads); self.decoder_bias=nn.Embedding(relative_attention_num_buckets,num_heads)
        self.encoder=nn.ModuleList([_Enc(d_model,num_heads,d_kv,d_ff,gated_ff,dropout,layer_norm_epsilon) for _ in range(num_layers)])
        self.decoder=nn.ModuleList([_Dec(d_model,num_heads,d_kv,d_ff,gated_ff,dropout,layer_norm_epsilon) for _ in range(num_decoder_layers)])
        self.encoder_norm,self.decoder_norm=_Norm(d_model,layer_norm_epsilon),_Norm(d_model,layer_norm_epsilon)
        self.lm_head=nn.Linear(d_model,vocab_size,bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        f=self.initializer_factor; nn.init.normal_(self.shared.weight,std=f)
        for bias in (self.encoder_bias,self.decoder_bias): nn.init.normal_(bias.weight,std=f)
        for stack in (self.encoder,self.decoder):
            for layer in stack:
                (layer.a if isinstance(layer,_Enc) else layer.sa).init_t5(self.d_model,f)
                if isinstance(layer,_Dec): layer.ca.init_t5(self.d_model,f)
                for ff in (layer.f,):
                    nn.init.normal_(ff.wi.weight,std=f*self.d_model**-.5)
                    nn.init.normal_(ff.wo.weight,std=f*ff.wo.weight.shape[1]**-.5)
                    if ff.wi_gate is not None: nn.init.normal_(ff.wi_gate.weight,std=f*self.d_model**-.5)
        if self.tie_word_embeddings: self.lm_head.weight=self.shared.weight
        else: nn.init.normal_(self.lm_head.weight, std=f)

    def encode(self, src, src_mask=None):
        if src.ndim != 2 or src.size(1)>self.context_length: raise ValueError("src must be [batch, sequence] within context")
        mask=src.ne(self.pad_token_id) if src_mask is None else src_mask.bool()
        if mask.shape!=src.shape or torch.any(~mask.any(-1)): raise ValueError("each source must have nonempty mask")
        x=F.dropout(self.shared(src),self.dropout,self.training)*mask.unsqueeze(-1)
        for layer in self.encoder: x=layer(x,mask,self.encoder_bias)*mask.unsqueeze(-1)
        return self.encoder_norm(x)*mask.unsqueeze(-1),mask

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, **kwargs):
        memory,mm=self.encode(src,src_mask)
        if tgt.ndim!=2 or tgt.size(1)==0 or tgt.size(1)>self.context_length: raise ValueError("tgt must be nonempty [batch, sequence] within context")
        mask=tgt.ne(self.pad_token_id) if tgt_mask is None else tgt_mask.bool()
        if mask.shape!=tgt.shape: raise ValueError("tgt_mask must match tgt")
        mask=mask.clone(); mask[:,0]=True
        x=F.dropout(self.shared(tgt),self.dropout,self.training)
        for layer in self.decoder: x=layer(x,memory,mask,mm,self.decoder_bias)*mask.unsqueeze(-1)
        x=self.decoder_norm(x)
        if self.tie_word_embeddings: x=x*self.d_model**-.5
        return F.linear(x,self.lm_head.weight)
