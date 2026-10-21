import torch
import torch.nn as nn
import torch.nn.functional as F


class _Block(nn.Module):
    def __init__(self, width, heads, ff, dropout, activation, encoder=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.cross_attn = None if encoder else nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(width, ff), nn.SiLU() if activation == "swish" else nn.ReLU(), nn.Linear(ff, width))
        self.self_norm = nn.LayerNorm(width)
        self.cross_norm = None if encoder else nn.LayerNorm(width)
        self.ff_norm = nn.LayerNorm(width)
        self.dropout = dropout

    def forward(self, x, memory, mask, memory_mask, causal):
        length = x.shape[1]
        causal_mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=1) if causal else None
        x = self.self_norm(x + F.dropout(self.self_attn(x, x, x, attn_mask=causal_mask,
            key_padding_mask=~mask, need_weights=False)[0], self.dropout, self.training))
        if self.cross_attn is not None:
            x = self.cross_norm(x + F.dropout(self.cross_attn(x, memory, memory,
                key_padding_mask=~memory_mask, need_weights=False)[0], self.dropout, self.training))
        return self.ff_norm(x + F.dropout(self.ff(x), self.dropout, self.training))


class MarianMT(nn.Module):
    architecture_id = "marian"
    PRESETS = {
        "marian": dict(vocab_size=58101, d_model=512, encoder_layers=6, decoder_layers=6, heads=8, ff=2048,
                       activation="swish", pad_token_id=58100, decoder_start_token_id=58100),
        "m2m100": dict(vocab_size=128112, d_model=1024, encoder_layers=12, decoder_layers=12, heads=16, ff=4096,
                        activation="relu", pad_token_id=1, decoder_start_token_id=2),
        "nllb": dict(vocab_size=256206, d_model=1024, encoder_layers=12, decoder_layers=12, heads=16, ff=4096,
                     activation="relu", pad_token_id=1, decoder_start_token_id=2),
        "tiny": dict(d_model=32, encoder_layers=1, decoder_layers=1, heads=4, ff=64),
    }

    def __init__(self, vocab_size=128, context_length=64, d_model=32, encoder_layers=1, decoder_layers=1,
                 heads=4, ff=64, dropout=0.0, pad_token_id=0, decoder_start_token_id=None,
                 tie_word_embeddings=True, activation="relu", scale_embedding=True):
        super().__init__()
        self.vocab_size, self.context_length, self.d_model = vocab_size, context_length, d_model
        self.pad_token_id, self.tie_word_embeddings = pad_token_id, tie_word_embeddings
        self.decoder_start_token_id = pad_token_id if decoder_start_token_id is None else decoder_start_token_id
        self.activation, self.scale_embedding = activation, scale_embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.register_buffer("position", self._sinusoidal_positions(context_length, d_model), persistent=False)
        self.encoder = nn.ModuleList(_Block(d_model, heads, ff, dropout, activation, encoder=True) for _ in range(encoder_layers))
        self.decoder = nn.ModuleList(_Block(d_model, heads, ff, dropout, activation) for _ in range(decoder_layers))
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        if tie_word_embeddings:
            self.output.weight = self.embedding.weight
        self.dropout = dropout

    @classmethod
    def from_preset(cls, name, **overrides):
        values = dict(cls.PRESETS[name])
        values.setdefault("context_length", 64)
        values["dropout"] = 0.0
        values.update(overrides)
        return cls(**values)

    @staticmethod
    def _sinusoidal_positions(length, width):
        positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        divisors = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / width))
        angles = positions * divisors
        result = torch.zeros(length, width)
        result[:, 0::2] = torch.sin(angles)
        result[:, 1::2] = torch.cos(angles[:, :result[:, 1::2].shape[1]])
        return result

    def _embed(self, ids):
        positions = torch.arange(ids.shape[1], device=ids.device)
        scale = self.d_model ** 0.5 if self.scale_embedding else 1.0
        return self.embedding(ids) * scale + self.position[positions][None]

    def encode(self, src, src_mask=None):
        mask = src.ne(self.pad_token_id) if src_mask is None else src_mask.bool()
        x = self._embed(src) * mask.unsqueeze(-1)
        for layer in self.encoder:
            x = layer(x, x, mask, mask, False) * mask.unsqueeze(-1)
        return x, mask

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory, memory_mask = self.encode(src, src_mask)
        mask = tgt.ne(self.pad_token_id) if tgt_mask is None else tgt_mask.bool()
        mask = mask.clone()
        mask[:, 0] = True
        x = self._embed(tgt)
        for layer in self.decoder:
            x = layer(x, memory, mask, memory_mask, True)
        return self.output(x)


def language_layout(family, source, target, tokens):
    if family == "m2m100":
        return [tokens[source]], [tokens[target]]
    if family == "nllb":
        return [], [tokens[target]]
    if family == "marian":
        return [], []
    raise ValueError(f"unknown translation family {family!r}")
