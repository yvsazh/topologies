import math

import torch
from torch import nn
from torch.nn import functional as F


def masked_nll(logits, targets, selected):
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise ValueError("logits must have shape (batch, sequence, vocabulary)")
    if not isinstance(targets, torch.Tensor) or targets.shape != logits.shape[:2]:
        raise ValueError("targets must match the logits batch and sequence dimensions")
    if not isinstance(selected, torch.Tensor) or selected.shape != logits.shape[:2]:
        raise ValueError("selected must match the logits batch and sequence dimensions")
    if selected.dtype != torch.bool:
        raise ValueError("selected must be a boolean tensor")
    if not selected.any():
        raise ValueError("at least one target position must be selected")
    return F.cross_entropy(logits[selected], targets[selected].long())


class ESM2(nn.Module):
    component_id = "esm"
    PRESETS = {
        "tiny": {"vocab_size": 33, "embed_dim": 32, "num_layers": 1, "num_heads": 4, "ffn_dim": 128},
        "small": {"vocab_size": 33, "embed_dim": 64, "num_layers": 2, "num_heads": 4, "ffn_dim": 256},
        "base": {"vocab_size": 33, "embed_dim": 320, "num_layers": 6, "num_heads": 20, "ffn_dim": 1280},
    }

    def __init__(
        self,
        vocab_size=33,
        embed_dim=32,
        num_layers=1,
        num_heads=4,
        ffn_dim=None,
        padding_idx=1,
        mask_token_id=32,
        token_dropout=True,
        layer_norm_eps=1e-5,
        rope_base=10000.0,
    ):
        super().__init__()
        dimensions = (vocab_size, embed_dim, num_layers, num_heads)
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in dimensions):
            raise ValueError("vocab_size, embed_dim, num_layers, and num_heads must be positive integers")
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        head_dim = embed_dim // num_heads
        if head_dim % 2:
            raise ValueError("the rotary attention head dimension must be even")
        if ffn_dim is None:
            ffn_dim = 4 * embed_dim
        if isinstance(ffn_dim, bool) or not isinstance(ffn_dim, int) or ffn_dim <= 0:
            raise ValueError("ffn_dim must be a positive integer")
        if isinstance(padding_idx, bool) or not isinstance(padding_idx, int) or not 0 <= padding_idx < vocab_size:
            raise ValueError("padding_idx must be a valid vocabulary index")
        if isinstance(mask_token_id, bool) or not isinstance(mask_token_id, int) or not 0 <= mask_token_id < vocab_size:
            raise ValueError("mask_token_id must be a valid vocabulary index")
        if padding_idx == mask_token_id:
            raise ValueError("padding_idx and mask_token_id must differ")
        if token_dropout is not True and token_dropout is not False:
            raise ValueError("token_dropout must be a boolean")
        if not isinstance(layer_norm_eps, (float, int)) or not math.isfinite(layer_norm_eps) or layer_norm_eps <= 0:
            raise ValueError("layer_norm_eps must be a finite positive number")
        if not isinstance(rope_base, (float, int)) or not math.isfinite(rope_base) or rope_base <= 0:
            raise ValueError("rope_base must be a finite positive number")
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.padding_idx = padding_idx
        self.mask_token_id = mask_token_id
        self.token_dropout = token_dropout
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.blocks = nn.ModuleList(
            nn.ModuleDict(
                {
                    "attention_norm": nn.LayerNorm(embed_dim, eps=layer_norm_eps),
                    "qkv": nn.Linear(embed_dim, 3 * embed_dim, bias=True),
                    "attention_output": nn.Linear(embed_dim, embed_dim, bias=True),
                    "ffn_norm": nn.LayerNorm(embed_dim, eps=layer_norm_eps),
                    "ffn_1": nn.Linear(embed_dim, ffn_dim, bias=True),
                    "ffn_2": nn.Linear(ffn_dim, embed_dim, bias=True),
                }
            )
            for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.lm_dense = nn.Linear(embed_dim, embed_dim, bias=True)
        self.lm_layer_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.output_projection = nn.Linear(embed_dim, vocab_size, bias=True)
        self.output_projection.weight = self.embed_tokens.weight
        inv_freq = 1.0 / (rope_base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

    def _embed_tokens(self, tokens):
        padding = tokens.eq(self.padding_idx)
        non_padding = (~padding).sum(dim=1)
        if (non_padding == 0).any():
            raise ValueError("each token row must contain at least one non-padding token")
        mask = tokens.eq(self.mask_token_id)
        if self.token_dropout:
            mask_count = mask.sum(dim=1)
            if (mask_count >= non_padding).any():
                raise ValueError("token dropout requires a non-mask token in every sequence")
            embedded = self.embed_tokens(tokens).masked_fill(mask.unsqueeze(-1), 0.0)
            observed_ratio = mask_count.to(embedded.dtype) / non_padding.to(embedded.dtype)
            scale = (1.0 - 0.15 * 0.80) / (1.0 - observed_ratio)
            embedded = embedded * scale[:, None, None]
        else:
            embedded = self.embed_tokens(tokens)
        return embedded.masked_fill(padding.unsqueeze(-1), 0.0)

    def _apply_rope(self, values):
        positions = torch.arange(values.shape[-2], device=values.device, dtype=self.rope_inv_freq.dtype)
        frequencies = torch.outer(positions, self.rope_inv_freq.to(values.device))
        angles = torch.cat((frequencies, frequencies), dim=-1)
        cosine = angles.cos().to(values.dtype)[None, None, :, :]
        sine = angles.sin().to(values.dtype)[None, None, :, :]
        first, second = values.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        return values * cosine + rotated * sine

    def _project_hidden(self, hidden):
        return self.output_projection(self.lm_layer_norm(F.gelu(self.lm_dense(hidden))))

    def forward(self, tokens):
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2:
            raise ValueError("tokens must be a rank-two tensor with shape (batch, sequence)")
        if tokens.shape[0] < 1 or tokens.shape[1] < 1:
            raise ValueError("tokens must have non-empty batch and sequence dimensions")
        if tokens.dtype != torch.long:
            raise ValueError("tokens must contain int64 vocabulary indices")
        if tokens.min().item() < 0 or tokens.max().item() >= self.vocab_size:
            raise ValueError("token IDs must be within the configured vocabulary")
        padding = tokens.eq(self.padding_idx)
        hidden = self._embed_tokens(tokens)
        batch_size, sequence_length, _ = hidden.shape
        for block in self.blocks:
            normalized = block["attention_norm"](hidden)
            qkv = block["qkv"](normalized).reshape(
                batch_size, sequence_length, 3, self.num_heads, self.head_dim
            ).permute(2, 0, 3, 1, 4)
            query, key, value = qkv.unbind(0)
            query = self._apply_rope(query)
            key = self._apply_rope(key)
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(padding[:, None, None, :], torch.finfo(scores.dtype).min)
            attended = torch.matmul(torch.softmax(scores, dim=-1), value)
            attended = attended.transpose(1, 2).reshape(batch_size, sequence_length, self.embed_dim)
            hidden = hidden + block["attention_output"](attended)
            feedforward = block["ffn_norm"](hidden)
            hidden = hidden + block["ffn_2"](F.gelu(block["ffn_1"](feedforward)))
        return self._project_hidden(self.final_norm(hidden))
