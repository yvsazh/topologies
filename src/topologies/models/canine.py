import math

import torch
from torch import nn
from torch.nn import functional as F


class CANINE(nn.Module):
    architecture_id = "canine"
    input_geometry = "unicode-codepoints"
    PRESETS = {"tiny": {"d_model": 64, "num_layers": 2, "num_heads": 4}, "small": {"d_model": 256, "num_layers": 6, "num_heads": 8}, "base": {"d_model": 768, "num_layers": 12, "num_heads": 12}}
    BUDGET_KNOBS = {"d_model": range(32, 257, 16)}
    HASH_PRIMES = (31, 43, 59, 61, 73, 97, 103, 113, 137, 149, 157, 173, 181, 193, 211, 223)

    def __init__(self, context_length=2048, d_model=128, num_layers=2, num_heads=4, intermediate_size=None, hash_buckets=16384, num_hashes=8, downsample_rate=4, local_block_size=128, dropout=0.0, pad_codepoint=0, cls_codepoint=2, mask_codepoint=0xF0000, preset=None):
        super().__init__()
        if preset is not None:
            if preset not in self.PRESETS:
                raise ValueError("preset must be tiny, small, or base")
            cfg = self.PRESETS[preset]
            d_model, num_layers, num_heads = cfg["d_model"], cfg["num_layers"], cfg["num_heads"]
        intermediate_size = 4 * d_model if intermediate_size is None else intermediate_size
        for name, value in (("context_length", context_length), ("d_model", d_model), ("num_layers", num_layers), ("num_heads", num_heads), ("intermediate_size", intermediate_size), ("hash_buckets", hash_buckets), ("num_hashes", num_hashes), ("downsample_rate", downsample_rate), ("local_block_size", local_block_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if num_hashes > 16:
            raise ValueError("num_hashes must be at most 16")
        if d_model % num_heads or d_model % num_hashes:
            raise ValueError("d_model must be divisible by num_heads and num_hashes")
        if local_block_size % 2:
            raise ValueError("local_block_size must be even")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")
        for name, value in (("pad_codepoint", pad_codepoint), ("cls_codepoint", cls_codepoint), ("mask_codepoint", mask_codepoint)):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0x10FFFF:
                raise ValueError(f"{name} must be a Unicode codepoint")
        if len({pad_codepoint, cls_codepoint, mask_codepoint}) != 3:
            raise ValueError("pad, CLS and mask codepoints must be distinct")
        self.context_length, self.d_model, self.num_layers = context_length, d_model, num_layers
        self.hash_buckets, self.num_hashes, self.downsample_rate = hash_buckets, num_hashes, downsample_rate
        self.pad_codepoint, self.cls_codepoint, self.mask_codepoint = pad_codepoint, cls_codepoint, mask_codepoint
        self.hash_embeddings = nn.ModuleList([nn.Embedding(hash_buckets, d_model // num_hashes) for _ in range(num_hashes)])
        self.position_embeddings = nn.Embedding(context_length + 1, d_model)
        self.segment_embeddings = nn.Embedding(2, d_model)
        self.embedding_norm = nn.LayerNorm(d_model)
        self.local = EncoderLayer(d_model, num_heads, intermediate_size, dropout, local_block_size)
        self.down = nn.Conv1d(d_model, d_model, kernel_size=downsample_rate, stride=downsample_rate)
        self.down_norm = nn.LayerNorm(d_model)
        self.deep = nn.ModuleList([EncoderLayer(d_model, num_heads, intermediate_size, dropout) for _ in range(num_layers)])
        self.up = nn.Conv1d(2 * d_model, d_model, kernel_size=downsample_rate, stride=1, padding=0)
        self.up_norm = nn.LayerNorm(d_model)
        self.final = EncoderLayer(d_model, num_heads, intermediate_size, dropout)
        self.mlm_transform = nn.Linear(d_model, d_model)
        self.mlm_norm = nn.LayerNorm(d_model)
        self.autoregressive = EncoderLayer(d_model, num_heads, intermediate_size, dropout, causal=True)
        self.target_context_projection = nn.Linear(2 * d_model, d_model)
        self.character_classifier = nn.Linear(d_model, hash_buckets)

    @staticmethod
    def hash_codepoints(codepoints, hash_buckets, hash_index):
        if hash_buckets <= 0 or hash_index < 0:
            raise ValueError("hash_buckets must be positive and hash_index nonnegative")
        if hash_index >= 16:
            raise ValueError("hash_index must be below 16")
        prime = CANINE.HASH_PRIMES[hash_index]
        return ((codepoints.to(torch.int64) + 1) * prime) % hash_buckets

    def embed_codepoints(self, codepoints):
        parts = [table(self.hash_codepoints(codepoints, self.hash_buckets, i)) for i, table in enumerate(self.hash_embeddings)]
        return torch.cat(parts, dim=-1)

    def encode(self, codepoints, mask=None, segments=None):
        if not isinstance(codepoints, torch.Tensor) or codepoints.ndim != 2 or codepoints.dtype not in (torch.int32, torch.int64):
            raise ValueError("codepoints must be an integer tensor [batch, time]")
        batch, length = codepoints.shape
        if not batch or not length or length > self.context_length:
            raise ValueError("non-empty codepoints must fit context_length")
        if bool((codepoints < 0).any()) or bool((codepoints > 0x10FFFF).any()):
            raise ValueError("codepoints must be Unicode scalar values")
        if mask is None:
            valid = codepoints.ne(self.pad_codepoint)
        else:
            if (not isinstance(mask, torch.Tensor) or mask.shape != codepoints.shape
                    or mask.device != codepoints.device
                    or not bool(((mask == 0) | (mask == 1)).all())):
                raise ValueError("mask must match codepoints/device and contain only 0/1 values")
            valid = mask.bool()
        if not bool(valid.any(dim=1).all()) or not bool(valid[:, 0].all()):
            raise ValueError("each sequence must contain a valid CLS character at position zero")
        if segments is None:
            segments = torch.zeros_like(codepoints)
        if (not isinstance(segments, torch.Tensor) or segments.shape != codepoints.shape
                or segments.device != codepoints.device or segments.dtype not in (torch.int32, torch.int64)
                or bool(((segments < 0) | (segments > 1)).any())):
            raise ValueError("segments must contain integer 0/1 values and match input/device")
        if not bool((codepoints[:, 0] == self.cls_codepoint).all()):
            raise ValueError("CANINE inputs must begin with the configured CLS codepoint")
        positions = torch.arange(length, device=codepoints.device).unsqueeze(0)
        x = self.embedding_norm(self.embed_codepoints(codepoints) + self.position_embeddings(positions) + self.segment_embeddings(segments))
        x = x * valid.unsqueeze(-1)
        local = self.local(x, valid)
        padded = (-local.shape[1]) % self.downsample_rate
        down_mask = F.pad(valid, (0, padded), value=False)
        down_x = F.pad(local, (0, 0, 0, padded))
        char_groups = self.down_norm(F.gelu(self.down(down_x.transpose(1, 2)).transpose(1, 2)))
        groups = F.max_pool1d(down_mask.float().unsqueeze(1), self.downsample_rate, self.downsample_rate).squeeze(1).bool()
        compressed = torch.cat((local[:, :1], char_groups[:, :-1]), dim=1)
        groups = torch.cat((valid[:, :1], groups[:, :-1]), dim=1)
        for layer in self.deep:
            compressed = layer(compressed, groups)
        expanded_body = compressed[:, 1:].repeat_interleave(self.downsample_rate, dim=1)
        remainder = local.shape[1] % self.downsample_rate
        expanded_tail = compressed[:, -1:].expand(-1, self.downsample_rate + remainder, -1)
        expanded = torch.cat((expanded_body, expanded_tail), dim=1)[:, :local.shape[1]]
        fused_in = torch.cat((local, expanded), dim=-1).transpose(1, 2)
        fused_in = F.pad(fused_in, ((self.downsample_rate - 1) // 2, self.downsample_rate // 2))
        fused = self.up_norm(F.gelu(self.up(fused_in).transpose(1, 2)))
        output = self.final(fused, valid)
        return output * valid.unsqueeze(-1)

    def forward(self, codepoints, mask=None, segments=None):
        hidden = self.encode(codepoints, mask, segments)
        logits = self.character_classifier(self.mlm_norm(F.gelu(self.mlm_transform(hidden))))
        valid = codepoints.ne(self.pad_codepoint) if mask is None else mask.bool()
        return logits * valid.unsqueeze(-1)

    def forward_features(self, codepoints, mask=None, segments=None):
        return self.encode(codepoints, mask, segments)

    def mlm_loss(self, codepoints, targets, prediction_mask, input_mask=None, segments=None, permutation=None):
        if (not isinstance(targets, torch.Tensor) or targets.shape != codepoints.shape
                or targets.device != codepoints.device
                or targets.dtype not in (torch.int32, torch.int64)):
            raise ValueError("targets must be an integer tensor matching codepoints and device")
        if bool(((targets < 0) | (targets > 0x10FFFF)).any()):
            raise ValueError("targets must contain Unicode codepoints")
        if (not isinstance(prediction_mask, torch.Tensor)
                or prediction_mask.shape != codepoints.shape
                or prediction_mask.device != codepoints.device
                or prediction_mask.dtype not in (torch.bool, torch.int32, torch.int64)
                or not bool(((prediction_mask == 0) | (prediction_mask == 1)).all())):
            raise ValueError("prediction_mask must match codepoints and contain only boolean/0/1 values")
        selected = prediction_mask.bool()
        if not bool(selected.any()):
            raise ValueError("prediction_mask must select at least one character")
        if bool(selected[:, 0].any()):
            raise ValueError("CANINE-C does not predict the leading CLS character")
        hidden = self.encode(codepoints, input_mask, segments)
        valid = codepoints.ne(self.pad_codepoint) if input_mask is None else input_mask.bool()
        if bool((selected & ~valid).any()):
            raise ValueError("prediction_mask may select only valid input characters")
        if not bool(codepoints[selected].eq(self.mask_codepoint).all()):
            raise ValueError("every selected CANINE-C target must be replaced by the mask codepoint")
        batch, length = targets.shape
        if permutation is None:
            random_keys = torch.rand((batch, length), device=targets.device).masked_fill(~selected, 2.0)
            order = torch.argsort(random_keys, dim=1)
            counts = selected.sum(1)
            ranks = torch.arange(length, device=targets.device).expand(batch, -1)
            permutation = torch.where(ranks < counts[:, None], order, torch.full_like(order, -1))
        if (not isinstance(permutation, torch.Tensor) or permutation.shape != targets.shape
                or permutation.dtype != torch.long or permutation.device != targets.device):
            raise ValueError("permutation must be LongTensor [batch, characters] on the target device")
        counts = selected.sum(1)
        ranks = torch.arange(length, device=targets.device).expand(batch, -1)
        prefix = ranks < counts[:, None]
        if bool((permutation[~prefix] != -1).any()):
            raise ValueError("permutation padding after each selected-target prefix must be -1")
        for row in range(batch):
            count = int(counts[row])
            indices = permutation[row, :count]
            if bool(((indices < 0) | (indices >= length)).any()):
                raise ValueError("permutation prefix indices must be inside the character sequence")
            if torch.unique(indices).numel() != count:
                raise ValueError("permutation prefix indices must be unique")
            if not bool(selected[row].index_select(0, indices).all()):
                raise ValueError("permutation prefix must contain exactly selected masked-target positions")
        gather = permutation.clamp_min(0)
        shuffled_gold = targets.gather(1, gather)
        shuffled_hidden = hidden.gather(1, gather.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
        context = shuffled_hidden
        decoder_valid = prefix
        logits = self.autoregressive_logits(context, shuffled_gold, decoder_valid)
        labels = shuffled_gold.remainder(self.hash_buckets)
        selected_order = decoder_valid
        return F.cross_entropy(logits[selected_order], labels[selected_order])

    def autoregressive_logits(self, context, gold_targets, valid):
        target_features = self.embed_codepoints(gold_targets)
        shifted_targets = torch.cat((torch.zeros_like(target_features[:, :1]), target_features[:, :-1]), dim=1)
        decoder_input = F.gelu(self.target_context_projection(torch.cat((shifted_targets, context), dim=-1)))
        decoded = self.autoregressive(decoder_input, valid)
        return self.character_classifier(decoded)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, intermediate_size, dropout, local_block_size=None, causal=False):
        super().__init__()
        self.num_heads, self.head_dim, self.local_block_size, self.causal = num_heads, d_model // num_heads, local_block_size, causal
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.qkv, self.proj = nn.Linear(d_model, 3 * d_model), nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, intermediate_size), nn.GELU(), nn.Linear(intermediate_size, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, valid):
        b, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, length, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        positions = torch.arange(length, device=x.device)
        allowed = valid[:, None, None, :]
        if self.local_block_size is not None:
            block = positions // self.local_block_size
            allowed = allowed & (block[:, None] == block[None, :])[None, None]
        if self.causal:
            allowed = allowed & (positions[None, :] <= positions[:, None])[None, None]
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        weights = scores.softmax(-1)
        y = (weights @ v).transpose(1, 2).reshape(b, length, width)
        x = x + self.dropout(self.proj(y))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x * valid.unsqueeze(-1)
