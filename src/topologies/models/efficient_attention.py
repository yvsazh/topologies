import math
import torch
from torch import nn
import torch.nn.functional as F


def positive_features(x, projection):
    projected = torch.einsum("bhtd,hdm->bhtm", x, projection)
    norm = (x * x).sum(dim=-1, keepdim=True) / 2
    return torch.exp(projected - norm) / math.sqrt(projection.shape[-1])


def performer_attention(q, k, v, projection):
    qf = positive_features(q, projection)
    kf = positive_features(k, projection)
    kv = torch.einsum("bhtm,bhtd->bhmd", kf, v)
    normalizer = torch.einsum("bhtm,bhm->bht", qf, kf.sum(dim=2)).unsqueeze(-1)
    return torch.einsum("bhtm,bhmd->bhtd", qf, kv) / normalizer.clamp_min(1e-9)


def linear_attention(q, k, v, causal=False):
    qf = F.elu(q) + 1
    kf = F.elu(k) + 1
    if causal:
        outputs = []
        kv = torch.zeros((*k.shape[:2], k.shape[-1], v.shape[-1]), dtype=v.dtype, device=v.device)
        ks = torch.zeros((*k.shape[:2], k.shape[-1]), dtype=k.dtype, device=k.device)
        for i in range(k.shape[2]):
            kv = kv + torch.einsum("bhd,bhe->bhde", kf[:, :, i], v[:, :, i])
            ks = ks + kf[:, :, i]
            outputs.append(torch.einsum("bhd,bhde->bhe", qf[:, :, i], kv) / torch.einsum("bhd,bhd->bh", qf[:, :, i], ks).unsqueeze(-1).clamp_min(1e-9))
        return torch.stack(outputs, dim=2)
    kv = torch.einsum("bhtd,bhte->bhde", kf, v)
    ks = kf.sum(dim=2)
    return torch.einsum("bhtd,bhde->bhte", qf, kv) / torch.einsum("bhtd,bhd->bht", qf, ks).unsqueeze(-1).clamp_min(1e-9)


def lsh_buckets(x, buckets, seed):
    generator = torch.Generator(device=x.device).manual_seed(seed)
    rotations = torch.randn(x.shape[-1], buckets, generator=generator, device=x.device, dtype=x.dtype)
    return (x @ rotations).argmax(dim=-1)


def reformer_attention(q, k, v, buckets, seed):
    ids = lsh_buckets(q, buckets, seed)
    scores = torch.einsum("bhtd,bhsd->bhts", q, k) / math.sqrt(q.shape[-1])
    visible = ids.unsqueeze(-1) == ids.unsqueeze(-2)
    weights = torch.softmax(scores.masked_fill(~visible, torch.finfo(scores.dtype).min), dim=-1)
    return weights @ v, ids


def linformer_attention(q, k, v, projection_k, projection_v):
    compressed_k = torch.einsum("bhtd,htk->bhkd", k, projection_k)
    compressed_v = torch.einsum("bhtd,htk->bhkd", v, projection_v)
    weights = torch.softmax(torch.einsum("bhtd,bhkd->bhtk", q, compressed_k) / math.sqrt(q.shape[-1]), dim=-1)
    return weights @ compressed_v


class EfficientAttentionEncoder(nn.Module):
    architecture_id = "efficient-attention"
    PRESETS = {"tiny": {"vocab_size": 64, "context_length": 16, "d_model": 16, "num_heads": 2, "num_layers": 1, "projection_rank": 4, "feature_count": 32, "bucket_count": 4}, "base": {"vocab_size": 30522, "context_length": 512, "d_model": 512, "num_heads": 8, "num_layers": 6, "projection_rank": 64, "feature_count": 256, "bucket_count": 64}}

    def __init__(self, vocab_size=64, context_length=16, d_model=16, num_heads=2, num_layers=1, projection_rank=4, feature_count=32, bucket_count=4, mode="linear", dropout=0.0, seed=17):
        super().__init__()
        if d_model % num_heads or projection_rank >= context_length or mode not in {"linformer", "performer", "reformer", "linear"}:
            raise ValueError("invalid attention geometry or mode")
        self.context_length, self.num_heads, self.head_dim = context_length, num_heads, d_model // num_heads
        self.mode, self.projection_rank, self.feature_count, self.bucket_count, self.seed = mode, projection_rank, feature_count, bucket_count, seed
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(context_length, d_model)
        self.layers = nn.ModuleList([nn.ModuleDict({"qkv": nn.Linear(d_model, 3 * d_model), "out": nn.Linear(d_model, d_model), "norm": nn.LayerNorm(d_model), "ff": nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)), "norm2": nn.LayerNorm(d_model)}) for _ in range(num_layers)])
        self.projection_k = nn.Parameter(torch.randn(num_heads, context_length, projection_rank) / math.sqrt(context_length))
        self.projection_v = nn.Parameter(torch.randn(num_heads, context_length, projection_rank) / math.sqrt(context_length))
        self.register_buffer("feature_projection", torch.randn(num_heads, self.head_dim, feature_count))
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def attend(self, q, k, v):
        if self.mode == "linformer":
            return linformer_attention(q, k, v, self.projection_k[:, :k.shape[2]], self.projection_v[:, :v.shape[2]])
        if self.mode == "performer":
            return performer_attention(q, k, v, self.feature_projection)
        if self.mode == "reformer":
            return reformer_attention(q, k, v, self.bucket_count, self.seed)[0]
        return linear_attention(q, k, v)

    def forward(self, input_ids):
        x = self.embedding(input_ids) + self.position(torch.arange(input_ids.shape[1], device=input_ids.device))
        for layer in self.layers:
            b, t, _ = x.shape
            q, k, v = layer["qkv"](x).view(b, t, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            attended = self.attend(q, k, v).transpose(1, 2).reshape(b, t, -1)
            x = layer["norm"](x + layer["out"](attended))
            x = layer["norm2"](x + layer["ff"](x))
        return self.lm_head(x)
