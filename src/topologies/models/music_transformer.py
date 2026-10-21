import torch
from torch import nn


class MusicTransformer(nn.Module):
    architecture_id = "music-transformer"
    def __init__(self, vocab_size=12, d_model=8, n_heads=2, max_distance=512):
        super().__init__()
        if vocab_size <= 0 or d_model <= 0 or n_heads <= 0 or max_distance < 0:
            raise ValueError("vocab_size, d_model, and n_heads must be positive; max_distance must be nonnegative")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_distance = max_distance
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.relative_key = nn.Parameter(torch.empty(n_heads, 2 * max_distance + 1, self.head_dim))
        nn.init.normal_(self.relative_key, std=self.head_dim ** -0.5)
        self.norm1 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.logit_projection = nn.Linear(d_model, vocab_size)

    def relative_attention(self, query, key, value):
        sequence_length = query.shape[-2]
        positions = torch.arange(sequence_length, device=query.device)
        relative_distance = positions[:, None] - positions[None, :]
        relative_index = relative_distance.clamp(-self.max_distance, self.max_distance) + self.max_distance
        relative_keys = self.relative_key[:, relative_index, :]
        content_scores = torch.matmul(query, key.transpose(-2, -1))
        relative_scores = torch.einsum("bhid,hijd->bhij", query, relative_keys)
        scores = (content_scores + relative_scores) / (self.head_dim ** 0.5)
        future_mask = torch.triu(torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=query.device), diagonal=1)
        scores = scores.masked_fill(future_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, value)

    def forward(self, tokens):
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (batch, sequence)")
        batch_size, sequence_length = tokens.shape
        hidden = self.token_embedding(tokens)
        shape = (batch_size, sequence_length, self.n_heads, self.head_dim)
        query = self.query(hidden).view(shape).transpose(1, 2)
        key = self.key(hidden).view(shape).transpose(1, 2)
        value = self.value(hidden).view(shape).transpose(1, 2)
        attended = self.relative_attention(query, key, value).transpose(1, 2).contiguous()
        attended = attended.view(batch_size, sequence_length, self.d_model)
        hidden = self.norm1(hidden + self.output(attended))
        hidden = self.norm2(hidden + self.feed_forward(hidden))
        return self.logit_projection(hidden)
