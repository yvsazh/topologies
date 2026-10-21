import math

import torch
from torch import nn


class PixArt(nn.Module):
    architecture_id = "pixart"
    def __init__(self, hidden_dim=8, text_dim=4, num_heads=2, mlp_ratio=4):
        super().__init__()
        if hidden_dim <= 0 or text_dim <= 0 or num_heads <= 0 or hidden_dim % num_heads:
            raise ValueError("hidden_dim and num_heads must be positive and hidden_dim divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.image_norm = nn.LayerNorm(hidden_dim)
        self.text_norm = nn.LayerNorm(text_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * mlp_ratio), nn.GELU(), nn.Linear(hidden_dim * mlp_ratio, hidden_dim))
        self.time_mlp = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))

    def forward(self, image_tokens, text_embedding, timestep):
        if image_tokens.ndim != 3 or text_embedding.ndim != 3:
            raise ValueError("image_tokens and text_embedding must have shape [batch, length, features]")
        if image_tokens.shape[0] != text_embedding.shape[0] or image_tokens.shape[-1] != self.hidden_dim:
            raise ValueError("batch dimensions must match and image token width must equal hidden_dim")
        if text_embedding.shape[-1] != self.text_norm.normalized_shape[0] or text_embedding.shape[1] == 0:
            raise ValueError("text embedding width must match text_dim and sequence must be nonempty")
        time = torch.as_tensor(timestep, dtype=image_tokens.dtype, device=image_tokens.device)
        if time.ndim == 0:
            time = time.expand(image_tokens.shape[0])
        time = time.reshape(image_tokens.shape[0], 1)
        shift_self, scale_self, gate_self, shift_mlp, scale_mlp, gate_mlp = self.time_mlp(time).chunk(6, dim=-1)
        normalized = self.image_norm(image_tokens) * (1 + scale_self[:, None]) + shift_self[:, None]
        self_out = self.self_attention(normalized, normalized, normalized, need_weights=False)[0]
        hidden = image_tokens + torch.tanh(gate_self[:, None]) * self_out
        text = self.text_projection(self.text_norm(text_embedding))
        cross = self.cross_attention(self.image_norm(hidden), text, text, need_weights=False)[0]
        hidden = hidden + cross
        normalized = self.mlp_norm(hidden) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        return hidden + torch.tanh(gate_mlp[:, None]) * self.mlp(normalized)
