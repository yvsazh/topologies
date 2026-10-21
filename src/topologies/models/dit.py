import math

import torch
from torch import nn


def _sincos_1d(positions, embedding_dim):
    half_dim = embedding_dim // 2
    frequencies = torch.arange(half_dim, device=positions.device, dtype=torch.float32)
    frequencies = torch.exp(-math.log(10000.0) * frequencies / half_dim)
    angles = positions.reshape(-1, 1).to(torch.float32) * frequencies.reshape(1, -1)
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


def get_2d_sincos_pos_embed(hidden_dim, grid_height, grid_width, *, device=None, dtype=None):
    if hidden_dim <= 0 or hidden_dim % 4:
        raise ValueError("hidden_dim must be a positive multiple of four")
    if grid_height <= 0 or grid_width <= 0:
        raise ValueError("grid dimensions must be positive")
    grid_x, grid_y = torch.meshgrid(
        torch.arange(grid_width, device=device),
        torch.arange(grid_height, device=device),
        indexing="xy",
    )
    embedding_dim_per_axis = hidden_dim // 2
    embedding = torch.cat(
        (
            _sincos_1d(grid_x.flatten(), embedding_dim_per_axis),
            _sincos_1d(grid_y.flatten(), embedding_dim_per_axis),
        ),
        dim=-1,
    )
    return embedding.to(dtype=dtype if dtype is not None else torch.float32)


class DiTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, conditioning_dim, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, hidden_dim))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(conditioning_dim, 6 * hidden_dim))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, tokens, condition):
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(condition).chunk(6, dim=-1)
        normalized = self.norm1(tokens) * (1 + scale_attn[:, None, :]) + shift_attn[:, None, :]
        attended = self.attn(normalized, normalized, normalized, need_weights=False)[0]
        tokens = tokens + gate_attn[:, None, :] * attended
        normalized = self.norm2(tokens) * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        return tokens + gate_mlp[:, None, :] * self.mlp(normalized)


class DiT(nn.Module):
    architecture_id = "diffusion-transformer"
    def __init__(self, in_channels=4, out_channels=4, patch_size=2, hidden_dim=128, num_heads=4, depth=4, conditioning_dim=128, mlp_ratio=4.0):
        super().__init__()
        if min(in_channels, out_channels, patch_size, hidden_dim, num_heads, depth, conditioning_dim) <= 0:
            raise ValueError("dimensions and depth must be positive")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if hidden_dim % 4:
            raise ValueError("hidden_dim must be divisible by four for 2-D sin/cos positions")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.conditioning_dim = conditioning_dim
        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.condition_embed = nn.Sequential(nn.Linear(conditioning_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, conditioning_dim))
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, num_heads, conditioning_dim, mlp_ratio) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(conditioning_dim, 2 * hidden_dim))
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        self.output_projection = nn.Linear(hidden_dim, patch_size * patch_size * out_channels)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, x, condition):
        if x.ndim != 4:
            raise ValueError("x must have shape (batch, channels, height, width)")
        batch, channels, height, width = x.shape
        if channels != self.in_channels or height % self.patch_size or width % self.patch_size:
            raise ValueError("input channels and spatial dimensions must match the patch configuration")
        if condition.ndim == 1:
            condition = condition.unsqueeze(0)
        if condition.ndim != 2 or condition.shape[-1] != self.conditioning_dim:
            raise ValueError("condition must have shape (batch, conditioning_dim)")
        if condition.shape[0] == 1 and batch != 1:
            condition = condition.expand(batch, -1)
        if condition.shape[0] != batch:
            raise ValueError("condition batch must match input batch")
        condition = self.condition_embed(condition)
        patches = self.patch_embed(x)
        grid_h, grid_w = patches.shape[-2:]
        tokens = patches.flatten(2).transpose(1, 2)
        tokens = tokens + get_2d_sincos_pos_embed(
            self.hidden_dim, grid_h, grid_w, device=tokens.device, dtype=tokens.dtype
        ).unsqueeze(0)
        for block in self.blocks:
            tokens = block(tokens, condition)
        shift, scale = self.final_modulation(condition).chunk(2, dim=-1)
        tokens = self.final_norm(tokens) * (1 + scale[:, None, :]) + shift[:, None, :]
        patches = self.output_projection(tokens).view(batch, grid_h, grid_w, self.patch_size, self.patch_size, self.out_channels)
        return patches.permute(0, 5, 1, 3, 2, 4).reshape(batch, self.out_channels, height, width)
