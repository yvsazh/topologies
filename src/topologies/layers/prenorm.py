import math

import torch
from torch import nn


class PreNormTransformerLayer(nn.Module):
    component_id = "prenorm"
    PRESETS = {
        "tiny": {"dim": 4, "ffn_dim": 4, "num_heads": 1, "norm_eps": 0.0},
        "small": {"dim": 16, "ffn_dim": 16, "num_heads": 1, "norm_eps": 0.0},
        "base": {"dim": 64, "ffn_dim": 64, "num_heads": 1, "norm_eps": 0.0},
    }

    def __init__(
        self,
        dim: int,
        ffn_dim: int | None = None,
        num_heads: int = 1,
        norm_eps: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 2:
            raise ValueError("dim must be an integer of at least 2")
        if ffn_dim is None:
            ffn_dim = dim
        if not isinstance(ffn_dim, int) or isinstance(ffn_dim, bool) or ffn_dim < 1:
            raise ValueError("ffn_dim must be a positive integer")
        if not isinstance(num_heads, int) or isinstance(num_heads, bool) or num_heads != 1:
            raise ValueError("num_heads must be 1 for the single-head layer")
        try:
            valid_norm_eps = (
                isinstance(norm_eps, (int, float))
                and not isinstance(norm_eps, bool)
                and math.isfinite(norm_eps)
                and norm_eps >= 0
            )
        except (TypeError, ValueError, OverflowError):
            valid_norm_eps = False
        if not valid_norm_eps:
            raise ValueError("norm_eps must be a finite nonnegative number")
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.norm_eps = float(norm_eps)
        self.attn_norm = nn.LayerNorm(dim, eps=self.norm_eps)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.ffn_norm = nn.LayerNorm(dim, eps=self.norm_eps)
        self.ffn_in = nn.Linear(dim, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, dim)
        for projection in (
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.out_proj,
            self.ffn_in,
            self.ffn_out,
        ):
            nn.init.xavier_normal_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or x.layout != torch.strided:
            raise ValueError("x must be a dense rank-3 floating-point tensor")
        if x.ndim != 3:
            raise ValueError("x must be a rank-3 [batch, sequence, dim] tensor")
        if x.shape[0] < 1 or x.shape[1] < 1:
            raise ValueError("x must have a nonempty batch and sequence")
        if x.shape[-1] != self.dim:
            raise ValueError("x final dimension must match dim")
        if not x.is_floating_point():
            raise ValueError("x must use a floating-point dtype")
        normalized = self.attn_norm(x)
        query = self.q_proj(normalized)
        key = self.k_proj(normalized)
        value = self.v_proj(normalized)
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.dim)
        attended = scores.softmax(dim=-1) @ value
        residual = x + self.out_proj(attended)
        normalized = self.ffn_norm(residual)
        feedforward = self.ffn_out(torch.relu(self.ffn_in(normalized)))
        return residual + feedforward
