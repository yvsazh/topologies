"""Reusable building blocks that follow the constructor rules.

Rules every block here (and every future architecture) follows:

1. Geometry is a constructor argument (``max_seq``, ``img_size``,
   ``patch_size``, ``in_channels``, ...) — never a hidden constant.
2. Constraints are validated in ``__init__`` with ``ConfigError`` naming the fix.
3. Regularization is configurable (``dropout=0.0`` keeps behavior identical).
4. Token models accept an optional padding ``mask`` (``1`` = real token).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from topologies.build.validation import (
    ConfigError,
    check_choice,
    check_divisible,
    check_positive,
    check_range,
)

ACTIVATIONS = ("relu", "gelu", "silu", "tanh", "swiglu", "geglu")
NORMS = ("layer", "rms", "none")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        check_positive("RMSNorm", dim=dim)
        from topologies.layers.rmsnorm import RMSNorm as ReviewedRMSNorm
        self.core = ReviewedRMSNorm(dim, eps=eps, affine=True)
        self.eps = self.core.eps

    @property
    def weight(self):
        return self.core.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(x)


def make_norm(kind: str | Callable[[int], nn.Module], dim: int) -> nn.Module:
    """``"layer"`` / ``"rms"`` / ``"none"``, or a factory ``norm(dim) -> nn.Module``."""
    if callable(kind):
        module = kind(dim)
        if not isinstance(module, nn.Module):
            raise ConfigError(f"norm factory returned {type(module).__name__}, not a Module.",
                              owner="make_norm")
        return module
    check_choice("norm", kind, NORMS)
    if kind == "layer":
        return nn.LayerNorm(dim)
    if kind == "rms":
        return RMSNorm(dim)
    return nn.Identity()


def make_activation(kind: str) -> nn.Module:
    check_choice("activation", kind, ("relu", "gelu", "silu", "tanh"))
    return {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh}[kind]()


class FeedForward(nn.Module):
    """Position-wise MLP; ``swiglu``/``geglu`` use a gated hidden layer."""

    def __init__(self, dim: int, hidden: int, *, activation: str = "gelu",
                 dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        check_positive("FeedForward", dim=dim, hidden=hidden)
        check_choice("activation", activation, ACTIVATIONS, owner="FeedForward")
        from topologies.layers.feed_forward import FeedForward as ReviewedFeedForward
        self.core = ReviewedFeedForward(dim, hidden, activation=activation,
                                        dropout=dropout, bias=bias)
        self.gated = activation in ("swiglu", "geglu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(x)


class RotaryEmbedding(nn.Module):
    """Rotary position embedding (RoPE) for ``[B, H, T, head_dim]`` tensors."""

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ConfigError(f"RoPE needs an even head_dim; got {head_dim}.",
                              hint="choose dim/heads so dim // heads is even.",
                              owner="RotaryEmbedding")
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        T = x.shape[-2]
        pos = torch.arange(offset, offset + T, device=x.device, dtype=torch.float32)
        freqs = torch.outer(pos, self.inv_freq.to(x.device))
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype)
        half = x.shape[-1] // 2
        rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
        return x * cos + rotated * sin


def padding_attention_mask(mask: torch.Tensor | None, T: int, *, causal: bool,
                            device: torch.device) -> torch.Tensor | None:
    """Boolean ``[B|1, 1, T, T]`` mask (True = attend) from a ``[B, T]`` padding mask."""
    allowed = None
    if causal:
        allowed = torch.ones(T, T, dtype=torch.bool, device=device).tril().view(1, 1, T, T)
    if mask is not None:
        keep = mask.to(device=device).bool().view(mask.shape[0], 1, 1, T)
        allowed = keep if allowed is None else (allowed & keep)
        # Every query must attend to at least itself, or softmax yields NaN rows.
        allowed = allowed | torch.eye(T, dtype=torch.bool, device=device).view(1, 1, T, T)
    return allowed


class MultiHeadAttention(nn.Module):
    """Self-attention with optional causal masking, padding mask, GQA and RoPE."""

    def __init__(self, dim: int, heads: int, *, kv_heads: int | None = None,
                 causal: bool = False, dropout: float = 0.0, bias: bool = True,
                 rope: bool = False) -> None:
        super().__init__()
        owner = "MultiHeadAttention"
        check_positive(owner, dim=dim, heads=heads, kv_heads=kv_heads)
        check_divisible(("dim", dim), ("heads", heads), owner=owner)
        kv_heads = kv_heads or heads
        check_divisible(("heads", heads), ("kv_heads", kv_heads), owner=owner)
        check_range("dropout", dropout, 0.0, 1.0, owner=owner)
        self.dim, self.heads, self.kv_heads = dim, heads, kv_heads
        self.head_dim = dim // heads
        self.causal, self.dropout = causal, float(dropout)
        from topologies.layers.multi_head_attention import MultiHeadAttention as ReviewedAttention
        self.core = ReviewedAttention(dim, heads, kv_heads=kv_heads, dropout=dropout, causal=causal)
        self.rope = None
        if rope:
            from topologies.layers.rotary_embedding import RotaryEmbedding as ReviewedRotary
            self.rope = ReviewedRotary(self.head_dim)
            self.core.q.register_forward_hook(self._rotate_projection(heads))
            self.core.k.register_forward_hook(self._rotate_projection(kv_heads))
        if not bias:
            for projection in (self.core.q, self.core.k, self.core.v, self.core.out):
                projection.register_parameter("bias", None)

    def _rotate_projection(self, heads: int):
        def rotate(module, inputs, output):
            batch, length, _ = output.shape
            shaped = output.reshape(batch, length, heads, self.head_dim).transpose(1, 2)
            return self.rope(shaped).transpose(1, 2).reshape(batch, length, -1)
        return rotate

    @property
    def q(self):
        return self.core.q

    @property
    def kv(self):
        return (self.core.k, self.core.v)

    @property
    def out(self):
        return self.core.out

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.core(x, mask=mask)


class TransformerBlock(nn.Module):
    """Pre-norm transformer layer: ``x + attn(norm(x))``, ``x + ffn(norm(x))``.

    ``attention`` / ``ffn`` / ``norm`` are slots: a name (``"mha"``, ``"mlp"``,
    ``"layer"`` ...) or a factory (see ``topologies.nn.slot``).
    """

    def __init__(self, dim: int, heads: int, *, kv_heads: int | None = None,
                 mlp_ratio: float = 4.0, activation: str = "gelu",
                 norm: str | Callable = "layer", dropout: float = 0.0, causal: bool = False,
                 rope: bool = False, bias: bool = True, attention: str | Callable = "mha",
                 ffn: str | Callable = "mlp") -> None:
        super().__init__()
        from topologies.nn.slots import build_attention, build_ffn

        check_positive("TransformerBlock", mlp_ratio=mlp_ratio)
        self.n1 = make_norm(norm, dim)
        self.attn = build_attention(attention, dim, heads, kv_heads=kv_heads, causal=causal,
                                    dropout=dropout, bias=bias, rope=rope)
        self.n2 = make_norm(norm, dim)
        self.ffn = build_ffn(ffn, dim, max(1, int(dim * mlp_ratio)), activation=activation,
                             dropout=dropout, bias=bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.n1(x), mask)
        return x + self.ffn(self.n2(x))


class LearnedPositionalEmbedding(nn.Module):
    """Learned absolute positions with an explicit, validated ``max_len``."""

    def __init__(self, max_len: int, dim: int) -> None:
        super().__init__()
        check_positive("LearnedPositionalEmbedding", max_len=max_len, dim=dim)
        self.max_len = int(max_len)
        self.emb = nn.Embedding(self.max_len, dim)

    def forward(self, T: int, device: torch.device | None = None) -> torch.Tensor:
        if T > self.max_len:
            raise ConfigError(
                f"sequence length {T} exceeds max_seq={self.max_len}.",
                hint=f"build the model with max_seq>={T}, or feed shorter blocks "
                     f"(e.g. lm_dataset(..., block_size={self.max_len})).",
                owner="LearnedPositionalEmbedding",
            )
        return self.emb(torch.arange(T, device=device or self.emb.weight.device))


class PatchEmbedding(nn.Module):
    """Image -> patch tokens via a strided convolution; geometry is validated."""

    def __init__(self, img_size: int, patch_size: int, in_channels: int, dim: int) -> None:
        super().__init__()
        owner = "PatchEmbedding"
        check_positive(owner, img_size=img_size, patch_size=patch_size,
                       in_channels=in_channels, dim=dim)
        if img_size % patch_size:
            divisors = [p for p in range(1, img_size + 1) if img_size % p == 0]
            near = sorted(divisors, key=lambda p: abs(p - patch_size))[:3]
            raise ConfigError(
                f"img_size={img_size} is not divisible by patch_size={patch_size}.",
                hint=f"use patch_size in {sorted(near)} or resize images.", owner=owner,
            )
        self.img_size, self.patch_size, self.in_channels = img_size, patch_size, in_channels
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(in_channels, dim, patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if x.dim() != 4 or x.shape[1] != self.in_channels:
            raise ConfigError(
                f"expected images [B, {self.in_channels}, H, W]; got {tuple(x.shape)}.",
                hint="set in_channels to your image channel count (1 for grayscale).",
                owner="PatchEmbedding",
            )
        H, W = x.shape[-2:]
        if H % self.patch_size or W % self.patch_size:
            raise ConfigError(
                f"image size {H}x{W} is not divisible by patch_size={self.patch_size}.",
                owner="PatchEmbedding",
            )
        h = self.proj(x)
        return h.flatten(2).transpose(1, 2), (h.shape[-2], h.shape[-1])


def resize_pos_embedding(pos: torch.Tensor, old_grid: int, new_grid: tuple[int, int],
                         extra_tokens: int = 1) -> torch.Tensor:
    """Bicubically resample a ``[1, extra + g*g, D]`` table to a new patch grid."""
    if (old_grid, old_grid) == tuple(new_grid):
        return pos
    extra, grid = pos[:, :extra_tokens], pos[:, extra_tokens:]
    D = grid.shape[-1]
    grid = grid.reshape(1, old_grid, old_grid, D).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=new_grid, mode="bicubic", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, new_grid[0] * new_grid[1], D)
    return torch.cat([extra, grid], dim=1)


class ResidualConvBlock(nn.Module):
    """Basic ResNet block (two 3x3 convs) with optional stride/width change."""

    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1,
                 dropout: float = 0.0) -> None:
        super().__init__()
        check_positive("ResidualConvBlock", in_ch=in_ch, out_ch=out_ch, stride=stride)
        check_range("dropout", dropout, 0.0, 1.0)
        from topologies.layers.residual_blocks import ResidualBlock as ReviewedResidualBlock
        self.core = ReviewedResidualBlock(in_ch, out_ch, stride=stride, kind="basic", dimension=2)
        if dropout:
            self.core.body = nn.Sequential(*list(self.core.body.children())[:3], nn.Dropout2d(dropout),
                                            *list(self.core.body.children())[3:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(x)


def init_weights(module: nn.Module, std: float = 0.02) -> None:
    """GPT-style init for Linear/Embedding (zero bias); used by the stacks."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=std)


__all__ = [
    "ACTIVATIONS",
    "NORMS",
    "FeedForward",
    "LearnedPositionalEmbedding",
    "MultiHeadAttention",
    "PatchEmbedding",
    "RMSNorm",
    "ResidualConvBlock",
    "RotaryEmbedding",
    "TransformerBlock",
    "init_weights",
    "make_activation",
    "make_norm",
    "padding_attention_mask",
    "resize_pos_embedding",
]
