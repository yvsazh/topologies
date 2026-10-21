"""Generic, size-configurable model stacks built from ``topologies.nn.layers``.

These are reusable constructor templates, not paper reproductions: every
dimension is a constructor argument, every one has ``tiny``/``small``/``base``
presets and declared budget knobs, so

    build(TransformerLM, "small", vocab_size=tok.vocab_size)
    build(ConvNet, target_params="50k", num_classes=10)

work the same way they do for your own architectures.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from topologies.build.config import Configurable
from topologies.build.presets import presets, register_budget_knobs
from topologies.build.validation import (
    ConfigError,
    check_choice,
    check_positive,
    check_range,
)
from topologies.models.mlp import MLP as _MLPCore
from topologies.models.recurrent_models import RecurrentPredictor as _RecurrentPredictorCore
from topologies.models.resnet import ResNet as _ResNetCore
from topologies.models.tabular_mlp import TabularMLP as _TabularMLPCore
from topologies.models.temporal_convnet import TemporalConvNet as _TemporalConvNetCore
from topologies.models.transformer_models import TransformerClassifier as _TransformerClassifierCore
from topologies.models.transformer_models import TransformerLM as _TransformerLMCore
from topologies.models.transformer_models import TransformerTagger as _TransformerTaggerCore
from topologies.models.vision_transformer import VisionTransformer as _VisionTransformerCore
from topologies.nn.layers import make_norm
from topologies.nn.slots import build_attention, build_ffn


class _CoreAttentionAdapter(nn.Module):
    """Adapt the toolbox attention slot protocol to the curated text core."""
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor, valid: torch.Tensor, causal: bool = False) -> torch.Tensor:
        return self.module(x, mask=valid)


class _VisionAttentionAdapter(nn.Module):
    """Adapt a sequence attention slot to torch MultiheadAttention's call shape."""
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, query, key=None, value=None, need_weights=False, **kwargs):
        return self.module(query, mask=None), None


def _configure_text_core(core: nn.Module, *, dim: int, heads: int, kv_heads: int | None,
                         mlp_ratio: float, activation: str, norm: str | Callable,
                         dropout: float, attention: str | Callable, ffn: str | Callable,
                         causal: bool, rope: bool = False, bias: bool = True) -> None:
    """Install optional reviewed toolbox components into a curated encoder core."""

    enc = core.encoder
    for block in enc.blocks:
        if attention != "mha" or kv_heads is not None:
            block.attn = _CoreAttentionAdapter(build_attention(
                attention, dim, heads, kv_heads=kv_heads, causal=causal,
                dropout=dropout, rope=rope, bias=bias))
        if ffn != "mlp" or mlp_ratio != 4.0 or activation != "gelu":
            block.ff = build_ffn(ffn, dim, int(dim * mlp_ratio), activation=activation,
                                 dropout=dropout, bias=bias)
        if norm != "layer":
            block.n1 = make_norm(norm, dim)
            block.n2 = make_norm(norm, dim)
    if norm != "layer":
        enc.norm = make_norm(norm, dim)


@presets(
    tiny={"width": 64, "depth": 1},
    small={"width": 256, "depth": 2},
    base={"width": 1024, "depth": 3},
)
class MLP(Configurable, _MLPCore):
    """Multi-layer perceptron. Flattens ``[B, ...]`` inputs by default.

    ``depth`` hidden layers of ``width`` units; works for tabular data and
    small images alike (``MLP(28 * 28, 10)`` for MNIST).
    """

    def __init__(self, in_features: int, out_features: int, *, width: int = 256,
                 depth: int = 2, activation: str = "relu", dropout: float = 0.0,
                 norm: str = "none", flatten: bool = True) -> None:
        owner = "MLP"
        check_positive(owner, in_features=in_features, out_features=out_features, width=width)
        if depth < 0:
            raise ConfigError(f"depth must be >= 0; got {depth}.", owner=owner)
        check_range("dropout", dropout, 0.0, 1.0, owner=owner)
        self.flatten = flatten
        _MLPCore.__init__(self, in_features, out_features, (width,) * depth or (width,),
                          activation, norm, dropout)
        if depth == 0:
            self.network = nn.Sequential(nn.Linear(in_features, out_features))
        self.net = self.network

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.flatten and x.ndim > 2:
            x = x.flatten(1)
        if x.shape[-1] != self.in_features:
            raise ConfigError(
                f"expected {self.in_features} input features; got {x.shape[-1]} "
                f"(input shape {tuple(x.shape)}).",
                hint="set in_features to the flattened input size, e.g. 28*28 for MNIST.",
                owner="MLP",
            )
        return self.net(x)


register_budget_knobs(MLP, width=range(4, 4097, 4), depth=range(9))


@presets(
    tiny={"width": 16, "stages": 2, "blocks_per_stage": 1},
    small={"width": 32, "stages": 3, "blocks_per_stage": 2},
    base={"width": 64, "stages": 4, "blocks_per_stage": 2},
)
class ConvNet(Configurable, nn.Module):
    """ResNet-style image classifier: stem, ``stages`` of residual blocks, GAP, head.

    Each stage after the first halves the resolution and doubles the width.
    Any input size works (global average pooling).
    """

    __topologies_spec__ = {"recipe": "vision-classify"}

    def __init__(self, num_classes: int, *, in_channels: int = 3, width: int = 32,
                 stages: int = 3, blocks_per_stage: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        owner = "ConvNet"
        check_positive(owner, num_classes=num_classes, in_channels=in_channels, width=width,
                       stages=stages, blocks_per_stage=blocks_per_stage)
        if isinstance(stages, bool) or stages < 1:
            raise ConfigError("stages must be a positive integer", owner=owner)
        check_range("dropout", dropout, 0.0, 1.0, owner=owner)
        self.in_channels, self.num_classes = in_channels, num_classes
        self.core = _ResNetCore(in_channels, num_classes, block="basic",
            layers=(blocks_per_stage,) * stages,
            widths=tuple(width * 2 ** i for i in range(stages)), dropout=dropout)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.dim() != 4 or pixels.shape[1] != self.in_channels:
            raise ConfigError(
                f"expected images [B, {self.in_channels}, H, W]; got {tuple(pixels.shape)}.",
                hint="set in_channels to your image channel count (1 for grayscale).",
                owner="ConvNet",
            )
        return self.core(pixels)


register_budget_knobs(ConvNet, width=range(4, 257, 4), blocks_per_stage=range(1, 5))


@presets(
    tiny={"dim": 128, "layers": 4, "heads": 4},
    small={"dim": 384, "layers": 6, "heads": 6},
    base={"dim": 768, "layers": 12, "heads": 12},
)
class TransformerLM(Configurable, nn.Module):
    """Decoder-only causal language model with configurable everything.

    ``forward(tokens, mask=None) -> logits [B, T, vocab_size]``. ``mask`` is a
    ``[B, T]`` padding mask (1 = real token). ``pos="learned"`` limits inputs
    to ``max_seq`` tokens (checked with a clear error); ``pos="rope"`` uses
    rotary embeddings (``max_seq`` is then the documented generation window).
    ``attention`` / ``ffn`` / ``norm`` accept names or factories
    (``topologies.nn.slot(SlidingWindowAttention, window=64)``).
    """

    __topologies_spec__ = {"recipe": "causal-lm"}

    def __init__(self, vocab_size: int, *, max_seq: int = 256, dim: int = 256,
                 layers: int = 4, heads: int = 4, kv_heads: int | None = None,
                 mlp_ratio: float = 4.0, activation: str = "gelu",
                 norm: str | Callable = "layer", pos: str = "learned", dropout: float = 0.0,
                 tie_embeddings: bool = True, bias: bool = True,
                 attention: str | Callable = "mha", ffn: str | Callable = "mlp") -> None:
        super().__init__()
        owner = "TransformerLM"
        check_positive(owner, vocab_size=vocab_size, max_seq=max_seq, dim=dim, layers=layers,
                       heads=heads)
        if dim % heads:
            raise ConfigError("dim must be divisible by heads", owner=owner)
        if kv_heads is not None and (kv_heads < 1 or kv_heads > heads or heads % kv_heads):
            raise ConfigError("kv_heads must be a positive divisor of heads", owner=owner)
        check_choice("pos", pos, ("learned", "rope"), owner=owner)
        check_range("dropout", dropout, 0.0, 1.0, owner=owner)
        self.vocab_size, self.max_seq, self.dim = vocab_size, max_seq, dim
        self.core = _TransformerLMCore(vocab_size, max_seq, dim, layers, heads,
                                       dropout=dropout, position=pos)
        _configure_text_core(self.core, dim=dim, heads=heads, kv_heads=kv_heads,
            mlp_ratio=mlp_ratio, activation=activation, norm=norm, dropout=dropout,
            attention=attention, ffn=ffn, causal=True, rope=pos == "rope", bias=bias)
        self.tok = self.core.encoder.embedding
        self.head = self.core.lm_head
        # The public bias flag controls transformer sublayer projections.  The
        # historical vocabulary projection is bias-free independently of it.
        self.head.bias = None
        if not bias:
            for module in self.core.encoder.modules():
                if isinstance(module, nn.Linear):
                    module.register_parameter("bias", None)
        if tie_embeddings:
            self.head.weight = self.tok.weight

    @property
    def context_length(self) -> int:
        return self.max_seq

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.ndim != 2 or not 0 < tokens.shape[1] <= self.max_seq:
            raise ConfigError(f"tokens must have shape [B,T] with 1 <= T <= max_seq={self.max_seq}", owner="TransformerLM")
        return self.core(tokens, mask)


register_budget_knobs(TransformerLM, dim=range(32, 4097, 32), layers=range(1, 49))


@presets(
    tiny={"dim": 96, "layers": 4, "heads": 3},
    small={"dim": 192, "layers": 6, "heads": 3},
    base={"dim": 384, "layers": 8, "heads": 6},
)
class VisionTransformer(Configurable, nn.Module):
    """ViT image classifier with validated geometry.

    Patch count comes from ``(img_size // patch_size) ** 2``; other input
    resolutions (divisible by ``patch_size``) are supported by bicubic
    resampling of the position table.
    """

    __topologies_spec__ = {"recipe": "vision-classify"}

    def __init__(self, num_classes: int, *, img_size: int = 32, patch_size: int = 4,
                 in_channels: int = 3, dim: int = 192, layers: int = 6, heads: int = 3,
                 mlp_ratio: float = 4.0, activation: str = "gelu",
                 norm: str | Callable = "layer", dropout: float = 0.0, pool: str = "cls",
                 attention: str | Callable = "mha", ffn: str | Callable = "mlp") -> None:
        super().__init__()
        owner = "VisionTransformer"
        check_positive(owner, num_classes=num_classes, in_channels=in_channels,
                       dim=dim, layers=layers, heads=heads)
        if dim % heads:
            raise ConfigError("dim must be divisible by heads", owner=owner)
        check_choice("pool", pool, ("cls", "mean"), owner=owner)
        check_range("dropout", dropout, 0.0, 1.0, owner=owner)
        def geometry_pair(value):
            if type(value) is int:
                return (value, value)
            if isinstance(value, (tuple, list)) and len(value) == 2:
                return tuple(value)
            return ()
        image_pair, patch_pair = geometry_pair(img_size), geometry_pair(patch_size)
        if len(image_pair) != 2 or len(patch_pair) != 2 or any(
            isinstance(x, bool) or not isinstance(x, int) or x < 1
            for x in (*image_pair, *patch_pair)
        ):
            raise ConfigError("img_size and patch_size must be positive integers or pairs", owner=owner)
        if any(size % patch for size, patch in zip(image_pair, patch_pair)):
            raise ConfigError("img_size must be divisible by patch_size", owner=owner)
        self.num_classes, self.pool = num_classes, pool
        self.patch_size = patch_pair
        self.core = _VisionTransformerCore(in_channels=in_channels, num_classes=num_classes,
            img_size=img_size, patch_size=patch_size, embed_dim=dim, depth=layers,
            num_heads=heads, mlp_ratio=mlp_ratio, dropout=dropout)
        for block in self.core.blocks:
            if attention != "mha":
                block.attn = _VisionAttentionAdapter(build_attention(
                    attention, dim, heads, kv_heads=None, causal=False,
                    dropout=dropout, rope=False, bias=True))
            if ffn != "mlp" or activation != "gelu":
                block.mlp = build_ffn(ffn, dim, int(dim * mlp_ratio),
                                      activation=activation, dropout=dropout, bias=True)
            if norm != "layer":
                block.n1, block.n2 = make_norm(norm, dim), make_norm(norm, dim)
        if norm != "layer":
            self.core.norm = make_norm(norm, dim)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.ndim != 4:
            raise ConfigError("pixels must have shape [B,C,H,W]", owner="VisionTransformer")
        if pixels.shape[-2] % self.patch_size[0] or pixels.shape[-1] % self.patch_size[1]:
            raise ConfigError("input image geometry must be divisible by patch_size", owner="VisionTransformer")
        if self.pool == "mean":
            return self.core.forward_features(pixels)[:, 1:].mean(1) @ self.core.head.weight.T + self.core.head.bias
        return self.core(pixels)


register_budget_knobs(VisionTransformer, dim=range(24, 1537, 24), layers=range(1, 25))


@presets(
    tiny={"dim": 64, "layers": 2, "heads": 2},
    small={"dim": 128, "layers": 4, "heads": 4},
    base={"dim": 384, "layers": 6, "heads": 6},
)
class TransformerClassifier(Configurable, nn.Module):
    """Bidirectional transformer encoder + pooled classification head.

    ``forward(tokens, mask=None) -> logits [B, num_classes]``. ``pool="cls"``
    prepends a learned CLS token; ``pool="mean"`` averages real (unmasked)
    positions. ``mask`` is ``[B, T]`` with 1 = token, 0 = padding.
    """

    __topologies_spec__ = {"recipe": "text-classify"}

    def __init__(self, vocab_size: int, num_classes: int, *, max_seq: int = 256,
                 dim: int = 128, layers: int = 4, heads: int = 4, mlp_ratio: float = 4.0,
                 activation: str = "gelu", norm: str | Callable = "layer",
                 pos: str = "learned", pool: str = "cls", dropout: float = 0.1,
                 attention: str | Callable = "mha", ffn: str | Callable = "mlp") -> None:
        super().__init__()
        owner = "TransformerClassifier"
        check_positive(owner, vocab_size=vocab_size, num_classes=num_classes, max_seq=max_seq,
                       dim=dim, layers=layers, heads=heads)
        if dim % heads:
            raise ConfigError("dim must be divisible by heads", owner=owner)
        check_choice("pos", pos, ("learned", "sinusoidal", "rope"), owner=owner)
        check_range("dropout", dropout, 0.0, 1.0, owner=owner)
        check_choice("pool", pool, ("cls", "mean"), owner=owner)
        self.vocab_size, self.num_classes, self.max_seq, self.pool = vocab_size, num_classes, \
            max_seq, pool
        self.core = _TransformerClassifierCore(vocab_size, num_classes, max_seq, dim,
            layers, heads, dropout=dropout, position=pos)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim)) if pool == "cls" else None
        if self.cls is not None and self.core.encoder.pos is not None:
            old = self.core.encoder.pos
            expanded = nn.Embedding(max_seq + 1, dim)
            with torch.no_grad():
                expanded.weight[1:].copy_(old.weight)
                expanded.weight[0].copy_(old.weight[0])
            self.core.encoder.pos = expanded
        _configure_text_core(self.core, dim=dim, heads=heads, kv_heads=None,
            mlp_ratio=mlp_ratio, activation=activation, norm=norm, dropout=dropout,
            attention=attention, ffn=ffn, causal=False, rope=pos == "rope")

    @property
    def context_length(self) -> int:
        return self.max_seq

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.cls is not None:
            from topologies.models.transformer_models import _tokens
            valid = _tokens(tokens, mask, self.vocab_size, self.max_seq)
            x = self.core.encoder.embedding(tokens)
            if self.core.encoder.pos is not None:
                pos = self.core.encoder.pos(torch.arange(tokens.shape[1] + 1, device=tokens.device))
                x = x + pos[1:][None]
                cls = self.cls + pos[0][None, None]
            else:
                cls = self.cls
            x = torch.cat((cls.expand(x.shape[0], -1, -1), x), dim=1)
            valid = torch.cat((torch.ones_like(valid[:, :1]), valid), dim=1)
            x = self.core.encoder.dropout(x) * valid.unsqueeze(-1)
            for block in self.core.encoder.blocks:
                x = block(x, valid, causal=False)
            return self.core.head(self.core.encoder.norm(x)[:, 0])
        return self.core(tokens, mask)


register_budget_knobs(TransformerClassifier, dim=range(32, 2049, 32), layers=range(1, 25))


@presets(
    tiny={"dim": 64, "layers": 2, "heads": 2},
    small={"dim": 128, "layers": 4, "heads": 4},
    base={"dim": 384, "layers": 6, "heads": 6},
)
class TransformerTagger(Configurable, nn.Module):
    """Bidirectional encoder with a per-token head: ``logits [B, T, num_classes]``."""

    __topologies_spec__ = {"recipe": "token-classify"}

    def __init__(self, vocab_size: int, num_classes: int, *, max_seq: int = 256,
                 dim: int = 128, layers: int = 4, heads: int = 4, mlp_ratio: float = 4.0,
                 activation: str = "gelu", norm: str | Callable = "layer",
                 pos: str = "learned", dropout: float = 0.1,
                 attention: str | Callable = "mha", ffn: str | Callable = "mlp") -> None:
        super().__init__()
        owner = "TransformerTagger"
        check_positive(owner, vocab_size=vocab_size, num_classes=num_classes, max_seq=max_seq,
                       dim=dim, layers=layers, heads=heads)
        if dim % heads:
            raise ConfigError("dim must be divisible by heads", owner=owner)
        check_choice("pos", pos, ("learned", "sinusoidal", "rope"), owner=owner)
        check_range("dropout", dropout, 0.0, 1.0, owner=owner)
        self.vocab_size, self.num_classes, self.max_seq = vocab_size, num_classes, max_seq
        self.core = _TransformerTaggerCore(vocab_size, num_classes, max_seq, dim,
            layers, heads, dropout=dropout, position=pos)
        _configure_text_core(self.core, dim=dim, heads=heads, kv_heads=None,
            mlp_ratio=mlp_ratio, activation=activation, norm=norm, dropout=dropout,
            attention=attention, ffn=ffn, causal=False, rope=pos == "rope")

    @property
    def context_length(self) -> int:
        return self.max_seq

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.core(tokens, mask)


register_budget_knobs(TransformerTagger, dim=range(32, 2049, 32), layers=range(1, 25))


@presets(
    tiny={"width": 64, "depth": 2},
    small={"width": 256, "depth": 3},
    base={"width": 512, "depth": 4},
)
class TabularMLP(Configurable, _TabularMLPCore):
    """MLP over numeric features plus embedded categorical columns.

    ``forward(features [B, F], categorical=None [B, K]) -> [B, out_features]``.
    ``cardinalities=[12, 4]`` declares K categorical columns (ids ``0..n-1``),
    each embedded with ``embed_dim``. Numeric features are batch-normalized.
    """

    __topologies_spec__ = {"recipe": "tabular-classify"}

    def __init__(self, in_features: int, out_features: int, *,
                 cardinalities: tuple[int, ...] | list[int] = (), embed_dim: int = 8,
                 width: int = 256, depth: int = 3, dropout: float = 0.1,
                 activation: str = "relu", norm: str = "layer") -> None:
        owner = "TabularMLP"
        check_positive(owner, in_features=in_features, out_features=out_features, width=width,
                       embed_dim=embed_dim)
        if depth < 1:
            raise ConfigError(f"depth must be >= 1; got {depth}.", owner=owner)
        check_range("dropout", dropout, 0.0, 1.0, owner=owner)
        self.in_features, self.out_features = in_features, out_features
        self.cardinalities = [int(c) for c in cardinalities]
        _TabularMLPCore.__init__(self, in_features, self.cardinalities, out_features,
                                 embed_dim, (width,) * depth, activation, dropout,
                                 "zero", norm=norm)
        self.net = self.network

    def forward(self, features: torch.Tensor,
                categorical: torch.Tensor | None = None) -> torch.Tensor:
        if features.shape[-1] != self.in_features:
            raise ConfigError(f"expected {self.in_features} features; got {features.shape[-1]}.",
                              hint="set in_features to the number of numeric columns.",
                              owner="TabularMLP")
        return _TabularMLPCore.forward(self, features, categorical)


register_budget_knobs(TabularMLP, width=range(8, 4097, 8), depth=range(1, 9))


@presets(
    tiny={"channels": 16, "levels": 3},
    small={"channels": 32, "levels": 4},
    base={"channels": 64, "levels": 6},
)
class TemporalConvNet(Configurable, nn.Module):
    """Dilated causal convolutions (TCN) for time series.

    ``forward(inputs [B, T] or [B, T, C]) -> [B, out_features]`` from the last
    time step (forecast horizon or class logits); ``per_step=True`` returns
    ``[B, T, out_features]`` (anomaly masks, sequence labels).
    """

    __topologies_spec__ = {"recipe": "regression"}

    def __init__(self, in_channels: int = 1, out_features: int = 1, *, channels: int = 32,
                 levels: int = 4, kernel_size: int = 3, dropout: float = 0.1,
                 per_step: bool = False) -> None:
        super().__init__()
        owner = "TemporalConvNet"
        check_positive(owner, in_channels=in_channels, out_features=out_features,
                       channels=channels, levels=levels, kernel_size=kernel_size)
        check_range("dropout", dropout, 0.0, 1.0, owner=owner)
        self.in_channels, self.out_features, self.per_step = in_channels, out_features, per_step
        self.core = _TemporalConvNetCore(in_features=in_channels, out_features=out_features,
            channels=(channels,) * levels, kernel_size=kernel_size, dropout=dropout,
            head="per_step" if per_step else "last")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.float()
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        if x.shape[-1] != self.in_channels:
            raise ConfigError(f"expected {self.in_channels} channels; got {x.shape[-1]} "
                              f"(input {tuple(inputs.shape)}).", owner="TemporalConvNet",
                              hint="inputs are [B, T] or [B, T, in_channels].")
        if x.ndim != 3:
            raise ConfigError("inputs must have shape [B,T] or [B,T,C]", owner="TemporalConvNet")
        return self.core(torch.nan_to_num(x))


register_budget_knobs(TemporalConvNet, channels=range(4, 1025, 4), levels=range(1, 13))


@presets(
    tiny={"dim": 32, "layers": 2, "heads": 2},
    small={"dim": 64, "layers": 3, "heads": 4},
    base={"dim": 128, "layers": 4, "heads": 4},
)
class TransformerForecaster(Configurable, nn.Module):
    """Compatibility facade backed by a reviewed recurrent predictor (not a Transformer).

    ``max_seq`` remains a maximum-input-length guard. ``out_features`` remains
    the forecast vector width; the reviewed predictor's horizon head is kept at
    one step to preserve the historical ``[B, out_features]`` output contract.
    """

    __topologies_spec__ = {"recipe": "regression"}

    def __init__(self, in_channels: int = 1, out_features: int = 1, *, max_seq: int = 512,
                 dim: int = 64, layers: int = 3, heads: int = 4, mlp_ratio: float = 4.0,
                 dropout: float = 0.1, norm: str | Callable = "layer") -> None:
        super().__init__()
        owner = "TransformerForecaster"
        check_positive(owner, in_channels=in_channels, out_features=out_features,
                       max_seq=max_seq, dim=dim, layers=layers, heads=heads)
        check_range("dropout", dropout, 0.0, 1.0, owner=owner)
        if norm != "layer":
            raise ConfigError("the recurrent predictor has no normalization slot", owner=owner)
        self.in_channels, self.out_features, self.max_seq = in_channels, out_features, max_seq
        self.core = _RecurrentPredictorCore(in_features=in_channels, hidden_size=dim,
            out_features=out_features, num_layers=layers, cell="gru", head="last", dropout=dropout)

    @property
    def context_length(self) -> int:
        return self.max_seq

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.float()
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        if x.shape[1] > self.max_seq:
            raise ConfigError(f"sequence length exceeds max_seq={self.max_seq}", owner="TransformerForecaster")
        return self.core(torch.nan_to_num(x))


register_budget_knobs(TransformerForecaster, dim=range(16, 1025, 16), layers=range(1, 13))


__all__ = ["MLP", "ConvNet", "TabularMLP", "TemporalConvNet", "TransformerClassifier",
           "TransformerForecaster", "TransformerLM", "TransformerTagger", "VisionTransformer"]
