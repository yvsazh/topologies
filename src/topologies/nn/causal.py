"""Composable causal decoder.

``CausalLM`` stacks sequence mixers and feed-forwards. ``layout="attention"``
is a Llama-style block (pre-norm, SwiGLU, RoPE). Other layouts reuse the same
block once their mixers are registered.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from topologies.build.config import Configurable
from topologies.build.presets import presets, register_budget_knobs
from topologies.build.validation import ConfigError, check_choice, check_positive, check_range
from topologies.nn.aux import collect_aux_losses
from topologies.nn.layers import FeedForward, RMSNorm
from topologies.nn.mixers import build_mixer

_FFN = ("mlp", "swiglu", "geglu", "relu2", "none")
_POS = ("rope", "none", "learned", "alibi")
_PLACEMENT = ("pre", "sandwich", "post")
_INIT = ("normal", "scaled", "small")


class _ReluSquared(nn.Module):
    def __init__(self, dim: int, hidden: int, *, dropout: float, bias: bool) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, hidden, bias=bias)
        self.out = nn.Linear(hidden, dim, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.out(self.drop(F.relu(self.fc(x)).square())))


class _CausalConv(nn.Module):
    """Depthwise causal convolution. ``cache.conv`` holds the last K-1 frames."""

    def __init__(self, dim: int, kernel: int, *, bias: bool) -> None:
        super().__init__()
        self.kernel = int(kernel)
        self.conv = nn.Conv1d(dim, dim, self.kernel, groups=dim, bias=bias)

    def forward(self, x: torch.Tensor, cache: BlockCache | None) -> torch.Tensor:
        xt = x.transpose(1, 2)
        left_n = self.kernel - 1
        if left_n:
            if cache is not None and cache.conv is not None:
                left = cache.conv
            else:
                left = xt.new_zeros(xt.shape[0], xt.shape[1], left_n)
            xt = torch.cat([left, xt], dim=-1)
            if cache is not None:
                cache.conv = xt[..., -left_n:]
        return self.conv(xt).transpose(1, 2)


class BlockCache:
    def __init__(self, mixer: object | None) -> None:
        self.mixer = mixer
        self.conv: torch.Tensor | None = None


class ModelCache:
    def __init__(self, slots: list[BlockCache]) -> None:
        self.slots = slots
        self.offset = 0


class DecoderBlock(nn.Module):
    def __init__(self, mixer: nn.Module, ffn: nn.Module | None, conv: nn.Module | None,
                 dim: int, *, norm: str, norm_eps: float, placement: str,
                 dropout: float) -> None:
        super().__init__()
        self.mixer = mixer
        self.ffn = ffn
        self.conv = conv
        self.placement = placement
        self.drop = nn.Dropout(dropout)
        self.norm1 = _norm(norm, dim, norm_eps)
        self.norm2 = _norm(norm, dim, norm_eps) if ffn is not None else None
        self.norm1b = _norm(norm, dim, norm_eps) if placement == "sandwich" else None
        self.norm2b = _norm(norm, dim, norm_eps) if placement == "sandwich" and ffn is not None else None

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None,
                cache: BlockCache | None) -> torch.Tensor:
        h = x if self.placement == "post" else self.norm1(x)
        if self.conv is not None:
            h = self.conv(h, cache)
        h = self.mixer(h, mask, None if cache is None else cache.mixer)
        if self.norm1b is not None:
            h = self.norm1b(h)
        h = self.drop(h)
        x = self.norm1(x + h) if self.placement == "post" else x + h
        if self.ffn is None or self.norm2 is None:
            return x
        h = x if self.placement == "post" else self.norm2(x)
        h = self.ffn(h)
        if self.norm2b is not None:
            h = self.norm2b(h)
        h = self.drop(h)
        return self.norm2(x + h) if self.placement == "post" else x + h


def _norm(kind: str, dim: int, eps: float) -> nn.Module:
    if kind == "rms":
        return RMSNorm(dim, eps=eps)
    if kind == "layer":
        return nn.LayerNorm(dim, eps=eps)
    return nn.Identity()


def _ffn(kind: str, dim: int, hidden: int, *, dropout: float, bias: bool) -> nn.Module | None:
    if kind == "none":
        return None
    if kind == "relu2":
        return _ReluSquared(dim, hidden, dropout=dropout, bias=bias)
    activation = {"mlp": "gelu", "swiglu": "swiglu", "geglu": "geglu"}[kind]
    return FeedForward(dim, hidden, activation=activation, dropout=dropout, bias=bias)


def _as_list(value: str | Sequence[str], n: int, owner: str) -> list:
    if isinstance(value, str):
        return [value] * n
    items = list(value)
    if len(items) == 1:
        return items * n
    if len(items) != n:
        raise ConfigError(f"expected 1 or {n} entries; got {len(items)}.", owner=owner)
    return items


def _layout_entries(layout: str | Sequence, layers: int,
                    codes: Mapping | None) -> list[tuple[str, object]]:
    """``(tie code, mixer spec)`` for each unique-stack layer."""
    if isinstance(layout, str) and codes is not None and layout and all(ch in codes for ch in layout):
        pairs = [(ch, codes[ch]) for ch in layout]
        if len(pairs) == 1:
            return pairs * layers
        if len(pairs) != layers:
            raise ConfigError(
                f"layout {layout!r} has {len(pairs)} layers; set layers={len(pairs)}.",
                owner="CausalLM",
            )
        return pairs
    specs = _as_list(layout, layers, "CausalLM")
    return [(spec if isinstance(spec, str) else spec[0], spec) for spec in specs]


def _spec_parts(spec: object, mixer_kwargs: Mapping[str, Mapping]) -> tuple[str, dict]:
    extra: dict = {}
    if isinstance(spec, str):
        name = spec
    elif isinstance(spec, (tuple, list)) and len(spec) == 2 and isinstance(spec[0], str):
        name, extra = spec[0], dict(spec[1])
    else:
        raise ConfigError(f"mixer spec must be a name or (name, kwargs); got {spec!r}.",
                          owner="CausalLM")
    kw = dict(mixer_kwargs.get(name, {}))
    kw.update(extra)
    return name, kw


def _share_index(share: str | None, layers: int) -> tuple[list[int], int]:
    if share is None:
        return list(range(layers)), layers
    kind, _, raw = str(share).partition(":")
    try:
        n = int(raw)
    except ValueError:
        n = 0
    if kind == "cycle" and 1 <= n <= layers:
        return [i % n for i in range(layers)], n
    if kind == "repeat" and n >= 1:
        return list(range(layers)) * n, layers
    raise ConfigError(
        f"share must be None, 'cycle:K', or 'repeat:R'; got {share!r}.", owner="CausalLM"
    )


def _scale_residual(module: nn.Module, scale: float, seen: set[int]) -> None:
    out = getattr(module, "out", None)
    if isinstance(out, nn.Linear) and id(out.weight) not in seen:
        with torch.no_grad():
            out.weight.mul_(scale)
        seen.add(id(out.weight))
        return
    core = getattr(module, "core", None)
    if core is not None and core is not module:
        _scale_residual(core, scale, seen)
        return
    linears = [m for m in module.modules() if isinstance(m, nn.Linear)]
    if linears and id(linears[-1].weight) not in seen:
        with torch.no_grad():
            linears[-1].weight.mul_(scale)
        seen.add(id(linears[-1].weight))


@presets(
    tiny={"dim": 128, "layers": 4, "heads": 4},
    small={"dim": 256, "layers": 8, "heads": 4},
    base={"dim": 512, "layers": 12, "heads": 8},
)
class CausalLM(Configurable, nn.Module):
    """Causal decoder. ``forward(tokens, mask=None) -> logits [B, T, V]``.

    Training with an active auxiliary loss (``z_loss`` or a submodule
    ``aux_loss``) returns ``{"logits", "aux_loss"}`` instead. ``step`` always
    returns logits and the same cache object.
    """

    __topologies_spec__ = {"recipe": "causal-lm"}

    def __init__(self, vocab_size: int, *, max_seq: int = 1024, dim: int = 256,
                 layers: int = 6, heads: int = 4, kv_heads: int | None = None,
                 head_dim: int | None = None,
                 layout: str | Sequence[str] = "attention",
                 codes: Mapping[str, object] | None = None,
                 mixer_kwargs: Mapping[str, Mapping] | None = None,
                 ffn: str | Sequence[str] = "swiglu", mlp_ratio: float = 8 / 3,
                 ffn_kwargs: Mapping | None = None, norm: str = "rms",
                 norm_eps: float = 1e-6, norm_placement: str = "pre", qk_norm: bool = False,
                 pos: str = "rope", rope_base: float = 10_000.0, rope_fraction: float = 1.0,
                 short_conv: int = 0, embed_dim: int | None = None,
                 tie_embeddings: bool = True, share: str | None = None,
                 tie_mixers: Sequence[str] = (), recurrence: Mapping[str, object] | None = None,
                 init: str = "scaled", init_std: float = 0.02,
                 logit_softcap: float | None = None, z_loss: float = 0.0,
                 dropout: float = 0.0, bias: bool = False) -> None:
        super().__init__()
        owner = "CausalLM"
        check_positive(owner, vocab_size=vocab_size, max_seq=max_seq, dim=dim, layers=layers,
                       heads=heads)
        check_choice("norm", norm, ("layer", "rms", "none"), owner=owner)
        check_choice("norm_placement", norm_placement, _PLACEMENT, owner=owner)
        check_choice("pos", pos, _POS, owner=owner)
        check_choice("init", init, _INIT, owner=owner)
        check_range("dropout", dropout, 0.0, 1.0, owner=owner)
        if recurrence:
            raise ConfigError("looped depth is added with the recurrence option.", owner=owner)
        if head_dim is None and dim % heads:
            raise ConfigError("dim must be divisible by heads.", owner=owner)
        if head_dim is not None and heads * head_dim != dim:
            raise ConfigError("heads * head_dim must equal dim.", owner=owner)
        if embed_dim is not None and tie_embeddings and embed_dim != dim:
            raise ConfigError("tie_embeddings requires embed_dim to equal dim.", owner=owner)
        if short_conv < 0:
            raise ConfigError("short_conv must be >= 0.", owner=owner)
        self.vocab_size, self.max_seq, self.dim = int(vocab_size), int(max_seq), int(dim)
        self.layers = int(layers)
        self.pos_kind = pos
        self.logit_softcap = None if logit_softcap is None else float(logit_softcap)
        self.z_loss = float(z_loss)
        self.tie_embeddings = bool(tie_embeddings)
        width = dim if embed_dim is None else int(embed_dim)
        self.tok_embed = nn.Embedding(self.vocab_size, width)
        self.embed_proj = None if width == dim else nn.Linear(width, dim, bias=False)
        self.pos_embed = nn.Embedding(self.max_seq, dim) if pos == "learned" else None
        self.final_norm = _norm(norm, dim, norm_eps)
        self.lm_head = None if tie_embeddings else nn.Linear(dim, self.vocab_size, bias=False)
        hidden = max(1, int(dim * float(mlp_ratio)))
        entries = _layout_entries(layout, self.layers, codes)
        specs = [spec for _, spec in entries]
        ffns = _as_list(ffn, self.layers, owner)
        for kind in ffns:
            if kind == "moe":
                raise ConfigError("ffn='moe' is not available yet.", owner=owner)
            check_choice("ffn", kind, _FFN, owner=owner)
        order, unique = _share_index(share, self.layers)
        for i in range(self.layers):
            if specs[i] != specs[order[i]] or ffns[i] != ffns[order[i]]:
                raise ConfigError(
                    "share='cycle:K' needs layout[i] and ffn[i] to match i % K.", owner=owner
                )
        self._order = order
        mixer_kwargs = mixer_kwargs or {}
        if ffn_kwargs:
            raise ConfigError("ffn_kwargs is used by MoE feed-forwards.", owner=owner)
        blocks: list[DecoderBlock] = []
        for i in range(unique):
            name, kw = _spec_parts(specs[i], mixer_kwargs)
            conv = _CausalConv(dim, short_conv, bias=bias) if short_conv else None
            mixer = build_mixer(
                name, dim, heads=heads, kv_heads=kv_heads, head_dim=head_dim,
                max_seq=self.max_seq, layer_idx=i, pos="none" if pos == "learned" else pos,
                rope_base=rope_base, rope_fraction=rope_fraction, qk_norm=qk_norm,
                dropout=dropout, bias=bias, **kw,
            )
            blocks.append(DecoderBlock(
                mixer, _ffn(ffns[i], dim, hidden, dropout=dropout, bias=bias), conv, dim,
                norm=norm, norm_eps=norm_eps, placement=norm_placement, dropout=dropout,
            ))
        self._tie_codes = tuple(tie_mixers)
        self._codes = [code for code, _ in entries[:unique]]
        self.blocks = nn.ModuleList(blocks)
        self._tie_mixer_weights()
        self._init_parameters(init, float(init_std))

    def _tie_mixer_weights(self) -> None:
        if not self._tie_codes:
            return
        wanted = set(self._tie_codes)
        groups: dict[str, list[DecoderBlock]] = {}
        for block, code in zip(self.blocks, self._codes, strict=True):
            if code in wanted:
                groups.setdefault(code, []).append(block)
        for members in groups.values():
            shared = members[0].mixer
            for other in members[1:]:
                other.mixer = shared

    def _init_parameters(self, kind: str, std: float) -> None:
        spread = std if kind != "small" else std / (self.dim ** 0.5)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=spread)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=spread)
        if kind != "scaled":
            return
        scale = (2 * max(1, len(self._order))) ** -0.5
        seen: set[int] = set()
        for block in self.blocks:
            _scale_residual(block.mixer, scale, seen)
            if block.ffn is not None:
                _scale_residual(block.ffn, scale, seen)

    def embed(self, tokens: torch.Tensor, offset: int = 0) -> torch.Tensor:
        h = self.tok_embed(tokens)
        if self.embed_proj is not None:
            h = self.embed_proj(h)
        if self.pos_embed is not None:
            if offset + tokens.shape[1] > self.max_seq:
                raise ConfigError(
                    f"learned positions stop at {self.max_seq}.",
                    owner="CausalLM",
                )
            pos = torch.arange(offset, offset + tokens.shape[1], device=tokens.device)
            h = h + self.pos_embed(pos)
        return h

    def head(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.tie_embeddings and self.embed_proj is None:
            logits = F.linear(hidden, self.tok_embed.weight)
        elif self.lm_head is None:
            raise ConfigError("tied factorized embeddings have no head.", owner="CausalLM")
        else:
            logits = self.lm_head(hidden)
        if self.logit_softcap is not None:
            logits = torch.tanh(logits / self.logit_softcap) * self.logit_softcap
        return logits

    def make_block(self, index: int) -> DecoderBlock:
        return self.blocks[self._order[index]]

    def init_cache(self, batch_size: int, *, device: torch.device | None = None,
                   dtype: torch.dtype | None = None) -> ModelCache:
        slots = []
        for index in self._order:
            block = self.blocks[index]
            init = getattr(block.mixer, "init_cache", None)
            mixer_cache = None if init is None else init(
                batch_size, max_seq=self.max_seq, device=device, dtype=dtype
            )
            slots.append(BlockCache(mixer_cache))
        return ModelCache(slots)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None, *,
                loops: int | None = None, cache: ModelCache | None = None):
        del loops
        offset = 0 if cache is None else cache.offset
        h = self.embed(tokens, offset)
        for step, index in enumerate(self._order):
            slot = None if cache is None else cache.slots[step]
            h = self.blocks[index](h, mask, slot)
        if cache is not None:
            cache.offset = offset + tokens.shape[1]
        logits = self.head(self.final_norm(h))
        if not self.training:
            return logits
        if self.z_loss:
            lse = torch.logsumexp(logits.float(), dim=-1)
            self.aux_loss = {"z_loss": self.z_loss * lse.square().mean()}
        aux = collect_aux_losses(self)
        if aux:
            return {"logits": logits, "aux_loss": aux}
        return logits

    def step(self, tokens: torch.Tensor, cache: ModelCache):
        out = self.forward(tokens, cache=cache)
        logits = out["logits"] if isinstance(out, dict) else out
        return logits, cache


register_budget_knobs(CausalLM, dim=range(32, 1025, 16), layers=range(1, 49))
