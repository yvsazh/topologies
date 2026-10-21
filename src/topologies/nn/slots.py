"""Injectable component slots for the toolbox stacks (attention / ffn / norm).

``TransformerBlock``, ``TransformerLM`` and ``VisionTransformer`` accept
``attention=``, ``ffn=`` and ``norm=`` as a name or a factory:

- ``attention(dim, heads, *, kv_heads, causal, dropout, rope, bias) -> nn.Module``
  whose ``forward(x, mask=None)`` returns ``[B, T, D]``;
- ``ffn(dim, hidden, *, activation, dropout, bias) -> nn.Module``;
- ``norm(dim) -> nn.Module``.

``slot(cls, **fixed)`` turns any class into such a factory: keyword arguments
are filtered by the class signature (with aliases ``dim``/``d_model``/
``embed_dim``, ``heads``/``num_heads``/``n_heads``, ``hidden``/``d_ff``/...),
``fixed`` kwargs always win, and modules whose ``forward`` takes no ``mask``
are wrapped (the mask is dropped: exact when every position is real; a
warning is emitted once otherwise).

    from topologies.layers import SlidingWindowAttention
    TransformerLM(vocab_size, attention=slot(SlidingWindowAttention, window=64))

Names: ``"mha"`` (default ``MultiHeadAttention``) / ``"mlp"`` (default
``FeedForward``), anything registered with ``register_attention`` /
``register_ffn``, or a catalog layer id/class name (``"linear-attention"``).
Importable factories (and ``slot(...)`` of importable classes) are recorded
by ``save_pretrained``; others need ``save_pretrained(..., model_kwargs=...)``.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from topologies._execution import primary_output
from topologies.build.validation import ConfigError

ALIASES: dict[str, tuple[str, ...]] = {
    "dim": ("dim", "d_model", "embed_dim", "hidden_size", "n_embd", "channels",
            "in_features", "normalized_shape"),
    "heads": ("heads", "num_heads", "n_heads", "n_head", "nhead"),
    "hidden": ("hidden", "hidden_dim", "d_ff", "ffn_dim", "dim_feedforward",
               "intermediate_size", "hidden_features", "mlp_dim"),
    "kv_heads": ("kv_heads", "num_kv_heads", "n_kv_heads", "kv"),
    "dropout": ("dropout", "attn_dropout", "p"),
}

_ATTENTION: dict[str, Callable[..., nn.Module]] = {}
_FFN: dict[str, Callable[..., nn.Module]] = {}
_WARNED: set[str] = set()


def _warn_once(msg: str) -> None:
    if msg not in _WARNED:
        _WARNED.add(msg)
        warnings.warn(msg, UserWarning, stacklevel=4)


def register_attention(name: str, factory: Callable[..., nn.Module]) -> None:
    """``attention=name`` builds ``factory(dim, heads, *, kv_heads, causal, ...)``."""
    _ATTENTION[str(name).lower()] = factory


def register_ffn(name: str, factory: Callable[..., nn.Module]) -> None:
    """``ffn=name`` builds ``factory(dim, hidden, *, activation, dropout, bias)``."""
    _FFN[str(name).lower()] = factory


def _signature(cls: Any) -> tuple[set[str], bool]:
    try:
        params = inspect.signature(cls).parameters.values()
    except (TypeError, ValueError):
        return set(), True
    names = {p.name for p in params if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    return names, any(p.kind is p.VAR_KEYWORD for p in params)


def _accepts_mask(module: nn.Module) -> bool:
    try:
        params = inspect.signature(module.forward).parameters
    except (TypeError, ValueError):
        return False
    return "mask" in params or any(p.kind is p.VAR_KEYWORD for p in params.values())


class MaskAdapter(nn.Module):
    """Wraps a module whose ``forward`` has no ``mask``: the padding mask is dropped."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None and not bool(mask.bool().all()):
            _warn_once(f"{type(self.inner).__name__} takes no padding mask; padded positions "
                       "are attended to (exact only when every position is real).")
        return primary_output(self.inner(x), prefer=None)


class Slot:
    """Factory built by ``slot(cls, **fixed)`` (see module docstring)."""

    def __init__(self, cls: Any, **fixed: Any) -> None:
        if not callable(cls):
            raise TypeError(f"slot() needs a class or factory, got {type(cls).__name__}")
        self.cls = cls
        self.fixed = dict(fixed)

    def _kwargs(self, values: dict[str, Any]) -> dict[str, Any]:
        names, var_kw = _signature(self.cls)
        out: dict[str, Any] = {}
        for canon, v in values.items():
            aliases = ALIASES.get(canon, (canon,))
            target = next((a for a in aliases if a in names), None)
            if target is None and var_kw:
                target = canon
            if target is not None and target not in out:
                out[target] = v
            elif target is None and canon in ("causal", "rope") and v:
                _warn_once(f"{getattr(self.cls, '__name__', self.cls)} takes no {canon!r} "
                           f"argument; it is used as-is ({canon}={v} is not applied).")
        for k, v in self.fixed.items():
            out[k] = v
        return out

    def __call__(self, dim: int, *args: Any, **kwargs: Any) -> nn.Module:
        values: dict[str, Any] = {"dim": dim}
        if args:
            key = "hidden" if "activation" in kwargs else "heads"
            values[key] = args[0]
        values.update(kwargs)
        module = self.cls(**self._kwargs(values))
        if not isinstance(module, nn.Module):
            raise TypeError(f"{self!r} built {type(module).__name__}, not an nn.Module")
        if ("causal" in kwargs or "heads" in values) and not _accepts_mask(module):
            return MaskAdapter(module)
        return module

    def to_config(self) -> dict[str, Any]:
        """JSON view (used for trial IDs and ``save_pretrained``)."""
        from topologies.build.config import _UNENCODABLE, encode_value

        cls = encode_value(self.cls)
        fixed = {k: encode_value(v) for k, v in self.fixed.items()}
        if cls is _UNENCODABLE or any(v is _UNENCODABLE for v in fixed.values()):
            name = getattr(self.cls, "__qualname__", type(self.cls).__name__)
            return {"cls": f"<unrecordable {name}>", "fixed": {k: repr(v) for k, v in
                                                               self.fixed.items()}}
        return {"cls": cls, "fixed": fixed}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Slot:
        from topologies.build.config import decode_value

        return cls(decode_value(config["cls"]), **{k: decode_value(v) for k, v in
                                                   (config.get("fixed") or {}).items()})

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Slot) and self.cls is other.cls and self.fixed == other.fixed

    def __hash__(self) -> int:
        return hash((id(self.cls), tuple(sorted(self.fixed))))

    def __repr__(self) -> str:
        extra = "".join(f", {k}={v!r}" for k, v in self.fixed.items())
        return f"slot({getattr(self.cls, '__name__', self.cls)}{extra})"


def slot(cls: Any, **fixed: Any) -> Slot:
    """Factory for ``attention=`` / ``ffn=`` / ``norm=`` from an arbitrary class."""
    return Slot(cls, **fixed)


def _catalog_layer(name: str) -> Any:
    try:
        from topologies.layers import get_layer

        return get_layer(name)
    except Exception:  # noqa: BLE001
        return None


def resolve_component(kind: str | Callable, registry: dict[str, Callable], default: str,
                      what: str) -> Callable[..., nn.Module] | None:
    """Name / factory -> factory; ``None`` means the built-in default."""
    if callable(kind):
        return kind
    key = str(kind).lower()
    if key == default:
        return None
    if key in registry:
        return registry[key]
    cls = _catalog_layer(str(kind))
    if cls is not None:
        return Slot(cls)
    import difflib

    options = [default, *sorted(registry)]
    sug = difflib.get_close_matches(key, options, n=1)
    raise ConfigError(f"Unknown {what} {kind!r}.",
                      hint=(f"did you mean {sug[0]!r}? " if sug else "")
                      + f"Use one of {options}, a catalog layer id, or slot(MyClass).")


def build_attention(kind: str | Callable, dim: int, heads: int, **kw: Any) -> nn.Module:
    from topologies.nn.layers import MultiHeadAttention

    factory = resolve_component(kind, _ATTENTION, "mha", "attention")
    if factory is None:
        return MultiHeadAttention(dim, heads, **kw)
    module = factory(dim, heads, **kw)
    if not _accepts_mask(module):
        module = MaskAdapter(module)
    return module


def build_ffn(kind: str | Callable, dim: int, hidden: int, **kw: Any) -> nn.Module:
    from topologies.nn.layers import FeedForward

    factory = resolve_component(kind, _FFN, "mlp", "ffn")
    if factory is None:
        return FeedForward(dim, hidden, **kw)
    return factory(dim, hidden, **kw)


__all__ = ["ALIASES", "MaskAdapter", "Slot", "build_attention", "build_ffn",
           "register_attention", "register_ffn", "slot"]
