"""Axis semantics: which tensor axis is batch, position, feature, ...

Semantics are never guessed from shapes. They are either DECLARED (by the user
or an adapter), INFERRED_FROM_MODULE_TYPE (a ``torch.nn`` layer with a documented
output layout), or UNKNOWN.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from topologies.introspection.errors import SemanticsRequired


class SemanticSource(str, enum.Enum):
    DECLARED = "declared"
    INFERRED_FROM_MODULE_TYPE = "inferred_from_module_type"
    UNKNOWN = "unknown"


class AxisRole(str, enum.Enum):
    BATCH = "batch"
    POSITION = "position"
    FEATURE = "feature"
    CHANNEL = "channel"
    HEIGHT = "height"
    WIDTH = "width"
    DEPTH = "depth"
    SPATIAL = "spatial"
    HEADS = "heads"
    QUERY = "query"
    KEY = "key"
    TIME = "time"
    ELLIPSIS = "..."
    UNKNOWN = "unknown"


_SHORT = {
    "b": "batch", "batch": "batch",
    "t": "position", "s": "position", "position": "position", "pos": "position",
    "time": "time",
    "d": "feature", "feature": "feature", "f": "feature",
    "c": "channel", "channel": "channel",
    "h": "height", "height": "height",
    "w": "width", "width": "width",
    "z": "depth", "depth": "depth",
    "heads": "heads", "head": "heads", "n": "heads",
    "q": "query", "query": "query",
    "k": "key", "key": "key",
    "...": "...", "spatial": "spatial",
}
_FEATURE_NAMES = ("feature", "channel")


@dataclass(frozen=True)
class Axes:
    """Named axes of one captured tensor and where the naming came from."""

    names: tuple[str, ...]
    source: SemanticSource = SemanticSource.DECLARED
    feature_axis: int | None = None
    batch_axis: int | None = 0

    @classmethod
    def parse(cls, spec: str | Axes | tuple[str, ...] | list[str],
              source: SemanticSource = SemanticSource.DECLARED) -> Axes:
        """Parse the einops-like shorthand ``"b t d"`` / ``"b c h w"`` / ``"... d"``."""
        if isinstance(spec, Axes):
            return spec
        tokens = spec.split() if isinstance(spec, str) else list(spec)
        if not tokens:
            raise ValueError("axes spec must name at least one axis, e.g. 'b t d'")
        names = []
        for token in tokens:
            name = _SHORT.get(token.lower())
            if name is None:
                raise ValueError(f"unknown axis name {token!r}; use one of "
                                 f"{sorted(k for k in _SHORT if len(k) <= 5)}")
            names.append(name)
        return cls.from_names(tuple(names), source)

    @classmethod
    def from_names(cls, names: tuple[str, ...], source: SemanticSource) -> Axes:
        feature = next((i for i, n in enumerate(names) if n in _FEATURE_NAMES), None)
        batch = next((i for i, n in enumerate(names) if n == "batch"), None)
        if "..." in names:
            # Resolve negative indices for axes after the ellipsis.
            cut = names.index("...")
            if feature is not None and feature > cut:
                feature = feature - len(names)
            if batch is not None and batch > cut:
                batch = batch - len(names)
        return cls(names, source, feature, batch)

    def resolve(self, ndim: int) -> Axes:
        """Expand an ellipsis against a concrete rank; raise on rank mismatch."""
        if "..." not in self.names:
            if len(self.names) != ndim:
                raise SemanticsRequired(
                    f"axes {self.short()} name {len(self.names)} axes but the tensor has "
                    f"{ndim}; declare matching axes with insp.declare(site, axes='...')")
            return self
        cut = self.names.index("...")
        fill = ndim - (len(self.names) - 1)
        if fill < 0:
            raise SemanticsRequired(f"axes {self.short()} do not fit a rank-{ndim} tensor")
        names = (*self.names[:cut], *(["unknown"] * fill), *self.names[cut + 1:])
        return Axes.from_names(tuple(names), self.source)

    def index(self, name: str) -> int | None:
        name = _SHORT.get(name, name)
        return self.names.index(name) if name in self.names else None

    def short(self) -> str:
        inverse = {"batch": "b", "position": "t", "feature": "d", "channel": "c",
                   "height": "h", "width": "w", "heads": "heads", "query": "q", "key": "k"}
        return " ".join(inverse.get(n, n) for n in self.names)

    def to_dict(self) -> dict[str, Any]:
        return {"names": list(self.names), "source": self.source.value,
                "feature_axis": self.feature_axis, "batch_axis": self.batch_axis}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Axes:
        return cls(tuple(data["names"]), SemanticSource(data["source"]),
                   data.get("feature_axis"), data.get("batch_axis"))

    def __repr__(self) -> str:
        return f"Axes({self.names!r}, {self.source.name})"


UNKNOWN_AXES = Axes(("unknown",), SemanticSource.UNKNOWN, None, None)

_INFERRED = SemanticSource.INFERRED_FROM_MODULE_TYPE
_RMS = getattr(nn, "RMSNorm", None)


def _feature_last(ndim: int) -> Axes:
    names = ("batch", *(["position"] if ndim == 3 else ["unknown"] * max(ndim - 2, 0)),
             "feature") if ndim >= 2 else ("feature",)
    axes = Axes.from_names(tuple(names), _INFERRED)
    return axes if ndim >= 2 else Axes(names, _INFERRED, 0, None)


def infer_axes(module: nn.Module | None, port: str, value: Any,
               batch_size: int | None = None) -> Axes:
    """Default axes for a captured value (the §4.3 inference table)."""
    if not isinstance(value, torch.Tensor):
        return UNKNOWN_AXES
    ndim = value.ndim
    if ndim == 0:
        return Axes((), _INFERRED, None, None)
    if module is not None and port in ("output", "output.0"):
        if isinstance(module, (nn.Linear, nn.Bilinear, nn.Embedding, nn.LayerNorm)) or (
                _RMS is not None and isinstance(module, _RMS)):
            return _feature_last(ndim)
        if isinstance(module, (nn.modules.conv._ConvNd, nn.modules.batchnorm._BatchNorm,
                               nn.modules.instancenorm._InstanceNorm, nn.GroupNorm)):
            spatial = {1: ("position",), 2: ("height", "width"),
                       3: ("depth", "height", "width")}.get(ndim - 2, ("spatial",) * (ndim - 2))
            if ndim >= 2:
                return Axes.from_names(("batch", "channel", *spatial), _INFERRED)
        if isinstance(module, nn.RNNBase) and ndim == 3:
            names = (("batch", "time", "feature") if module.batch_first
                     else ("time", "batch", "feature"))
            return Axes.from_names(names, _INFERRED)
        if isinstance(module, nn.MultiheadAttention) and ndim == 3:
            names = (("batch", "position", "feature") if module.batch_first
                     else ("position", "batch", "feature"))
            return Axes.from_names(names, _INFERRED)
    batch_axis = 0 if batch_size is not None and value.shape[0] == batch_size else None
    return Axes(("unknown",) * ndim, SemanticSource.UNKNOWN, None, batch_axis)


def module_feature_axes(module: nn.Module) -> Axes | None:
    """Feature-axis-only axes for a module type (no tensor needed), or ``None``."""
    if isinstance(module, (nn.Linear, nn.Bilinear, nn.Embedding, nn.LayerNorm)) or (
            _RMS is not None and isinstance(module, _RMS)):
        return Axes(("...", "feature"), _INFERRED, -1, None)
    if isinstance(module, (nn.modules.conv._ConvNd, nn.modules.batchnorm._BatchNorm, nn.GroupNorm)):
        return Axes(("batch", "channel", "..."), _INFERRED, 1, 0)
    return None


@dataclass(frozen=True)
class Unit:
    """One unit (neuron/channel) of a site: index along the feature axis, optional position."""

    site: str
    index: int | tuple[int, ...]
    at: Any = None

    def __str__(self) -> str:
        return f"{self.site}[{self.index}]" + (f"@{self.at}" if self.at is not None else "")


def require_feature_axis(site: str, axes: Axes | None) -> int:
    if axes is None or axes.feature_axis is None:
        raise SemanticsRequired(
            f"site {site!r} has no known feature axis (axes={axes!r}). Declare one, e.g. "
            f"insp.declare({site!r}, axes='b t d')  or  axes='b c h w'")
    return axes.feature_axis


__all__ = [
    "UNKNOWN_AXES",
    "Axes",
    "AxisRole",
    "SemanticSource",
    "Unit",
    "infer_axes",
    "module_feature_axes",
    "require_feature_axis",
]
