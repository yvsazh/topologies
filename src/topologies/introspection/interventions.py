"""Intervention specs: explicit, composable edits applied by forward hooks.

Interventions are the only sanctioned way to change outputs. They select a site
(``"blocks.3"``, ``"blocks.3:output.0"``, ``"head@2"``), optionally ``units``
(feature-axis indices) and ``positions`` (position-axis indices), both of which need
known axes, and return the edited value. Specs are frozen dataclasses.

    with insp.intervene(Ablate("blocks.4.attn", value="zero")):
        out = insp.model(x)
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from topologies.introspection import _tree
from topologies.introspection.errors import IntrospectionError, SemanticsRequired
from topologies.introspection.semantics import Axes, SemanticSource
from topologies.introspection.sites import parse_term

OUT_PROJECTIONS = ("out", "out_proj", "o_proj", "c_proj", "dense", "proj")
HEAD_ATTRS = ("heads", "num_heads", "n_head", "num_attention_heads")


class Intervention:
    """Base: subclasses implement ``edit(value, ctx) -> value``."""

    site: str
    needs_grad = False

    # -- session protocol -----------------------------------------------------------------

    def _spec(self) -> Any:
        return parse_term(self.site)

    def paths(self, index: Any) -> list[str]:
        spec = self._spec()
        paths = index._match_term(spec.module, frozenset()) if spec.module else [""]
        if not paths:
            raise ValueError(index.unknown_message(spec.module))
        return paths

    def kwarg_paths(self, index: Any) -> list[str]:
        return []

    def install(self, session: Any, hooks: Any) -> None:
        index = session.subject.index
        paths = set(self.paths(index))
        spec = self._spec()
        port = spec.port or "output"
        if not port.startswith("output"):
            raise IntrospectionError(f"{type(self).__name__} edits outputs; got port {port!r}")
        sub = port[len("output."):] if port.startswith("output.") else None

        def post(frame: Any, module: nn.Module, args: tuple, kwargs: dict, output: Any) -> Any:
            if frame.path not in paths:
                return None
            if isinstance(spec.occurrence, int) and frame.occurrence not in (spec.occurrence, -1):
                return None
            leaf = sub if sub is not None else ("" if isinstance(output, torch.Tensor)
                                                else _tree.primary_path(output))
            if leaf is None:
                return None
            value = _tree.get_path(output, leaf) if leaf else output
            ctx = {"session": session, "frame": frame, "module": module, "path": frame.path,
                   "axes": _axes(session, frame.path, module, value)}
            new = self.edit(value, ctx)
            if new is value:
                return None
            return _tree.set_path(output, leaf, new) if leaf else new
        hooks.post_handlers.append(post)

    def cleanup(self, session: Any) -> None:
        return

    def edit(self, value: torch.Tensor, ctx: dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError

    def describe(self) -> str:
        return repr(self)


def _axes(session: Any, path: str, module: nn.Module, value: torch.Tensor) -> Axes | None:
    from topologies.introspection.semantics import infer_axes

    declared = session.subject.declared_axes(path)
    axes = declared or infer_axes(module, "output", value, None)
    if axes is None or axes.source == SemanticSource.UNKNOWN:
        return None
    try:
        return axes.resolve(value.ndim)
    except SemanticsRequired:
        return None


def selection_mask(value: torch.Tensor, axes: Axes | None, units: Any, positions: Any,
                   site: str) -> torch.Tensor | None:
    """Boolean mask broadcastable to ``value`` selecting units/positions (``None`` = all)."""
    if units is None and positions is None:
        return None
    mask = torch.ones(value.shape, dtype=torch.bool, device=value.device)
    if units is not None:
        if axes is None or axes.feature_axis is None:
            raise SemanticsRequired(f"units= at {site!r} needs a feature axis; declare it, e.g. "
                                    f"insp.declare({site!r}, axes='b t d')")
        mask &= _axis_mask(value, axes.feature_axis, units)
    if positions is not None:
        axis = axes.index("position") if axes is not None else None
        if axis is None:
            raise SemanticsRequired(f"positions= at {site!r} needs a position axis; declare it, "
                                    f"e.g. insp.declare({site!r}, axes='b t d')")
        mask &= _axis_mask(value, axis, positions)
    return mask


def _axis_mask(value: torch.Tensor, axis: int, index: Any) -> torch.Tensor:
    axis = axis % value.ndim
    size = value.shape[axis]
    picks = torch.zeros(size, dtype=torch.bool, device=value.device)
    if isinstance(index, slice):
        picks[index] = True
    else:
        picks[torch.as_tensor([index] if isinstance(index, int) else list(index),
                              device=value.device) % size] = True
    shape = [1] * value.ndim
    shape[axis] = size
    return picks.reshape(shape).expand(value.shape)


def _const(value: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    other = other.to(value.device, value.dtype)
    if torch.is_inference(other) and not torch.is_inference_mode_enabled():
        other = other.clone()
    return other


def _apply(value: torch.Tensor, new: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    new = new.expand_as(value) if new.shape != value.shape else new
    return new if mask is None else torch.where(mask, new, value)


@dataclass(frozen=True)
class Ablate(Intervention):
    """Replace values with zero, a constant, a mean (reference collection or batch) or resamples."""

    site: str
    units: Any = None
    positions: Any = None
    value: Any = "zero"
    reference: Any = None

    def edit(self, value: torch.Tensor, ctx: dict[str, Any]) -> torch.Tensor:
        axes = ctx["axes"]
        mask = selection_mask(value, axes, self.units, self.positions, self.site)
        mode = self.value
        if isinstance(mode, (int, float)) and not isinstance(mode, bool):
            new = torch.full_like(value, float(mode))
        elif mode == "zero":
            new = torch.zeros_like(value)
        elif mode == "mean":
            new = self._mean(value, axes, ctx)
        elif mode == "resample":
            new = self._resample(value, ctx)
        elif isinstance(mode, torch.Tensor):
            new = _const(value, mode)
        else:
            raise ValueError("value must be 'zero', 'mean', 'resample', a number or a tensor")
        return _apply(value, new, mask)

    def _mean(self, value: torch.Tensor, axes: Axes | None, ctx: dict[str, Any]) -> torch.Tensor:
        ref = self.reference
        if ref is not None:
            stats = _reference_stats(ref, ctx["path"])
            mean = stats.get("mean")
            if isinstance(mean, torch.Tensor):
                if axes is None or axes.feature_axis is None:
                    raise SemanticsRequired(f"per-unit mean ablation at {self.site!r} needs a "
                                            "feature axis")
                shape = [1] * value.ndim
                shape[axes.feature_axis % value.ndim] = mean.numel()
                return _const(value, mean.reshape(shape))
            return torch.full_like(value, float(mean))
        note = f"{self.site}: mean ablation uses the current batch mean (no reference=)"
        session = ctx["session"]
        if note not in session.assumptions:
            session.assumptions.append(note)
        return value.mean(0, keepdim=True).expand_as(value)

    def _resample(self, value: torch.Tensor, ctx: dict[str, Any]) -> torch.Tensor:
        ref = self.reference
        if ref is not None:
            stats = _reference_stats(ref, ctx["path"])
            pool = stats.get("values")
            if isinstance(pool, torch.Tensor) and pool.shape[1:] == value.shape[1:]:
                generator = torch.Generator().manual_seed(0)
                picks = torch.randint(0, pool.shape[0], (value.shape[0],), generator=generator)
                return _const(value, pool[picks])
        if value.shape[0] < 2:
            raise IntrospectionError("resample ablation without reference needs batch size >= 2")
        note = f"{self.site}: resample ablation takes each value from the next example in the batch"
        session = ctx["session"]
        if note not in session.assumptions:
            session.assumptions.append(note)
        return value.roll(1, dims=0)


def _reference_stats(reference: Any, path: str) -> dict[str, Any]:
    results = getattr(reference, "results", None)
    if results is None and isinstance(reference, dict):
        results = reference
    if results is None:
        raise TypeError("reference must be a Collection (insp.collect) or {site: {reducer: ...}}")
    site = path or "<root>"
    entry = results.get(site)
    if entry is None:
        raise KeyError(f"reference has no statistics for {site!r}; collect them with "
                       f"insp.collect(data, sites={site!r}, reducers=['moments'])")
    if "moments" in entry:
        return {"mean": entry["moments"].get("mean"), **({"values": entry["collect"]["values"]}
                                                          if "collect" in entry else {})}
    if "collect" in entry:
        return {"values": entry["collect"]["values"]}
    raise KeyError(f"reference for {site!r} needs 'moments' (mean) or 'collect' (resample)")


@dataclass(frozen=True)
class Patch(Intervention):
    """Copy values recorded in another run (a ``Trace`` with kept tensors) into this run."""

    site: str
    source: Any = None
    units: Any = None
    positions: Any = None
    pool: bool = False

    def edit(self, value: torch.Tensor, ctx: dict[str, Any]) -> torch.Tensor:
        source = self._source_value(ctx)
        mask = selection_mask(value, ctx["axes"], self.units, self.positions, self.site)
        if source.shape != value.shape:
            if not self.pool:
                raise IntrospectionError(
                    f"Patch at {self.site!r}: source shape {tuple(source.shape)} != "
                    f"{tuple(value.shape)}; positional patching needs matching shapes (use "
                    "pool=True for position-agnostic patching)")
            axes = ctx["axes"]
            axis = axes.index("position") if axes is not None else None
            if axis is None:
                raise SemanticsRequired(f"pooled patching at {self.site!r} needs a position axis")
            source = source.mean(axis, keepdim=True)
        return _apply(value, _const(value, source), mask)

    def _source_value(self, ctx: dict[str, Any]) -> torch.Tensor:
        src = self.source
        if isinstance(src, torch.Tensor):
            return src
        if src is None or not hasattr(src, "records"):
            raise TypeError("Patch(source=...) must be a Trace with kept tensors or a tensor")
        frame = ctx["frame"]
        occurrence = max(frame.occurrence, 0)
        port = self._spec().port or "output"
        for key, record in src.records.items():
            if key.path == ctx["path"] and key.port == port and key.op is None and \
                    key.occurrence == occurrence:
                return record.tensor
        raise KeyError(f"source trace has no kept tensor for {ctx['path'] or '<root>'}@"
                       f"{occurrence}; trace it with keep='tensors' at that site")


@dataclass(frozen=True)
class Steer(Intervention):
    """Add ``alpha * vector`` along the feature axis (optionally at some positions)."""

    site: str
    vector: Any = None
    alpha: float = 1.0
    positions: Any = None

    def edit(self, value: torch.Tensor, ctx: dict[str, Any]) -> torch.Tensor:
        axes = ctx["axes"]
        if axes is None or axes.feature_axis is None:
            raise SemanticsRequired(f"Steer at {self.site!r} needs a feature axis; declare it")
        vector = torch.as_tensor(self.vector).to(value.device, value.dtype).reshape(-1)
        shape = [1] * value.ndim
        shape[axes.feature_axis % value.ndim] = vector.numel()
        delta = self.alpha * vector.reshape(shape)
        mask = selection_mask(value, axes, None, self.positions, self.site)
        steered = value + delta
        return steered if mask is None else torch.where(mask, steered, value)


@dataclass(frozen=True)
class Scale(Intervention):
    site: str
    factor: float = 1.0
    units: Any = None
    positions: Any = None

    def edit(self, value: torch.Tensor, ctx: dict[str, Any]) -> torch.Tensor:
        mask = selection_mask(value, ctx["axes"], self.units, self.positions, self.site)
        return _apply(value, value * self.factor, mask)


@dataclass(frozen=True)
class Clamp(Intervention):
    site: str
    lo: float | None = None
    hi: float | None = None
    units: Any = None

    def edit(self, value: torch.Tensor, ctx: dict[str, Any]) -> torch.Tensor:
        mask = selection_mask(value, ctx["axes"], self.units, None, self.site)
        return _apply(value, value.clamp(self.lo, self.hi), mask)


@dataclass(frozen=True)
class Replace(Intervention):
    """Arbitrary ``fn(value, ctx) -> value`` (``ctx``: session, frame, module, path, axes)."""

    site: str
    fn: Callable[[torch.Tensor, dict[str, Any]], torch.Tensor] = field(default=lambda v, c: v)

    def edit(self, value: torch.Tensor, ctx: dict[str, Any]) -> torch.Tensor:
        return self.fn(value, ctx)


@dataclass(frozen=True)
class HeadAblate(Intervention):
    """Zero (or mean-replace) attention heads at the input of the output projection (tier A).

    The output projection (``out``/``out_proj``/``o_proj``/``c_proj``) must be a submodule
    that is *called* (not ``F.linear`` on its weight), and the head count must be readable
    from the attention module. ``nn.MultiheadAttention`` does neither, so it is refused.
    """

    site: str
    heads: Any = (0,)
    value: str = "zero"

    def _projection(self, index: Any, path: str) -> tuple[str, int]:
        module = index.module(path)
        if isinstance(module, nn.MultiheadAttention):
            raise SemanticsRequired("HeadAblate cannot edit nn.MultiheadAttention (its output "
                                    "projection is applied functionally); ablate its output instead")
        heads = next((getattr(module, a) for a in HEAD_ATTRS if isinstance(getattr(module, a, None),
                                                                           int)), None)
        proj = next((f"{path}.{n}" if path else n for n in OUT_PROJECTIONS
                     if isinstance(getattr(module, n, None), nn.Module)), None)
        if heads is None or proj is None:
            raise SemanticsRequired(f"HeadAblate at {path!r} needs a head count attribute "
                                    f"({'/'.join(HEAD_ATTRS)}) and an output projection submodule "
                                    f"({'/'.join(OUT_PROJECTIONS)})")
        return proj, heads

    def paths(self, index: Any) -> list[str]:
        return [self._projection(index, p)[0] for p in super().paths(index)]

    def install(self, session: Any, hooks: Any) -> None:
        index = session.subject.index
        targets = {}
        for path in super().paths(index):
            proj, heads = self._projection(index, path)
            targets[proj] = heads
        chosen = [self.heads] if isinstance(self.heads, int) else list(self.heads)

        def pre(frame: Any, module: nn.Module, args: tuple, kwargs: dict) -> Any:
            if frame.path not in targets or not args or not isinstance(args[0], torch.Tensor):
                return None
            x = args[0]
            heads = targets[frame.path]
            if x.shape[-1] % heads:
                raise SemanticsRequired(f"{frame.path}: width {x.shape[-1]} is not divisible by "
                                        f"{heads} heads")
            split = x.reshape(*x.shape[:-1], heads, x.shape[-1] // heads)
            mask = torch.zeros(heads, dtype=torch.bool, device=x.device)
            mask[chosen] = True
            fill = split.mean(dim=tuple(range(split.ndim - 2)), keepdim=True) \
                if self.value == "mean" else torch.zeros_like(split)
            edited = torch.where(mask[:, None], fill.expand_as(split), split).reshape(x.shape)
            return (edited, *args[1:]), kwargs
        hooks.pre_handlers.append(pre)


__all__ = ["Ablate", "Clamp", "HeadAblate", "Intervention", "Patch", "Replace", "Scale", "Steer",
           "selection_mask"]
