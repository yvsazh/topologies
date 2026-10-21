"""Streaming reducers: summarise activations over a dataset without storing them.

Every reducer implements ``init(meta) -> state``, ``update(state, value, batch) ->
state``, ``merge(a, b) -> state`` and ``finalize(state) -> result``. States are
plain dicts of tensors and numbers, so ``torch.save`` / ``torch.load(weights_only=True)``
round-trips them (resumable collection). Reducers work on the value's device and
accumulate in float32, or float64 for moments.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch

from topologies.introspection.errors import SemanticsRequired
from topologies.introspection.semantics import Axes, SemanticSource


@dataclass(frozen=True)
class SiteMeta:
    site: str
    axes: Axes | None
    dtype: torch.dtype
    device: torch.device
    shape: tuple[int, ...]
    module_type: str | None = None


@dataclass
class BatchMeta:
    batch_index: int
    example_ids: torch.Tensor | None = None   # [B] global example ids, when the batch axis is known
    batch_size: int | None = None
    occurrence: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Reducer(Protocol):
    name: str

    def init(self, meta: SiteMeta) -> Any: ...
    def update(self, state: Any, value: torch.Tensor, batch: BatchMeta) -> Any: ...
    def merge(self, a: Any, b: Any) -> Any: ...
    def finalize(self, state: Any) -> Any: ...


# --------------------------------------------------------------------------- layout helpers


def feature_last(value: torch.Tensor, axes: Axes | None, *, site: str = "?") -> torch.Tensor:
    """Return ``value`` as ``[N, F]`` with the feature axis last (requires a known feature axis)."""
    if axes is None or axes.feature_axis is None or axes.source == SemanticSource.UNKNOWN:
        raise SemanticsRequired(
            f"per-unit statistics at {site!r} need a feature axis; declare one, e.g. "
            f"insp.declare({site!r}, axes='b t d')")
    resolved = axes.resolve(value.ndim)
    axis = resolved.feature_axis % value.ndim
    moved = value.movedim(axis, -1)
    return moved.reshape(-1, moved.shape[-1])


def batch_positions_features(value: torch.Tensor, axes: Axes | None, *, site: str = "?"
                             ) -> torch.Tensor:
    """``[B, P, F]``: batch first, every non-batch non-feature axis flattened into positions."""
    if axes is None or axes.feature_axis is None or axes.batch_axis is None or \
            axes.source == SemanticSource.UNKNOWN:
        raise SemanticsRequired(
            f"top-k examples at {site!r} need batch and feature axes; declare them, e.g. "
            f"insp.declare({site!r}, axes='b t d')")
    resolved = axes.resolve(value.ndim)
    f = resolved.feature_axis % value.ndim
    b = resolved.batch_axis % value.ndim
    moved = value.movedim((b, f), (0, -1))
    return moved.reshape(moved.shape[0], -1, moved.shape[-1])


def _per(per: str, axes: Axes | None) -> str:
    if per != "auto":
        return per
    known = axes is not None and axes.feature_axis is not None and \
        axes.source != SemanticSource.UNKNOWN
    return "unit" if known else "site"


# --------------------------------------------------------------------------- reducers


@dataclass
class Moments:
    """count, mean, variance (Chan/Welford parallel merge), min, max — float64."""

    per: str = "auto"
    name: str = "moments"
    approximate: bool = False

    def init(self, meta: SiteMeta) -> dict:
        return {"per": _per(self.per, meta.axes), "site": meta.site, "axes": meta.axes,
                "count": 0.0, "mean": None, "m2": None, "min": None, "max": None}

    def _rows(self, state: dict, value: torch.Tensor) -> torch.Tensor:
        x = value.detach().to(torch.float64)
        if state["per"] == "unit":
            return feature_last(x, state["axes"], site=state["site"])
        return x.reshape(-1, 1)

    def update(self, state: dict, value: torch.Tensor, batch: BatchMeta) -> dict:
        rows = self._rows(state, value)
        if rows.shape[0] == 0:
            return state
        n = float(rows.shape[0])
        mean = rows.mean(0)
        m2 = ((rows - mean) ** 2).sum(0)
        new = {**state, "count": n, "mean": mean, "m2": m2, "min": rows.amin(0), "max": rows.amax(0)}
        return self.merge(state, new) if state["mean"] is not None else new

    def merge(self, a: dict, b: dict) -> dict:
        if a["mean"] is None:
            return b
        if b["mean"] is None:
            return a
        na, nb = a["count"], b["count"]
        n = na + nb
        delta = b["mean"].to(a["mean"].device) - a["mean"]
        mean = a["mean"] + delta * (nb / n)
        m2 = a["m2"] + b["m2"].to(a["m2"].device) + delta ** 2 * (na * nb / n)
        return {**a, "count": n, "mean": mean, "m2": m2,
                "min": torch.minimum(a["min"], b["min"].to(a["min"].device)),
                "max": torch.maximum(a["max"], b["max"].to(a["max"].device))}

    def finalize(self, state: dict) -> dict:
        if state["mean"] is None:
            return {"count": 0, "per": state["per"]}
        var = state["m2"] / max(state["count"], 1.0)
        squeeze = (lambda t: float(t[0])) if state["per"] == "site" else (lambda t: t.float().cpu())
        return {"per": state["per"], "count": int(state["count"]), "mean": squeeze(state["mean"]),
                "var": squeeze(var), "std": squeeze(var.sqrt()), "min": squeeze(state["min"]),
                "max": squeeze(state["max"])}


@dataclass
class Histogram:
    """Fixed-edge histogram; ``range="auto"`` takes the first batch's range plus 2 overflow bins."""

    bins: int = 64
    range: tuple[float, float] | str = "auto"
    name: str = "histogram"
    approximate: bool = True
    site_ranges: Mapping[str, tuple[float, float]] | None = None

    def init(self, meta: SiteMeta) -> dict:
        fixed = (self.site_ranges or {}).get(meta.site)
        return {"edges": None, "counts": None, "under": 0.0, "over": 0.0, "nonfinite": 0.0,
                "lo_seen": math.inf, "hi_seen": -math.inf, "fixed": list(fixed) if fixed else None}

    def update(self, state: dict, value: torch.Tensor, batch: BatchMeta) -> dict:
        x = value.detach().reshape(-1).float()
        finite = torch.isfinite(x)
        nonfinite = float((~finite).sum())
        x = x[finite]
        if x.numel() == 0:
            return {**state, "nonfinite": state["nonfinite"] + nonfinite}
        lo, hi = float(x.min()), float(x.max())
        edges = state["edges"]
        if edges is None:
            if state.get("fixed"):
                low, high = state["fixed"]
            elif self.range == "auto":
                span = hi - lo
                pad = 0.05 * span if span > 0 else max(abs(lo) * 0.05, 1e-6)
                low, high = lo - pad, hi + pad
            else:
                low, high = float(self.range[0]), float(self.range[1])
            edges = torch.linspace(low, high, self.bins + 1, dtype=torch.float64)
            counts = torch.zeros(self.bins, dtype=torch.float64)
        else:
            counts = state["counts"].clone()
        low, high = float(edges[0]), float(edges[-1])
        under = float((x < low).sum())
        over = float((x > high).sum())
        inside = x[(x >= low) & (x <= high)]
        counts += torch.histc(inside.cpu().double(), bins=self.bins, min=low, max=high)
        return {**state, "edges": edges, "counts": counts, "under": state["under"] + under,
                "over": state["over"] + over, "nonfinite": state["nonfinite"] + nonfinite,
                "lo_seen": min(state["lo_seen"], lo), "hi_seen": max(state["hi_seen"], hi)}

    def merge(self, a: dict, b: dict) -> dict:
        if a["edges"] is None:
            return b
        if b["edges"] is None:
            return a
        counts_b = b["counts"]
        approx = a.get("rebinned", False) or b.get("rebinned", False)
        if not torch.equal(a["edges"], b["edges"]):
            counts_b = _rebin(b["edges"], counts_b, a["edges"])
            approx = True
        return {**a, "edges": a["edges"], "counts": a["counts"] + counts_b, "rebinned": approx,
                "under": a["under"] + b["under"], "over": a["over"] + b["over"],
                "nonfinite": a["nonfinite"] + b["nonfinite"],
                "lo_seen": min(a["lo_seen"], b["lo_seen"]), "hi_seen": max(a["hi_seen"], b["hi_seen"])}

    def finalize(self, state: dict) -> dict:
        if state["edges"] is None:
            return {"edges": [], "counts": [], "approximate": True}
        return {"edges": state["edges"].tolist(), "counts": state["counts"].tolist(),
                "underflow": state["under"], "overflow": state["over"],
                "nonfinite": state["nonfinite"], "min_seen": state["lo_seen"],
                "max_seen": state["hi_seen"],
                "approximate": (self.range == "auto" and not state.get("fixed"))
                or state.get("rebinned", False)}


def _rebin(edges: torch.Tensor, counts: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Move each source bin's mass to the target bin holding its centre (approximate)."""
    centres = (edges[:-1] + edges[1:]) / 2
    index = torch.bucketize(centres, target[1:-1])
    out = torch.zeros(len(target) - 1, dtype=counts.dtype)
    out.index_add_(0, index, counts)
    return out


@dataclass
class Quantiles:
    """Quantiles of a seeded reservoir sample (Algorithm R, vectorised)."""

    q: tuple[float, ...] = (0.01, 0.5, 0.99)
    reservoir: int = 100_000
    seed: int = 0
    name: str = "quantiles"
    approximate: bool = True

    def init(self, meta: SiteMeta) -> dict:
        return {"sample": torch.empty(0), "seen": 0,
                "rng": torch.Generator().manual_seed(self.seed).get_state()}

    def update(self, state: dict, value: torch.Tensor, batch: BatchMeta) -> dict:
        x = value.detach().reshape(-1).float().cpu()
        x = x[torch.isfinite(x)]
        sample, seen = state["sample"], state["seen"]
        generator = torch.Generator()
        generator.set_state(state["rng"])
        room = self.reservoir - sample.numel()
        if room > 0:
            sample = torch.cat([sample, x[:room]])
            seen += min(room, x.numel())
            x = x[room:]
        if x.numel():
            positions = seen + torch.arange(1, x.numel() + 1, dtype=torch.float64)
            slots = (torch.rand(x.numel(), generator=generator, dtype=torch.float64) * positions).long()
            keep = slots < self.reservoir
            sample = sample.clone()
            sample[slots[keep]] = x[keep]
            seen += x.numel()
        return {"sample": sample, "seen": seen, "rng": generator.get_state()}

    def merge(self, a: dict, b: dict) -> dict:
        total = a["seen"] + b["seen"]
        if total == 0:
            return a
        generator = torch.Generator()
        generator.set_state(a["rng"])
        take_a = round(self.reservoir * a["seen"] / total) if total > self.reservoir else a["sample"].numel()
        take_b = min(self.reservoir - take_a, b["sample"].numel())
        take_a = min(take_a, a["sample"].numel())
        pa = a["sample"][torch.randperm(a["sample"].numel(), generator=generator)[:take_a]]
        pb = b["sample"][torch.randperm(b["sample"].numel(), generator=generator)[:take_b]]
        return {"sample": torch.cat([pa, pb]), "seen": total, "rng": generator.get_state()}

    def finalize(self, state: dict) -> dict:
        sample = state["sample"]
        if sample.numel() == 0:
            return {"q": list(self.q), "values": [], "seen": 0, "approximate": True}
        values = torch.quantile(sample.double(), torch.tensor(self.q, dtype=torch.float64))
        return {"q": list(self.q), "values": values.tolist(), "seen": state["seen"],
                "sample_size": int(sample.numel()),
                "approximate": state["seen"] > sample.numel()}


@dataclass
class Sparsity:
    """Fraction of values ``<= threshold`` (per unit or per site); dead units have fraction 1."""

    threshold: float = 0.0
    per: str = "auto"
    name: str = "sparsity"
    approximate: bool = False

    def init(self, meta: SiteMeta) -> dict:
        return {"per": _per(self.per, meta.axes), "site": meta.site, "axes": meta.axes,
                "hits": None, "count": 0.0}

    def update(self, state: dict, value: torch.Tensor, batch: BatchMeta) -> dict:
        x = value.detach()
        rows = feature_last(x, state["axes"], site=state["site"]) if state["per"] == "unit" \
            else x.reshape(-1, 1)
        hits = (rows <= self.threshold).sum(0).double()
        old = state["hits"]
        return {**state, "hits": hits if old is None else old + hits.to(old.device),
                "count": state["count"] + rows.shape[0]}

    def merge(self, a: dict, b: dict) -> dict:
        if a["hits"] is None:
            return b
        if b["hits"] is None:
            return a
        return {**a, "hits": a["hits"] + b["hits"].to(a["hits"].device),
                "count": a["count"] + b["count"]}

    def finalize(self, state: dict) -> dict:
        if state["hits"] is None:
            return {"per": state["per"], "count": 0}
        fraction = (state["hits"] / max(state["count"], 1.0)).float().cpu()
        out = {"per": state["per"], "count": int(state["count"]), "threshold": self.threshold}
        if state["per"] == "unit":
            out.update(fraction=fraction, dead_units=torch.nonzero(fraction >= 1.0).flatten().tolist(),
                       dead_fraction=float((fraction >= 1.0).float().mean()))
        else:
            out["fraction"] = float(fraction[0])
        return out


_SATURATION_DEFAULTS = {"Sigmoid": (0.0, 1.0), "Tanh": (-1.0, 1.0), "Hardtanh": (-1.0, 1.0),
                        "Softsign": (-1.0, 1.0), "Hardsigmoid": (0.0, 1.0), "ReLU6": (0.0, 6.0)}


@dataclass
class Saturation:
    """Fraction of values within ``tol`` of the bounds; defaults come from the activation type."""

    low: float | None = None
    high: float | None = None
    tol: float = 1e-2
    per: str = "auto"
    name: str = "saturation"
    approximate: bool = False

    def init(self, meta: SiteMeta) -> dict:
        low, high = self.low, self.high
        if low is None or high is None:
            default = _SATURATION_DEFAULTS.get(meta.module_type or "")
            if default is None:
                raise ValueError(f"Saturation at {meta.site!r} ({meta.module_type}) needs explicit "
                                 "low=/high= bounds")
            low, high = default
        return {"per": _per(self.per, meta.axes), "site": meta.site, "axes": meta.axes,
                "low": low, "high": high, "hits": None, "count": 0.0}

    def update(self, state: dict, value: torch.Tensor, batch: BatchMeta) -> dict:
        x = value.detach().float()
        rows = feature_last(x, state["axes"], site=state["site"]) if state["per"] == "unit" \
            else x.reshape(-1, 1)
        span = (state["high"] - state["low"]) * self.tol
        hits = ((rows <= state["low"] + span) | (rows >= state["high"] - span)).sum(0).double()
        old = state["hits"]
        return {**state, "hits": hits if old is None else old + hits,
                "count": state["count"] + rows.shape[0]}

    merge = Sparsity.merge

    def finalize(self, state: dict) -> dict:
        if state["hits"] is None:
            return {"count": 0}
        fraction = (state["hits"] / max(state["count"], 1.0)).float().cpu()
        return {"per": state["per"], "low": state["low"], "high": state["high"], "tol": self.tol,
                "fraction": fraction if state["per"] == "unit" else float(fraction[0]),
                "mean_fraction": float(fraction.mean()), "count": int(state["count"])}


@dataclass
class NormStats:
    """Moments of per-position vector norms over the feature axis (residual-stream growth)."""

    ord: float = 2.0
    name: str = "norms"
    approximate: bool = False

    def init(self, meta: SiteMeta) -> dict:
        return {"site": meta.site, "axes": meta.axes, "inner": Moments(per="site").init(
            SiteMeta(meta.site, None, meta.dtype, meta.device, meta.shape))}

    def update(self, state: dict, value: torch.Tensor, batch: BatchMeta) -> dict:
        rows = feature_last(value.detach().float(), state["axes"], site=state["site"])
        norms = torch.linalg.vector_norm(rows, ord=self.ord, dim=-1)
        return {**state, "inner": Moments(per="site").update(state["inner"], norms, batch)}

    def merge(self, a: dict, b: dict) -> dict:
        return {**a, "inner": Moments(per="site").merge(a["inner"], b["inner"])}

    def finalize(self, state: dict) -> dict:
        return {"ord": self.ord, **Moments(per="site").finalize(state["inner"])}


@dataclass
class TopK:
    """Per unit, the k largest activations with (example id, position)."""

    k: int = 20
    per: str = "example"     # "example": max over positions per example; "position": every position
    key: Callable[[torch.Tensor], torch.Tensor] | None = None
    units: tuple[int, ...] | None = None
    name: str = "topk"
    approximate: bool = False

    def init(self, meta: SiteMeta) -> dict:
        return {"site": meta.site, "axes": meta.axes, "values": None, "ids": None, "pos": None}

    def update(self, state: dict, value: torch.Tensor, batch: BatchMeta) -> dict:
        x = batch_positions_features(value.detach().float(), state["axes"], site=state["site"])
        if self.units is not None:
            x = x[..., list(self.units)]
        score = self.key(x) if self.key is not None else x
        bsz, positions, _ = x.shape
        ids = batch.example_ids.to(x.device) if batch.example_ids is not None else \
            torch.arange(bsz, device=x.device) + batch.batch_index * 1_000_000
        if self.per == "example":
            best, where = score.max(dim=1)                      # [B, F]
            cand_ids = ids[:, None].expand_as(best)
            cand_pos = where
        else:
            best = score.reshape(bsz * positions, -1)
            cand_ids = ids.repeat_interleave(positions)[:, None].expand_as(best)
            cand_pos = torch.arange(positions, device=x.device).repeat(bsz)[:, None].expand_as(best)
        if state["values"] is not None:
            best = torch.cat([state["values"].to(best.device), best])
            cand_ids = torch.cat([state["ids"].to(best.device), cand_ids])
            cand_pos = torch.cat([state["pos"].to(best.device), cand_pos])
        k = min(self.k, best.shape[0])
        values, order = best.topk(k, dim=0)
        return {**state, "values": values, "ids": cand_ids.gather(0, order),
                "pos": cand_pos.gather(0, order)}

    def merge(self, a: dict, b: dict) -> dict:
        if a["values"] is None:
            return b
        if b["values"] is None:
            return a
        values = torch.cat([a["values"], b["values"].to(a["values"].device)])
        ids = torch.cat([a["ids"], b["ids"].to(a["ids"].device)])
        pos = torch.cat([a["pos"], b["pos"].to(a["pos"].device)])
        k = min(self.k, values.shape[0])
        top, order = values.topk(k, dim=0)
        return {**a, "values": top, "ids": ids.gather(0, order), "pos": pos.gather(0, order)}

    def finalize(self, state: dict) -> dict:
        if state["values"] is None:
            return {"k": self.k, "values": None}
        return {"k": self.k, "per": self.per, "values": state["values"].cpu(),
                "example_ids": state["ids"].cpu(), "positions": state["pos"].cpu(),
                "units": list(self.units) if self.units is not None else None}


@dataclass
class Covariance:
    """Streaming mean + covariance; above ``max_dim`` a rank-``rank`` Frequent-Directions sketch."""

    rank: int | None = None
    max_dim: int = 16_384
    name: str = "covariance"
    approximate: bool = False

    def init(self, meta: SiteMeta) -> dict:
        return {"site": meta.site, "axes": meta.axes, "count": 0.0, "sum": None, "gram": None,
                "sketch": None}

    def update(self, state: dict, value: torch.Tensor, batch: BatchMeta) -> dict:
        rows = feature_last(value.detach().to(torch.float64), state["axes"], site=state["site"])
        d = rows.shape[-1]
        if d > self.max_dim and self.rank is None:
            raise ValueError(f"Covariance at {state['site']!r}: d={d} > {self.max_dim}; pass "
                             "rank=... for a randomised sketch")
        total = rows.sum(0) if state["sum"] is None else state["sum"] + rows.sum(0)
        new = {**state, "count": state["count"] + rows.shape[0], "sum": total}
        if self.rank is None:
            gram = rows.T @ rows
            new["gram"] = gram if state["gram"] is None else state["gram"] + gram
        else:
            sketch = rows if state["sketch"] is None else torch.cat([state["sketch"], rows])
            new["sketch"] = _shrink(sketch, self.rank)
        return new

    def merge(self, a: dict, b: dict) -> dict:
        if a["sum"] is None:
            return b
        if b["sum"] is None:
            return a
        out = {**a, "count": a["count"] + b["count"], "sum": a["sum"] + b["sum"]}
        if self.rank is None:
            out["gram"] = a["gram"] + b["gram"]
        else:
            out["sketch"] = _shrink(torch.cat([a["sketch"], b["sketch"]]), self.rank)
        return out

    def finalize(self, state: dict) -> dict:
        n = max(state["count"], 1.0)
        if state["sum"] is None:
            return {"count": 0}
        mean = state["sum"] / n
        if self.rank is None:
            cov = state["gram"] / n - torch.outer(mean, mean)
            eig = torch.linalg.eigvalsh(cov).flip(0).clamp_min(0)
            return {"count": int(n), "mean": mean.float(), "covariance": cov.float(),
                    "eigenvalues": eig.float(), "approximate": False}
        _, s, vh = torch.linalg.svd(state["sketch"], full_matrices=False)
        second = (vh.T * (s ** 2 / n)) @ vh
        cov = second - torch.outer(mean, mean)
        eig = torch.linalg.eigvalsh(cov).flip(0).clamp_min(0)[: self.rank]
        return {"count": int(n), "mean": mean.float(), "eigenvalues": eig.float(),
                "components": vh[: self.rank].float(), "approximate": True}


def _shrink(sketch: torch.Tensor, rank: int) -> torch.Tensor:
    """Frequent Directions: keep the sketch at most ``2 * rank`` rows (mergeable)."""
    if sketch.shape[0] <= 2 * rank:
        return sketch
    _, s, vh = torch.linalg.svd(sketch, full_matrices=False)
    delta = s[rank] ** 2 if s.numel() > rank else 0.0
    shrunk = (s[:rank] ** 2 - delta).clamp_min(0).sqrt()
    return shrunk[:, None] * vh[:rank]


@dataclass
class NonFinite:
    """Counts of NaN/Inf and the first batch/occurrence where they appeared."""

    name: str = "nonfinite"
    approximate: bool = False

    def init(self, meta: SiteMeta) -> dict:
        return {"nan": 0.0, "inf": 0.0, "count": 0.0, "first": None}

    def update(self, state: dict, value: torch.Tensor, batch: BatchMeta) -> dict:
        x = value.detach()
        if not (x.is_floating_point() or x.is_complex()):
            return {**state, "count": state["count"] + x.numel()}
        nan = float(torch.isnan(x).sum())
        inf = float(torch.isinf(x).sum())
        first = state["first"]
        if first is None and (nan or inf):
            first = [batch.batch_index, batch.occurrence, batch.extra.get("order", 0)]
        return {"nan": state["nan"] + nan, "inf": state["inf"] + inf,
                "count": state["count"] + x.numel(), "first": first}

    def merge(self, a: dict, b: dict) -> dict:
        firsts = [f for f in (a["first"], b["first"]) if f is not None]
        return {"nan": a["nan"] + b["nan"], "inf": a["inf"] + b["inf"],
                "count": a["count"] + b["count"], "first": min(firsts) if firsts else None}

    def finalize(self, state: dict) -> dict:
        return {"nan": int(state["nan"]), "inf": int(state["inf"]), "count": int(state["count"]),
                "first": state["first"], "finite": not (state["nan"] or state["inf"])}


@dataclass
class Collect:
    """Raw values (rows along the batch axis) up to ``limit_bytes``; explicit only."""

    limit_bytes: int = 256 * 2**20
    pool: str | None = None   # None keeps per-example tensors; "mean"/"last"/"flatten" pool to [B, F]
    name: str = "collect"
    approximate: bool = False

    def init(self, meta: SiteMeta) -> dict:
        return {"site": meta.site, "axes": meta.axes, "chunks": [], "ids": [], "bytes": 0,
                "truncated": False}

    def update(self, state: dict, value: torch.Tensor, batch: BatchMeta) -> dict:
        x = pool_features(value.detach(), state["axes"], self.pool, site=state["site"]) \
            if self.pool else value.detach()
        size = x.numel() * x.element_size()
        if state["bytes"] + size > self.limit_bytes:
            return {**state, "truncated": True}
        ids = batch.example_ids if batch.example_ids is not None else torch.arange(x.shape[0])
        return {**state, "chunks": [*state["chunks"], x.cpu()], "ids": [*state["ids"], ids.cpu()],
                "bytes": state["bytes"] + size}

    def merge(self, a: dict, b: dict) -> dict:
        return {**a, "chunks": a["chunks"] + b["chunks"], "ids": a["ids"] + b["ids"],
                "bytes": a["bytes"] + b["bytes"], "truncated": a["truncated"] or b["truncated"]}

    def finalize(self, state: dict) -> dict:
        if not state["chunks"]:
            return {"values": None, "example_ids": None, "truncated": state["truncated"]}
        return {"values": torch.cat(state["chunks"]), "example_ids": torch.cat(state["ids"]),
                "truncated": state["truncated"], "pool": self.pool}


def pool_features(value: torch.Tensor, axes: Axes | None, pool: str, *, site: str = "?"
                  ) -> torch.Tensor:
    """Pool to ``[B, F]``: ``mean``/``last``/``cls`` over positions, or ``flatten``."""
    if pool == "flatten":
        return value.reshape(value.shape[0], -1)
    x = batch_positions_features(value, axes, site=site)
    if pool == "mean":
        return x.mean(1)
    if pool == "last":
        return x[:, -1]
    if pool in ("cls", "first"):
        return x[:, 0]
    raise ValueError("pool must be 'mean', 'last', 'cls'/'first' or 'flatten'")


@dataclass
class Custom:
    """Wrap user functions into the reducer protocol (default: running sum of ``fn``)."""

    fn: Callable[[torch.Tensor, BatchMeta], Any]
    merge_fn: Callable[[Any, Any], Any] | None = None
    finalize_fn: Callable[[Any], Any] | None = None
    name: str = "custom"
    approximate: bool = False

    def init(self, meta: SiteMeta) -> dict:
        return {"value": None}

    def update(self, state: dict, value: torch.Tensor, batch: BatchMeta) -> dict:
        out = self.fn(value.detach(), batch)
        return {"value": out if state["value"] is None else self.merge({"value": state["value"]},
                                                                      {"value": out})["value"]}

    def merge(self, a: dict, b: dict) -> dict:
        if a["value"] is None:
            return b
        if b["value"] is None:
            return a
        combine = self.merge_fn or (lambda x, y: x + y)
        return {"value": combine(a["value"], b["value"])}

    def finalize(self, state: dict) -> Any:
        return self.finalize_fn(state["value"]) if self.finalize_fn else state["value"]


# --------------------------------------------------------------------------- CKA


def _unbiased_hsic(k: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
    """Unbiased HSIC estimator (Song et al. 2012), as used by minibatch CKA (Nguyen et al. 2021)."""
    n = k.shape[0]
    k = k.clone()
    l = l.clone()
    k.fill_diagonal_(0)
    l.fill_diagonal_(0)
    ones = torch.ones(n, 1, dtype=k.dtype, device=k.device)
    trace = (k * l.T).sum()
    middle = (ones.T @ k @ ones) * (ones.T @ l @ ones) / ((n - 1) * (n - 2))
    last = 2.0 / (n - 2) * (ones.T @ k @ l @ ones)
    return ((trace + middle - last) / (n * (n - 3))).reshape(())


class CKAAccumulator:
    """Minibatch linear CKA for every pair of sites: O(L² · batch²) memory.

    ``update(features)`` takes ``{site: [B, F]}`` for one batch (B >= 4); ``finalize`` returns the
    ``[L, L]`` matrix ``mean HSIC(K, L) / sqrt(mean HSIC(K, K) · mean HSIC(L, L))``.
    """

    name = "cka"
    approximate = True

    def __init__(self, sites: list[str], other: list[str] | None = None):
        self.rows = list(sites)
        self.cols = list(other) if other is not None else list(sites)
        self.cross = torch.zeros(len(self.rows), len(self.cols), dtype=torch.float64)
        self.self_rows = torch.zeros(len(self.rows), dtype=torch.float64)
        self.self_cols = torch.zeros(len(self.cols), dtype=torch.float64)
        self.batches = 0

    def update(self, features: dict[str, torch.Tensor], other: dict[str, torch.Tensor] | None = None
               ) -> None:
        other = features if other is None else other
        grams_r = {s: _gram(features[s]) for s in self.rows}
        grams_c = grams_r if other is features and self.cols == self.rows else \
            {s: _gram(other[s]) for s in self.cols}
        if next(iter(grams_r.values())).shape[0] < 4:
            raise ValueError("minibatch CKA needs batches of at least 4 examples")
        for i, a in enumerate(self.rows):
            self.self_rows[i] += _unbiased_hsic(grams_r[a], grams_r[a])
            for j, b in enumerate(self.cols):
                self.cross[i, j] += _unbiased_hsic(grams_r[a], grams_c[b])
        for j, b in enumerate(self.cols):
            self.self_cols[j] += _unbiased_hsic(grams_c[b], grams_c[b])
        self.batches += 1

    def merge(self, other: CKAAccumulator) -> CKAAccumulator:
        self.cross += other.cross
        self.self_rows += other.self_rows
        self.self_cols += other.self_cols
        self.batches += other.batches
        return self

    def finalize(self) -> torch.Tensor:
        denom = torch.outer(self.self_rows.clamp_min(1e-30), self.self_cols.clamp_min(1e-30)).sqrt()
        return (self.cross / denom).float()


def _gram(x: torch.Tensor) -> torch.Tensor:
    x = x.double().reshape(x.shape[0], -1)
    return x @ x.T


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Full-batch linear CKA (Kornblith et al. 2019) with centred features."""
    x = x.double().reshape(x.shape[0], -1)
    y = y.double().reshape(y.shape[0], -1)
    x = x - x.mean(0)
    y = y - y.mean(0)
    cross = (y.T @ x).norm() ** 2
    return float(cross / ((x.T @ x).norm() * (y.T @ y).norm()).clamp_min(1e-30))


# --------------------------------------------------------------------------- registry

_REDUCERS: dict[str, Callable[..., Any]] = {
    "moments": Moments, "histogram": Histogram, "quantiles": Quantiles, "sparsity": Sparsity,
    "saturation": Saturation, "norms": NormStats, "topk": TopK, "covariance": Covariance,
    "nonfinite": NonFinite, "collect": Collect,
}


def register_reducer(name: str, factory: Callable[..., Any]) -> None:
    """Register a reducer factory usable by name in ``insp.collect(reducers=[name])``."""
    if not callable(factory):
        raise TypeError("reducer factory must be callable")
    _REDUCERS[name] = factory


def make_reducer(spec: Any) -> Any:
    if isinstance(spec, str):
        try:
            return _REDUCERS[spec]()
        except KeyError:
            raise ValueError(f"unknown reducer {spec!r}; known: {sorted(_REDUCERS)}") from None
    if isinstance(spec, Mapping):
        spec = dict(spec)
        return _REDUCERS[spec.pop("name")](**spec)
    for attr in ("init", "update", "merge", "finalize"):
        if not callable(getattr(spec, attr, None)):
            raise TypeError(f"reducer {spec!r} lacks {attr}()")
    return spec


def topk(k: int = 20, **kwargs: Any) -> TopK:
    return TopK(k=k, **kwargs)


def reducer_names() -> list[str]:
    return sorted(_REDUCERS)


__all__ = ["BatchMeta", "CKAAccumulator", "Collect", "Covariance", "Custom", "Histogram", "Moments", "NonFinite",
           "NormStats", "Quantiles", "Reducer", "Saturation", "SiteMeta", "Sparsity", "TopK",
           "batch_positions_features", "feature_last", "linear_cka", "make_reducer", "pool_features",
           "reducer_names", "register_reducer", "topk"]
