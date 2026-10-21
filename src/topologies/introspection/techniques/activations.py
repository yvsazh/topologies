"""Activation statistics, norm growth, dead/saturated units and distribution drift (U)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from topologies.introspection.inputs import is_collection
from topologies.introspection.payloads import Payload, input_fingerprint, make_result
from topologies.introspection.reducers import Histogram, Moments, Saturation, SiteMeta, Sparsity
from topologies.introspection.techniques import Technique, builtin
from topologies.introspection.views import bars, finish, lines, table
from topologies.visualization.views import ViewSpec


class AutoSaturation(Saturation):
    """Saturation only where the activation type has known bounds; silent elsewhere."""

    def init(self, meta: SiteMeta) -> dict:
        try:
            return super().init(meta)
        except ValueError:
            return {"skip": True}

    def update(self, state: dict, value: torch.Tensor, batch: Any) -> dict:
        return state if state.get("skip") else super().update(state, value, batch)

    def merge(self, a: dict, b: dict) -> dict:
        return a if a.get("skip") else super().merge(a, b)

    def finalize(self, state: dict) -> Any:
        return None if state.get("skip") else super().finalize(state)


@dataclass
class ActivationsPayload(Payload):
    sites: dict[str, dict[str, Any]]
    norm_growth: list[dict[str, Any]]
    dead_units: dict[str, dict[str, Any]]
    saturation: dict[str, float]
    histograms: dict[str, dict[str, Any]] = field(default_factory=dict)
    drift: dict[str, dict[str, Any]] = field(default_factory=dict)
    examples: int = 0

    def to_records(self) -> list[dict[str, Any]]:
        rows = []
        for site, stats in self.sites.items():
            row = {"site": site, **{k: v for k, v in stats.items() if isinstance(v, (int, float))}}
            dead = self.dead_units.get(site)
            if dead:
                row["dead_units"] = len(dead["dead"])
                row["dead_basis"] = dead["basis"]
            if site in self.saturation:
                row["saturation"] = self.saturation[site]
            if site in self.drift:
                row.update({f"drift_{k}": v for k, v in self.drift[site].items()
                            if isinstance(v, (int, float))})
            rows.append(row)
        return rows

    def views(self, **options: Any) -> list[ViewSpec]:
        names = [r["site"] for r in self.norm_growth]
        views = [lines("Layer-by-layer RMS (norm growth)",
                       [{"y": [r["rms"] for r in self.norm_growth], "label": "rms"}],
                       x_ticks=names, y_label="RMS",
                       caption="RMS = sqrt(mean² + var) over all values at each site")]
        if self.dead_units:
            views.append(bars("Dead-unit fraction", list(self.dead_units),
                              [d["fraction"] for d in self.dead_units.values()],
                              caption=f"dead = zero activation on all {self.examples} example(s); "
                                      "heuristic below min_examples"))
        if self.saturation:
            views.append(bars("Saturated fraction (bounded activations)", list(self.saturation),
                              list(self.saturation.values())))
        series = [{"edges": h["edges"], "counts": h["counts"], "label": s}
                  for s, h in list(self.histograms.items())[:6] if h.get("edges")]
        if series:
            views.append(ViewSpec("histogram", "Activation distributions", {"series": series},
                                  {"log_y": True}))
        if self.drift:
            views.append(table("Distribution drift vs compare_to",
                               [{"site": s, **d} for s, d in self.drift.items()]))
        return finish(views, **options)


@builtin
class Activations(Technique):
    name = "activations"
    tier = "U"
    result_kind = "introspection.activations"
    summary = "per-site/per-unit stats, norm growth, dead/saturated units, drift"

    def run(self, subject: Any, inputs: Any = None, *, sites: Any = "leaf",
            stats: tuple[str, ...] = ("moments", "histogram", "sparsity"),
            compare_to: Any = None, min_examples: int = 1000, max_batches: int | None = None,
            bins: int = 64, threshold: float = 0.0, **params: Any) -> Any:
        inspector = subject.inspector
        data = inputs if is_collection(inputs) else [inputs]
        reducers = [Moments(per="auto")]
        if "histogram" in stats:
            reducers.append(Histogram(bins=bins))
        if "sparsity" in stats:
            reducers.append(Sparsity(threshold=threshold, per="auto"))
        reducers.append(AutoSaturation())
        collection = inspector.collect(data, sites, reducers, max_batches=max_batches,
                                       progress=False, **params)
        payload = summarize(collection, min_examples=min_examples)
        warnings = list(collection.warnings)
        if compare_to is not None:
            other_data = compare_to if is_collection(compare_to) else [compare_to]
            ranges = {s: (h["edges"][0], h["edges"][-1]) for s, h in payload.histograms.items()
                      if h.get("edges")}
            other = inspector.collect(other_data, sites, [Moments(per="auto"),
                                                          Histogram(bins=bins, site_ranges=ranges)],
                                      max_batches=max_batches, progress=False, **params)
            payload.drift = drift(collection, other)
        assumptions = [*collection.assumptions,
                       (f"dead units: zero (<= {threshold}) on every collected value; measured only "
                       f"with >= {min_examples} examples, heuristic below")]
        if compare_to is not None:
            assumptions.append("drift: per-unit standardised mean difference and histogram "
                               "Jensen-Shannon distance (base 2) on the first input's bins")
        return make_result(self, payload, subject, warnings=warnings, assumptions=assumptions,
                           provenance={"inputs": input_fingerprint(inputs), "sites": str(sites),
                                       "examples": collection.examples})


def summarize(collection: Any, *, min_examples: int = 1000) -> ActivationsPayload:
    sites: dict[str, dict[str, Any]] = {}
    growth = []
    dead: dict[str, dict[str, Any]] = {}
    saturation: dict[str, float] = {}
    histograms: dict[str, dict[str, Any]] = {}
    for site in collection.sites():
        result = collection[site]
        moments = result.get("moments", {})
        stats = _site_stats(moments)
        sites[site] = stats
        if "rms" in stats:
            growth.append({"site": site, "rms": stats["rms"]})
        sparsity = result.get("sparsity")
        if sparsity and sparsity.get("per") == "unit" and sparsity.get("count"):
            dead[site] = {"dead": sparsity["dead_units"], "fraction": sparsity["dead_fraction"],
                          "units": int(sparsity["fraction"].numel()),
                          "basis": "measured" if collection.examples >= min_examples else "heuristic"}
            stats["dead_fraction"] = sparsity["dead_fraction"]
        elif sparsity and sparsity.get("per") == "site":
            stats["zero_fraction"] = sparsity.get("fraction")
        sat = result.get("saturation")
        if sat:
            saturation[site] = sat["mean_fraction"]
        if result.get("histogram"):
            histograms[site] = result["histogram"]
    return ActivationsPayload(sites, growth, dead, saturation, histograms, {}, collection.examples)


def _site_stats(moments: dict[str, Any]) -> dict[str, Any]:
    if not moments or not moments.get("count"):
        return {}
    mean, var = moments["mean"], moments["var"]
    if isinstance(mean, torch.Tensor):
        # Pool per-unit moments into site-level ones (equal counts per unit).
        grand = float(mean.double().mean())
        total_var = float((var.double() + (mean.double() - grand) ** 2).mean())
        mn, mx = float(moments["min"].min()), float(moments["max"].max())
        unit_std = moments["std"]
        extra = {"units": int(mean.numel()), "unit_std_min": float(unit_std.min()),
                 "unit_std_max": float(unit_std.max())}
    else:
        grand, total_var, mn, mx, extra = float(mean), float(var), moments["min"], moments["max"], {}
    return {"mean": grand, "std": math.sqrt(max(total_var, 0.0)),
            "rms": math.sqrt(max(grand ** 2 + total_var, 0.0)), "min": mn, "max": mx,
            "count": moments["count"], **extra}


def drift(first: Any, second: Any) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for site in first.sites():
        if site not in second.results:
            continue
        a, b = first[site].get("moments", {}), second[site].get("moments", {})
        entry: dict[str, Any] = {}
        if a.get("count") and b.get("count"):
            ma, mb = _as_tensor(a["mean"]), _as_tensor(b["mean"])
            va, vb = _as_tensor(a["var"]), _as_tensor(b["var"])
            if ma.shape == mb.shape:
                smd = (ma - mb) / ((va + vb) / 2).clamp_min(1e-12).sqrt()
                entry.update(smd_mean_abs=float(smd.abs().mean()), smd_max_abs=float(smd.abs().max()))
        ha, hb = first[site].get("histogram"), second[site].get("histogram")
        if ha and hb and ha.get("edges") and len(ha["counts"]) == len(hb["counts"]):
            entry["js_distance"] = js_distance(ha["counts"], hb["counts"])
        if entry:
            out[site] = entry
    return out


def _as_tensor(value: Any) -> torch.Tensor:
    return value.double() if isinstance(value, torch.Tensor) else torch.tensor([float(value)],
                                                                                dtype=torch.float64)


def js_distance(p: list[float], q: list[float]) -> float:
    """Jensen-Shannon distance (sqrt of divergence, base 2) between two count vectors."""
    pa, qa = torch.tensor(p, dtype=torch.float64), torch.tensor(q, dtype=torch.float64)
    if pa.sum() <= 0 or qa.sum() <= 0:
        return float("nan")
    pa, qa = pa / pa.sum(), qa / qa.sum()
    m = (pa + qa) / 2

    def kl(x: torch.Tensor, y: torch.Tensor) -> float:
        mask = x > 0
        return float((x[mask] * (x[mask] / y[mask]).log2()).sum())
    return math.sqrt(max((kl(pa, m) + kl(qa, m)) / 2, 0.0))
