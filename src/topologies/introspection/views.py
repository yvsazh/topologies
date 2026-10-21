"""Backend-neutral ``ViewSpec`` builders for introspection payloads (stdlib + torch only)."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from topologies.visualization.views import ViewSpec, select_views


def plain(value: Any) -> Any:
    """CPU float lists for tensors; everything else unchanged."""
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().tolist()
    return value


def heatmap(title: str, matrix: Any, *, rows: Any = None, cols: Any = None, center: Any = None,
            caption: str | None = None, site: str | None = None, **options: Any) -> ViewSpec:
    data = {"matrix": plain(matrix), "row_labels": rows, "col_labels": cols, "center": center}
    if site:
        data["site"] = site
    return ViewSpec("heatmap", title, data, options, caption)


def lines(title: str, series: list[dict[str, Any]], *, x_label: str = "", y_label: str = "",
          x_ticks: Any = None, caption: str | None = None, **options: Any) -> ViewSpec:
    clean = [{"x": plain(s.get("x")), "y": plain(s["y"]), "label": s.get("label", "")}
             for s in series]
    return ViewSpec("lines", title, {"series": clean, "x_label": x_label, "y_label": y_label,
                                     "x_ticks": x_ticks}, options, caption)


def bars(title: str, labels: list[Any], values: list[Any], *, errors: Any = None,
         y_label: str = "", caption: str | None = None, **options: Any) -> ViewSpec:
    return ViewSpec("bars", title, {"labels": [str(x) for x in labels], "values": plain(values),
                                    "errors": plain(errors), "y_label": y_label}, options, caption)


def table(title: str, rows: list[Mapping[str, Any]], *, columns: list[str] | None = None,
          caption: str | None = None) -> ViewSpec:
    columns = columns or list(dict.fromkeys(k for r in rows for k in r))
    return ViewSpec("table", title, {"columns": columns,
                                     "rows": [[_cell(r.get(c, "")) for c in columns] for r in rows]},
                    {}, caption)


def text(title: str, body: str) -> ViewSpec:
    return ViewSpec("text", title, {"text": body})


def _cell(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return float(value) if value.numel() == 1 else f"tensor{tuple(value.shape)}"
    if isinstance(value, float) and math.isfinite(value):
        return round(value, 6)
    return value


def finish(views: list[ViewSpec], **options: Any) -> list[ViewSpec]:
    return select_views(views, kinds=options.get("kinds") or options.get("views"),
                        sites=options.get("sites"), max_panels=options.get("max_panels"))


# --------------------------------------------------------------------------- traces


def trace_views(payload: Any, **options: Any) -> list[ViewSpec]:
    records = [r for r in payload.records if r.get("summary") and r["port"] == "output"]
    views: list[ViewSpec] = []
    if records:
        first = [r for r in records if r["occurrence"] == 0]
        names = [r["site"] for r in first]
        rms = [math.sqrt(max(_f(r["summary"].get("mean")) ** 2 + _f(r["summary"].get("std")) ** 2, 0))
               for r in first]
        views.append(lines("Activation RMS by call order", [{"y": rms, "label": "rms"}],
                           x_label="site (call order)", y_label="RMS", x_ticks=names,
                           caption="RMS = sqrt(mean² + std²) of each module's first-call output"))
        views.append(bars("Fraction of zeros", names,
                          [_f(r["summary"].get("zero_frac")) for r in first], y_label="fraction"))
        nonfinite = [r for r in records if _f(r["summary"].get("nonfinite_frac")) > 0]
        if nonfinite:
            views.append(table("Non-finite outputs", [{"site": r["site"], "occurrence": r["occurrence"],
                                                       "fraction": r["summary"]["nonfinite_frac"]}
                                                      for r in nonfinite]))
    nodes = [{"path": c["path"], "depth": c["depth"],
              "label": f"{c['path'].rsplit('.', 1)[-1]}@{c['occurrence']} ({c['type']})",
              "metrics": {"out": _shape(c["outputs"])}} for c in payload.calls]
    if nodes:
        views.append(ViewSpec("tree", "Call tree", {"nodes": nodes}))
    if payload.never_called:
        views.append(table("Never-called modules", [{"module": p} for p in payload.never_called]))
    return finish(views, **options)


def _shape(value: Any) -> str:
    if isinstance(value, (list, tuple)) and all(isinstance(v, int) for v in value):
        return "x".join(map(str, value))
    return str(value)[:30]


def _f(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


# --------------------------------------------------------------------------- collections


def collection_views(payload: Any, **options: Any) -> list[ViewSpec]:
    stages = payload.stages or [payload]
    views: list[ViewSpec] = []
    for stage in stages:
        name, sites = stage.reducer, stage.sites
        if not sites:
            continue
        labels = list(sites)
        if name == "moments":
            means = [_scalar(v.get("mean")) for v in sites.values()]
            stds = [_scalar(v.get("std")) for v in sites.values()]
            views.append(bars("Mean activation per site", labels, means, errors=stds,
                              caption="error bars: std over all collected values"))
        elif name == "sparsity":
            values = [v.get("dead_fraction", v.get("fraction")) for v in sites.values()]
            views.append(bars("Dead-unit fraction" if any("dead_fraction" in v for v in sites.values())
                              else "Fraction <= threshold", labels, [_scalar(v) for v in values]))
        elif name == "histogram":
            series = [{"edges": v["edges"], "counts": v["counts"], "label": s}
                      for s, v in list(sites.items())[:6] if v.get("edges")]
            if series:
                views.append(ViewSpec("histogram", "Activation distributions", {"series": series},
                                      {}, "approximate (auto range)" if any(
                                          v.get("approximate") for v in sites.values()) else None))
        elif name == "norms":
            views.append(lines("Per-position norm by site", [{"y": [_scalar(v.get("mean"))
                                                                     for v in sites.values()],
                                                               "label": "mean norm"}],
                               x_ticks=labels, y_label="norm"))
        elif name == "topk":
            for site, value in list(sites.items())[:4]:
                values = value.get("values")
                if values is not None:
                    views.append(heatmap(f"Top-k activations: {site}", values, site=site,
                                         x_label="unit", y_label="rank"))
        elif name == "nonfinite":
            views.append(table("Non-finite values", [{"site": s, **{k: v.get(k) for k in (
                "nan", "inf", "first")}} for s, v in sites.items()]))
        else:
            rows = [{"site": s, **{k: _cell(v) for k, v in (val.items() if isinstance(
                val, Mapping) else [("value", val)]) if not isinstance(v, (list, dict))}}
                    for s, val in sites.items()]
            views.append(table(f"{name}", rows))
    return finish(views, **options)


def _scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.float().mean()) if value.numel() else float("nan")
    return _f(value)


__all__ = ["bars", "collection_views", "finish", "heatmap", "lines", "plain", "table", "text",
           "trace_views"]
