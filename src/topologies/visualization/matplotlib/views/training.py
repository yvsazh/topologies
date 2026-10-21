"""Training views: static history (metric families as small multiples).

Families: loss, accuracy-like (direction ``max`` with values in [0, 1]), perplexity,
learning rate (log y), grad norm, then one panel per other metric (at most six panels;
the rest share one "other metrics" panel). Train is always slot 1, validation slot 2.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from topologies.visualization.format import direction, fmt_number
from topologies.visualization.matplotlib.style import (
    colors,
    direct_labels,
    empty_state,
    header,
    integer_x,
    legend_if_many,
    number_formatter,
)
from topologies.visualization.matplotlib.views import builtin_view

_IGNORE = {"epoch", "step", "global_step"}
_LR = ("lr", "learning_rate")
MAX_PANELS = 6
FAMILY_ALIASES = {"loss": "loss", "validation": "loss", "validation_loss": "loss",
                  "accuracy": "accuracy", "acc": "accuracy", "lr": "learning rate",
                  "learning_rate": "learning rate", "learning rate": "learning rate",
                  "grad_norm": "grad norm", "grad norm": "grad norm",
                  "perplexity": "perplexity", "ppl": "perplexity",
                  "throughput": "throughput", "memory": "memory"}


def split_name(key: str) -> tuple[str, str]:
    """``("train" | "validation", base)`` for ``train_x``/``val_x``/``validation:x`` keys."""
    if key.startswith("validation:"):
        return "validation", key.split(":", 1)[1]
    if key.startswith("train:"):
        key = key.split(":", 1)[1]
    if key.startswith("val_"):
        return "validation", key[4:]
    if key.startswith("train_"):
        return "train", key[6:]
    return "train", key


def family_of(base: str, values: Iterable[float]) -> str:
    lower = base.lower()
    if "loss" in lower:
        return "loss"
    if lower in _LR or lower.endswith("_lr"):
        return "learning rate"
    if "ppl" in lower or "perplexity" in lower:
        return "perplexity"
    if "grad" in lower:
        return "grad norm"
    if "throughput" in lower or "steps_per_second" in lower:
        return "throughput"
    if "memory" in lower or lower.startswith("mem_"):
        return "memory"
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if vals and direction(base) == "max" and min(vals) >= 0 and max(vals) <= 1:
        return "accuracy"
    return f"metric:{base}"


def group_families(series: Mapping[str, Sequence[tuple[float, float]]], *,
                   charts: Sequence[str] | None = None, max_panels: int = MAX_PANELS
                   ) -> list[tuple[str, list[str]]]:
    """``[(family, [series keys])]`` in display order; extra families fold into "other"."""
    order = ["loss", "accuracy", "perplexity", "learning rate", "grad norm", "throughput",
             "memory"]
    groups: dict[str, list[str]] = {}
    for key, points in series.items():
        _split, base = split_name(key)
        fam = family_of(base, (p[1] for p in points))
        groups.setdefault(fam, []).append(key)
    names = sorted(groups, key=lambda f: (order.index(f) if f in order else len(order), f))
    if charts:
        wanted = {FAMILY_ALIASES.get(c, c) for c in charts} | {f"metric:{c}" for c in charts}
        names = [n for n in names if n in wanted or n.removeprefix("metric:") in wanted]
    if len(names) > max_panels:
        keep, rest = names[: max_panels - 1], names[max_panels - 1:]
        groups["other metrics"] = [k for f in rest for k in groups[f]]
        names = [*keep, "other metrics"]
    return [(n, groups[n]) for n in names]


FAMILY_TITLES = {"throughput": "throughput (steps/s)", "memory": "memory",
                 "learning rate": "learning rate", "grad norm": "grad norm"}


def panel_title(family: str, keys: Sequence[str]) -> str:
    if family.startswith("metric:"):
        return family.split(":", 1)[1]
    if family in FAMILY_TITLES:
        return FAMILY_TITLES[family]
    bases = {split_name(k)[1] for k in keys}
    if family == "loss" and bases == {"loss"}:
        return "loss"
    return family if len(bases) > 1 or family in ("learning rate", "grad norm") else \
        next(iter(bases))


def series_label(key: str, keys: Sequence[str]) -> str:
    split, base = split_name(key)
    bases = {split_name(k)[1] for k in keys}
    return split if len(bases) == 1 else f"{split} {base}" if split == "validation" else base


def slot_for(key: str, keys: Sequence[str], palette: Sequence[str], seen: dict[str, int]) -> str:
    """Train = slot 1, validation = slot 2; other bases continue in palette order."""
    split, base = split_name(key)
    bases = sorted({split_name(k)[1] for k in keys})
    if len(bases) == 1:
        return palette[1 if split == "validation" else 0]
    if base not in seen:
        seen[base] = len(seen)
    index = seen[base] * 2 + (1 if split == "validation" else 0)
    return palette[index % len(palette)]


def smooth(values: Sequence[float], *, window: int = 1, ema: float | None = None) -> list[float]:
    vals = list(values)
    if ema:
        out, acc = [], None
        for v in vals:
            acc = v if acc is None else ema * acc + (1 - ema) * v
            out.append(acc)
        return out
    if window > 1:
        return [sum(vals[max(0, i - window + 1): i + 1]) / len(vals[max(0, i - window + 1): i + 1])
                for i in range(len(vals))]
    return vals


def _history_series(rows: Sequence[Mapping[str, Any]], metrics: Sequence[str] | None
                    ) -> tuple[dict[str, list[tuple[float, float]]], str]:
    xkey = "epoch" if any("epoch" in r for r in rows) else "step"
    series: dict[str, list[tuple[float, float]]] = {}
    for i, row in enumerate(rows):
        x = row.get(xkey, i)
        if not isinstance(x, (int, float)):
            x = i
        for key, value in row.items():
            if key in _IGNORE or isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if metrics and key not in metrics and split_name(key)[1] not in metrics:
                continue
            if math.isfinite(value):
                series.setdefault(key, []).append((float(x), float(value)))
    return series, xkey


def monitored_metric(series: Mapping[str, Any]) -> str | None:
    for key in ("val_loss", "validation:loss"):
        if key in series:
            return key
    return next((k for k in series if split_name(k)[0] == "validation"), None)


def best_point(points: Sequence[tuple[float, float]], name: str) -> tuple[float, float] | None:
    if not points:
        return None
    maximize = direction(split_name(name)[1]) == "max"
    return (max if maximize else min)(points, key=lambda p: p[1])


def draw_family(ax: Any, family: str, keys: Sequence[str],
                series: Mapping[str, Sequence[tuple[float, float]]], *, theme: str,
                options: Any, markers: bool) -> list[Any]:
    """Draw one family panel; returns the main Line2D objects."""
    c = colors(theme)
    seen: dict[str, int] = {}
    lines = []
    ema = getattr(options, "ema", None)
    window = int(getattr(options, "smoothing", 1) or 1)
    for key in keys[: int(getattr(options, "max_series", 8) or 8)]:
        pts = series[key]
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        color = slot_for(key, keys, c.series, seen)
        label = series_label(key, keys)
        if ema:
            ax.plot(xs, ys, color=color, alpha=0.25, linewidth=1.0, label="_raw")
        ys_s = smooth(ys, window=window, ema=ema)
        (ln,) = ax.plot(xs, ys_s, color=color, label=label,
                        marker="o" if markers else None, markersize=3.5)
        lines.append(ln)
    yscale = getattr(options, "y_scale", "linear")
    values = [p[1] for k in keys for p in series[k]]
    if family == "learning rate" and values and min(values) > 0 or (
            yscale == "auto" and values and min(values) > 0 and max(values) / min(values) > 100):
        ax.set_yscale("log")
    elif yscale not in ("auto", None):
        ax.set_yscale(yscale)
    if family == "learning rate" and ax.get_yscale() == "linear":
        from topologies.visualization.matplotlib.style import lr_formatter

        ax.yaxis.set_major_formatter(lr_formatter())
        return _finish_family(ax, lines, theme)
    xscale = getattr(options, "x_scale", "linear")
    if xscale not in ("auto", None, "linear"):
        ax.set_xscale(xscale)
    if ax.get_yscale() == "linear":
        ax.yaxis.set_major_formatter(number_formatter())
    return _finish_family(ax, lines, theme)


def _finish_family(ax: Any, lines: list[Any], theme: str) -> list[Any]:
    legend_if_many(ax, loc="best")
    direct_labels(ax, lines, theme)
    return lines


@builtin_view("training", "history")
def history_view(fig: Any, data: Any, *, theme: str, options: Any) -> Any:
    rows = list(data if isinstance(data, (list, tuple)) else data.get("history", ()))
    rows = [r for r in rows if isinstance(r, Mapping)]
    series, xkey = _history_series(rows, getattr(options, "metrics", None))
    title = getattr(options, "title", None) or "Training history"
    if not series:
        ax = fig.subplots()
        empty_state(ax, "No numeric metrics in this history", None, theme)
        return fig
    if getattr(options, "layout", "auto") == "single":
        families = [("metrics", list(series))]
    else:
        families = group_families(series, charts=getattr(options, "charts", None))
    if not families:
        ax = fig.subplots()
        empty_state(ax, "No metric matches the selected charts", None, theme)
        return fig
    n = len(families)
    ncols = 1 if n <= 2 else 2
    nrows = math.ceil(n / ncols)
    from topologies.visualization.matplotlib.style import header_gridspec

    gs = header_gridspec(fig, nrows, ncols)
    axes = []
    for i in range(n):
        share = axes[0] if axes else None
        axes.append(fig.add_subplot(gs[1 + i // ncols, i % ncols], sharex=share))
    markers = len(rows) <= 30
    monitored = monitored_metric(series)
    best = best_point(series[monitored], monitored) if monitored else None
    c = colors(theme)
    for ax, (family, keys) in zip(axes, families):
        draw_family(ax, family, keys, series, theme=theme, options=options, markers=markers)
        ax.set_title(panel_title(family, keys))
        if xkey == "epoch":
            integer_x(ax)
        if best is not None:
            ax.vlines([best[0]], 0, 1, transform=ax.get_xaxis_transform(), colors=c.muted,
                      linewidth=0.8, linestyles="dotted", zorder=0)
        if monitored in keys and best is not None:
            ax.scatter([best[0]], [best[1]], s=36, color=c.ink, zorder=5)
            ax.annotate(f"best {fmt_number(best[1])} @ {best[0]:g}", best, xytext=(-8, 12),
                        textcoords="offset points", ha="right", fontsize=8, color=c.ink2)
    for k, ax in enumerate(axes):
        if k >= len(axes) - ncols:
            ax.set_xlabel(xkey)
    subtitle = f"{len(rows)} {xkey}s"
    if best is not None and monitored:
        name = monitored.replace("validation:", "val_")
        subtitle = f"best {name} {fmt_number(best[1])} @ {xkey} {best[0]:g} · {subtitle}"
    if len(families) >= 1 and families[-1][0] == "other metrics":
        subtitle += " · further metrics folded into “other metrics”"
    header(fig, gs[0, :], title, subtitle, theme)
    return fig


def default_figsize(data: Any) -> tuple[float, float]:
    rows = list(data if isinstance(data, (list, tuple)) else data.get("history", ()))
    series, _ = _history_series([r for r in rows if isinstance(r, Mapping)], None)
    n = len(group_families(series)) if series else 1
    return (9, 4.5) if n <= 2 else (11, 7)


__all__ = ["default_figsize", "draw_family", "family_of", "group_families", "history_view",
           "integer_x", "panel_title", "series_label", "smooth", "split_name"]
