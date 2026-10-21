"""Experiment views: leaderboard dot plot, learning curves, best so far, pivot heatmap.

One 2×2 GridSpec with constrained layout (no hand-placed axes). Configs keep their
color slot across panels (slot = position in the summaries, i.e. plan order of first
appearance is not available here, so the stable summary order is used); the leaderboard
uses the emphasis pattern (best in slot 1, others neutral).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from topologies.visualization.format import fmt_number
from topologies.visualization.matplotlib.style import (
    colors,
    empty_state,
    header,
    integer_x,
    legend_if_many,
    number_formatter,
)
from topologies.visualization.matplotlib.views import builtin_view

MAX_ROWS = 15
MAX_CURVES = 6


def varying_axes(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    keys: list[str] = []
    for row in rows:
        for k in (row.get("labels") or {}):
            if k != "seed" and k not in keys:
                keys.append(k)
    return [k for k in keys if len({str((r.get("labels") or {}).get(k)) for r in rows}) > 1] \
        or keys


def config_label(labels: Mapping[str, Any], axes: Sequence[str]) -> str:
    parts = [f"{k}={labels.get(k)}" for k in axes if k in labels]
    return " · ".join(parts) or "(all)"


def _natural(value: Any) -> tuple:
    try:
        return (0, float(value), "")
    except (TypeError, ValueError):
        return (1, 0.0, str(value))


def config_slots(rows: Sequence[Mapping[str, Any]]) -> dict[int, int]:
    """Stable color slot per config: natural order of its labels (never its rank)."""
    axes = varying_axes(rows)
    keyed = sorted(rows, key=lambda r: tuple(_natural((r.get("labels") or {}).get(a))
                                             for a in axes))
    return {id(r): i for i, r in enumerate(keyed)}


def _arrow(mode: str) -> str:
    return "↓ (lower is better)" if mode == "min" else "↑ (higher is better)"


def _ranked(rows: Sequence[Mapping[str, Any]], mode: str) -> list[Mapping[str, Any]]:
    valid = [r for r in rows if isinstance(r.get("mean"), (int, float))]
    return sorted(valid, key=lambda r: r["mean"], reverse=(mode == "max"))


def _trial_curves(row: Mapping[str, Any]) -> list[list[tuple[float, float]]]:
    out = []
    for trial in row.get("trials") or ():
        if trial.get("status") != "completed":
            continue
        pts = [(float(p[0]), float(p[1])) for p in trial.get("history") or ()
               if isinstance(p, (list, tuple)) and len(p) > 1 and isinstance(p[1], (int, float))]
        if pts:
            out.append(pts)
    return out


def mean_curve(curves: Sequence[Sequence[tuple[float, float]]]
               ) -> list[tuple[float, float, float]]:
    by_x: dict[float, list[float]] = {}
    for curve in curves:
        for x, y in curve:
            by_x.setdefault(x, []).append(y)
    out = []
    for x, values in sorted(by_x.items()):
        m = sum(values) / len(values)
        sd = math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1)) \
            if len(values) > 1 else 0.0
        out.append((x, m, sd))
    return out


def leaderboard_panel(ax: Any, data: Mapping[str, Any], *, theme: str) -> None:
    c = colors(theme)
    goal = data.get("goal") or {}
    metric, mode = goal.get("metric", "metric"), goal.get("mode", "min")
    rows = list(data.get("summaries") or ())
    axes = varying_axes(rows)
    ranked = _ranked(rows, mode)
    extra = max(0, len(ranked) - MAX_ROWS)
    ranked = ranked[:MAX_ROWS]
    statuses: dict[str, int] = {}
    for row in rows:
        for k, v in (row.get("statuses") or {}).items():
            if k != "completed":
                statuses[k] = statuses.get(k, 0) + int(v)
    title = "leaderboard"
    if statuses:
        title += "  ·  " + ", ".join(f"{v} {k}" for k, v in sorted(statuses.items()))
    ax.set_title(title)
    if not ranked:
        empty_state(ax, "No completed trials", None, theme)
        return
    ys = list(range(len(ranked)))[::-1]
    labels = []
    ci_used = False
    for y, row in zip(ys, ranked):
        values = [v for v in row.get("values") or () if isinstance(v, (int, float))]
        mean = float(row["mean"])
        color = c.series[0] if row is ranked[0] else c.ink2
        if values:
            ax.scatter(values, [y] * len(values), s=14, color=color, alpha=0.3, linewidths=0,
                       zorder=2)
        if len(values) >= 2:
            sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
            half = 1.96 * sd / math.sqrt(len(values)) if len(values) >= 3 else sd
            ci_used = ci_used or len(values) >= 3
            ax.hlines(y, mean - half, mean + half, colors=color, linewidth=1.2, zorder=3)
        ax.scatter([mean], [y], s=44, color=color, zorder=4)
        labels.append(config_label(row.get("labels") or {}, axes))
    ax.set_yticks(ys, labels)
    ax.set_ylim(-0.7, len(ranked) - 0.3)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(number_formatter())
    whisker = "95 % CI (n ≥ 3), else ± std" if ci_used else "± std over seeds"
    ax.set_xlabel(f"{metric} {_arrow(mode)}  ·  whisker {whisker}")
    if extra:
        ax.annotate(f"+{extra} more", (0, 0), xycoords="axes fraction", xytext=(0, -26),
                    textcoords="offset points", fontsize=8, color=c.muted)


def curves_panel(ax: Any, data: Mapping[str, Any], *, theme: str) -> None:
    c = colors(theme)
    goal = data.get("goal") or {}
    metric, mode = goal.get("metric", "metric"), goal.get("mode", "min")
    rows = list(data.get("summaries") or ())
    axes = varying_axes(rows)
    ranked = _ranked(rows, mode)
    slot = config_slots(rows)
    ax.set_title(f"learning curves  ·  {metric}")
    drawn = 0
    lines = []
    for rank, row in enumerate(ranked):
        curve = mean_curve(_trial_curves(row))
        if not curve:
            continue
        xs = [p[0] for p in curve]
        ms = [p[1] for p in curve]
        if rank < MAX_CURVES:
            color = c.series[slot[id(row)] % len(c.series)]
            (ln,) = ax.plot(xs, ms, color=color, label=config_label(row.get("labels") or {},
                                                                     axes),
                            marker="o" if len(xs) <= 30 else None, markersize=3)
            lines.append(ln)
            sds = [p[2] for p in curve]
            if any(sds):
                ax.fill_between(xs, [m - s for m, s in zip(ms, sds)],
                                [m + s for m, s in zip(ms, sds)], color=color, alpha=0.15,
                                linewidth=0)
        else:
            ax.plot(xs, ms, color=c.muted, linewidth=0.8, alpha=0.6, label="_other")
        drawn += 1
    if not drawn:
        empty_state(ax, "No learning curves recorded", "trials report per-epoch histories",
                    theme)
        return
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    integer_x(ax)
    ax.yaxis.set_major_formatter(number_formatter())
    legend_if_many(ax, loc="best", fontsize=7.5)


def best_so_far_panel(ax: Any, data: Mapping[str, Any], *, theme: str) -> None:
    c = colors(theme)
    goal = data.get("goal") or {}
    points = [p for p in data.get("best_so_far") or () if isinstance(p, Mapping)]
    ax.set_title("best so far")
    if not points:
        empty_state(ax, "No completed trials", None, theme)
        return
    xs = [p["order"] for p in points]
    best = [p["best"] for p in points]
    ax.scatter(xs, [p["value"] for p in points], s=14, color=c.muted, alpha=0.6, zorder=2,
               label="trial")
    ax.step(xs, best, where="post", color=c.series[0], label="best so far")
    improved = [(x, b) for i, (x, b) in enumerate(zip(xs, best)) if i == 0 or b != best[i - 1]]
    ax.scatter([p[0] for p in improved], [p[1] for p in improved], s=30, color=c.series[0],
               zorder=3)
    order = data.get("best_so_far_order") or "completed trial order"
    ax.set_xlabel(f"completed trials · {order}")
    ax.set_ylabel(goal.get("metric", "metric"))
    integer_x(ax)
    ax.yaxis.set_major_formatter(number_formatter())
    legend_if_many(ax, loc="best")


def _pivot_axes(rows: Sequence[Mapping[str, Any]]) -> tuple[str, str] | None:
    axes = varying_axes(rows)
    return (axes[0], axes[1]) if len(axes) >= 2 else None


def pivot_panel(ax: Any, data: Mapping[str, Any], *, theme: str) -> None:
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    c = colors(theme)
    goal = data.get("goal") or {}
    metric, mode = goal.get("metric", "metric"), goal.get("mode", "min")
    rows = [r for r in data.get("summaries") or () if isinstance(r.get("mean"), (int, float))]
    pair = _pivot_axes(list(data.get("summaries") or ()))
    if pair is None:
        strip_panel(ax, data, theme=theme)
        return
    ra, ca = pair
    rvals = sorted(dict.fromkeys(str((r.get("labels") or {}).get(ra)) for r in rows),
                   key=_natural)
    cvals = sorted(dict.fromkeys(str((r.get("labels") or {}).get(ca)) for r in rows),
                   key=_natural)
    cells: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        labels = r.get("labels") or {}
        cells.setdefault((str(labels.get(ra)), str(labels.get(ca))), []).append(float(r["mean"]))
    grid = [[(sum(cells[(rv, cv)]) / len(cells[(rv, cv)])) if (rv, cv) in cells else math.nan
             for cv in cvals] for rv in rvals]
    flat = [v for row in grid for v in row if not math.isnan(v)]
    ax.set_title(f"{metric} by {ra} × {ca}  ·  darker is better")
    scale_note = None
    if not flat:
        empty_state(ax, "No completed trials", None, theme)
        return
    lo, hi = min(flat), max(flat)
    use_log = lo > 0 and hi / lo > 10

    def scaled(v: float) -> float:
        return math.log10(v) if use_log else v

    slo, shi = scaled(lo), scaled(hi)
    span = (shi - slo) or 1.0
    goodness = [[math.nan if math.isnan(v) else ((shi - scaled(v)) / span if mode == "min" else
                                                (scaled(v) - slo) / span) for v in row]
                for row in grid]
    cmap = LinearSegmentedColormap.from_list("topologies_seq", list(c.sequential))
    cmap.set_bad(c.surface)
    import numpy as np

    ax.imshow(np.array(goodness, dtype=float), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    best_v = lo if mode == "min" else hi
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            if math.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color=c.muted)
                continue
            g = goodness[i][j]
            ax.text(j, i, fmt_number(v, sig=3), ha="center", va="center", fontsize=8,
                    color="#ffffff" if g > 0.55 else c.ink)
            if v == best_v:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=c.ink,
                                       linewidth=1.6))
    ax.set_xticks(range(len(cvals)), cvals)
    ax.set_yticks(range(len(rvals)), rvals)
    if use_log:
        scale_note = "shading on a log scale"
    if scale_note:
        ax.set_title(f"{metric} by {ra} × {ca}  ·  darker is better ({scale_note})")
    ax.set_xlabel(ca)
    ax.set_ylabel(ra)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def strip_panel(ax: Any, data: Mapping[str, Any], *, theme: str) -> None:
    c = colors(theme)
    goal = data.get("goal") or {}
    metric, mode = goal.get("metric", "metric"), goal.get("mode", "min")
    rows = list(data.get("summaries") or ())
    axes = varying_axes(rows)
    ranked = _ranked(rows, mode)[:MAX_ROWS]
    slot = config_slots(rows)
    ax.set_title(f"seed distribution  ·  {metric}")
    if not ranked:
        empty_state(ax, "No completed trials", None, theme)
        return
    for x, row in enumerate(ranked):
        values = [v for v in row.get("values") or () if isinstance(v, (int, float))]
        color = c.series[slot[id(row)] % len(c.series)]
        jitter = [x + (k - (len(values) - 1) / 2) * 0.06 for k in range(len(values))]
        ax.scatter(jitter, values, s=18, color=color, alpha=0.8, zorder=3)
        ax.hlines(row["mean"], x - 0.25, x + 0.25, colors=c.ink, linewidth=1.2, zorder=4)
    ax.set_xticks(range(len(ranked)),
                  [config_label(r.get("labels") or {}, axes) for r in ranked],
                  rotation=20 if len(ranked) > 4 else 0, ha="right" if len(ranked) > 4
                  else "center", fontsize=7.5)
    ax.set_ylabel(f"{metric} {_arrow(mode).split(' ')[0]}")
    ax.yaxis.set_major_formatter(number_formatter())


@builtin_view("experiments", "overview")
def overview_view(fig: Any, data: Any, *, theme: str, options: Any) -> Any:
    goal = data.get("goal") or {}
    metric, mode = goal.get("metric", "metric"), goal.get("mode", "min")
    rows = list(data.get("summaries") or ())
    trials = sum(len(r.get("trials") or ()) for r in rows)
    from topologies.visualization.matplotlib.style import header_gridspec

    gs = header_gridspec(fig, 2, 2, width_ratios=[1.05, 1])
    lead = fig.add_subplot(gs[1, 0])
    curves = fig.add_subplot(gs[1, 1])
    best = fig.add_subplot(gs[2, 0])
    pivot = fig.add_subplot(gs[2, 1])
    leaderboard_panel(lead, data, theme=theme)
    curves_panel(curves, data, theme=theme)
    best_so_far_panel(best, data, theme=theme)
    pivot_panel(pivot, data, theme=theme)
    title = getattr(options, "title", None) or "Experiment results"
    arrow = "↓" if mode == "min" else "↑"
    subtitle = f"goal {metric} {arrow} · {trials} trials · {len(rows)} configs"
    header(fig, gs[0, :], title, subtitle, theme)
    return fig


@builtin_view("experiments", "leaderboard")
def leaderboard_view(fig: Any, data: Any, *, theme: str, options: Any) -> Any:
    leaderboard_panel(fig.subplots(), data, theme=theme)
    return fig


__all__ = ["best_so_far_panel", "config_label", "curves_panel", "leaderboard_panel",
           "mean_curve", "overview_view", "pivot_panel", "strip_panel", "varying_axes"]
