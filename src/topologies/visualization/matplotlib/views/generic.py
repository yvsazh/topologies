"""Generic fallback: sorted horizontal bars of numeric leaves, or an empty state."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from topologies.visualization.format import fmt_number
from topologies.visualization.matplotlib.style import colors, empty_state, number_formatter

from . import builtin_view

MAX_BARS = 20


def numeric_leaves(value: Any, prefix: str = "") -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    if isinstance(value, Mapping):
        for k, v in value.items():
            if k == "kind" and not prefix:
                continue
            out.extend(numeric_leaves(v, f"{prefix}{k}."))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out.append((prefix.rstrip("."), float(value)))
    return out


@builtin_view("result:generic", "default")
def generic_view(fig: Any, data: Any, *, theme: str, options: Any, kind: str = "result") -> Any:
    c = colors(theme)
    ax = fig.subplots()
    pairs = numeric_leaves(data) if isinstance(data, Mapping) else []
    title = getattr(options, "title", None) or kind
    if not pairs:
        empty_state(ax, "No numeric values in this payload",
                    "register a view with register_view(\"result:<kind>\", ...)", theme)
        ax.set_title(title)
        return fig
    pairs = sorted(pairs, key=lambda kv: -abs(kv[1]))
    extra = len(pairs) - MAX_BARS
    pairs = pairs[:MAX_BARS][::-1]
    bars = ax.barh([k for k, _ in pairs], [v for _, v in pairs], color=c.series[0], height=0.6)
    for bar, (_k, v) in zip(bars, pairs):
        ax.annotate(fmt_number(v, sig=3), (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(3 if v >= 0 else -3, 0), textcoords="offset points", va="center",
                    ha="left" if v >= 0 else "right", fontsize=8, color=c.ink2)
    if any(v < 0 for _k, v in pairs):
        ax.axvline(0, color=c.baseline, linewidth=0.8)
    ax.xaxis.set_major_formatter(number_formatter())
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_title(title + (f"  ·  +{extra} more values" if extra > 0 else ""))
    return fig


@builtin_view("result:evaluation_report", "default")
def evaluation_view(fig: Any, data: Any, *, theme: str, options: Any) -> Any:
    """Metric values of an evaluation report (support in the label, counts in the title)."""
    c = colors(theme)
    metrics = data.get("metrics") if isinstance(data, Mapping) else None
    rows = []
    for name, item in (metrics or {}).items():
        value = item.get("value") if isinstance(item, Mapping) else item
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            support = item.get("support") if isinstance(item, Mapping) else None
            rows.append((f"{name} (n={support:g})" if isinstance(support, (int, float))
                         else name, float(value)))
    if not rows:
        return generic_view(fig, data, theme=theme, options=options, kind="evaluation")
    ax = fig.subplots()
    bars = ax.barh([r[0] for r in rows][::-1], [r[1] for r in rows][::-1], color=c.series[0],
                   height=0.55)
    for bar, (_n, v) in zip(bars, rows[::-1]):
        ax.annotate(fmt_number(v, sig=4), (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(3, 0), textcoords="offset points", va="center", fontsize=8,
                    color=c.ink2)
    ax.xaxis.set_major_formatter(number_formatter())
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    counts = [f"{k} {data[k]:,}" for k in ("records", "batches") if isinstance(data.get(k), int)]
    ax.set_title(getattr(options, "title", None) or
                 "evaluation" + (f"  ·  {' · '.join(counts)}" if counts else ""))
    fig.set_size_inches(8, 1.4 + 0.5 * len(rows))
    return fig


__all__ = ["evaluation_view", "generic_view", "numeric_leaves"]
