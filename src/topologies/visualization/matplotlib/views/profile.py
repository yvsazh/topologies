"""Profile view: latency distribution with percentile hairlines and a stat line."""

from __future__ import annotations

import math
from typing import Any

from topologies.visualization.format import fmt_bytes
from topologies.visualization.matplotlib.style import colors, empty_state, header

from . import builtin_view


def _fmt_seconds(value: float) -> str:
    if value >= 1:
        return f"{value:.3g} s"
    if value >= 1e-3:
        return f"{value * 1e3:.3g} ms"
    return f"{value * 1e6:.3g} µs"


def percentile(values: list[float], q: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)


@builtin_view("result:profile", "default")
def profile_view(fig: Any, data: Any, *, theme: str, options: Any) -> Any:
    c = colors(theme)
    p = data.get("payload", data)
    samples = [float(v) for v in p.get("latency_seconds", ()) if isinstance(v, (int, float))]
    from topologies.visualization.matplotlib.style import header_gridspec, seconds_formatter

    gs = header_gridspec(fig, 1, 1)
    ax = fig.add_subplot(gs[1])
    title = getattr(options, "title", None) or "Measured runtime profile"
    if samples:
        mean = sum(samples) / len(samples)
        sd = math.sqrt(sum((v - mean) ** 2 for v in samples) / max(1, len(samples) - 1))
        if len(samples) < 30:
            ax.scatter(samples, [0.5 + ((i % 7) - 3) * 0.04 for i in range(len(samples))],
                       s=24, color=c.series[0], alpha=0.8)
            ax.set_yticks([])
            ax.set_ylim(0, 1)
        else:
            ax.hist(samples, bins=min(40, max(10, len(samples) // 5)), color=c.series[0],
                    alpha=0.85)
            ax.set_ylabel("calls")
        for k, (q, name) in enumerate(((0.5, "p50"), (0.9, "p90"), (0.99, "p99"))):
            v = percentile(samples, q)
            ax.axvline(v, color=c.ink2, linewidth=0.8, linestyle="dotted")
            ax.annotate(f"{name} {_fmt_seconds(v)}", (v, 1), xycoords=("data", "axes fraction"),
                        xytext=(-3, -12 - 12 * k), textcoords="offset points", ha="right",
                        fontsize=8, color=c.ink2,
                        bbox={"boxstyle": "round,pad=0.15", "facecolor": c.surface,
                              "edgecolor": "none", "alpha": 0.85})
        ax.xaxis.set_major_formatter(seconds_formatter())
        ax.set_xlabel("latency (seconds / call)")
        stat = f"mean {_fmt_seconds(mean)} ± {_fmt_seconds(sd)} · n={len(samples)}"
    elif isinstance(p.get("mean_latency_seconds"), (int, float)):
        ax.barh(["mean latency"], [p["mean_latency_seconds"]], color=c.series[0])
        ax.set_xlabel("latency (seconds / call)")
        stat = f"mean {_fmt_seconds(float(p['mean_latency_seconds']))}"
    else:
        empty_state(ax, "Latency unavailable", "profile with RuntimeProfile(...)", theme)
        stat = "no samples"
    peak = p.get("peak_memory_bytes")
    stat += f" · CUDA peak {fmt_bytes(peak)}" if isinstance(peak, (int, float)) \
        else " · CPU peak unknown"
    ax.set_title("latency")
    header(fig, gs[0], title, stat, theme)
    return fig


__all__ = ["percentile", "profile_view"]
