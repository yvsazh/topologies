"""Hardware view: per-device memory capacity (used vs free when measured)."""

from __future__ import annotations

from typing import Any

from topologies.visualization.format import fmt_bytes
from topologies.visualization.matplotlib.style import (
    bytes_formatter,
    colors,
    empty_state,
    header,
)

from . import builtin_view


@builtin_view("result:hardware", "default")
def hardware_view(fig: Any, data: Any, *, theme: str, options: Any) -> Any:
    c = colors(theme)
    p = data.get("payload", data)
    rows: list[tuple[str, float, float | None]] = []
    cpu = (p.get("cpu") or {}).get("ram_bytes")
    if isinstance(cpu, (int, float)):
        rows.append(("CPU RAM", float(cpu), None))
    for i, gpu in enumerate((p.get("cuda") or {}).get("gpus", ())):
        total = gpu.get("total_bytes")
        if isinstance(total, (int, float)):
            free = gpu.get("free_bytes")
            rows.append((f"GPU {i}" + (f" · {gpu['name']}" if gpu.get("name") else ""),
                         float(total), float(free) if isinstance(free, (int, float)) else None))
    from topologies.visualization.matplotlib.style import header_gridspec

    fig.set_size_inches(9, 1.6 + 0.6 * max(1, len(rows)))
    gs = header_gridspec(fig, 1, 1)
    ax = fig.add_subplot(gs[1])
    if not rows:
        empty_state(ax, "Hardware memory capacity unavailable", None, theme)
    else:
        labels = [r[0] for r in rows][::-1]
        for y, (_name, total, free) in zip(range(len(rows) - 1, -1, -1), rows):
            if free is None:
                ax.barh([y], [total], color=c.series[0], height=0.5)
                text = fmt_bytes(total)
            else:
                ax.barh([y], [total - free], color=c.series[0], height=0.5, label="used")
                ax.barh([y], [free], left=[total - free], color=c.series[0], alpha=0.3,
                        height=0.5, label="free")
                text = f"{fmt_bytes(total)} · {fmt_bytes(free)} free"
            ax.annotate(text, (total, y), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=c.ink2)
        ax.set_yticks(range(len(rows)), labels)
        ax.xaxis.set_major_formatter(bytes_formatter())
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.set_title("memory capacity")
    platform = (p.get("cpu") or {}).get("platform")
    accel = "MPS available" if (p.get("mps") or {}).get("available") else None
    subtitle = " · ".join(x for x in (platform, accel) if x) or "measured on this machine"
    header(fig, gs[0], getattr(options, "title", None) or "Hardware", subtitle, theme)
    return fig


__all__ = ["hardware_view"]
