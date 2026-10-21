"""Engineering view: memory breakdown, top layers by parameters and MACs, feasibility.

Honesty rules: only known components are drawn, unknown ones are listed explicitly and
totals are omitted when anything is unknown.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from topologies.visualization.format import fmt_bytes, fmt_count
from topologies.visualization.matplotlib.style import (
    bytes_formatter,
    colors,
    count_formatter,
    empty_state,
    header,
)

from . import builtin_view

MEMORY_COMPONENTS = ("parameter_storage", "buffer_storage", "gradient_storage",
                     "optimizer_state", "master_weight_copy", "activation_bytes",
                     "workspace_bytes")
TOP_LAYERS = 15


def memory_parts(payload: Mapping[str, Any]) -> tuple[list[tuple[str, float]], list[str]]:
    memory = payload.get("memory")
    if memory is None and "parameter_storage" in payload:
        memory = payload
    known, unknown = [], []
    if isinstance(memory, Mapping):
        for name in MEMORY_COMPONENTS:
            item = memory.get(name)
            if isinstance(item, Mapping):
                value = item.get("value")
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    known.append((name, float(value)))
                else:
                    unknown.append(name)
    for name in payload.get("unknown_components") or ():
        if name not in unknown:
            unknown.append(str(name))
    return known, unknown


def layer_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    structure = payload.get("structure", payload if "module_parameters" in payload else None)
    compute = payload.get("compute", payload if "modules" in payload and "macs" in payload
                          else None)
    rows = list(structure.get("module_parameters", ())) if isinstance(structure, Mapping) else []
    macs = ({r.get("module"): r.get("macs") for r in compute.get("modules", ())
             if isinstance(r.get("macs"), (int, float))} if isinstance(compute, Mapping) else {})
    return [{**r, "macs": macs.get(r.get("module"))} for r in rows]


def short_name(name: str, width: int = 28) -> str:
    """Left-truncate long module paths, keeping the leaf name."""
    name = name or "root"
    return name if len(name) <= width else "…" + name[-(width - 1):]


def _bars(ax: Any, items: Sequence[tuple[str, float]], *, theme: str, unit: str,
          title: str) -> None:
    c = colors(theme)
    items = sorted(items, key=lambda kv: -kv[1])
    other = items[TOP_LAYERS:]
    items = items[:TOP_LAYERS]
    if other:
        items.append((f"other ({len(other)} layers)", sum(v for _, v in other)))
    labels = [short_name(k) for k, _ in items][::-1]
    values = [v for _, v in items][::-1]
    bars = ax.barh(labels, values, color=c.series[0], height=0.6)
    if other:
        bars[0].set_color(c.muted)
    fmt = fmt_bytes if unit == "bytes" else fmt_count
    for bar, value in zip(bars, values):
        ax.annotate(fmt(value), (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(3, 0), textcoords="offset points", va="center", fontsize=7.5,
                    color=c.ink2)
    ax.xaxis.set_major_formatter(bytes_formatter() if unit == "bytes" else count_formatter())
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    ax.tick_params(axis="y", labelsize=7.5)


def memory_panel(ax: Any, known: list[tuple[str, float]], unknown: list[str], *,
                 theme: str) -> None:
    c = colors(theme)
    total = sum(v for _, v in known)
    left = 0.0
    for i, (name, value) in enumerate(known):
        color = c.series[i % len(c.series)]
        ax.barh([0], [value], left=left, color=color, height=0.5, edgecolor=c.surface,
                linewidth=2, label=name.replace("_", " "))
        if total and value / total >= 0.08:
            ax.text(left + value / 2, 0, f"{name.replace('_', ' ')}\n{fmt_bytes(value)}",
                    ha="center", va="center", fontsize=7.5, color="#ffffff")
        left += value
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(bytes_formatter())
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    title = "known memory components"
    if unknown:
        title += "  ·  total omitted (unknown components)"
    else:
        title += f"  ·  total {fmt_bytes(total)}"
    ax.set_title(title)
    from matplotlib.patches import Patch

    small = [n for n, v in known if not total or v / total < 0.08]
    handles, labels = ax.get_legend_handles_labels()
    keep = [(h, lab) for h, lab in zip(handles, labels) if lab.replace(" ", "_") in small]
    if unknown:
        keep.append((Patch(facecolor="none", edgecolor=c.muted, hatch="///"),
                     "unknown: " + ", ".join(u.replace("_", " ") for u in unknown)))
    if keep:
        ax.legend(*zip(*keep), loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8,
                  ncols=min(4, len(keep)), frameon=False)


def feasibility_panel(ax: Any, payload: Mapping[str, Any], *, theme: str) -> None:
    c = colors(theme)
    bound = float(payload["known_lower_bound_bytes"])
    capacity = payload.get("capacity_bytes")
    status = str(payload.get("status") or "indeterminate")
    good = status.startswith(("within", "fits", "feasible"))
    bad = status in ("insufficient", "exceeds", "infeasible") or status.startswith("exceed")
    color = c.good if good else (c.critical if bad else c.warning)
    ax.barh([0], [bound], color=c.series[0], height=0.35, label="known lower bound")
    if isinstance(capacity, (int, float)):
        ax.axvline(capacity, color=c.ink, linewidth=1.6, ymin=0.15, ymax=0.85)
        ax.annotate(f"capacity {fmt_bytes(capacity)}", (capacity, 0.3), xytext=(4, 0),
                    textcoords="offset points", fontsize=8, color=c.ink2)
        cap_label = "lower-bound comparison"
    else:
        cap_label = "capacity unknown"
    ax.annotate(f"lower bound ≥ {fmt_bytes(bound)}", (bound, -0.3), xytext=(4, 0),
                textcoords="offset points", fontsize=8, color=c.ink2)
    ax.set_yticks([])
    ax.set_ylim(-0.7, 0.7)
    ax.xaxis.set_major_formatter(bytes_formatter())
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"feasibility · {status} · {cap_label}")
    ax.text(0.99, 0.95, f"● {status}", transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color=color, fontweight=600)
    if payload.get("unknown_components"):
        ax.text(0.99, 0.05, "Unknown: " + ", ".join(payload["unknown_components"]),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=c.muted)


@builtin_view("result:engineering", "default")
def engineering_view(fig: Any, data: Any, *, theme: str, options: Any) -> Any:
    payload = data.get("payload", data)
    result = data.get("result")
    known, unknown = memory_parts(payload)
    rows = layer_rows(payload)
    params = [(r.get("module") or "root", float(r["unique_parameters"])) for r in rows
              if isinstance(r.get("unique_parameters"), (int, float)) and r["unique_parameters"]]
    macs = [(r.get("module") or "root", float(r["macs"])) for r in rows
            if isinstance(r.get("macs"), (int, float))]
    panels: list[tuple[str, float]] = []
    if known or unknown and "known_lower_bound_bytes" not in payload:
        panels.append(("memory", 1.0))
    if params or macs:
        panels.append(("layers", 0.6 + 0.35 * min(TOP_LAYERS + 1, max(len(params),
                                                                      len(macs)))))
    if "known_lower_bound_bytes" in payload:
        panels.append(("feasibility", 1.2))
    if not panels:
        ax = fig.subplots()
        empty_state(ax, "No supported known engineering measurements",
                    "analyze_model(..., input_shape=...) adds structure and compute", theme)
        return fig
    from topologies.visualization.matplotlib.style import header_gridspec

    fig.set_size_inches(11, 1.0 + 1.3 * sum(h for _, h in panels))
    gs = header_gridspec(fig, len(panels), 2, height_ratios=[h for _, h in panels])
    for row, (name, _h) in enumerate(panels, start=1):
        if name == "memory":
            memory_panel(fig.add_subplot(gs[row, :]), known, unknown, theme=theme)
        elif name == "feasibility":
            feasibility_panel(fig.add_subplot(gs[row, :]), payload, theme=theme)
        elif params and macs:
            _bars(fig.add_subplot(gs[row, 0]), params, theme=theme, unit="count",
                  title="top layers by parameters")
            _bars(fig.add_subplot(gs[row, 1]), macs, theme=theme, unit="count",
                  title="top layers by MACs")
        else:
            _bars(fig.add_subplot(gs[row, :]), params or macs, theme=theme, unit="count",
                  title="top layers by parameters" if params else "top layers by MACs")
    warnings = list(getattr(result, "warnings", ()) or ())
    subtitle = "known measurements only" + (f" · warnings: {'; '.join(warnings)}"
                                            if warnings else "")
    header(fig, gs[0, :], getattr(options, "title", None) or "Model engineering", subtitle,
           theme)
    return fig


__all__ = ["engineering_view", "feasibility_panel", "layer_rows", "memory_panel",
           "memory_parts", "short_name"]
