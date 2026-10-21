"""Matplotlib renderers for every generic ``ViewSpec`` kind (imported lazily).

A renderer has the signature ``renderer(fig, cell, spec)``, where ``cell`` is a
``SubplotSpec``. Single-axes kinds call ``fig.add_subplot(cell)``. Grid kinds
subdivide the cell with ``cell.subgridspec``.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from .views import DEFAULT_VIEW_RENDERERS, ViewSpec, register_view_renderer


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=float)


def _image(value: Any) -> np.ndarray:
    arr = _array(value)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 3:
        lo, hi = np.nanmin(arr), np.nanmax(arr)
        arr = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
    return arr


def _finish(ax: Any, spec: ViewSpec) -> None:
    ax.set_title(spec.title, fontsize=9)
    if spec.caption:
        ax.annotate(spec.caption, (0, 0), xycoords="axes fraction", xytext=(0, -24),
                    textcoords="offset points", fontsize=7, color="0.35", va="top", wrap=True)


def heatmap(fig: Any, cell: Any, spec: ViewSpec, ax: Any = None) -> Any:
    ax = ax or fig.add_subplot(cell)
    data = spec.data
    matrix = _array(data["matrix"])
    if matrix.ndim == 1:
        matrix = matrix[None]
    center = data.get("center", spec.options.get("center"))
    kwargs = {"cmap": spec.options.get("cmap", "RdBu_r" if center is not None else "viridis"),
              "aspect": "auto", "interpolation": "nearest"}
    if center is not None:
        bound = np.nanmax(np.abs(matrix - center)) or 1.0
        kwargs.update(vmin=center - bound, vmax=center + bound)
    image = ax.imshow(matrix, **kwargs)
    _labels(ax, data.get("row_labels"), data.get("col_labels"))
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    ax.set_xlabel(spec.options.get("x_label", ""))
    ax.set_ylabel(spec.options.get("y_label", ""))
    _finish(ax, spec)
    return ax


def _labels(ax: Any, rows: Any, cols: Any, limit: int = 40) -> None:
    if rows is not None and 0 < len(rows) <= limit:
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([str(r) for r in rows], fontsize=7)
    if cols is not None and 0 < len(cols) <= limit:
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([str(c) for c in cols], fontsize=7, rotation=90)


def heatmap_grid(fig: Any, cell: Any, spec: ViewSpec) -> Any:
    maps = list(spec.data.get("maps", []))
    labels = list(spec.data.get("labels", [""] * len(maps)))
    if not maps:
        return text(fig, cell, ViewSpec("text", spec.title, {"text": "no maps"}))
    cols = int(spec.data.get("columns") or min(4, len(maps)))
    rows = math.ceil(len(maps) / cols)
    grid = cell.subgridspec(rows, cols, wspace=0.3, hspace=0.5)
    axes = []
    for i, (entry, label) in enumerate(zip(maps, labels)):
        ax = fig.add_subplot(grid[i // cols, i % cols])
        matrix = _array(entry["matrix"] if isinstance(entry, Mapping) else entry)
        ax.imshow(matrix if matrix.ndim > 1 else matrix[None], aspect="auto",
                  cmap=spec.options.get("cmap", "viridis"), interpolation="nearest")
        ax.set_title(str(label), fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        axes.append(ax)
    axes[0].set_title(f"{spec.title}\n{labels[0]}", fontsize=7)
    return axes


def image_grid(fig: Any, cell: Any, spec: ViewSpec) -> Any:
    images = list(spec.data.get("images", []))
    labels = list(spec.data.get("labels", [""] * len(images)))
    overlays = spec.data.get("overlay")
    if not images:
        return text(fig, cell, ViewSpec("text", spec.title, {"text": "no images"}))
    cols = int(spec.data.get("columns") or min(8, len(images)))
    rows = math.ceil(len(images) / cols)
    grid = cell.subgridspec(rows, cols, wspace=0.05, hspace=0.3)
    axes = []
    for i, image in enumerate(images):
        ax = fig.add_subplot(grid[i // cols, i % cols])
        arr = _image(image)
        ax.imshow(arr, cmap=None if arr.ndim == 3 else spec.options.get("cmap", "gray"))
        if overlays is not None and i < len(overlays) and overlays[i] is not None:
            ax.imshow(_array(overlays[i]), cmap="jet", alpha=0.45,
                      extent=(-0.5, arr.shape[1] - 0.5, arr.shape[0] - 0.5, -0.5))
        ax.set_title(str(labels[i]) if i < len(labels) else "", fontsize=7)
        ax.axis("off")
        axes.append(ax)
    axes[0].set_title(f"{spec.title}\n{labels[0] if labels else ''}", fontsize=7)
    return axes


def token_heat(fig: Any, cell: Any, spec: ViewSpec) -> Any:
    ax = fig.add_subplot(cell)
    tokens = [str(t) for t in spec.data.get("tokens", [])]
    scores = _array(spec.data.get("scores", []))
    if scores.ndim == 1:
        scores = scores[None]
    bound = np.nanmax(np.abs(scores)) or 1.0
    ax.imshow(scores, cmap=spec.options.get("cmap", "RdBu_r"), vmin=-bound, vmax=bound,
              aspect="auto", interpolation="nearest")
    if scores.shape[1] == len(tokens) and len(tokens) <= 60:
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels([repr(t)[1:-1] for t in tokens], rotation=90, fontsize=7)
    rows = spec.data.get("rows")
    if rows is not None and len(rows) == scores.shape[0] <= 40:
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([str(r) for r in rows], fontsize=7)
    else:
        ax.set_yticks([])
    _finish(ax, spec)
    return ax


def lines(fig: Any, cell: Any, spec: ViewSpec, ax: Any = None) -> Any:
    ax = ax or fig.add_subplot(cell)
    for series in spec.data.get("series", []):
        y = _array(series["y"])
        x = _array(series["x"]) if series.get("x") is not None else np.arange(len(y))
        ax.plot(x, y, marker="." if len(y) < 40 else None, label=str(series.get("label", "")))
    ax.set_xlabel(spec.data.get("x_label", ""))
    ax.set_ylabel(spec.data.get("y_label", ""))
    if spec.options.get("log_y"):
        ax.set_yscale("log")
    if spec.data.get("x_ticks"):
        ticks = spec.data["x_ticks"]
        if len(ticks) <= 40:
            ax.set_xticks(range(len(ticks)))
            ax.set_xticklabels([str(t) for t in ticks], rotation=90, fontsize=7)
    if any(s.get("label") for s in spec.data.get("series", [])):
        ax.legend(fontsize=7)
    _finish(ax, spec)
    return ax


def bars(fig: Any, cell: Any, spec: ViewSpec, ax: Any = None) -> Any:
    ax = ax or fig.add_subplot(cell)
    labels = [str(label) for label in spec.data.get("labels", [])]
    values = _array(spec.data.get("values", []))
    errors = spec.data.get("errors")
    positions = np.arange(len(values))
    ax.bar(positions, values, yerr=_array(errors) if errors is not None else None,
           color=spec.options.get("color", "C0"))
    if len(labels) <= 50:
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel(spec.data.get("y_label", ""))
    _finish(ax, spec)
    return ax


def histogram(fig: Any, cell: Any, spec: ViewSpec, ax: Any = None) -> Any:
    ax = ax or fig.add_subplot(cell)
    series = spec.data.get("series") or [{"edges": spec.data.get("edges", []),
                                          "counts": spec.data.get("counts", []),
                                          "label": spec.data.get("label", "")}]
    for entry in series:
        edges, counts = _array(entry["edges"]), _array(entry["counts"])
        if len(edges) == len(counts) + 1 and len(counts):
            ax.stairs(counts, edges, label=str(entry.get("label", "")), fill=len(series) == 1,
                      alpha=0.6 if len(series) == 1 else 1.0)
    if spec.options.get("log_y"):
        ax.set_yscale("log")
    if len(series) > 1:
        ax.legend(fontsize=7)
    _finish(ax, spec)
    return ax


def scatter(fig: Any, cell: Any, spec: ViewSpec, ax: Any = None) -> Any:
    ax = ax or fig.add_subplot(cell)
    points = _array(spec.data.get("points", []))
    labels = spec.data.get("labels")
    if points.size:
        if labels is not None and len(labels) == len(points):
            classes = sorted({str(label) for label in labels})
            for c in classes:
                mask = np.array([str(label) == c for label in labels])
                ax.scatter(points[mask, 0], points[mask, 1], s=8, label=c)
            if len(classes) <= 12:
                ax.legend(fontsize=7)
        else:
            ax.scatter(points[:, 0], points[:, 1], s=8)
    ax.set_xlabel(spec.data.get("x_label", ""))
    ax.set_ylabel(spec.data.get("y_label", ""))
    _finish(ax, spec)
    return ax


def spectrum(fig: Any, cell: Any, spec: ViewSpec, ax: Any = None) -> Any:
    ax = ax or fig.add_subplot(cell)
    values = _array(spec.data.get("values", []))
    if values.size:
        ax.plot(np.arange(1, len(values) + 1), values, marker=".")
        ax.set_yscale("log")
        ax.set_xscale("log")
    fit = spec.data.get("fit")
    if isinstance(fit, Mapping) and fit.get("x") and fit.get("y"):
        ax.plot(fit["x"], fit["y"], "--", label=str(fit.get("label", "fit")))
        ax.legend(fontsize=7)
    ax.set_xlabel("rank")
    ax.set_ylabel("singular value")
    _finish(ax, spec)
    return ax


def tree(fig: Any, cell: Any, spec: ViewSpec, ax: Any = None) -> Any:
    """Indented table chart (no graphviz dependency)."""
    ax = ax or fig.add_subplot(cell)
    nodes = list(spec.data.get("nodes", []))[: int(spec.options.get("max_rows", 40))]
    ax.axis("off")
    for i, node in enumerate(nodes):
        indent = "  " * int(node.get("depth", 0))
        label = node.get("label") or node.get("path", "")
        metrics = node.get("metrics") or {}
        extra = "  ".join(f"{k}={_short(v)}" for k, v in list(metrics.items())[:4])
        ax.text(0.0, 1 - (i + 0.5) / max(len(nodes), 1), f"{indent}{label}   {extra}",
                family="monospace", fontsize=7, va="center", transform=ax.transAxes)
    _finish(ax, spec)
    return ax


def table(fig: Any, cell: Any, spec: ViewSpec, ax: Any = None) -> Any:
    ax = ax or fig.add_subplot(cell)
    ax.axis("off")
    columns = [str(c) for c in spec.data.get("columns", [])]
    rows = [[_short(v) for v in (r.values() if isinstance(r, Mapping) else r)]
            for r in list(spec.data.get("rows", []))[: int(spec.options.get("max_rows", 25))]]
    if columns and rows:
        tbl = ax.table(cellText=rows, colLabels=columns, loc="upper center", cellLoc="left")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
    _finish(ax, spec)
    return ax


def text(fig: Any, cell: Any, spec: ViewSpec, ax: Any = None) -> Any:
    ax = ax or fig.add_subplot(cell)
    ax.axis("off")
    ax.text(0.0, 1.0, str(spec.data.get("text", "")), va="top", ha="left", fontsize=8,
            wrap=True, transform=ax.transAxes)
    _finish(ax, spec)
    return ax


def _short(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    text_value = str(value)
    return text_value if len(text_value) <= 40 else text_value[:37] + "..."


_SINGLE = {"heatmap": heatmap, "lines": lines, "bars": bars, "histogram": histogram,
           "scatter": scatter, "spectrum": spectrum, "tree": tree, "table": table, "text": text}
_ALL = {**_SINGLE, "heatmap_grid": heatmap_grid, "image_grid": image_grid,
        "token_heat": token_heat}

for _kind, _renderer in _ALL.items():
    if DEFAULT_VIEW_RENDERERS.get("matplotlib", _kind) is None:
        register_view_renderer("matplotlib", _kind, _renderer)


def render_views(fig: Any, views: list[ViewSpec], *, registry: Any = None, columns: int | None = None
                 ) -> list[Any]:
    """Lay out ``views`` as a grid of panels on ``fig``."""
    registry = registry or DEFAULT_VIEW_RENDERERS
    count = len(views)
    if count == 0:
        ax = fig.add_subplot(1, 1, 1)
        ax.axis("off")
        ax.text(0.5, 0.5, "no views", ha="center", va="center")
        return [ax]
    cols = columns or (1 if count == 1 else 2 if count <= 4 else 3)
    rows = math.ceil(count / cols)
    width, height = fig.get_size_inches()
    fig.set_size_inches(max(width, 5.5 * cols), max(height, 4.2 * rows))
    grid = fig.add_gridspec(rows, cols, hspace=0.7, wspace=0.35)
    out = []
    for i, spec in enumerate(views):
        renderer = registry.get("matplotlib", spec.kind) or _fallback
        out.append(renderer(fig, grid[i // cols, i % cols], spec))
    return out


def render_view_on_axes(ax: Any, spec: ViewSpec) -> Any:
    """Render a single-axes view onto caller-owned axes (``analysis.render(ax=...)``)."""
    renderer = _SINGLE.get(spec.kind)
    if renderer is None:
        return text(ax.figure, None, ViewSpec("text", spec.title,
                                              {"text": f"{spec.kind} needs its own figure"}), ax=ax)
    return renderer(ax.figure, None, spec, ax=ax)


def _fallback(fig: Any, cell: Any, spec: ViewSpec) -> Any:
    return text(fig, cell, ViewSpec("text", spec.title,
                                    {"text": f"no renderer for view kind {spec.kind!r}"}))


__all__ = ["render_view_on_axes", "render_views"]
