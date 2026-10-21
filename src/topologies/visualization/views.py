"""Producer-neutral view contract: ``ViewSpec`` + ``(backend, kind)`` renderer registry.

Any producer can emit ``ViewSpec`` objects; any visualization sink can render
them. This module is stdlib-only. ``data`` holds plain Python values or CPU
tensors and never holds live model handles.

View kinds and their ``data`` keys:

=============  ==============================================================
``heatmap``       ``matrix``, ``row_labels``, ``col_labels``, ``center``
``heatmap_grid``  ``maps`` (list of heatmap data), ``labels``, ``columns``
``image_grid``    ``images`` (C,H,W or H,W), ``labels``, optional ``overlay``
``token_heat``    ``tokens``, ``scores`` (1-D, or 2-D with ``rows``)
``lines``         ``series``: list of ``{"x", "y", "label"}``; ``x_label``/``y_label``
``bars``          ``labels``, ``values``, optional ``errors``
``histogram``     ``edges``, ``counts`` (or ``series`` of those)
``scatter``       ``points`` (list of (x, y)), optional ``labels``/``colors``
``spectrum``      ``values`` (singular values), optional ``fit``
``tree``          ``nodes``: list of ``{"path", "parent", "depth", "label", "metrics"}``
``table``         ``columns``, ``rows``
``text``          ``text`` (markdown)
=============  ==============================================================
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

VIEW_KINDS = ("heatmap", "heatmap_grid", "image_grid", "token_heat", "lines", "bars", "histogram",
              "scatter", "spectrum", "tree", "table", "text")


@dataclass(frozen=True)
class ViewSpec:
    """One backend-neutral figure panel."""

    kind: str
    title: str
    data: Mapping[str, Any]
    options: Mapping[str, Any] = field(default_factory=dict)
    caption: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("ViewSpec.kind must be a non-empty string")

    def with_options(self, **options: Any) -> ViewSpec:
        return ViewSpec(self.kind, self.title, self.data, {**self.options, **options}, self.caption)


def is_view_list(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and bool(value) and all(
        isinstance(v, ViewSpec) for v in value)


class ViewRendererRegistry:
    """``(backend, kind) -> renderer``; same shape as ``analysis.RendererRegistry``."""

    def __init__(self) -> None:
        self.renderers: dict[tuple[str, str], Callable] = {}

    def register(self, backend: str, kind: str, renderer: Callable) -> None:
        if not callable(renderer):
            raise TypeError("renderer must be callable")
        self.renderers[(backend, kind)] = renderer

    def get(self, backend: str, kind: str) -> Callable | None:
        return self.renderers.get((backend, kind))

    def kinds(self, backend: str) -> list[str]:
        return sorted(k for b, k in self.renderers if b == backend)


DEFAULT_VIEW_RENDERERS = ViewRendererRegistry()


def register_view_renderer(backend: str, kind: str, renderer: Callable) -> None:
    """Register (or restyle) a renderer globally; prefer a local registry in plugins."""
    DEFAULT_VIEW_RENDERERS.register(backend, kind, renderer)


def select_views(views: list[ViewSpec], *, kinds: Any = None, sites: Any = None,
                 max_panels: int | None = None) -> list[ViewSpec]:
    """Filter by kind and by a site substring/glob in the title or ``data["site"]``."""
    import fnmatch

    out = list(views)
    if kinds:
        wanted = {kinds} if isinstance(kinds, str) else set(kinds)
        out = [v for v in out if v.kind in wanted]
    if sites:
        patterns = [sites] if isinstance(sites, str) else list(sites)

        def matches(view: ViewSpec) -> bool:
            site = str(view.data.get("site", view.title)) if isinstance(view.data, Mapping) else \
                view.title
            return any(fnmatch.fnmatch(site, p) or p in site for p in patterns)
        out = [v for v in out if matches(v)]
    if max_panels is not None:
        out = out[:max_panels]
    return out


def describe_views(views: list[ViewSpec]) -> str:
    """Plain-text rendering used by terminal sinks (tables/text shown, figures listed)."""
    lines: list[str] = []
    figures = []
    for view in views:
        if view.kind == "text":
            lines.append(f"## {view.title}\n{view.data.get('text', '')}")
        elif view.kind == "table":
            lines.append(format_table(view.title, view.data.get("columns", []),
                                      view.data.get("rows", [])))
        else:
            figures.append(f"{view.kind}: {view.title}")
    if figures:
        lines.append(f"{len(figures)} figure(s) available; render with visualizer='matplotlib':")
        lines.extend(f"  - {f}" for f in figures)
    return "\n".join(lines)


def format_table(title: str, columns: list[str], rows: list[Any], *, limit: int = 40) -> str:
    rows = [list(r.values()) if isinstance(r, Mapping) else list(r) for r in rows[:limit]]
    cells = [[_fmt(c) for c in row] for row in rows]
    widths = [max([len(str(c)), *(len(r[i]) for r in cells if i < len(r))])
              for i, c in enumerate(columns)]
    head = " | ".join(str(c).ljust(w) for c, w in zip(columns, widths))
    rule = "-+-".join("-" * w for w in widths)
    body = [" | ".join(r[i].ljust(w) if i < len(r) else "".ljust(w) for i, w in enumerate(widths))
            for r in cells]
    return "\n".join([f"## {title}", head, rule, *body])


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


__all__ = ["DEFAULT_VIEW_RENDERERS", "VIEW_KINDS", "ViewRendererRegistry", "ViewSpec",
           "describe_views", "format_table", "is_view_list", "register_view_renderer",
           "select_views"]
