"""Optional downstream visualization of analysis artifacts; never called by analyze()."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from topologies.analysis.core import field_value
from topologies.analysis.statistics import FitArtifact


class RendererRegistry:
    """Local renderer extension point, keyed by result kind or payload type."""

    def __init__(self) -> None:
        self.renderers: dict[Any, Callable] = {}

    def register(self, key: Any, renderer: Callable) -> None:
        if not callable(renderer):
            raise TypeError("renderer must be callable")
        self.renderers[key] = renderer

    def get(self, key: Any) -> Callable | None:
        return self.renderers.get(key)


_DEFAULT_RENDERERS = RendererRegistry()


def _views_spec(result: Any, *, x: str | None = None, y: str | None = None) -> dict[str, Any]:
    return {"kind": "views", "views": list(result.payload.views())}


_DEFAULT_RENDERERS.register(("spec", "introspection.*"), _views_spec)


def register_renderer(kind: Any, renderer: Callable) -> None:
    """Register a convenience global renderer; prefer a local registry in plugins.

    A renderer registered for a result ``kind`` is also exposed to the Matplotlib backend
    as the ``result:<kind>`` view, so ``visualize(result)`` uses it as well.
    """
    _DEFAULT_RENDERERS.register(kind, renderer)
    if isinstance(kind, str):
        from topologies.visualization.matplotlib.views import register_view

        def view(fig: Any, data: Any, *, theme: str, options: Any) -> Any:
            source = data.get("result", data) if isinstance(data, Mapping) else data
            ax = fig.subplots()
            out = renderer(source, ax=ax)
            return getattr(out, "figure", None) or fig

        register_view(f"result:{kind}", "default")(view)


def plot_spec(result: Any, *, x: str | None = None, y: str | None = None,
              registry: RendererRegistry | None = None) -> dict[str, Any]:
    """Return backend-neutral fit/record data or invoke a registered spec factory."""
    payload = getattr(result, "payload", result)
    kind = getattr(result, "kind", None)
    registry = registry or _DEFAULT_RENDERERS
    factory = registry.get(("spec", kind)) or registry.get(("spec", type(payload)))
    if factory is None and isinstance(kind, str) and "." in kind:
        # Prefix factories such as ("spec", "introspection.*") cover a family of kinds.
        prefix = kind.split(".", 1)[0]
        factory = registry.get(("spec", f"{prefix}.*"))
    if factory:
        return factory(result, x=x, y=y)
    if isinstance(payload, FitArtifact):
        xs = [point[0] for point in payload.observations]
        grid = _grid(xs)
        curve = [payload.predict(value) for value in grid]
        intervals = [point.get("confidence_interval") for point in curve]
        return {"kind": "fit", "x_label": x or "x", "y_label": y or "y",
                "observations": list(payload.observations), "curve": curve,
                "confidence_band": ({"x": grid, "lower": [i[0] for i in intervals],
                                     "upper": [i[1] for i in intervals],
                                     "label": f"{payload.interval['confidence']:.0%} confidence interval"}
                                    if intervals and all(i is not None for i in intervals) else None),
                "diagnostics": {"rmse": payload.rmse, "r_squared": payload.r_squared,
                                "residual_domain": payload.residual_domain},
                "method": payload.method, "parameters": dict(payload.parameters),
                "n": payload.n, "residuals": list(payload.residuals)}
    if isinstance(payload, (list, tuple)) and all(isinstance(row, Mapping) for row in payload):
        points = [(field_value(row, x), field_value(row, y)) for row in payload] if x and y else []
        return {"kind": "records", "x": x, "y": y, "points": points, "rows": list(payload)}
    raise TypeError(f"No plot specification for {type(payload).__name__}; register a spec factory")


def render(result: Any, *, ax: Any = None, renderer: Callable | None = None,
           x: str | None = None, y: str | None = None,
           registry: RendererRegistry | None = None) -> Any:
    """Render on demand, returning caller-owned axes; never displays a figure."""
    payload = getattr(result, "payload", result)
    kind = getattr(result, "kind", None)
    registry = registry or _DEFAULT_RENDERERS
    custom = renderer or registry.get(kind) or registry.get(type(payload))
    if custom:
        return custom(result, ax=ax)
    spec = plot_spec(result, x=x, y=y, registry=registry)
    from topologies.visualization.matplotlib import draw_analysis_spec
    from topologies.visualization.matplotlib.style import themed

    with themed("light"):
        if ax is None:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure

            fig = Figure(figsize=(8, 5))
            FigureCanvasAgg(fig)
            ax = fig.subplots()
        draw_analysis_spec(ax, spec, x=x, y=y)
    return ax


def visualize(result: Any, *, visualizer="matplotlib", views: Any = None,
              view_options: Any = None, **options: Any) -> Any:
    """Explicit convenience rendering; analyze/report never plots implicitly."""
    from topologies.visualization import create_visualizer, render_result

    view_options = dict(view_options or {})
    if views is not None:
        view_options["kinds"] = views
    backend = create_visualizer(visualizer, **options)
    try:
        backend.prepare()
        return render_result(result, backend, view_options=view_options)
    finally:
        backend.close()


def _grid(xs: list[float], count: int = 100) -> list[float]:
    lo, hi = min(xs), max(xs)
    if lo == hi:
        return xs
    return [lo + (hi - lo) * index / (count - 1) for index in range(count)]
