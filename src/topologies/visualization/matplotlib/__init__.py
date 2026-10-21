"""Optional Matplotlib backend (imported lazily: ``matplotlib`` loads on first use).

``MatplotlibVisualizer`` renders static results through the view registry
(:func:`register_view`) and consumes live events into an incremental
:class:`~topologies.visualization.matplotlib.live.LiveFigure`. Figures are styled with
the shared palette inside ``rc_context`` (no global rcParams changes).
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from topologies.visualization.core import VisualizationResult, Visualizer

from .views import get_view, has_domain, register_view, view_names

_SCALES = {"linear", "log", "symlog", "logit", "auto"}
_LAYOUTS = {"auto", "grid", "single"}
_FIGSIZES = {"history": None, "experiments": (12, 8), "engineering": (11, 8),
             "analysis": (9, 6), "profile": (9, 4.5), "hardware": (9, 4)}


def topologies_style(theme: str = "light", **kwargs: Any) -> dict[str, Any]:
    """rc mapping for ``matplotlib.rc_context`` (see :mod:`.style`)."""
    from .style import topologies_style as _style

    return _style(theme, **kwargs)


def draw_analysis_spec(ax: Any, spec: Any, *, theme: str = "light", **kwargs: Any) -> Any:
    """Draw an analysis plot spec into a caller-owned axes (public API)."""
    from .views.analysis import draw_analysis_spec as _draw

    return _draw(ax, spec, theme=theme, **kwargs)


class MatplotlibVisualizer(Visualizer):
    """Render semantic results or consume normalized events into live charts.

    Returned figures belong to the caller for static rendering. Live-session
    figures belong to this visualizer and are closed by :meth:`close`.
    """

    def __init__(
        self,
        *,
        live=False,
        show_on_finish=False,
        save=None,
        format="png",
        metrics=None,
        charts=None,
        refresh_seconds=1.0,
        refresh_steps=10,
        history_limit=None,
        smoothing=1,
        title=None,
        x_scale="linear",
        y_scale="linear",
        figsize=None,
        theme="light",
        style=None,
        dpi=150,
        ema=None,
        max_series=8,
        layout="auto",
        views=None,
        max_panels=12,
        **options,
    ):
        if options:
            raise TypeError(f"unknown MatplotlibVisualizer option(s): {', '.join(sorted(options))}")
        self.live = bool(live)
        self.show_on_finish = bool(show_on_finish)
        self.save = save
        self.format = format.lower()
        if isinstance(metrics, str):
            metrics = [metrics]
        elif metrics is not None and (
            not isinstance(metrics, (list, tuple))
            or any(not isinstance(item, str) for item in metrics)
        ):
            raise TypeError("metrics must be a string, a list/tuple of strings, or None")
        if isinstance(charts, str):
            charts = [charts]
        elif charts is not None and (
            not isinstance(charts, (list, tuple))
            or any(not isinstance(item, str) for item in charts)
        ):
            raise TypeError("charts must be a string, a list/tuple of strings, or None")
        self.metrics = metrics
        self.charts = charts
        self.refresh_seconds = max(0.0, float(refresh_seconds))
        self.refresh_steps = max(1, int(refresh_steps))
        self.history_limit = history_limit
        self.smoothing = max(1, int(smoothing))
        self.title = title
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.figsize = figsize
        self.theme = theme
        self.style = style
        self.dpi = int(dpi)
        self.ema = ema
        self.max_series = max(1, int(max_series))
        self.layout = layout
        self.views = [views] if isinstance(views, str) else list(views) if views else None
        if isinstance(max_panels, bool) or not isinstance(max_panels, int) or max_panels < 1:
            raise ValueError("max_panels must be a positive integer")
        self.max_panels = max_panels
        if self.format not in {"png", "svg", "pdf"}:
            raise ValueError("format must be PNG, SVG, or PDF")
        if x_scale not in _SCALES or y_scale not in _SCALES:
            raise ValueError("x_scale and y_scale must be valid Matplotlib scales or 'auto'")
        if history_limit is not None and (not isinstance(history_limit, int)
                                          or isinstance(history_limit, bool) or history_limit < 1):
            raise ValueError("history_limit must be a positive integer or None")
        if theme not in ("light", "dark"):
            raise ValueError("theme must be 'light' or 'dark'")
        if ema is not None and not (0 < float(ema) < 1):
            raise ValueError("ema must be a float in (0, 1)")
        if layout not in _LAYOUTS:
            raise ValueError(f"layout must be one of {sorted(_LAYOUTS)}")
        if self.views:
            known = {name.split("/", 1)[1] for name in view_names()}
            unknown = [v for v in self.views if v not in known]
            if unknown:
                raise ValueError(f"unknown view(s) {unknown}; available: {sorted(known)}")
        self._plt = None
        self._events: list[Any] = []
        self._figure = None
        self._live = None
        self._last_render = 0.0
        self._last_step = 0
        self._refresh_dirty = True
        self._shown = False
        self._finalized = False
        self._stream_live_session = self.live
        self._tracker = None
        self._extra: dict[str, list[tuple[float, float]]] = {}
        self._completed: list[tuple[int, str, float]] = []
        self._best: list[tuple[int, float]] = []
        self._goal: tuple[str | None, str] = (None, "min")
        self._load()

    # -- setup ------------------------------------------------------------------
    def _load(self):
        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
        except ImportError as exc:
            raise ImportError(
                "Matplotlib visualization requires matplotlib; install topologies[visualization]"
            ) from exc
        self._Figure = Figure
        self._Agg = FigureCanvasAgg
        if self.live or self.show_on_finish:
            try:
                import matplotlib.pyplot as plt
            except ImportError as exc:
                raise ImportError("Interactive Matplotlib display requires pyplot") from exc
            self._plt = plt

    @property
    def options(self) -> SimpleNamespace:
        return SimpleNamespace(metrics=self.metrics, charts=self.charts, smoothing=self.smoothing,
                               ema=self.ema, title=self.title, x_scale=self.x_scale,
                               y_scale=self.y_scale, max_series=self.max_series,
                               layout=self.layout)

    def _themed(self):
        from .style import themed

        return themed(self.theme, self.style, dpi=self.dpi)

    def prepare(self, source=None, context=None):
        if self._figure is not None:
            self.close()
        from topologies.visualization.state import ExperimentTracker, TrainingTracker

        self._events = []
        self._last_render = 0.0
        self._last_step = 0
        self._refresh_dirty = True
        self._shown = False
        self._finalized = False
        self._stream_live_session = self.live
        self._extra = {}
        self._completed = []
        self._best = []
        self._goal = (None, "min")
        self._tracker = ExperimentTracker() if source == "experiment" else TrainingTracker()
        self._load()
        if self.live:
            self._make_live_figure()

    def _new_figure(self, size=None):
        size = self.figsize or size or (9, 5)
        with self._themed():
            if self.show_on_finish and self._plt:
                fig = self._plt.figure(figsize=size)
            else:
                fig = self._Figure(figsize=size)
                self._Agg(fig)
        return fig

    def _save(self, fig, suffix=""):
        if self.save is None:
            return
        path = Path(self.save)
        if path.suffix:
            out = path if not suffix else path.with_name(path.stem + suffix + path.suffix)
        else:
            path.mkdir(parents=True, exist_ok=True)
            out = path / f"visualization{suffix}.{self.format}"
        out.parent.mkdir(parents=True, exist_ok=True)
        with self._themed():
            fig.savefig(out, format=self.format, dpi=self.dpi)

    # -- static rendering ----------------------------------------------------------
    def _resolve(self, result) -> list[tuple[str, Any, Any]]:
        """``[(name, view, data)]`` for a result (one per selected view)."""
        kind = result.kind
        if kind in ("history", "training", "training_history"):
            domain, size = "training", None
        elif kind == "experiments":
            domain, size = "experiments", _FIGSIZES["experiments"]
        elif kind == "analysis":
            domain, size = "analysis", _FIGSIZES["analysis"]
        elif has_domain(f"result:{kind}"):
            domain, size = f"result:{kind}", _FIGSIZES.get(kind)
        else:
            domain, size = "result:generic", (9, 5)
        names = [v for v in self.views or () if get_view(domain, v) is not None] or [None]
        out = []
        for name in names:
            view = get_view(domain, name)
            if view is None:
                view = get_view("result:generic")
            if domain == "training" and size is None:
                from .views.training import default_figsize

                size = default_figsize(result.data)
            out.append((name or "default", view, size, domain))
        return out

    def render(self, result):
        self._load()
        if not isinstance(result, VisualizationResult):
            from topologies.visualization.adapters import adapt_result

            result = adapt_result(result)
        if result.kind == "views":
            return self._render_view_specs(result)
        figures = []
        for name, view, size, domain in self._resolve(result):
            fig = self._new_figure(size)
            try:
                with self._themed():
                    kwargs = {"kind": result.kind} if domain == "result:generic" else {}
                    returned = view(fig, result.data, theme=self.theme, options=self.options,
                                    **kwargs)
                    fig = returned if returned is not None else fig
                self._save(fig, "" if len(figures) == 0 else f"-{name}")
            except Exception:
                fig.clear()
                raise
            figures.append(fig)
        if self.show_on_finish and self._plt:
            _show(self._plt)
            self._shown = True
        return figures[0] if len(figures) == 1 else figures

    def _render_view_specs(self, result):
        """Lay out generic ``ViewSpec`` panels (introspection results, ``visualize``).

        At most ``max_panels`` panels go on one figure; with ``save`` set to a
        directory every further page is saved as ``visualization-<n>``.
        """
        from topologies.visualization.matplotlib_views import render_views

        views = list(result.data.get("views", ()))
        fig = self._new_figure((9, 5))
        with self._themed():
            render_views(fig, views[: self.max_panels])
            if self.title:
                fig.suptitle(self.title)
            if len(views) > self.max_panels:
                if self.save is not None and not Path(self.save).suffix:
                    pages = range(self.max_panels, len(views), self.max_panels)
                    for index, start in enumerate(pages, 1):
                        extra = self._new_figure((9, 5))
                        render_views(extra, views[start:start + self.max_panels])
                        self._save(extra, f"-{index}")
                        extra.clear()
                else:
                    fig.suptitle(f"{self.title or ''} (showing {self.max_panels} of {len(views)} "
                                 "views; save to a directory for every page)".strip())
        self._save(fig)
        if self.show_on_finish and self._plt:
            _show(self._plt)
            self._shown = True
        return fig

    def render_session(self, result):
        """Render the session result; live/stream sessions finish their own figure."""
        if self._events and any(event.source == "training" for event in self._events):
            self._draw_stream_final()
            return self._figure
        return self.render(result)

    def _draw_stream_final(self):
        if not self.live:
            self._make_live_figure(force_agg=True)
            self._refresh(force=True)
            if self.show_on_finish and self._plt and not self._shown:
                _show(self._plt)
                self._shown = True
            return
        self._refresh(force=True)

    # -- live ------------------------------------------------------------------------
    def _make_live_figure(self, force_agg=False):
        from .live import LiveFigure

        if self._figure is None:
            with self._themed():
                if self._plt and not force_agg:
                    self._figure = self._plt.figure(figsize=self.figsize or (11, 7))
                    _show(self._plt)
                    self._shown = True
                else:
                    self._figure = self._Figure(figsize=self.figsize or (11, 7))
                    self._Agg(self._figure)
            self._live = LiveFigure(self._figure, theme=self.theme, options=self.options,
                                    pyplot=self._plt if not force_agg else None)
        return self._figure

    def handle(self, event):
        self._events.append(event)
        self._refresh_dirty = True
        if self.history_limit and len(self._events) > self.history_limit:
            self._events = self._events[-self.history_limit:]
        if self._tracker is None:
            from topologies.visualization.state import ExperimentTracker, TrainingTracker

            self._tracker = ExperimentTracker() if event.source == "experiment" and \
                event.kind.startswith("experiment.") else TrainingTracker()
        self._track(event)
        if event.kind in {"training.completed", "experiment.completed",
                          "experiment.trial_completed"}:
            self._refresh(force=True)
            return
        step = event.progress.current if event.progress and event.progress.current is not None \
            else len(self._events)
        now = time.monotonic()
        if self.live and (step - self._last_step >= self.refresh_steps
                          or now - self._last_render >= self.refresh_seconds):
            self._refresh(force=False)

    def _track(self, event):
        tracker = self._tracker
        if event.source == "experiment" and event.kind.startswith("experiment."):
            if event.kind == "experiment.started":
                goal = event.data.get("goal")
                if isinstance(goal, (list, tuple)) and len(goal) == 2:
                    self._goal = (str(goal[0]), str(goal[1]))
            if event.kind in ("experiment.trial_completed", "experiment.completed"):
                order = len({o for o, _n, _v in self._completed}) + 1
                for key, value in event.metrics.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and (
                            self.metrics is None or key in self.metrics):
                        self._completed.append((order, key, float(value)))
                goal_metric = event.data.get("goal_metric") or self._goal[0]
                mode = event.data.get("mode") or self._goal[1]
                value = event.metrics.get(goal_metric) if goal_metric else None
                if isinstance(value, (int, float)):
                    prev = self._best[-1][1] if self._best else None
                    better = prev is None or (value < prev if mode == "min" else value > prev)
                    self._best.append((order, float(value) if better else prev))
            if hasattr(tracker, "update") and isinstance(tracker, _experiment_tracker_type()):
                tracker.update(event)
            return
        if event.trial_id is not None:
            return
        if not isinstance(tracker, _training_tracker_type()):
            return
        tracker.update(event)
        coordinate = event.step
        if coordinate is None:
            coordinate = tracker.step
        for key in ("throughput_steps_per_second", "memory_bytes", "memory_allocated_bytes"):
            value = event.data.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and \
                    math.isfinite(value) and (self.metrics is None or key in self.metrics):
                self._extra.setdefault(key, []).append((float(coordinate), float(value)))

    def _training_series(self) -> dict[str, list[tuple[float, float]]]:
        tracker = self._tracker
        series: dict[str, list[tuple[float, float]]] = {}
        for name in tracker.metric_names():
            from .views.training import split_name

            if self.metrics is not None and name not in self.metrics and \
                    split_name(name)[1] not in self.metrics:
                continue
            s = tracker.series(name)
            key = f"validation:{split_name(name)[1]}" if s.split == "val" else name
            series[key] = list(zip(s.x, s.y))
        series.update(self._extra)
        return series

    def _refresh(self, force=False):
        if not (self.live or self._figure is not None):
            return
        if not self.live and self._figure is None:
            return
        if not force and not self._refresh_dirty:
            return
        self._make_live_figure(force_agg=not self.live)
        live = self._live
        from .live import training_status

        with self._themed():
            if self._completed or self._best:
                metric = self._goal[0]
                live.update_experiment(self._completed, self._best, metric=metric,
                                       title=self.title or ("Live experiment" if self.live
                                                            else "Experiment trials"),
                                       status=f"{len({o for o, _n, _v in self._completed})} "
                                              "completed trials")
            elif isinstance(self._tracker, _training_tracker_type()):
                steps = [float(getattr(row, "step", 0) or 0) for row in self._tracker.epochs
                         if getattr(row, "step", None)]
                live.update_training(self._training_series(), epoch_steps=steps,
                                     status=training_status(self._tracker),
                                     title=self.title or ("Live training metrics"
                                                          if self.live else "Training metrics"))
        self._last_render = time.monotonic()
        last = next((e.progress.current for e in reversed(self._events)
                     if e.progress and e.progress.current is not None), 0)
        self._last_step = int(last or 0)
        self._refresh_dirty = False
        if force:
            self._save(self._figure, "-final" if self._stream_live_session else "")
            self._finalized = bool(self._events and self._events[-1].kind in {
                "training.completed", "experiment.completed"})
            if self.show_on_finish and self._plt and not self._shown:
                _show(self._plt)
                self._shown = True

    def close(self):
        if self.live and self._events and self._figure is not None and not self._finalized:
            self._refresh(force=True)
        if self._figure is not None:
            if self._plt:
                self._plt.close(self._figure)
            self._figure.clear()
            self._figure = None
            self._live = None


def _show(plt: Any) -> None:
    """``plt.show(block=False)`` on interactive backends only (Agg would warn)."""
    import matplotlib

    try:
        from matplotlib.backends import backend_registry  # type: ignore[attr-defined]

        interactive = matplotlib.get_backend().lower() in {
            b.lower() for b in backend_registry.list_builtin("interactive")}
    except ImportError:
        from matplotlib.rcsetup import interactive_bk

        interactive = matplotlib.get_backend().lower() in {b.lower() for b in interactive_bk}
    if interactive or getattr(plt.show, "__module__", "").startswith("unittest") or \
            not getattr(plt.show, "__module__", "matplotlib").startswith("matplotlib"):
        plt.show(block=False)


def _training_tracker_type():
    from topologies.visualization.state import TrainingTracker

    return TrainingTracker


def _experiment_tracker_type():
    from topologies.visualization.state import ExperimentTracker

    return ExperimentTracker


__all__ = ["MatplotlibVisualizer", "draw_analysis_spec", "register_view", "topologies_style",
           "view_names"]
