"""Incremental live figures for training and experiment sessions.

The layout (one axes per metric family) is rebuilt only when the set of families
changes; otherwise each refresh updates ``Line2D`` data in place, rescales, refreshes
the status subtitle and calls ``canvas.draw_idle()``. Interactive (pyplot) canvases get
``flush_events()`` so windows never freeze; notebooks get a display handle updated at
most once per second.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

from topologies.visualization.format import fmt_count, fmt_rate
from topologies.visualization.matplotlib.style import (
    colors,
    integer_x,
    legend_if_many,
    number_formatter,
)
from topologies.visualization.matplotlib.views.training import (
    group_families,
    panel_title,
    series_label,
    slot_for,
    smooth,
    split_name,
)


class LiveFigure:
    """Owns one figure and updates it in place from tracked series."""

    def __init__(self, fig: Any, *, theme: str = "light", options: Any = None,
                 pyplot: Any = None) -> None:
        self.fig = fig
        self.theme = theme
        self.options = options
        self.plt = pyplot
        self.axes: dict[str, Any] = {}
        self.lines: dict[str, Any] = {}
        self.collections: dict[str, Any] = {}
        self._layout_key: tuple | None = None
        self._status = None
        self._title = None
        self.rebuilds = 0
        self._notebook = None
        self._notebook_t = 0.0

    # -- layout -------------------------------------------------------------------
    def _rebuild(self, panels: Sequence[tuple[str, str]], title: str) -> None:
        fig = self.fig
        fig.clear()
        self.axes, self.lines, self.collections = {}, {}, {}
        n = max(1, len(panels))
        ncols = 1 if n <= 2 else 2
        nrows = math.ceil(n / ncols)
        from topologies.visualization.matplotlib.style import header_gridspec

        gs = header_gridspec(fig, nrows, ncols)
        first = None
        for i, (key, panel_title_text) in enumerate(panels):
            ax = fig.add_subplot(gs[1 + i // ncols, i % ncols], sharex=first)
            first = first or ax
            ax.set_title(panel_title_text)
            self.axes[key] = ax
        head = fig.add_subplot(gs[0, :])
        head.set_axis_off()
        head.set_label("header")
        c = colors(self.theme)
        self._title = head.text(0, 1.0, title, transform=head.transAxes, ha="left", va="top",
                                fontsize=13, fontweight=600, color=c.ink)
        self._status = head.text(0, 0.0, "", transform=head.transAxes, ha="left",
                                 va="bottom", fontsize=9, color=c.muted)
        self.rebuilds += 1

    def _line(self, ax: Any, key: str, **style: Any) -> Any:
        ln = self.lines.get(key)
        if ln is None:
            (ln,) = ax.plot([], [], **style)
            self.lines[key] = ln
        return ln

    def _finish(self, status: str) -> None:
        from matplotlib.collections import PathCollection

        for ax in self.axes.values():
            ax.relim()
            for coll in ax.collections:  # relim() ignores scatter collections
                if isinstance(coll, PathCollection) and len(coll.get_offsets()):
                    ax.update_datalim(coll.get_offsets())
            ax.autoscale_view()
            legend_if_many(ax, loc="best")
        if self._status is not None:
            self._status.set_text(status)
        self.draw()

    def draw(self) -> None:
        canvas = self.fig.canvas
        try:
            canvas.draw_idle()
        except Exception:  # noqa: BLE001, S110
            pass
        if self.plt is not None and getattr(canvas, "manager", None) is not None:
            try:
                canvas.flush_events()
            except (NotImplementedError, AttributeError):
                pass
        self._update_notebook()

    def _update_notebook(self) -> None:
        from topologies.visualization.terminal.caps import in_notebook

        if not in_notebook() or self.plt is not None:
            return
        now = time.monotonic()
        if now - self._notebook_t < 1.0 and self._notebook is not None:
            return
        self._notebook_t = now
        try:
            from IPython.display import display

            if self._notebook is None:
                self._notebook = display(self.fig, display_id=True)
            else:
                self._notebook.update(self.fig)
        except Exception:  # noqa: BLE001, S110
            pass

    # -- training -------------------------------------------------------------------
    def update_training(self, series: Mapping[str, Sequence[tuple[float, float]]], *,
                        epoch_steps: Sequence[float] = (), status: str = "",
                        title: str = "Live training metrics") -> None:
        opts = self.options
        families = group_families(series, charts=getattr(opts, "charts", None))
        if not families:
            families = [("metrics", [])]
        key = tuple((f, tuple(k)) for f, k in families)
        layout = tuple(f for f, _ in families)
        if layout != self._layout_key:
            self._rebuild([(f, panel_title(f, keys) if keys else "metrics")
                           for f, keys in families], title)
            self._layout_key = layout
        c = colors(self.theme)
        window = int(getattr(opts, "smoothing", 1) or 1)
        ema = getattr(opts, "ema", None)
        for family, keys in families:
            ax = self.axes[family]
            seen: dict[str, int] = {}
            for name in keys:
                pts = series.get(name) or ()
                split, _base = split_name(name)
                color = slot_for(name, keys, c.series, seen)
                label = series_label(name, keys)
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                if ema and split == "train":
                    raw = self._line(ax, f"raw:{name}", color=color, alpha=0.25, linewidth=1.0,
                                     label="_raw")
                    raw.set_data(xs, ys)
                ln = self._line(ax, name, color=color, label=label,
                                marker="o" if split == "validation" else None, markersize=4)
                ln.set_data(xs, smooth(ys, window=window, ema=ema) if split == "train" else ys)
            values = [p[1] for k in keys for p in series.get(k) or ()]
            if family == "learning rate":
                from topologies.visualization.matplotlib.style import lr_formatter

                if values and min(values) > 0:
                    ax.set_yscale("log")
                else:
                    ax.set_yscale("linear")
                    ax.yaxis.set_major_formatter(lr_formatter())
            elif ax.get_yscale() == "linear":
                ax.yaxis.set_major_formatter(number_formatter())
            ax.set_xlabel("global step")
            hair = self.collections.get(family)
            segments = [[(x, 0), (x, 1)] for x in epoch_steps]
            if hair is None and segments:
                hair = ax.vlines(list(epoch_steps), 0, 1, transform=ax.get_xaxis_transform(),
                                 colors=c.muted, linewidth=0.6, linestyles="dotted", zorder=0)
                self.collections[family] = hair
            elif hair is not None:
                hair.set_segments(segments)
        del key
        self._finish(status)

    # -- experiments -------------------------------------------------------------
    def update_experiment(self, completed: Sequence[tuple[int, str, float]],
                          best: Sequence[tuple[int, float]], *, metric: str | None,
                          status: str = "", title: str = "Live experiment") -> None:
        panels = [("trials", "completed trial metrics")]
        if best:
            panels.append(("best", "best so far"))
        layout = tuple(p for p, _ in panels)
        if layout != self._layout_key:
            self._rebuild(panels, title)
            self._layout_key = layout
        c = colors(self.theme)
        ax = self.axes["trials"]
        by_metric: dict[str, list[tuple[int, float]]] = {}
        for order, name, value in completed:
            by_metric.setdefault(name, []).append((order, value))
        for i, (name, pts) in enumerate(by_metric.items()):
            coll = self.collections.get(f"trials:{name}")
            offsets = [(o, v) for o, v in pts]
            if coll is None:
                coll = ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=22,
                                  color=c.series[i % len(c.series)], label=name, zorder=3)
                self.collections[f"trials:{name}"] = coll
            else:
                coll.set_offsets(offsets)
        ax.set_xlabel("completed trial order")
        integer_x(ax)
        if best:
            bax = self.axes["best"]
            ln = self._line(bax, "best", color=c.series[0], drawstyle="steps-post",
                            marker=".", label="best so far")
            ln.set_data([b[0] for b in best], [b[1] for b in best])
            bax.set_xlabel("completed trial order")
            bax.set_ylabel(metric or "goal")
            integer_x(bax)
        self._finish(status)

    def update_status(self, text: str) -> None:
        if self._status is not None:
            self._status.set_text(text)


def training_status(tracker: Any) -> str:
    bits = []
    if tracker.total_epochs:
        bits.append(f"epoch {min(tracker.epoch + 1, tracker.total_epochs)}/{tracker.total_epochs}")
    if tracker.total_steps:
        bits.append(f"step {fmt_count(tracker.step)}/{fmt_count(tracker.total_steps)}")
    elif tracker.step:
        bits.append(f"step {tracker.step:,}")
    if tracker.throughput:
        bits.append(fmt_rate(tracker.throughput))
    if tracker.status in ("done", "failed"):
        bits.append(tracker.status)
    return " · ".join(bits)


__all__ = ["LiveFigure", "training_status"]
