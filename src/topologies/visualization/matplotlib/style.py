"""Matplotlib style: rc dictionaries, theme colors and tick formatters.

``topologies_style`` returns an rc mapping that is applied with
``matplotlib.rc_context`` around figure creation, drawing and saving. Global
``rcParams`` are never modified and ``matplotlib.use`` is never called.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping
from types import SimpleNamespace
from typing import Any

from topologies.visualization.format import fmt_bytes, fmt_count, fmt_number
from topologies.visualization.theme import CHROME, SEQUENTIAL, STATUS, series_colors

FONTS = ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"]
THEMES = ("light", "dark")


def available_fonts() -> list[str]:
    """``FONTS`` filtered to installed families (avoids ``findfont`` warning spam)."""
    global _AVAILABLE
    if _AVAILABLE is None:
        try:
            from matplotlib import font_manager

            installed = {f.name for f in font_manager.fontManager.ttflist}
            _AVAILABLE = [f for f in FONTS if f in installed] or ["DejaVu Sans"]
        except Exception:  # noqa: BLE001 - style must never break drawing
            _AVAILABLE = ["DejaVu Sans"]
    return list(_AVAILABLE)


_AVAILABLE: list[str] | None = None


def check_theme(theme: str) -> str:
    if theme not in THEMES:
        raise ValueError(f"theme must be one of {THEMES}; got {theme!r}")
    return theme


def colors(theme: str = "light") -> SimpleNamespace:
    """Series palette, chrome and status colors for a theme."""
    chrome = CHROME["dark" if theme == "dark" else "light"]
    return SimpleNamespace(series=series_colors(theme), surface=chrome["surface"],
                           ink=chrome["ink"], ink2=chrome["ink2"], muted=chrome["muted"],
                           grid=chrome["grid"], baseline=chrome["baseline"],
                           good=STATUS["good"], warning=STATUS["warning"],
                           serious=STATUS["serious"], critical=STATUS["critical"],
                           sequential=SEQUENTIAL)


def topologies_style(theme: str = "light", *, base: Any = None,
                     overrides: Mapping[str, Any] | None = None, dpi: int = 150) -> dict[str, Any]:
    """rc mapping for ``matplotlib.rc_context`` (``base`` = extra rc dict applied first)."""
    from cycler import cycler

    c = colors(check_theme(theme))
    rc: dict[str, Any] = dict(base or {}) if isinstance(base, Mapping) else {}
    rc.update({
        "figure.facecolor": c.surface,
        "figure.edgecolor": c.surface,
        "savefig.facecolor": c.surface,
        "savefig.edgecolor": c.surface,
        "axes.facecolor": c.surface,
        "axes.prop_cycle": cycler(color=list(c.series)),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": c.baseline,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": c.grid,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "axes.titlelocation": "left",
        "axes.titlesize": 11,
        "axes.titleweight": 600,
        "axes.titlecolor": c.ink,
        "axes.labelsize": 9.5,
        "axes.labelcolor": c.ink2,
        "xtick.color": c.muted,
        "ytick.color": c.muted,
        "xtick.labelcolor": c.muted,
        "ytick.labelcolor": c.muted,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "text.color": c.ink,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.labelcolor": c.ink2,
        "font.family": available_fonts(),  # a list: Matplotlib falls back per glyph along it
        "figure.constrained_layout.use": True,
        "figure.dpi": 100,
        "savefig.dpi": dpi,
        "savefig.bbox": "tight",
    })
    if overrides:
        rc.update(dict(overrides))
    return rc


@contextlib.contextmanager
def themed(theme: str = "light", style: Any = None, *, dpi: int = 150) -> Iterator[None]:
    """Apply the topologies style (then ``style``: rc dict or style name) temporarily."""
    import logging

    import matplotlib

    rc = topologies_style(theme, dpi=dpi)
    font_logger = logging.getLogger("matplotlib.font_manager")
    level = font_logger.level
    font_logger.setLevel(logging.ERROR)  # missing preferred fonts fall back silently
    try:
        with matplotlib.rc_context(rc):
            if isinstance(style, Mapping):
                with matplotlib.rc_context(dict(style)):
                    yield
            elif isinstance(style, str):
                import matplotlib.style as mstyle

                with mstyle.context(style):
                    yield
            else:
                yield
    finally:
        font_logger.setLevel(level)


def number_formatter() -> Any:
    """Tick labels with one shared number of decimals per axis (``fmt_column`` rules)."""
    from matplotlib.ticker import Formatter

    from topologies.visualization.terminal.widgets import fmt_ticks

    class _Shared(Formatter):
        labels: dict[float, str] = {}

        def set_locs(self, locs: Any) -> None:
            super().set_locs(locs)
            values = [float(v) for v in locs]
            try:
                self.labels = dict(zip(values, fmt_ticks(values))) if values else {}
            except Exception:  # noqa: BLE001
                self.labels = {}

        def __call__(self, value: float, pos: Any = None) -> str:
            return self.labels.get(float(value)) or fmt_number(value, sig=3)

    return _Shared()


def lr_formatter() -> Any:
    from matplotlib.ticker import FuncFormatter

    from topologies.visualization.format import fmt_lr

    return FuncFormatter(lambda v, _pos: "0" if v == 0 else fmt_lr(v))


def seconds_formatter() -> Any:
    from matplotlib.ticker import FuncFormatter

    def fmt(v: float, _pos: Any = None) -> str:
        if v == 0:
            return "0"
        if abs(v) >= 1:
            return f"{v:.3g} s"
        if abs(v) >= 1e-3:
            return f"{v * 1e3:.3g} ms"
        return f"{v * 1e6:.3g} µs"

    return FuncFormatter(fmt)


def header_gridspec(fig: Any, nrows: int, ncols: int = 1, *,
                    height_ratios: Any = None, width_ratios: Any = None,
                    inches: float = 0.6) -> Any:
    """GridSpec with a header row of a fixed physical height (title + subtitle)."""
    ratios = list(height_ratios or [1] * nrows)
    fig_h = max(1.0, float(fig.get_figheight()))
    frac = min(0.35, inches / fig_h)
    header_ratio = frac / (1 - frac) * sum(ratios)
    return fig.add_gridspec(nrows + 1, ncols, height_ratios=[header_ratio, *ratios],
                            width_ratios=width_ratios)


def bytes_formatter() -> Any:
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda v, _pos: fmt_bytes(v) if v else "0")


def count_formatter() -> Any:
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda v, _pos: fmt_count(v) if v else "0")


def integer_x(ax: Any) -> None:
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def header(fig: Any, slot: Any, title: str, subtitle: str | None, theme: str) -> Any:
    """Left-aligned figure title + muted subtitle in a frameless header axes."""
    c = colors(theme)
    ax = fig.add_subplot(slot)
    ax.set_axis_off()
    ax.set_label("header")
    ax.text(0.0, 1.0 if subtitle else 0.5, title, transform=ax.transAxes, ha="left",
            va="top" if subtitle else "center", fontsize=13, fontweight=600, color=c.ink)
    if subtitle:
        ax.text(0.0, 0.0, subtitle, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=9, color=c.muted)
    return ax


def empty_state(ax: Any, message: str, hint: str | None, theme: str) -> None:
    """Centered muted message (+ hint line) for views without data."""
    c = colors(theme)
    ax.set_axis_off()
    ax.text(0.5, 0.55, message, transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color=c.muted)
    if hint:
        ax.text(0.5, 0.42, hint, transform=ax.transAxes, ha="center", va="center",
                fontsize=8.5, color=c.muted)


def legend_if_many(ax: Any, **kwargs: Any) -> None:
    """Legend only for ≥ 2 labelled series (one series: the title names it)."""
    handles, labels = ax.get_legend_handles_labels()
    keep = [(h, lab) for h, lab in zip(handles, labels) if lab and not lab.startswith("_")]
    if len(keep) >= 2:
        ax.legend(*zip(*keep), **kwargs)


def direct_labels(ax: Any, lines: list[Any], theme: str, *, fmt: Any = None) -> None:
    """Label the last point of ≤ 4 lines at the line end (ink color, not series color)."""
    if not 0 < len(lines) <= 4:
        return
    c = colors(theme)
    for ln in lines:
        xs, ys = ln.get_xdata(), ln.get_ydata()
        if len(xs) == 0:
            continue
        text = (fmt or (lambda v: fmt_number(v, sig=3)))(float(ys[-1]))
        ax.annotate(text, (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=c.ink2)


__all__ = ["FONTS", "THEMES", "bytes_formatter", "colors", "count_formatter", "direct_labels",
           "empty_state", "header", "header_gridspec", "integer_x", "legend_if_many",
           "lr_formatter", "number_formatter", "seconds_formatter", "themed",
           "topologies_style"]
