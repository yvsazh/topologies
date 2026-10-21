"""Analysis views: fits (observations, fit line, confidence band, residuals) and records."""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import Any

from topologies.visualization.format import fmt_number
from topologies.visualization.matplotlib.style import colors, empty_state, header

from . import builtin_view


def _sig3(value: Any) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{f:.3g}"


def fit_equation(spec: Mapping[str, Any]) -> str | None:
    params = spec.get("parameters") or {}
    method = spec.get("method")
    if method == "linear" and {"slope", "intercept"} <= set(params):
        b = float(params["intercept"])
        sign = "−" if b < 0 else "+"
        return f"y = {_sig3(params['slope'])}·x {sign} {_sig3(abs(b))}"
    if method == "power_law" and {"a", "b"} <= set(params):
        return f"y = {_sig3(params['a'])}·x^{_sig3(params['b'])}"
    return None


def fit_title(spec: Mapping[str, Any]) -> str:
    return {"linear": "Linear fit", "power_law": "Power-law fit"}.get(spec.get("method"), "Fit")


def _log_log(spec: Mapping[str, Any], x_scale: str, y_scale: str) -> tuple[str, str]:
    if spec.get("method") == "power_law":
        obs = spec.get("observations") or ()
        positive = all(p[0] > 0 and p[1] > 0 for p in obs)
        if positive:
            return ("log" if x_scale == "auto" else x_scale,
                    "log" if y_scale == "auto" else y_scale)
    return ("linear" if x_scale == "auto" else x_scale,
            "linear" if y_scale == "auto" else y_scale)


def draw_analysis_spec(ax: Any, spec: Mapping[str, Any], *, theme: str = "light",
                       x: str | None = None, y: str | None = None, x_scale: str = "auto",
                       y_scale: str = "auto", annotate: bool = True) -> Any:
    """Draw a fit/records plot spec into a caller-owned axes (public API)."""
    if spec.get("kind") == "views":
        from topologies.visualization.matplotlib_views import render_view_on_axes

        views = spec.get("views") or []
        return render_view_on_axes(ax, views[0]) if views else ax
    c = colors(theme)
    kind = spec.get("kind")
    if kind == "fit":
        obs = spec.get("observations") or ()
        curve = spec.get("curve") or ()
        band = spec.get("confidence_band")
        if band:
            ax.fill_between(band["x"], band["lower"], band["upper"], color=c.series[0],
                            alpha=0.18, linewidth=0, label=band.get("label", "confidence band"))
        if obs:
            ax.scatter([p[0] for p in obs], [p[1] for p in obs], color=c.series[0], s=22,
                       zorder=3, label="observations")
        if curve:
            ax.plot([p["x"] for p in curve], [p["value"] for p in curve], color=c.ink,
                    linewidth=1.6, label="fit")
        ax.set_xlabel(spec.get("x_label", "x"))
        ax.set_ylabel(spec.get("y_label", "y"))
        xs, ys = _log_log(spec, x_scale, y_scale)
        ax.set_xscale(xs)
        ax.set_yscale(ys)
        ax.grid(True, axis="both")
        rising = _rising(spec)
        if annotate:
            diag = spec.get("diagnostics") or {}
            lines = [fit_equation(spec), f"RMSE {fmt_number(diag.get('rmse'), sig=3)}"
                     + (f" ({diag.get('residual_domain')})" if diag.get("residual_domain")
                        not in (None, "y") else ""),
                     "R² " + fmt_number(diag.get("r_squared"), sig=3),
                     f"n {spec.get('n') or len(obs)}"]
            ax.text(0.02, 0.97 if rising else 0.03, "\n".join(x for x in lines if x),
                    transform=ax.transAxes, ha="left", va="top" if rising else "bottom",
                    fontsize=8.5, color=c.muted)
        if ax.collections or ax.lines:
            ax.legend(loc="lower right" if rising else "upper right")
        return ax
    points = spec.get("points")
    if kind in {"records", "points"} and points is not None:
        pts = [p for p in points if p[0] is not None and p[1] is not None]
        if pts:
            xs_ = [p[0] for p in pts]
            if all(a <= b for a, b in itertools.pairwise(xs_)):
                ax.plot(xs_, [p[1] for p in pts], color=c.series[0], linewidth=1.2)
            ax.scatter(xs_, [p[1] for p in pts], color=c.series[0], s=20, zorder=3)
        ax.set_xlabel(spec.get("x") or x or "x")
        ax.set_ylabel(spec.get("y") or y or "y")
        ax.grid(True, axis="both")
        return ax
    raise ValueError(f"Unsupported or unrenderable analysis plot spec kind: {kind!r}")


def _rising(spec: Mapping[str, Any]) -> bool:
    obs = sorted(spec.get("observations") or ())
    return len(obs) < 2 or obs[-1][1] >= obs[0][1]


def _residuals(spec: Mapping[str, Any]) -> list[tuple[float, float]]:
    obs = spec.get("observations") or ()
    res = spec.get("residuals") or ()
    if res and len(res) == len(obs):
        return [(p[0], float(r)) for p, r in zip(obs, res)]
    return []


@builtin_view("analysis", "fit")
def fit_view(fig: Any, data: Any, *, theme: str, options: Any) -> Any:
    spec = data.get("spec", data)
    c = colors(theme)
    res = _residuals(spec)
    xs_scale = getattr(options, "x_scale", "auto")
    ys_scale = getattr(options, "y_scale", "auto")
    from topologies.visualization.matplotlib.style import header_gridspec

    if res:
        gs = header_gridspec(fig, 2, 1, height_ratios=[3, 1])
        top = fig.add_subplot(gs[1])
        bottom = fig.add_subplot(gs[2], sharex=top)
        draw_analysis_spec(top, spec, theme=theme, x_scale=xs_scale, y_scale=ys_scale)
        top.label_outer()
        top.set_xlabel("")
        bottom.axhline(0, color=c.baseline, linewidth=0.8)
        bottom.vlines([p[0] for p in res], 0, [p[1] for p in res], colors=c.series[0],
                      linewidth=1.0)
        bottom.scatter([p[0] for p in res], [p[1] for p in res], color=c.series[0], s=14,
                       zorder=3)
        domain = (spec.get("diagnostics") or {}).get("residual_domain") or "y"
        bottom.set_ylabel(f"residual ({domain})" if domain != "y" else "residual")
        bottom.set_xlabel(spec.get("x_label", "x"))
        bottom.grid(True, axis="both")
        head = gs[0]
    else:
        gs = header_gridspec(fig, 1, 1)
        top = fig.add_subplot(gs[1])
        draw_analysis_spec(top, spec, theme=theme, x_scale=xs_scale, y_scale=ys_scale)
        head = gs[0]
    subtitle = f"n = {spec.get('n') or len(spec.get('observations') or ())}"
    header(fig, head, getattr(options, "title", None) or fit_title(spec), subtitle, theme)
    return fig


@builtin_view("analysis", "records")
def records_view(fig: Any, data: Any, *, theme: str, options: Any) -> Any:
    spec = data.get("spec", data)
    ax = fig.subplots()
    draw_analysis_spec(ax, spec, theme=theme)
    ax.set_title(getattr(options, "title", None) or "records")
    return fig


@builtin_view("analysis", "auto")
def analysis_view(fig: Any, data: Any, *, theme: str, options: Any) -> Any:
    spec = data.get("spec", {}) if isinstance(data, Mapping) else {}
    kind = spec.get("kind")
    if kind == "fit":
        return fit_view(fig, data, theme=theme, options=options)
    if kind in ("records", "points"):
        return records_view(fig, data, theme=theme, options=options)
    ax = fig.subplots()
    empty_state(ax, f"No view for analysis spec {kind!r}",
                "register one with register_view('analysis', name)", theme)
    return fig


__all__ = ["analysis_view", "draw_analysis_spec", "fit_equation", "fit_title", "fit_view",
           "records_view"]
