"""Dashboard views (compatibility): state -> plain text lines (pure, never raise)."""

from __future__ import annotations

from typing import Any

DEFAULT_CHARTS = ("loss", "grad_norm", "lr")


def _tracker(state: Any) -> Any:
    return getattr(state, "tracker", state)


def render_dashboard(state: Any, *, width: int = 100, charts: list[str] | None = None,
                     events: int = 3) -> list[str]:
    """Compact one-column dashboard as plain lines, each at most ``width`` cells."""
    try:
        from topologies.visualization.terminal.text import to_plain
        from topologies.visualization.training_views import compact_frame

        width = max(40, int(width))
        system = getattr(state, "system", None) or None
        lines = compact_frame(_tracker(state), width, charts=charts, system=system)
        return [to_plain(ln) for ln in lines]
    except Exception as e:  # noqa: BLE001
        return [f"dashboard render error: {e}"[:width]]


def render_summary(state: Any) -> str:
    """One-line summary (kept for compatibility)."""
    try:
        t = _tracker(state)
        latest = {k: v[0] for k, v in getattr(t, "_latest", {}).items()}
        return f"epoch={t.epoch} step={t.step} metrics={latest}"
    except Exception:  # noqa: BLE001
        return "dashboard"
