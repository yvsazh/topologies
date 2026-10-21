"""Dependency-free terminal dashboard monitor (facade over ``TrainingDashboard``).

On a terminal the dashboard runs fullscreen (alternate screen, keyboard shortcuts) or,
when fullscreen is unavailable, as a compact inline frame; off a terminal one snapshot
is printed per epoch. Read-only by default, rank-0 only (the reporter gates ranks), and
fail-safe: rendering errors never reach the training loop.
"""

from __future__ import annotations

from typing import Any

from topologies.training.monitoring.dashboard.state import DashboardState
from topologies.training.monitoring.standard import LegacyFacade


def is_tui_available() -> bool:
    """Kept for compatibility: the console dashboard has no optional deps."""
    return True


def _stream_is_tty(stream: Any) -> bool:
    try:
        from topologies.visualization.terminal.caps import probe

        return probe(stream).is_tty
    except Exception:  # noqa: BLE001
        return False


class DashboardMonitor(LegacyFacade):
    """Live multi-panel console dashboard: progress, metric charts, epochs, system."""

    def __init__(
        self,
        refresh_rate: float = 5.0,
        charts: list[str] | None = None,
        *,
        stream: Any = None,
        width: int | None = None,
        tty: bool | None = None,
        **options: Any,
    ) -> None:
        super().__init__()
        from topologies.visualization.training_views import TrainingDashboard, view_options

        try:
            self.refresh_rate = max(0.1, float(refresh_rate or 5.0))
        except (TypeError, ValueError):
            self.refresh_rate = 5.0
        self.charts = list(charts or [])
        self.stream = stream
        self._width = width
        opts = view_options("dashboard", {**options, "stream": stream, "width": width,
                                          "tty": tty})
        self.visualizer = TrainingDashboard(self.refresh_rate, self.charts or None, **opts)
        self._standalone: DashboardState | None = None

    @property
    def state(self) -> DashboardState:
        tracker = getattr(self.visualizer, "state", None)
        if tracker is None:
            if self._standalone is None:
                self._standalone = DashboardState()
            return self._standalone
        cached = self._standalone
        if cached is None or cached.tracker is not tracker:
            self._standalone = DashboardState(tracker)
        return self._standalone

    @property
    def _tty(self) -> bool:
        return getattr(self.visualizer, "screen_type", None) not in ("plain", "null", None)

    def frame(self) -> list[str]:
        """Current dashboard as plain text lines (also handy in notebooks/tests)."""
        from topologies.training.monitoring.dashboard.views import render_dashboard

        width = self._width
        if width is None:
            try:
                from topologies.visualization.terminal.caps import terminal_size

                width = min(120, terminal_size(None, (100, 24))[0])
            except Exception:  # noqa: BLE001
                width = 100
        return render_dashboard(self.state, width=width, charts=self.charts or None)
