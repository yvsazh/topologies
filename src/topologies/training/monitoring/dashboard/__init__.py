"""Dependency-free console dashboard (state + pure views/widgets + monitor)."""

from __future__ import annotations

from topologies.training.monitoring.dashboard.app import DashboardMonitor, is_tui_available
from topologies.training.monitoring.dashboard.charts import braille_sparkline, sparkline
from topologies.training.monitoring.dashboard.state import DashboardState
from topologies.training.monitoring.dashboard.views import render_dashboard

__all__ = [
    "DashboardMonitor",
    "DashboardState",
    "braille_sparkline",
    "is_tui_available",
    "render_dashboard",
    "sparkline",
]
