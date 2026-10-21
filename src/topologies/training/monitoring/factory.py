"""Monitor factory."""

from __future__ import annotations

from typing import Any

from topologies.training.errors import TrainingError
from topologies.training.monitoring.base import BaseMonitor, NullMonitor
from topologies.training.monitoring.standard import StandardMonitor, StandardMonitorOptions

_MODES = ["none", "plain", "compact", "standard", "dashboard"]


def build_monitor(
    kind: Any = None,
    options: StandardMonitorOptions | dict | None = None,
    **kwargs: Any,
) -> BaseMonitor:
    """Build a monitor. Never writes training history.

    Accepts None / "none" -> NullMonitor,
    "standard" / "auto" -> StandardMonitor (behaves like plain off a terminal),
    "plain" / "compact" -> StandardMonitor in the log-friendly plain tier,
    "dashboard" -> DashboardMonitor (fullscreen / inline / epoch snapshots),
    BaseMonitor instance -> returned as-is.
    """
    if kind is None:
        return NullMonitor()
    if isinstance(kind, BaseMonitor):
        return kind
    if not isinstance(kind, str):
        raise TrainingError(
            f"Unknown monitor {kind!r}.",
            stage="config",
            available=_MODES,
            hint="Use monitor='none'|'plain'|'standard'|'dashboard'|BaseMonitor(...).",
        )
    k = kind.strip().lower()
    if k in ("none", "null", "off", "quiet", ""):
        return NullMonitor()
    if k in ("standard", "auto"):
        return _make_standard(options, kwargs, plain=False)
    if k in ("plain", "compact"):
        return _make_standard(options, kwargs, plain=True)
    if k == "dashboard":
        return _make_dashboard(options, kwargs)
    raise TrainingError(
        f"Unknown monitor {kind!r}.",
        stage="config",
        available=_MODES,
        hint="Use monitor='none'|'plain'|'standard'|'dashboard'|BaseMonitor(...).",
    )


def _merged(options: StandardMonitorOptions | dict | None, kwargs: dict) -> dict:
    merged: dict = {}
    if isinstance(options, dict):
        merged.update(options)
    elif options is not None:
        import dataclasses as _dc

        try:
            merged.update(_dc.asdict(options))
        except Exception:  # noqa: BLE001, S110
            pass
    merged.update(kwargs)
    merged.pop("mode", None)
    return merged


def _make_standard(options: StandardMonitorOptions | dict | None, kwargs: dict, *,
                   plain: bool) -> StandardMonitor:
    return StandardMonitor(_merged(options, kwargs), plain=plain)


def _make_dashboard(
    options: StandardMonitorOptions | dict | None, kwargs: dict
) -> BaseMonitor:
    """Dependency-free console dashboard; every dashboard option is forwarded."""
    from topologies.training.monitoring.dashboard.app import DashboardMonitor

    merged = _merged(options, kwargs)
    try:
        rr = float(merged.pop("refresh_rate", 5.0) or 5.0)
    except Exception:  # noqa: BLE001
        rr = 5.0
    charts = merged.pop("charts", None) or merged.get("columns")
    return DashboardMonitor(refresh_rate=rr, charts=list(charts) if charts else None, **merged)
