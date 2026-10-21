"""Dashboard widgets: pure, dependency-free text renderers (never raise)."""

from __future__ import annotations

import math
from typing import Any

from topologies.training.monitoring.dashboard.charts import sparkline


def fmt_value(v: Any, precision: int = 4) -> str:
    """Compatibility wrapper over :func:`topologies.visualization.format.fmt_number`."""
    from topologies.visualization.format import fmt_number

    return fmt_number(v, sig=precision)


def fmt_duration(seconds: float | None) -> str:
    """Compatibility wrapper over :func:`topologies.visualization.format.fmt_clock`."""
    from topologies.visualization.format import fmt_clock

    return fmt_clock(seconds)


def progress_bar(fraction: float | None, width: int = 30) -> str:
    """``[━━━━━╸────]`` thin line bar (compatibility wrapper over the shared widget)."""
    from topologies.visualization.terminal.text import to_plain
    from topologies.visualization.terminal.widgets import progress_bar as _bar

    try:
        width = max(4, int(width))
    except (TypeError, ValueError):
        width = 30
    return "[" + to_plain(_bar(fraction, width)) + "]"


def metric_row(name: str, value: Any, history: list[float] | None, *, name_width: int = 12,
               chart_width: int = 30) -> str:
    """``loss        2.3101  ▇▆▅▄▄▃▂▂▁  (min 1.98 max 2.41)``."""
    try:
        hist = [h for h in (history or []) if isinstance(h, (int, float)) and math.isfinite(h)]
        chart = sparkline(hist, width=chart_width) if len(hist) > 1 else ""
        rng = f"  min {fmt_value(min(hist))} max {fmt_value(max(hist))}" if len(hist) > 1 else ""
        return f"{name[:name_width]:<{name_width}} {fmt_value(value):>10}  {chart}{rng}"
    except Exception:  # noqa: BLE001
        return f"{name}: {value}"


def kv_line(pairs: list[tuple[str, Any]], sep: str = " | ") -> str:
    """``device=mps | params=1.1M`` skipping ``None`` values."""
    try:
        return sep.join(f"{k}={v}" if k else str(v) for k, v in pairs if v is not None)
    except Exception:  # noqa: BLE001
        return ""


def rule(width: int, title: str = "") -> str:
    try:
        if not title:
            return "─" * width
        t = f" {title} "
        return "──" + t + "─" * max(0, width - len(t) - 2)
    except Exception:  # noqa: BLE001
        return ""
