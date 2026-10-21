"""Dependency-free sparkline rendering (compatibility wrappers over the shared widgets)."""

from __future__ import annotations

_BLOCKS = " ▁▂▃▄▅▆▇█"
_BRAILLE_BASE = 0x2800


def _width(width: int | None) -> int:
    try:
        return max(1, min(int(width or 20), 200))
    except (TypeError, ValueError):
        return 20


def sparkline(values: list[float] | tuple[float, ...], width: int = 20) -> str:
    """Block sparkline (bucket means, robust range). Never raises; pure function."""
    try:
        from topologies.visualization.terminal.text import to_plain
        from topologies.visualization.terminal.widgets import sparkline as _spark

        return to_plain(_spark(values, _width(width)))
    except Exception:  # noqa: BLE001
        return ""


def braille_sparkline(values: list[float] | tuple[float, ...], width: int = 20) -> str:
    """Braille variant (bucket means, robust range). Never raises; pure function."""
    try:
        from topologies.visualization.terminal.widgets import _finite, bucket, robust_range

        vals = _finite(values)
        if not vals:
            return ""
        cells = bucket(vals, _width(width))
        lo, hi = robust_range(vals)
        if hi <= lo:
            return chr(_BRAILLE_BASE + 0x24) * len(cells)
        out = []
        # One braille column per cell: fill dots from the bottom up to the level.
        column = (0x40, 0x04, 0x02, 0x01)
        for v in cells:
            level = max(1, min(4, 1 + round((v - lo) / (hi - lo) * 3)))
            bits = 0
            for k in range(level):
                bits |= column[k]
            out.append(chr(_BRAILLE_BASE + bits))
        return "".join(out)
    except Exception:  # noqa: BLE001
        return ""
