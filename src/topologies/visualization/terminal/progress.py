"""One-line, resize-safe progress for simple producers (synth runs, downloads).

On a terminal the line lives in an :class:`InlineRegion` (clipped to the width, redrawn
in place, throttled); elsewhere nothing is drawn and callers print plain lines.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from typing import Any

from topologies.visualization.terminal.caps import TerminalCaps, probe
from topologies.visualization.terminal.screen import InlineRegion
from topologies.visualization.terminal.text import Line, as_line


class ProgressLine:
    """Sticky single progress line; ``live`` is False off a terminal."""

    def __init__(self, stream: Any = None, *, fps: float = 10.0,
                 caps: TerminalCaps | None = None) -> None:
        self.stream = stream if stream is not None else sys.stderr
        self.caps = caps if caps is not None else probe(self.stream)
        self.interval = 1.0 / max(0.5, fps)
        self.live = self.caps.is_tty
        self._region = InlineRegion(self.stream, caps=self.caps) if self.live else None
        self._last = 0.0
        self._started = False
        self._closed = False

    def width(self) -> int:
        return (self._region.size()[0] if self._region is not None else self.caps.columns) - 1

    def update(self, value: Any, *, force: bool = False) -> None:
        """Redraw the line (``value``: Line/str) at most ``fps`` times per second."""
        if self._region is None or self._closed:
            return
        now = time.monotonic()
        if not force and now - self._last < self.interval:
            return
        self._last = now
        if not self._started:
            self._region.begin()
            self._started = True
        try:
            self._region.draw([as_line(value)])
        except (OSError, ValueError):
            pass

    def print(self, value: Any) -> None:
        """Permanent output above the live line (or a plain line off a terminal)."""
        if self._region is not None and self._started and not self._closed:
            self._region.print(value)
            return
        from topologies.visualization.terminal.text import to_plain

        try:
            self.stream.write((value if isinstance(value, str) else to_plain(value)) + "\n")
            self.stream.flush()
        except (OSError, ValueError):
            pass

    def close(self, final: Sequence[Line | str] | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        if self._region is not None and self._started:
            self._region.end(final)
        elif final:
            for item in final:
                self.print(item)


__all__ = ["ProgressLine"]
