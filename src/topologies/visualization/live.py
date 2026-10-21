"""``LiveVisualizer``: base class for live terminal views (built-in and custom).

Subclasses implement ``draw(state, width, height) -> list[str | Line]`` (and optionally
``final(state)`` for the persistent summary). The base class owns the tracker (chosen
from the session source), the screen choice, the render driver, resizing, output
capture, fail-safety and ``close()``::

    class LossLine(LiveVisualizer):
        refresh_rate = 4
        def draw(self, state, width, height):
            bar = progress_bar(state.fraction, 20)
            trend = sparkline(state.series("loss").y, 24, smoothing=0.6)
            return [line(bar, f" {state.fraction or 0:4.0%}  loss ", trend)]
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Sequence
from typing import Any

from topologies.visualization.core import (
    SessionInfo,
    VisualizationEvent,
    Visualizer,
    dispatch_hooks,
    session_info,
)
from topologies.visualization.state import make_tracker
from topologies.visualization.terminal.caps import TerminalCaps, probe
from topologies.visualization.terminal.driver import LiveDriver, OutputCapture, debug_enabled
from topologies.visualization.terminal.screen import Screen, choose_screen
from topologies.visualization.terminal.text import Line, as_line
from topologies.visualization.theme import get_theme

DEFAULT_FPS = {"inline": 10.0, "fullscreen": 8.0, "notebook": 2.0, "plain": 1.0, "null": 1.0}


class LiveVisualizer(Visualizer):
    """Live terminal view driven by a tracker; subclasses implement :meth:`draw`.

    Options: ``screen`` (``"auto"``, ``"inline"``, ``"fullscreen"``, ``"plain"``),
    ``stream`` (default: stderr for training, stdout for experiments), ``refresh_rate``,
    ``capture_output``, ``theme``, ``threaded`` and ``caps`` (testing).
    """

    owns_terminal = True
    refresh_rate: float | None = None
    screen: str = "auto"
    live_screen: str = "inline"
    capture_output: bool = True

    def __init__(self, *, screen: str | None = None, stream: Any = None,
                 refresh_rate: float | None = None, capture_output: bool | None = None,
                 theme: Any = "default", threaded: bool | None = None,
                 caps: TerminalCaps | None = None) -> None:
        if screen is not None:
            self.screen = screen
        if refresh_rate is not None:
            self.refresh_rate = float(refresh_rate)
        if capture_output is not None:
            self.capture_output = bool(capture_output)
        self.stream = stream
        self.theme = get_theme(theme)
        self.threaded = threaded
        self.caps = caps
        self.state: Any = None
        self.session: SessionInfo | None = None
        self.result: Any = None
        self._driver: LiveDriver | None = None
        self._screen: Screen | None = None
        self._capture: OutputCapture | None = None
        self._released = False
        self._closed = True
        self._errors: list[BaseException] = []

    # -- session ----------------------------------------------------------------
    def make_state(self, source: str | None) -> Any:
        """Tracker for the session (override to supply a custom state object)."""
        return make_tracker(source)

    def default_stream(self, source: str | None) -> Any:
        return sys.stdout if source == "experiment" else sys.stderr

    def screen_kind(self, caps: TerminalCaps) -> str:
        """Resolve ``screen`` (``auto`` → notebook / live screen on a TTY / plain)."""
        if self._released:
            return "null"
        kind = (self.screen or "auto").lower()
        if kind == "auto":
            if caps.notebook:
                return "notebook"
            if self.live_screen == "fullscreen" and caps.fullscreen_ok:
                return "fullscreen"
            return "inline" if caps.is_tty else "plain"
        if kind == "fullscreen" and not caps.fullscreen_ok:
            return "inline" if caps.is_tty else "plain"
        if kind == "inline" and not caps.is_tty:
            return "plain"
        return kind

    def prepare(self, source: str | None = None, context: Any = None) -> None:
        if not self._closed:
            self.close()
        self._closed = False
        self.session = session_info(source, context)
        self.state = self.make_state(self.session.source)
        self.result = None
        stream = self.stream if self.stream is not None else \
            self.default_stream(self.session.source)
        caps = self.caps if self.caps is not None else probe(stream)
        kind = self.screen_kind(caps)
        self._screen = choose_screen(kind, stream, caps=caps, theme=self.theme)
        fps = self.refresh_rate or DEFAULT_FPS.get(self._screen.kind, 8.0)
        threaded = self.threaded if self._screen.kind in ("inline", "fullscreen",
                                                          "notebook") else False
        self._driver = LiveDriver(self._screen, self._render_frame, fps=fps, threaded=threaded)
        self._driver.start()
        if self.capture_output and self._screen.kind in ("inline", "fullscreen") \
                and not debug_enabled():
            self._capture = OutputCapture(self._driver.print).start()
        self.on_session_start()

    def on_session_start(self) -> None:
        """Hook: called after the screen and driver started."""
        return

    @property
    def screen_type(self) -> str | None:
        """Kind of the active screen (``inline``, ``fullscreen``, ``plain`` ...)."""
        return None if self._screen is None else self._screen.kind

    @property
    def driver(self) -> LiveDriver | None:
        return self._driver

    def release_terminal(self) -> None:
        """Give up the terminal (another live view owns it); events are still tracked."""
        self._released = True
        if self._driver is not None and self._screen is not None and \
                self._screen.kind != "null":
            if self._capture is not None:
                self._capture.stop()
                self._capture = None
            self._driver.stop(render_final=False)
            self._screen = choose_screen("null", self._screen.stream, caps=self._screen.caps)
            self._driver = LiveDriver(self._screen, self._render_frame, threaded=False)
            self._driver.start()

    # -- events -----------------------------------------------------------------
    def handle(self, event: VisualizationEvent) -> None:
        if self.state is None or self._closed:
            self.prepare(getattr(event, "source", None))
        changed = self.state.update(event)
        dispatch_hooks(self, event)
        if changed and self._driver is not None:
            self._driver.invalidate()

    def _render_frame(self, cols: int, rows: int) -> list[Line]:
        with self.state.lock:
            lines = self.draw(self.state, cols, rows)
        return [as_line(x) for x in lines or ()]

    def draw(self, state: Any, width: int, height: int) -> Sequence[Any]:
        """Return the live frame (list of ``str``/``Line``); override in subclasses."""
        return []

    def final(self, state: Any) -> Sequence[Any] | None:
        """Persistent lines printed when the session ends (override)."""
        return None

    def print(self, item: Any) -> None:
        """Permanent output above the live region (inline) or after it (fullscreen)."""
        if self._driver is not None:
            self._driver.print(item)

    def refresh(self) -> None:
        if self._driver is not None:
            self._driver.refresh()

    def render(self, result: Any) -> Any:
        self.result = result
        return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._capture is not None:
            self._capture.stop()
            self._capture = None
        final = None
        if self.state is not None:
            try:
                with self.state.lock:
                    final = self.final(self.state)
            except Exception as exc:
                if debug_enabled():
                    raise
                self._errors.append(exc)
        if self._driver is not None:
            self._driver.stop(final)
            self._errors.extend(self._driver.errors)
        self.on_session_end()
        self._report_errors()

    def on_session_end(self) -> None:
        """Hook: called after the screen ended."""
        return

    def _report_errors(self) -> None:
        if not self._errors:
            return
        exc = self._errors[0]
        self._errors = []
        with contextlib.suppress(Exception):
            sys.stderr.write(f"[topologies] {type(self).__name__} display error "
                             f"({type(exc).__name__}: {exc}); set TOPOLOGIES_VIZ_DEBUG=1 "
                             "for a traceback.\n")


__all__ = ["DEFAULT_FPS", "LiveVisualizer"]
