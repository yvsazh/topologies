"""Live rendering driver (render thread, throttle, resize polling) and output capture.

The producer thread only updates a tracker and calls :meth:`LiveDriver.invalidate`
(microseconds). A daemon thread renders when the frame is dirty, the terminal size
changed, or ``clock_tick`` elapsed (so elapsed/ETA keep ticking during slow steps).
With ``threaded=False`` (tests, ``TOPOLOGIES_VIZ_THREADS=0``) rendering happens
synchronously inside ``invalidate()`` with the same throttle. Children created by
``fork`` never draw.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import sys
import threading
import time
import weakref
from collections.abc import Callable, Sequence
from typing import Any

from topologies.visualization.terminal.text import Line

_DRIVERS: weakref.WeakSet = weakref.WeakSet()
_FORKED = False


def _after_fork_in_child() -> None:
    global _FORKED
    _FORKED = True
    for driver in list(_DRIVERS):
        driver._forked = True


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


def threads_enabled() -> bool:
    return os.environ.get("TOPOLOGIES_VIZ_THREADS", "1").strip().lower() not in (
        "0", "false", "no", "off")


def debug_enabled() -> bool:
    return os.environ.get("TOPOLOGIES_VIZ_DEBUG", "").strip().lower() not in (
        "", "0", "false", "no")


class LiveDriver:
    """Drive ``render(cols, rows) -> list[Line]`` onto ``screen`` at most ``fps`` per second."""

    def __init__(self, screen: Any, render: Callable[[int, int], Sequence[Any]], *,
                 fps: float = 8.0, threaded: bool | None = True, clock_tick: float = 1.0,
                 name: str = "topologies-live") -> None:
        self.screen = screen
        self.render = render
        self.fps = max(0.5, float(fps or 8.0))
        self.interval = 1.0 / self.fps
        self.threaded = threads_enabled() if threaded is None else bool(threaded) and \
            threads_enabled()
        self.clock_tick = float(clock_tick) if clock_tick else 0.0
        self.name = name
        self.paused = False
        self.errors: list[BaseException] = []
        self._cond = threading.Condition()
        self._dirty = False
        self._stopping = False
        self._started = False
        self._stopped = False
        self._thread: threading.Thread | None = None
        self._last_render = 0.0
        self._pending_prints: list[Any] = []
        self._forked = _FORKED
        self.frames = 0
        _DRIVERS.add(self)

    # -- lifecycle -------------------------------------------------------------
    @property
    def active(self) -> bool:
        return self._started and not self._stopped and not self._forked

    def start(self) -> None:
        if self._started or self._forked:
            return
        self._started = True
        self.screen.begin()
        if self.threaded:
            try:
                self._thread = threading.Thread(target=self._run, name=self.name, daemon=True)
                self._thread.start()
            except RuntimeError:
                self.threaded = False
                self._thread = None

    def invalidate(self) -> None:
        """Mark the frame dirty (cheap). Synchronous mode renders here, throttled."""
        if not self.active:
            return
        if self.threaded:
            with self._cond:
                self._dirty = True
                self._cond.notify()
            return
        self._dirty = True
        if time.monotonic() - self._last_render >= self.interval:
            self._render_now()

    def refresh(self) -> None:
        """Render immediately (unthrottled), e.g. before printing an epoch line."""
        if not self.active:
            return
        if self.threaded and threading.current_thread() is not self._thread:
            with self._cond:
                self._dirty = True
                self._force = True
                self._cond.notify()
            return
        self._render_now()

    def print(self, item: Any) -> None:
        """Permanent output above (inline) or after (fullscreen) the live area."""
        if self._forked:
            return
        if not self.active:
            with contextlib.suppress(Exception):
                self.screen.print(item)
            return
        if self.threaded and threading.current_thread() is not self._thread:
            with self._cond:
                self._pending_prints.append(item)
                self._cond.notify()
            return
        self._flush_prints()
        self._safe(self.screen.print, item)

    def print_now(self, items: Sequence[Any], *, refresh: bool = True) -> None:
        """Synchronously flush a frame then print ``items`` (ordering-sensitive callers)."""
        if not self.active:
            for item in items:
                with contextlib.suppress(Exception):
                    self.screen.print(item)
            return
        with self._cond:
            self._flush_prints()
            if refresh:
                self._render_locked()
            for item in items:
                self._safe(self.screen.print, item)

    def stop(self, final: Sequence[Any] | None = None, *, render_final: bool = True) -> None:
        """Stop the thread, draw one last frame, then end the screen. Idempotent."""
        if self._stopped:
            return
        thread = self._thread
        with self._cond:
            self._stopping = True
            self._cond.notify()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._stopped = True
        if self._forked:
            return
        if self._started:
            with self._cond:
                self._flush_prints()
                if render_final and not self.paused:
                    self._render_locked()
        self._safe(self.screen.end, final)

    # -- internals -------------------------------------------------------------
    _force = False

    def _safe(self, fn: Callable, *args: Any) -> Any:
        try:
            return fn(*args)
        except Exception as exc:
            if debug_enabled():
                raise
            self.errors.append(exc)
            return None

    def _flush_prints(self) -> None:
        prints, self._pending_prints = self._pending_prints, []
        for item in prints:
            self._safe(self.screen.print, item)

    def _render_locked(self) -> None:
        if self._forked:
            return
        try:
            cols, rows = self.screen.size()
            lines = self.render(cols, rows)
            self.screen.draw(list(lines or ()))
            self.frames += 1
        except Exception as exc:
            if debug_enabled():
                raise
            self.errors.append(exc)
        self._last_render = time.monotonic()
        self._dirty = False
        self._force = False

    def _render_now(self) -> None:
        if self.paused:
            self._dirty = False
            return
        with self._cond:
            self._render_locked()

    def _size_changed(self) -> bool:
        check = getattr(self.screen, "needs_redraw", None)
        try:
            return bool(check()) if callable(check) else False
        except Exception:  # noqa: BLE001
            return False

    def _run(self) -> None:
        with self._cond:
            while not self._stopping:
                timeout = self.interval
                self._cond.wait(timeout)
                if self._stopping or self._forked:
                    break
                if self._pending_prints:
                    self._flush_prints()
                if self.paused:
                    continue
                elapsed = time.monotonic() - self._last_render
                resized = self._size_changed()
                due = elapsed >= self.interval
                clock = bool(self.clock_tick) and elapsed >= self.clock_tick
                if self._force or resized or (due and (self._dirty or clock)):
                    try:
                        self._render_locked()
                    except Exception as exc:  # noqa: BLE001
                        self.errors.append(exc)
                        if debug_enabled():
                            break


# ---------------------------------------------------------------- output capture


class _Proxy(io.TextIOBase):
    """Line-buffered stand-in for stdout/stderr while a live session owns the terminal."""

    def __init__(self, original: Any, sink: Callable[[str], None]) -> None:
        super().__init__()
        self._original = original
        self._sink = sink
        self._buffer = ""
        self._lock = threading.Lock()

    def write(self, text: str) -> int:  # type: ignore[override]
        if not isinstance(text, str):
            text = str(text)
        with self._lock:
            self._buffer += text
            if "\n" not in self._buffer:
                return len(text)
            *complete, self._buffer = self._buffer.split("\n")
        for item in complete:
            self._sink(item.rstrip("\r"))
        return len(text)

    def flush_partial(self) -> None:
        with self._lock:
            rest, self._buffer = self._buffer, ""
        if rest:
            self._sink(rest)

    def flush(self) -> None:
        return

    def isatty(self) -> bool:
        try:
            return bool(self._original.isatty())
        except Exception:  # noqa: BLE001
            return False

    def fileno(self) -> int:
        return self._original.fileno()

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return getattr(self._original, "encoding", None) or "utf-8"

    def writable(self) -> bool:
        return True

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class OutputCapture:
    """Redirect ``sys.stdout``/``sys.stderr`` (and root logging handlers) into ``sink``.

    Complete lines are forwarded as they arrive; partial lines are flushed at
    :meth:`stop`. Inline sessions print them above the live line; fullscreen sessions
    keep them in the log panel and replay them after leaving the alternate screen.
    """

    def __init__(self, sink: Callable[[str], None], *, streams: Sequence[str] = ("stdout",
                                                                              "stderr")) -> None:
        self.sink = sink
        self.streams = tuple(streams)
        self._saved: dict[str, Any] = {}
        self._proxies: dict[str, _Proxy] = {}
        self._handlers: list[tuple[logging.StreamHandler, Any]] = []
        self.active = False

    def start(self) -> OutputCapture:
        if self.active or debug_enabled():
            return self
        self.active = True
        for name in self.streams:
            original = getattr(sys, name)
            proxy = _Proxy(original, self.sink)
            self._saved[name] = original
            self._proxies[name] = proxy
            setattr(sys, name, proxy)
        originals = {id(v): self._proxies[k] for k, v in self._saved.items()}
        for handler in logging.getLogger().handlers:
            stream = getattr(handler, "stream", None)
            if isinstance(handler, logging.StreamHandler) and id(stream) in originals:
                with contextlib.suppress(Exception):
                    handler.setStream(originals[id(stream)])
                    self._handlers.append((handler, stream))
        return self

    def stop(self) -> None:
        if not self.active:
            return
        self.active = False
        for handler, stream in self._handlers:
            with contextlib.suppress(Exception):
                handler.setStream(stream)
        self._handlers = []
        for name, original in self._saved.items():
            proxy = self._proxies.get(name)
            if getattr(sys, name) is proxy:
                setattr(sys, name, original)
            if proxy is not None:
                proxy.flush_partial()
        self._saved, self._proxies = {}, {}

    def __enter__(self) -> OutputCapture:  # noqa: PYI034
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


__all__ = ["Line", "LiveDriver", "OutputCapture", "debug_enabled", "threads_enabled"]
