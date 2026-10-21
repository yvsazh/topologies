"""Screen backends: where encoded frames go.

All screens implement ``begin()``, ``draw(lines)``, ``print(item)`` (permanent output),
``end(final)`` and ``size()``. Widgets never write; screens are the only encoders.

* :class:`PlainScreen` — append-only lines; ``draw`` is ignored (logs, CI, pipes).
* :class:`InlineRegion` — a small live region at the bottom of normal scrollback. Lines
  are clipped to ``cols - 1`` cells, autowrap is disabled while it is active and the
  previous frame's physical height is recomputed after a resize (reflow-aware), so
  resizing never duplicates or corrupts frames.
* :class:`FullscreenScreen` — alternate screen, absolute row painting, terminal state
  restored on ``end()``, ``atexit`` and ``SIGTERM``/``SIGHUP``.
* :class:`NotebookScreen` — IPython display handles updated with HTML frames (≤ 2 Hz).
"""

from __future__ import annotations

import atexit
import contextlib
import math
import os
import sys
import threading
import time
from collections import deque
from collections.abc import Sequence
from typing import Any

from topologies.visualization.terminal.caps import TerminalCaps, probe
from topologies.visualization.terminal.text import Line, as_line, styled, to_ansi, to_plain
from topologies.visualization.theme import Theme, get_theme

SYNC_ON = "\x1b[?2026h"
SYNC_OFF = "\x1b[?2026l"
HIDE_CURSOR = "\x1b[?25l"
SHOW_CURSOR = "\x1b[?25h"
WRAP_OFF = "\x1b[?7l"
WRAP_ON = "\x1b[?7h"
ALT_ON = "\x1b[?1049h"
ALT_OFF = "\x1b[?1049l"
RESET = "\x1b[0m"


def reflow_assumed() -> bool:
    """Most modern terminals re-wrap lines on shrink; ``TOPOLOGIES_TERM_REFLOW=0`` opts out."""
    return os.environ.get("TOPOLOGIES_TERM_REFLOW", "1").strip().lower() not in (
        "0", "false", "no", "off")


class Screen:
    """Base screen: encoding helpers and a lock serializing writes."""

    kind = "screen"

    def __init__(self, stream: Any = None, *, caps: TerminalCaps | None = None,
                 theme: Theme | str | None = None, color: int | None = None) -> None:
        self.stream = stream if stream is not None else sys.stdout
        self.caps = caps if caps is not None else probe(self.stream)
        self.theme = get_theme(theme)
        self.color = self.caps.color if color is None else int(color)
        self.active = False
        self.ended = False
        self.lock = threading.RLock()

    # -- encoding -------------------------------------------------------------
    def encode(self, item: Any) -> str:
        ln = as_line(item)
        if self.color > 0 or (self.theme.mono and self.caps.is_tty):
            return to_ansi(ln, self.theme, self.color, unicode=self.caps.unicode)
        return to_plain(ln, unicode=self.caps.unicode)

    def _write(self, text: str) -> None:
        if not text:
            return
        try:
            self.stream.write(text)
            self.stream.flush()
        except UnicodeEncodeError:
            enc = getattr(self.stream, "encoding", None) or "ascii"
            self.stream.write(text.encode(enc, "replace").decode(enc, "replace"))
            self.stream.flush()
        except (OSError, ValueError):
            pass

    def size(self, *, fresh: bool = False) -> tuple[int, int]:
        if not self.caps.is_tty:
            return self.caps.columns, self.caps.rows
        return self.caps.size(fresh=fresh)

    # -- protocol -------------------------------------------------------------
    def begin(self) -> None:
        self.active = True

    def draw(self, lines: Sequence[Any]) -> None:
        return

    def print(self, item: Any) -> None:
        with self.lock:
            for part in _permanent_parts(item):
                self._write(self._encode_permanent(part) + "\n")

    def _encode_permanent(self, part: Any) -> str:
        return part if isinstance(part, str) else self.encode(part)

    def end(self, final: Sequence[Any] | None = None) -> None:
        with self.lock:
            if self.ended:
                return
            self.ended = True
            self.active = False
            for item in final or ():
                self.print(item)


def _permanent_parts(item: Any) -> list[Any]:
    if isinstance(item, (list, tuple)) and not isinstance(item, Line):
        return list(item)
    if isinstance(item, str):
        return item.split("\n") if "\n" in item else [item]
    return [item]


class PlainScreen(Screen):
    """Append-only output: ``draw`` is ignored, ``print`` appends one line."""

    kind = "plain"


class NullScreen(Screen):
    """Draws and prints nothing (a visualizer that lost the terminal to another owner)."""

    kind = "null"

    def print(self, item: Any) -> None:
        return

    def end(self, final: Sequence[Any] | None = None) -> None:
        self.ended = True
        self.active = False


class InlineRegion(Screen):
    """Resize-safe live region below the normal scrollback (see module docstring)."""

    kind = "inline"

    def __init__(self, stream: Any = None, *, caps: TerminalCaps | None = None,
                 theme: Theme | str | None = None, color: int | None = None,
                 reflow: bool | None = None, max_height: int | None = None) -> None:
        super().__init__(stream, caps=caps, theme=theme, color=color)
        self.reflow = reflow_assumed() if reflow is None else bool(reflow)
        self.max_height = max_height
        self._widths: list[int] = []
        self._cols_at_draw = 0
        self._last: list[Line] = []
        self._last_size: tuple[int, int] | None = None

    def begin(self) -> None:
        with self.lock:
            if self.active:
                return
            self.active = True
            self.ended = False
            self._write(HIDE_CURSOR + WRAP_OFF)

    def _rows_old(self, cols_now: int) -> int:
        n = len(self._widths)
        if not n:
            return 0
        if cols_now >= self._cols_at_draw or not self.reflow:
            return n
        return sum(max(1, math.ceil(w / max(1, cols_now))) for w in self._widths)

    def _clear_seq(self, cols_now: int) -> str:
        rows_old = self._rows_old(cols_now)
        self._widths = []
        return "\r" + (f"\x1b[{rows_old}F" if rows_old else "") + "\x1b[J"

    def _frame(self, lines: Sequence[Any], cols: int, rows: int) -> str:
        width = max(1, cols - 1)
        limit = max(1, rows - 1)
        if self.max_height is not None:
            limit = max(1, min(limit, int(self.max_height)))
        body = [as_line(x).clip(width) for x in lines]
        if len(body) > limit:
            hidden = len(body) - limit + 1
            body = body[: limit - 1] + [styled(f"… {hidden} more", "muted").clip(width)]
        self._widths = [ln.width for ln in body]
        self._cols_at_draw = cols
        if not body:
            return ""
        return "\r\n".join(self.encode(ln) for ln in body) + "\r\n"

    def draw(self, lines: Sequence[Any]) -> None:
        with self.lock:
            if not self.active or self.ended:
                return
            cols, rows = self.size(fresh=True)
            self._last = [as_line(x) for x in lines]
            self._last_size = (cols, rows)
            out = SYNC_ON + self._clear_seq(cols) + WRAP_OFF + self._frame(self._last, cols, rows) \
                + SYNC_OFF
            self._write(out)

    def needs_redraw(self) -> bool:
        """True when the terminal size changed since the last frame."""
        return self._last_size is not None and self.size() != self._last_size

    def print(self, item: Any) -> None:
        with self.lock:
            if not self.active or self.ended:
                super().print(item)
                return
            cols, rows = self.size(fresh=True)
            text = "".join(self._encode_permanent(p) + "\n" for p in _permanent_parts(item))
            out = SYNC_ON + self._clear_seq(cols) + WRAP_ON + text + WRAP_OFF
            out += self._frame(self._last, cols, rows) + SYNC_OFF
            self._last_size = (cols, rows)
            self._write(out)

    def end(self, final: Sequence[Any] | None = None) -> None:
        with self.lock:
            if self.ended:
                return
            cols, _rows = self.size(fresh=True)
            out = (self._clear_seq(cols) if self.active else "") + WRAP_ON
            out += "".join(self._encode_permanent(p) + "\n" for item in final or ()
                           for p in _permanent_parts(item))
            out += SHOW_CURSOR
            self.ended = True
            self.active = False
            self._last = []
            self._write(out)


class FullscreenScreen(Screen):
    """Alternate-screen frontend with absolute painting and guaranteed restoration."""

    kind = "fullscreen"

    def __init__(self, stream: Any = None, *, caps: TerminalCaps | None = None,
                 theme: Theme | str | None = None, color: int | None = None,
                 install_signals: bool = True, log_size: int = 2000,
                 stdin: Any = None) -> None:
        super().__init__(stream, caps=caps, theme=theme, color=color)
        self.install_signals = install_signals
        self.log: deque[str] = deque(maxlen=log_size)
        self._replay: list[str] = []
        self._prev: list[str] = []
        self._prev_size: tuple[int, int] | None = None
        self._stdin = stdin if stdin is not None else sys.stdin
        self._termios: Any = None
        self._handlers: dict[int, Any] = {}
        self._atexit = False
        self.on_end: list[Any] = []

    def begin(self) -> None:
        with self.lock:
            if self.active:
                return
            self.active = True
            self.ended = False
            self._save_termios()
            self._write(ALT_ON + HIDE_CURSOR + WRAP_OFF + "\x1b[H\x1b[2J")
            atexit.register(self._atexit_end)
            self._atexit = True
            if self.install_signals:
                self._install_signals()

    def _save_termios(self) -> None:
        try:
            import termios

            if self._stdin is not None and self._stdin.isatty():
                self._termios = termios.tcgetattr(self._stdin.fileno())
        except Exception:  # noqa: BLE001
            self._termios = None

    def _restore_termios(self) -> None:
        if self._termios is None:
            return
        try:
            import termios

            termios.tcsetattr(self._stdin.fileno(), termios.TCSADRAIN, self._termios)
        except Exception:  # noqa: BLE001, S110
            pass
        self._termios = None

    def _install_signals(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        import signal

        for name in ("SIGTERM", "SIGHUP"):
            signum = getattr(signal, name, None)
            if signum is None:
                continue
            try:
                previous = signal.getsignal(signum)
                signal.signal(signum, self._on_signal)
                self._handlers[signum] = previous
            except (ValueError, OSError):
                continue

    def _uninstall_signals(self) -> None:
        if not self._handlers:
            return
        import signal

        if threading.current_thread() is threading.main_thread():
            for signum, previous in self._handlers.items():
                with contextlib.suppress(ValueError, OSError, TypeError):
                    signal.signal(signum, previous)
        self._handlers = {}

    def _on_signal(self, signum: int, frame: Any) -> None:
        import signal

        previous = self._handlers.get(signum, signal.SIG_DFL)
        self.end()
        if callable(previous):
            previous(signum, frame)
        elif previous == signal.SIG_DFL:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    def _atexit_end(self) -> None:
        self.end()

    def draw(self, lines: Sequence[Any]) -> None:
        with self.lock:
            if not self.active or self.ended:
                return
            cols, rows = self.size()
            rows = max(1, rows)
            encoded = []
            for i in range(rows):
                width = cols - 1 if i == rows - 1 else cols
                ln = as_line(lines[i]) if i < len(lines) else Line()
                encoded.append(self.encode(ln.clip(max(0, width)).pad(max(0, width))))
            full = (cols, rows) != self._prev_size
            parts = [SYNC_ON]
            if full:
                parts.append("\x1b[2J")
            changed = False
            for i, enc in enumerate(encoded):
                if full or i >= len(self._prev) or self._prev[i] != enc:
                    parts.append(f"\x1b[{i + 1};1H{enc}{RESET}")
                    changed = True
            parts.append(SYNC_OFF)
            self._prev = encoded
            self._prev_size = (cols, rows)
            if changed or full:
                self._write("".join(parts))

    def needs_redraw(self) -> bool:
        return self._prev_size is not None and self.size() != self._prev_size

    def print(self, item: Any) -> None:
        with self.lock:
            parts = [p if isinstance(p, str) else self._encode_permanent(p)
                     for p in _permanent_parts(item)]
            if not self.active or self.ended:
                for p in parts:
                    self._write(p + "\n")
                return
            for p in parts:
                self.log.append(p)
                self._replay.append(p)

    def end(self, final: Sequence[Any] | None = None) -> None:
        with self.lock:
            if self.ended:
                return
            self.ended = True
            was_active = self.active
            self.active = False
            if was_active:
                self._write(SYNC_OFF + RESET + WRAP_ON + SHOW_CURSOR + ALT_OFF)
            self._restore_termios()
            self._uninstall_signals()
            if self._atexit:
                with contextlib.suppress(Exception):
                    atexit.unregister(self._atexit_end)
                self._atexit = False
            for callback in list(self.on_end):
                with contextlib.suppress(Exception):
                    callback()
            out = "".join(p + "\n" for p in self._replay)
            self._replay = []
            out += "".join(self._encode_permanent(p) + "\n" for item in final or ()
                           for p in _permanent_parts(item))
            self._write(out)


class NotebookScreen(Screen):
    """Jupyter frontend: one HTML display handle updated in place (≤ ``fps`` per second)."""

    kind = "notebook"

    def __init__(self, stream: Any = None, *, caps: TerminalCaps | None = None,
                 theme: Theme | str | None = None, fps: float = 2.0,
                 columns: int = 120, rows: int = 40) -> None:
        super().__init__(stream, caps=caps, theme=theme, color=16)
        self.interval = 1.0 / max(0.1, float(fps))
        self.columns, self.rows = columns, rows
        self._frame = None
        self._log = None
        self._log_lines: list[str] = []
        self._last = 0.0
        self._pending: list[Line] | None = None

    def size(self, *, fresh: bool = False) -> tuple[int, int]:
        return self.columns, self.rows

    def _html(self, lines: Sequence[Any]) -> Any:
        from IPython.display import HTML

        from topologies.visualization.terminal.text import to_html

        body = "\n".join(to_html(as_line(x), self.theme) for x in lines)
        return HTML('<pre style="font-family:ui-monospace,Menlo,monospace;line-height:1.25;'
                    f'font-size:12px;margin:0">{body}</pre>')

    def begin(self) -> None:
        from IPython.display import display

        self.active = True
        self.ended = False
        self._log = display(self._html([]), display_id=True)
        self._frame = display(self._html([]), display_id=True)

    def draw(self, lines: Sequence[Any], *, force: bool = False) -> None:
        if not self.active or self._frame is None:
            return
        now = time.monotonic()
        self._pending = [as_line(x) for x in lines]
        if not force and now - self._last < self.interval:
            return
        self._last = now
        self._frame.update(self._html(self._pending))
        self._pending = None

    def print(self, item: Any) -> None:
        for p in _permanent_parts(item):
            self._log_lines.append(p if isinstance(p, str) else to_plain(p))
        if self._log is not None:
            from topologies.visualization.terminal.text import line

            self._log.update(self._html([line(x) for x in self._log_lines[-500:]]))

    def end(self, final: Sequence[Any] | None = None) -> None:
        if self.ended:
            return
        self.ended = True
        if self._frame is not None:
            if final:
                self._frame.update(self._html(list(final)))
            elif self._pending is not None:
                self._frame.update(self._html(self._pending))
        self.active = False


def choose_screen(kind: str = "auto", stream: Any = None, *, caps: TerminalCaps | None = None,
                  theme: Theme | str | None = None, live: str = "inline",
                  **options: Any) -> Screen:
    """Pick a screen: ``auto`` → notebook, ``live`` (inline/fullscreen) on a TTY, else plain."""
    stream = stream if stream is not None else sys.stdout
    caps = caps if caps is not None else probe(stream)
    kind = (kind or "auto").lower()
    if kind == "null":
        return NullScreen(stream, caps=caps, theme=theme)
    if kind == "auto":
        if caps.notebook:
            kind = "notebook"
        elif live == "fullscreen" and caps.fullscreen_ok:
            kind = "fullscreen"
        elif caps.is_tty:
            kind = "inline"
        else:
            kind = "plain"
    if kind == "fullscreen" and not caps.is_tty:
        kind = "plain"
    if kind == "inline" and not caps.is_tty:
        kind = "plain"
    if kind == "notebook":
        return NotebookScreen(stream, caps=caps, theme=theme)
    if kind == "fullscreen":
        return FullscreenScreen(stream, caps=caps, theme=theme,
                                **{k: v for k, v in options.items()
                                   if k in ("install_signals", "log_size", "stdin", "color")})
    if kind == "inline":
        return InlineRegion(stream, caps=caps, theme=theme,
                            **{k: v for k, v in options.items()
                               if k in ("reflow", "max_height", "color")})
    return PlainScreen(stream, caps=caps, theme=theme,
                       **{k: v for k, v in options.items() if k == "color"})


__all__ = ["FullscreenScreen", "InlineRegion", "NotebookScreen", "NullScreen", "PlainScreen",
           "Screen",
           "choose_screen", "reflow_assumed"]
