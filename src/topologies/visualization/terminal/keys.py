"""Non-blocking keyboard input for fullscreen views (POSIX termios / Windows msvcrt).

POSIX uses cbreak mode (``ISIG`` kept, so Ctrl+C still raises ``KeyboardInterrupt`` in
the main thread) and a daemon reader thread polling ``select`` every 100 ms. The
original terminal attributes are always restored by :meth:`KeyReader.stop`.
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
from collections.abc import Callable
from typing import Any

_SEQUENCES = {
    "\x1b[A": "up", "\x1b[B": "down", "\x1b[C": "right", "\x1b[D": "left",
    "\x1bOA": "up", "\x1bOB": "down", "\x1bOC": "right", "\x1bOD": "left",
    "\x1b[5~": "pageup", "\x1b[6~": "pagedown", "\x1b[H": "home", "\x1b[F": "end",
    "\x1b[Z": "backtab",
}
_SINGLE = {"\t": "tab", "\r": "enter", "\n": "enter", "\x1b": "escape", "\x7f": "backspace",
           " ": "space"}


def decode_keys(data: str) -> list[str]:
    """Split raw terminal input into key names (``"q"``, ``"S"``, ``"up"``, ``"tab"`` ...)."""
    keys: list[str] = []
    i = 0
    while i < len(data):
        if data[i] == "\x1b":
            for seq, name in _SEQUENCES.items():
                if data.startswith(seq, i):
                    keys.append(name)
                    i += len(seq)
                    break
            else:
                keys.append("escape")
                i += 1
            continue
        ch = data[i]
        keys.append(_SINGLE.get(ch, ch))
        i += 1
    return keys


class KeyReader:
    """Deliver key names to ``callback`` from a daemon thread while active."""

    def __init__(self, callback: Callable[[str], None], stdin: Any = None) -> None:
        self.callback = callback
        self.stdin = stdin if stdin is not None else sys.stdin
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._saved: Any = None
        self.active = False

    def available(self) -> bool:
        try:
            return bool(self.stdin is not None and self.stdin.isatty())
        except Exception:  # noqa: BLE001
            return False

    def start(self) -> bool:
        if self.active or not self.available():
            return False
        if os.name == "nt":
            target = self._run_windows
        else:
            try:
                import termios
                import tty

                fd = self.stdin.fileno()
                self._saved = termios.tcgetattr(fd)
                tty.setcbreak(fd, termios.TCSANOW)
            except Exception:  # noqa: BLE001
                self._saved = None
                return False
            target = self._run_posix
        self.active = True
        self._stop.clear()
        self._thread = threading.Thread(target=target, name="topologies-keys", daemon=True)
        self._thread.start()
        return True

    def _emit(self, data: str) -> None:
        for key in decode_keys(data):
            with contextlib.suppress(Exception):
                self.callback(key)

    def _run_posix(self) -> None:
        import select

        fd = self.stdin.fileno()
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([fd], [], [], 0.1)
            except (OSError, ValueError):
                return
            if not ready:
                continue
            try:
                data = os.read(fd, 64)
            except OSError:
                return
            if not data:
                return
            self._emit(data.decode("utf-8", "replace"))

    def _run_windows(self) -> None:  # pragma: no cover - Windows only
        import msvcrt
        import time

        names = {"H": "up", "P": "down", "K": "left", "M": "right"}
        while not self._stop.is_set():
            if not msvcrt.kbhit():  # type: ignore[attr-defined]
                time.sleep(0.05)
                continue
            ch = msvcrt.getwch()  # type: ignore[attr-defined]
            if ch in ("\x00", "\xe0"):
                code = msvcrt.getwch()  # type: ignore[attr-defined]
                with contextlib.suppress(Exception):
                    self.callback(names.get(code, "unknown"))
                continue
            self._emit(ch)

    def stop(self) -> None:
        if not self.active:
            return
        self.active = False
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=0.5)
        if self._saved is not None:
            try:
                import termios

                termios.tcsetattr(self.stdin.fileno(), termios.TCSADRAIN, self._saved)
            except Exception:  # noqa: BLE001, S110
                pass
            self._saved = None


__all__ = ["KeyReader", "decode_keys"]
