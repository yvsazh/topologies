"""One terminal capability probe shared by every display surface.

Color depth, in order: ``NO_COLOR`` (any value) → none; ``TOPOLOGIES_COLOR`` =
``never|always|16|256|truecolor`` overrides; ``FORCE_COLOR`` → at least 16 colors
even off a TTY; not a TTY → none; ``COLORTERM`` truecolor/24bit → truecolor;
``TERM`` containing ``256color`` → 256; Windows Terminal → truecolor; else 16.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

TRUECOLOR = 16_777_216
_CI_VARS = ("CI", "CONTINUOUS_INTEGRATION", "GITHUB_ACTIONS", "BUILDKITE", "GITLAB_CI",
            "TF_BUILD", "JENKINS_URL", "TEAMCITY_VERSION", "CIRCLECI", "TRAVIS")
_FALSY = ("", "0", "false", "no", "off")


def in_ci(env: Any = None) -> bool:
    env = os.environ if env is None else env
    return any(str(env.get(name, "")).strip().lower() not in _FALSY for name in _CI_VARS)


def in_notebook() -> bool:
    """True inside a Jupyter (ZMQ) IPython kernel; never imports IPython itself."""
    if "ipykernel" not in sys.modules:
        return False
    try:
        ipython = sys.modules.get("IPython")
        get = getattr(ipython, "get_ipython", None)
        shell = get() if callable(get) else None
        return shell is not None and type(shell).__name__ == "ZMQInteractiveShell"
    except Exception:  # noqa: BLE001
        return False


def _isatty(stream: Any) -> bool:
    try:
        isatty = getattr(stream, "isatty", None)
        return bool(callable(isatty) and isatty())
    except Exception:  # noqa: BLE001
        return False


def _enable_windows_vt(stream: Any) -> bool:
    """Enable VT processing on a Windows console; True when escapes are usable."""
    if os.name != "nt":
        return True
    try:
        import ctypes
        import msvcrt

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = msvcrt.get_osfhandle(stream.fileno())  # type: ignore[attr-defined]
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:  # noqa: BLE001
        return False


def _encoding_ok(stream: Any) -> bool:
    enc = getattr(stream, "encoding", None) or "utf-8"
    try:
        "✓✗★━─╭╯⣿▁█·…±≈".encode(enc)
        return True
    except (LookupError, UnicodeEncodeError):
        return False


def color_depth(stream: Any = None, *, is_tty: bool | None = None, env: Any = None) -> int:
    """Number of colors usable on ``stream`` (0, 16, 256 or 16 777 216)."""
    env = os.environ if env is None else env
    if env.get("NO_COLOR") is not None:
        return 0
    forced = str(env.get("TOPOLOGIES_COLOR", "")).strip().lower()
    if forced in ("never", "none", "0", "off", "false"):
        return 0
    if forced in ("truecolor", "24bit", "16m"):
        return TRUECOLOR
    if forced == "256":
        return 256
    if forced == "16":
        return 16
    tty = _isatty(stream if stream is not None else sys.stdout) if is_tty is None else is_tty
    force_env = env.get("FORCE_COLOR")
    force = force_env is not None and str(force_env).strip().lower() not in ("0", "false")
    if not tty and not force and forced not in ("always", "yes", "1", "on", "true"):
        return 0
    colorterm = str(env.get("COLORTERM", "")).lower()
    term = str(env.get("TERM", "")).lower()
    if colorterm in ("truecolor", "24bit"):
        return TRUECOLOR
    if "256color" in term:
        return 256
    if env.get("WT_SESSION"):
        return TRUECOLOR
    return 16


@dataclass(frozen=True)
class TerminalCaps:
    """What a stream supports; ``size()`` re-reads the live terminal size (cached ≤50 ms)."""

    is_tty: bool = False
    color: int = 0
    unicode: bool = True
    columns: int = 80
    rows: int = 24
    notebook: bool = False
    ci: bool = False
    fullscreen_ok: bool = False
    windows_vt: bool = True
    stream: Any = field(default=None, repr=False, compare=False)
    size_source: Any = field(default=None, repr=False, compare=False)
    _cache: list = field(default_factory=lambda: [0.0, None], repr=False, compare=False)

    def size(self, *, fresh: bool = False) -> tuple[int, int]:
        """Current ``(columns, rows)`` (cached ≤ 50 ms unless ``fresh``)."""
        if self.size_source is not None:
            return tuple(self.size_source())  # type: ignore[return-value]
        now = time.monotonic()
        cached = self._cache[1]
        if not fresh and cached is not None and now - self._cache[0] < 0.05:
            return cached
        value = terminal_size(self.stream, (self.columns, self.rows))
        self._cache[0], self._cache[1] = now, value
        return value


def terminal_size(stream: Any = None, default: tuple[int, int] = (80, 24)) -> tuple[int, int]:
    """``(columns, rows)`` of the terminal behind ``stream``; else ``COLUMNS``/``LINES``/default."""
    if stream is not None:
        try:
            size = os.get_terminal_size(stream.fileno())
            if size.columns > 0 and size.lines > 0:
                return size.columns, size.lines
        except Exception:  # noqa: BLE001, S110
            pass
    try:
        cols = int(os.environ.get("COLUMNS") or default[0])
        rows = int(os.environ.get("LINES") or default[1])
        return max(1, cols), max(1, rows)
    except ValueError:
        return default


def probe(stream: Any = None, *, env: Any = None) -> TerminalCaps:
    """Probe the capabilities of ``stream`` (default ``sys.stdout``)."""
    env = os.environ if env is None else env
    stream = sys.stdout if stream is None else stream
    term = str(env.get("TERM", "")).lower()
    notebook = in_notebook()
    ci = in_ci(env)
    raw_tty = _isatty(stream)
    windows_vt = _enable_windows_vt(stream) if raw_tty else True
    is_tty = raw_tty and not ci and not notebook and term != "dumb"
    depth = color_depth(stream, is_tty=is_tty, env=env)
    if raw_tty and not windows_vt:
        depth = 0
    unicode = _encoding_ok(stream) and term != "linux" and \
        str(env.get("TOPOLOGIES_ASCII", "")).strip().lower() in _FALSY
    cols, rows = terminal_size(stream, (80, 24)) if raw_tty else _fallback_size(env)
    stdin_tty = _isatty(sys.stdin)
    no_full = str(env.get("TOPOLOGIES_NO_FULLSCREEN", "")).strip().lower() not in _FALSY
    fullscreen_ok = (is_tty and windows_vt and stdin_tty and rows >= 12 and cols >= 60
                     and term not in ("dumb", "unknown") and not no_full)
    return TerminalCaps(is_tty=is_tty and windows_vt, color=depth, unicode=unicode,
                        columns=cols, rows=rows, notebook=notebook, ci=ci,
                        fullscreen_ok=fullscreen_ok, windows_vt=windows_vt, stream=stream)


def _fallback_size(env: Any) -> tuple[int, int]:
    try:
        cols = int(env.get("COLUMNS") or 100)
        rows = int(env.get("LINES") or 24)
        return max(20, cols), max(5, rows)
    except ValueError:
        return 100, 24


def plain_caps(columns: int = 100, rows: int = 24, *, color: int = 0,
               unicode: bool = True) -> TerminalCaps:
    """Caps for deterministic rendering (tests, files, snapshots)."""
    return TerminalCaps(is_tty=False, color=color, unicode=unicode, columns=columns,
                        rows=rows)


__all__ = ["TRUECOLOR", "TerminalCaps", "color_depth", "in_ci", "in_notebook", "plain_caps",
           "probe", "terminal_size"]
