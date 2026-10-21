"""Styled text model with display-cell widths and ANSI/plain/HTML encoders.

Widgets build :class:`Line` objects (tuples of :class:`Span`); only screens encode
them. Widths are measured in terminal cells: wide East-Asian characters count 2,
combining marks and zero-width joiners count 0. ``TOPOLOGIES_AMBIGUOUS_WIDTH=2``
counts East-Asian *ambiguous* characters as 2 cells for CJK terminals.
"""

from __future__ import annotations

import html
import os
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from topologies.visualization.theme import SERIES_16, Style, Theme, get_theme

_ZERO_CATEGORIES = frozenset({"Mn", "Me", "Cf", "Cc"})
_ZERO_WIDTH = frozenset({0x200B, 0x200C, 0x200D, 0x2060, 0xFE0E, 0xFE0F})


def _ambiguous_wide() -> bool:
    return os.environ.get("TOPOLOGIES_AMBIGUOUS_WIDTH", "1").strip() == "2"


def char_width(ch: str, ambiguous: int = 1) -> int:
    """Display cells of one code point (0, 1 or 2)."""
    code = ord(ch)
    if code < 0x7F:
        return 1 if code >= 0x20 else 0
    if 0x2800 <= code <= 0x28FF:  # braille: always narrow
        return 1
    if code in _ZERO_WIDTH or unicodedata.combining(ch):
        return 0
    if unicodedata.category(ch) in _ZERO_CATEGORIES:
        return 0
    eaw = unicodedata.east_asian_width(ch)
    if eaw in ("W", "F"):
        return 2
    if eaw == "A":
        return ambiguous
    return 1


@lru_cache(maxsize=8192)
def _width_cached(s: str, ambiguous: int) -> int:
    return sum(char_width(ch, ambiguous) for ch in s)


def cell_width(s: str) -> int:
    """Display width of ``s`` in terminal cells (no escape sequences expected)."""
    if not s:
        return 0
    amb = 2 if _ambiguous_wide() else 1
    if s.isascii():
        return sum(1 for ch in s if ch >= " " and ch != "\x7f")
    if len(s) <= 256:
        return _width_cached(s, amb)
    return sum(char_width(ch, amb) for ch in s)


def clip_text(s: str, width: int) -> str:
    """Longest prefix of ``s`` that fits ``width`` cells (never splits a wide char)."""
    if width <= 0:
        return ""
    if cell_width(s) <= width:
        return s
    amb = 2 if _ambiguous_wide() else 1
    out, used = [], 0
    for ch in s:
        w = char_width(ch, amb)
        if used + w > width:
            break
        out.append(ch)
        used += w
    return "".join(out)


@dataclass(frozen=True)
class Span:
    """A run of text with one style (token name, :class:`Style` or ``None``)."""

    text: str
    style: Any = None

    @property
    def width(self) -> int:
        cached = self.__dict__.get("_width")
        if cached is None:
            cached = cell_width(self.text)
            object.__setattr__(self, "_width", cached)
        return cached


class Line(tuple):
    """An immutable sequence of :class:`Span` forming one terminal row."""

    __slots__ = ()

    def __new__(cls, spans: Iterable[Span] = ()) -> Line:  # noqa: PYI034
        return super().__new__(cls, tuple(spans))

    @property
    def width(self) -> int:
        return sum(span.width for span in self)

    @property
    def plain(self) -> str:
        return "".join(span.text for span in self)

    def clip(self, width: int, ellipsis: str = "…") -> Line:
        """Fit ``width`` cells; the cut is marked with a ``muted`` ellipsis."""
        if self.width <= width:
            return self
        if width <= 0:
            return Line()
        ew = cell_width(ellipsis)
        budget = max(0, width - ew)
        out: list[Span] = []
        used = 0
        for span in self:
            w = span.width
            if used + w <= budget:
                out.append(span)
                used += w
                continue
            part = clip_text(span.text, budget - used)
            if part:
                out.append(Span(part, span.style))
            break
        if ew and ew <= width:
            out.append(Span(ellipsis, "muted"))
        return Line(out)

    def crop(self, width: int) -> Line:
        """Hard cut at ``width`` cells without an ellipsis."""
        if self.width <= width:
            return self
        out: list[Span] = []
        used = 0
        for span in self:
            w = span.width
            if used + w <= width:
                out.append(span)
                used += w
                continue
            part = clip_text(span.text, width - used)
            if part:
                out.append(Span(part, span.style))
            break
        return Line(out)

    def pad(self, width: int, align: str = "<", *, clip: bool = True) -> Line:
        """Pad with spaces to exactly ``width`` cells (``<``, ``>`` or ``^``)."""
        line = self.clip(width) if clip else self
        gap = max(0, width - line.width)
        if not gap:
            return line
        if align == ">":
            return Line((Span(" " * gap), *line))
        if align == "^":
            left = gap // 2
            return Line((Span(" " * left), *line, Span(" " * (gap - left))))
        return Line((*line, Span(" " * gap)))

    def __add__(self, other: Any) -> Line:  # type: ignore[override]
        return line(self, other)

    def __repr__(self) -> str:
        return f"Line({self.plain!r})"


def line(*parts: Any) -> Line:
    """Build a :class:`Line` from str, :class:`Span`, Line or ``(text, style)`` parts."""
    spans: list[Span] = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, Span):
            if part.text:
                spans.append(part)
        elif isinstance(part, Line):
            spans.extend(part)
        elif isinstance(part, str):
            if part:
                spans.append(Span(part))
        elif isinstance(part, tuple) and len(part) == 2 and isinstance(part[0], str):
            if part[0]:
                spans.append(Span(part[0], part[1]))
        elif isinstance(part, (list, tuple)):
            spans.extend(line(*part))
        else:
            spans.append(Span(str(part)))
    return Line(spans)


def as_line(value: Any) -> Line:
    """Coerce str/Span/Line/parts to a Line (newlines become spaces)."""
    if isinstance(value, Line):
        return value
    if isinstance(value, str):
        return Line((Span(value.replace("\n", " ").replace("\r", " ")),)) if value else Line()
    return line(value)


def styled(text: str, style: Any) -> Line:
    return Line((Span(text, style),)) if text else Line()


# ---------------------------------------------------------------- color encoding

_NAMED = {
    "black": 30, "red": 31, "green": 32, "yellow": 33, "blue": 34, "magenta": 35,
    "cyan": 36, "white": 37, "bright_black": 90, "bright_red": 91, "bright_green": 92,
    "bright_yellow": 93, "bright_blue": 94, "bright_magenta": 95, "bright_cyan": 96,
    "bright_white": 97, "gray": 90, "grey": 90,
}
# Approximate RGB of the 16 named colors (xterm defaults) for nearest-match fallback.
_NAMED_RGB = {
    30: (0, 0, 0), 31: (205, 49, 49), 32: (13, 188, 121), 33: (229, 229, 16),
    34: (36, 114, 200), 35: (188, 63, 188), 36: (17, 168, 205), 37: (229, 229, 229),
    90: (102, 102, 102), 91: (241, 76, 76), 92: (35, 209, 139), 93: (245, 245, 67),
    94: (59, 142, 234), 95: (214, 112, 214), 96: (41, 184, 219), 97: (255, 255, 255),
}


def hex_rgb(value: str) -> tuple[int, int, int]:
    v = value.lstrip("#")
    return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)


_CUBE = (0, 95, 135, 175, 215, 255)


def rgb_to_256(r: int, g: int, b: int) -> int:
    """Nearest xterm-256 index (6×6×6 cube or gray ramp)."""
    def nearest(v: int) -> int:
        return min(range(6), key=lambda i: abs(_CUBE[i] - v))

    ri, gi, bi = nearest(r), nearest(g), nearest(b)
    cube = (_CUBE[ri], _CUBE[gi], _CUBE[bi])
    cube_idx = 16 + 36 * ri + 6 * gi + bi
    gray = round(((r + g + b) / 3 - 8) / 10)
    gray = max(0, min(23, gray))
    gray_v = 8 + 10 * gray

    def dist(c: tuple[int, int, int]) -> int:
        return (c[0] - r) ** 2 + (c[1] - g) ** 2 + (c[2] - b) ** 2

    return cube_idx if dist(cube) <= dist((gray_v, gray_v, gray_v)) else 232 + gray


def _rgb_of_256(index: int) -> tuple[int, int, int]:
    if index >= 232:
        v = 8 + 10 * (index - 232)
        return v, v, v
    if index >= 16:
        i = index - 16
        return _CUBE[i // 36], _CUBE[(i // 6) % 6], _CUBE[i % 6]
    code = (30 + index) if index < 8 else (90 + index - 8)
    return _NAMED_RGB[code]


def rgb_to_16(r: int, g: int, b: int) -> int:
    """Nearest named SGR foreground code (30–37, 90–97)."""
    return min(_NAMED_RGB, key=lambda c: sum((x - y) ** 2 for x, y in
                                             zip(_NAMED_RGB[c], (r, g, b))))


def _fg_codes(fg: str | None, theme: Theme, depth: int) -> str:
    if not fg or depth <= 0:
        return ""
    if fg.startswith("series:"):
        index = int(fg.split(":", 1)[1])
        if depth >= 256:
            fg = theme.series_hex(index)
        else:
            fg = theme.series16[index % len(theme.series16)] if theme.series16 else SERIES_16[0]
    if fg in _NAMED:
        return str(_NAMED[fg])
    if fg.startswith("#") and len(fg) == 7:
        r, g, b = hex_rgb(fg)
        if depth >= 16_777_216:
            return f"38;2;{r};{g};{b}"
        if depth >= 256:
            return f"38;5;{rgb_to_256(r, g, b)}"
        return str(rgb_to_16(r, g, b))
    return ""


@lru_cache(maxsize=1024)
def _sgr(style: Style, theme: Theme, depth: int) -> str:
    codes = []
    if style.bold:
        codes.append("1")
    if style.dim:
        codes.append("2")
    if style.italic:
        codes.append("3")
    if style.underline:
        codes.append("4")
    fg = _fg_codes(style.fg, theme, depth)
    if fg:
        codes.append(fg)
    return f"\x1b[{';'.join(codes)}m" if codes else ""


def to_ansi(value: Any, theme: Theme | str | None = None, color_depth: int = 16, *,
            unicode: bool = True) -> str:
    """Encode a Line with SGR styles; always ends with a reset when styled.

    ``color_depth=0`` still emits bold/dim/underline (the ``mono`` look) only when the
    theme is the mono theme; otherwise depth 0 produces plain text.
    """
    ln = as_line(value)
    th = get_theme(theme)
    out = []
    for span in ln:
        text = span.text if unicode else _ascii(span.text)
        st = th.style(span.style)
        seq = _sgr(st, th, color_depth) if st is not None and (color_depth > 0 or th.mono) \
            else ""
        if seq:
            out.append(seq + text + "\x1b[0m")
        else:
            out.append(text)
    return "".join(out)


def _ascii(text: str) -> str:
    from topologies.visualization.terminal.glyphs import to_ascii

    return to_ascii(text)


def to_plain(value: Any, *, unicode: bool = True) -> str:
    """Encode a Line as plain text (no escape sequences)."""
    text = as_line(value).plain
    return text if unicode else _ascii(text)


def _css(style: Style, theme: Theme) -> str:
    parts = []
    fg = style.fg
    if fg and fg.startswith("series:"):
        fg = theme.series_hex(int(fg.split(":", 1)[1]))
    if fg in _NAMED:
        r, g, b = _NAMED_RGB[_NAMED[fg]]
        fg = f"var(--ansi-{fg}, #{r:02x}{g:02x}{b:02x})"
    if fg:
        parts.append(f"color:{fg}")
    if style.bold:
        parts.append("font-weight:600")
    if style.dim:
        parts.append("opacity:.65")
    if style.italic:
        parts.append("font-style:italic")
    if style.underline:
        parts.append("text-decoration:underline")
    return ";".join(parts)


def to_html(value: Any, theme: Theme | str | None = None) -> str:
    """Encode a Line as HTML ``<span>`` runs (for notebooks)."""
    th = get_theme(theme)
    out = []
    for span in as_line(value):
        text = html.escape(span.text)
        st = th.style(span.style)
        css = _css(st, th) if st is not None else ""
        out.append(f'<span style="{css}">{text}</span>' if css else text)
    return "".join(out)


def strip_ansi(text: str) -> str:
    """Remove CSI/OSC escape sequences from ``text``."""
    import re

    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07|\x1b[=>78]", "", text)


__all__ = [
    "Line",
    "Span",
    "as_line",
    "cell_width",
    "char_width",
    "clip_text",
    "hex_rgb",
    "line",
    "rgb_to_16",
    "rgb_to_256",
    "strip_ansi",
    "styled",
    "to_ansi",
    "to_html",
    "to_plain",
]
