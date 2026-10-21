"""Unicode and ASCII glyph sets (every glyph is exactly one display cell).

Renderers always build frames with the Unicode set; when the output stream cannot
encode it (``caps.unicode`` is false) the screen encoders translate every glyph to its
ASCII stand-in with :func:`to_ascii`. Both sets have identical widths, so a frame laid
out once is valid in either encoding.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Glyphs:
    completed: str
    failed: str
    diverged: str
    oom: str
    timeout: str
    pruned: str
    interrupted: str
    best: str
    running: str
    bar_done: str
    bar_half: str
    bar_todo: str
    box_tl: str
    box_tr: str
    box_bl: str
    box_br: str
    box_h: str
    box_v: str
    rule: str
    up: str
    down: str
    flat: str
    blocks: str
    markers: str
    bullet: str
    dot: str
    ellipsis: str

    def status(self, status: str) -> str:
        """Glyph for a trial/run status; unknown statuses get ``?``."""
        return getattr(self, str(status), None) if str(status) in STATUSES else "?"


STATUSES = ("completed", "failed", "diverged", "oom", "timeout", "pruned", "interrupted")

UNICODE = Glyphs(
    completed="✓", failed="✗", diverged="↯", oom="▲", timeout="◔", pruned="✂",
    interrupted="‖", best="★", running="▸", bar_done="━", bar_half="╸", bar_todo="─",
    box_tl="╭", box_tr="╮", box_bl="╰", box_br="╯", box_h="─", box_v="│", rule="─",
    up="↑", down="↓", flat="→", blocks="▁▂▃▄▅▆▇█", markers="●▲■◆○△□◇", bullet="•",
    dot="·", ellipsis="…",
)

ASCII = Glyphs(
    completed="+", failed="x", diverged="~", oom="!", timeout="t", pruned="-",
    interrupted="=", best="*", running=">", bar_done="#", bar_half=">", bar_todo="-",
    box_tl="+", box_tr="+", box_bl="+", box_br="+", box_h="-", box_v="|", rule="-",
    up="^", down="v", flat="-", blocks=".:-=+*##", markers="o^#+OA=%", bullet="*",
    dot=".", ellipsis=".",
)

_EXTRA = {
    "┤": "|", "├": "|", "└": "+", "┘": "+", "┌": "+", "┐": "+", "┴": "+", "┬": "+",
    "┼": "+", "╴": "-", "╶": "-", "│": "|", "─": "-", "━": "#", "╸": ">", "░": ".",
    "▒": ":", "▓": "+", "█": "#", "▏": "|", "▕": "|", "●": "o", "◌": "o", "±": "~",
    "−": "-", "×": "x", "≈": "~", "Δ": "d", "∞": "~", "•": "*", "·": ".", "…": ".",
    "▸": ">", "★": "*", "✓": "+", "✗": "x", "↯": "~", "▲": "!", "◔": "t", "✂": "-",
    "‖": "=", "↑": "^", "↓": "v", "→": "-", "←": "-", "⚠": "!", "╭": "+", "╮": "+",
    "╰": "+", "╯": "+", "▁": ".", "▂": ":", "▃": "-", "▄": "=", "▅": "+", "▆": "*",
    "▇": "#", "■": "#", "◆": "+", "○": "O", "△": "A", "□": "=", "◇": "%", "┄": "-",
    "┈": "-", "╌": "-", "⋯": ".", "≤": "<", "≥": ">", "²": "2", "µ": "u", "é": "e",
}


def _braille_ascii(code: int) -> str:
    bits = (code - 0x2800).bit_count()
    return " " if bits == 0 else ("." if bits <= 2 else (":" if bits <= 4 else "#"))


_TABLE = {ord(k): v for k, v in _EXTRA.items()}
_TABLE.update({code: _braille_ascii(code) for code in range(0x2800, 0x2900)})


def to_ascii(text: str) -> str:
    """Translate the library's Unicode glyphs to one-cell ASCII stand-ins."""
    out = text.translate(_TABLE)
    if out.isascii():
        return out
    return "".join(ch if ch.isascii() else "?" for ch in out)


def glyphs(unicode: bool = True) -> Glyphs:
    """The glyph set for a capability flag."""
    return UNICODE if unicode else ASCII


__all__ = ["ASCII", "STATUSES", "UNICODE", "Glyphs", "glyphs", "to_ascii"]
