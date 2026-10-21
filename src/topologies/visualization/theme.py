"""Shared palette and semantic style tokens (terminal, Matplotlib, HTML).

UI chrome maps semantic tokens onto the 16 named ANSI colors so the user's terminal
palette is respected. Only chart series use the hex categorical palette, and the same
slot order is used by every backend so an entity has one color everywhere.
The categorical palette was checked for color-vision deficiencies in this order;
never reorder or cycle it (fold extra series into "other" instead).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

# Categorical slots: (name, light-surface step, dark-surface step).
SERIES = (
    ("blue", "#2a78d6", "#3987e5"),
    ("orange", "#eb6834", "#d95926"),
    ("aqua", "#1baf7a", "#199e70"),
    ("yellow", "#eda100", "#c98500"),
    ("magenta", "#e87ba4", "#d55181"),
    ("green", "#008300", "#008300"),
    ("violet", "#4a3aa7", "#9085e9"),
    ("red", "#e34948", "#e66767"),
)
SERIES_LIGHT = tuple(light for _, light, _ in SERIES)
SERIES_DARK = tuple(dark for _, _, dark in SERIES)
MAX_SERIES = len(SERIES)

# 16-color fallback for series: only four hues stay distinguishable.
SERIES_16 = ("blue", "yellow", "cyan", "magenta",
             "bright_blue", "bright_yellow", "bright_cyan", "bright_magenta")

STATUS = MappingProxyType({
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
})

CHROME = MappingProxyType({
    "light": MappingProxyType({
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "baseline": "#c3c2b7",
    }),
    "dark": MappingProxyType({
        "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "baseline": "#383835",
    }),
})

SEQUENTIAL = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
              "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b")

TRAIN_SLOT = 0
VALIDATION_SLOT = 1


@dataclass(frozen=True)
class Style:
    """Terminal text style; ``fg`` is a named ANSI color, ``"#rrggbb"`` or ``None``."""

    fg: str | None = None
    bold: bool = False
    dim: bool = False
    italic: bool = False
    underline: bool = False

    def merge(self, other: Style | None) -> Style:
        if other is None:
            return self
        return Style(fg=other.fg or self.fg, bold=self.bold or other.bold,
                     dim=self.dim or other.dim, italic=self.italic or other.italic,
                     underline=self.underline or other.underline)


TOKENS = ("title", "subtitle", "label", "value", "muted", "border", "accent", "good", "warn",
          "bad", "best", "progress.done", "progress.todo", "key", "selected")


@dataclass(frozen=True, eq=False)
class Theme:
    """Semantic token → :class:`Style` map plus categorical series colors."""

    name: str
    tokens: Mapping[str, Style] = field(default_factory=dict)
    series: tuple[str, ...] = SERIES_DARK
    series16: tuple[str, ...] = SERIES_16
    mono: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "tokens", MappingProxyType(dict(self.tokens)))

    def with_(self, **tokens: Style | str) -> Theme:
        """Copy with tokens replaced; dotted tokens use ``__`` (``progress__done``)."""
        merged = dict(self.tokens)
        for key, value in tokens.items():
            merged[key.replace("__", ".")] = value if isinstance(value, Style) else Style(value)
        return replace(self, tokens=merged)

    def style(self, token: Any) -> Style | None:
        """Resolve a token name, ``series.N`` slot or :class:`Style` to a Style."""
        if token is None:
            return None
        if isinstance(token, Style):
            return token
        name = str(token)
        if name.startswith("series."):
            try:
                index = int(name.split(".", 1)[1])
            except ValueError:
                return None
            if self.mono:
                return Style(bold=index == 0)
            return Style(fg=f"series:{index}")
        if "+" in name:
            out = Style()
            for part in name.split("+"):
                out = out.merge(self.style(part))
            return out
        if name in ("bold", "dim", "italic", "underline"):
            return Style(**{name: True})
        return self.tokens.get(name)

    def series_hex(self, index: int) -> str:
        return self.series[index % len(self.series)]


def _default_tokens() -> dict[str, Style]:
    return {
        "title": Style(bold=True),
        "subtitle": Style(dim=True),
        "label": Style(bold=True),
        "value": Style(),
        "muted": Style(fg="bright_black"),
        "border": Style(fg="bright_black"),
        "accent": Style(fg="cyan"),
        "good": Style(fg="green"),
        "warn": Style(fg="yellow"),
        "bad": Style(fg="red"),
        "best": Style(fg="green", bold=True),
        "progress.done": Style(fg="cyan"),
        "progress.todo": Style(fg="bright_black"),
        "key": Style(fg="cyan", bold=True),
        "selected": Style(bold=True, underline=True),
    }


def _mono_tokens() -> dict[str, Style]:
    return {
        "title": Style(bold=True),
        "subtitle": Style(dim=True),
        "label": Style(bold=True),
        "value": Style(),
        "muted": Style(dim=True),
        "border": Style(dim=True),
        "accent": Style(bold=True),
        "good": Style(bold=True),
        "warn": Style(underline=True),
        "bad": Style(bold=True, underline=True),
        "best": Style(bold=True),
        "progress.done": Style(),
        "progress.todo": Style(dim=True),
        "key": Style(bold=True),
        "selected": Style(underline=True),
    }


def light_background() -> bool:
    """True when ``TOPOLOGIES_THEME=light`` or ``COLORFGBG`` reports a light background."""
    forced = os.environ.get("TOPOLOGIES_THEME", "").strip().lower()
    if forced in ("light", "dark"):
        return forced == "light"
    fgbg = os.environ.get("COLORFGBG", "")
    if fgbg:
        last = fgbg.split(";")[-1].strip()
        return last in ("7", "15")
    return False


DEFAULT = Theme("default", _default_tokens(), SERIES_DARK)
DEFAULT_LIGHT = Theme("default", _default_tokens(), SERIES_LIGHT)
MONO = Theme("mono", _mono_tokens(), SERIES_DARK, mono=True)
_THEMES: dict[str, Theme] = {}


def register_theme(theme: Theme) -> Theme:
    """Make ``theme`` selectable by name (``theme="<name>"`` options)."""
    _THEMES[theme.name] = theme
    return theme


def get_theme(theme: str | Theme | None = None) -> Theme:
    """Resolve a theme name/object; ``"default"`` follows the terminal background."""
    if isinstance(theme, Theme):
        return theme
    name = (theme or "default").strip().lower()
    if name in ("mono", "monochrome"):
        return MONO
    if name in _THEMES:
        return _THEMES[name]
    if name == "light":
        return DEFAULT_LIGHT
    if name == "dark":
        return DEFAULT
    if name != "default":
        raise ValueError(f"Unknown theme {theme!r}; use 'default', 'mono', 'light', 'dark' "
                         f"or one of {sorted(_THEMES)}")
    return DEFAULT_LIGHT if light_background() else DEFAULT


def series_colors(mode: str = "light") -> tuple[str, ...]:
    """Categorical hex colors for ``mode`` ``"light"`` or ``"dark"`` surfaces."""
    return SERIES_DARK if mode == "dark" else SERIES_LIGHT


def css_variables(mode: str = "light") -> str:
    """CSS custom properties for the chrome tokens and series palette of one mode."""
    chrome = CHROME["dark" if mode == "dark" else "light"]
    lines = [f"--surface:{chrome['surface']}", f"--ink:{chrome['ink']}",
             f"--ink2:{chrome['ink2']}", f"--muted:{chrome['muted']}",
             f"--grid:{chrome['grid']}", f"--baseline:{chrome['baseline']}"]
    lines += [f"--s{i + 1}:{c}" for i, c in enumerate(series_colors(mode))]
    lines += [f"--{k}:{v}" for k, v in STATUS.items()]
    return ";".join(lines) + ";"


__all__ = [
    "CHROME",
    "DEFAULT",
    "MAX_SERIES",
    "MONO",
    "SEQUENTIAL",
    "SERIES",
    "SERIES_16",
    "SERIES_DARK",
    "SERIES_LIGHT",
    "STATUS",
    "TOKENS",
    "TRAIN_SLOT",
    "VALIDATION_SLOT",
    "Style",
    "Theme",
    "css_variables",
    "get_theme",
    "light_background",
    "register_theme",
    "series_colors",
]
