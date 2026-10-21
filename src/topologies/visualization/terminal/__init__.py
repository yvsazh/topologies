"""Stdlib-only terminal toolkit: capabilities, styled text, glyphs, widgets, layout.

Widgets return :class:`Line` objects; screens (``screen.py``) encode and write them.
Everything here is safe to use in user scripts::

    from topologies.visualization.terminal import progress_bar, sparkline, line, to_ansi
    print(to_ansi(line(progress_bar(0.4, 20), " 40%")))
"""

from topologies.visualization.terminal.caps import (
    TerminalCaps,
    color_depth,
    in_ci,
    in_notebook,
    plain_caps,
    probe,
    terminal_size,
)
from topologies.visualization.terminal.driver import LiveDriver, OutputCapture
from topologies.visualization.terminal.glyphs import ASCII, UNICODE, Glyphs, glyphs, to_ascii
from topologies.visualization.terminal.keys import KeyReader, decode_keys
from topologies.visualization.terminal.layout import (
    Fixed,
    Min,
    Ratio,
    Rect,
    columns_for_width,
    grid_layout,
    split_cols,
    split_rows,
)
from topologies.visualization.terminal.panels import Panel, get_panel, panel, resolve_panels
from topologies.visualization.terminal.screen import (
    FullscreenScreen,
    InlineRegion,
    NotebookScreen,
    NullScreen,
    PlainScreen,
    Screen,
    choose_screen,
)
from topologies.visualization.terminal.text import (
    Line,
    Span,
    as_line,
    cell_width,
    clip_text,
    line,
    strip_ansi,
    styled,
    to_ansi,
    to_html,
    to_plain,
)
from topologies.visualization.terminal.widgets import (
    Column,
    badge,
    box,
    fmt_percent,
    hstack,
    kv_line,
    line_chart,
    meter,
    nice_ticks,
    progress_bar,
    rule,
    sparkline,
    table,
    vstack,
)

__all__ = [
    "ASCII",
    "UNICODE",
    "Column",
    "Fixed",
    "FullscreenScreen",
    "Glyphs",
    "InlineRegion",
    "KeyReader",
    "Line",
    "LiveDriver",
    "Min",
    "NotebookScreen",
    "NullScreen",
    "OutputCapture",
    "Panel",
    "PlainScreen",
    "Ratio",
    "Rect",
    "Screen",
    "Span",
    "TerminalCaps",
    "as_line",
    "badge",
    "box",
    "cell_width",
    "choose_screen",
    "clip_text",
    "color_depth",
    "columns_for_width",
    "decode_keys",
    "fmt_percent",
    "get_panel",
    "glyphs",
    "grid_layout",
    "hstack",
    "in_ci",
    "in_notebook",
    "kv_line",
    "line",
    "line_chart",
    "meter",
    "nice_ticks",
    "panel",
    "plain_caps",
    "probe",
    "progress_bar",
    "resolve_panels",
    "rule",
    "sparkline",
    "split_cols",
    "split_rows",
    "strip_ansi",
    "styled",
    "table",
    "terminal_size",
    "to_ansi",
    "to_ascii",
    "to_html",
    "to_plain",
    "vstack",
]
