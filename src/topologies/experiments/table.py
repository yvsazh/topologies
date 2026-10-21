"""``Table``: the one value object every experiment view returns, plus formatters."""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------- Table


@dataclass
class Table:
    """``columns`` + raw ``rows`` (+ optional pre-formatted ``display``) + ``meta``.

    ``meta`` keys used by renderers: ``title``, ``notes`` (footnotes),
    ``align`` ({column: "<"|">"}), ``heatmap`` ({"values": [[...]], "mode":
    "min"|"max", "row_offset", "col_offset"}), ``json`` (machine-readable payload).
    """

    columns: list[str]
    rows: list[list[Any]]
    meta: dict[str, Any] = field(default_factory=dict)
    display: list[list[str]] | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def cells(self) -> list[list[str]]:
        if self.display is not None:
            return [list(r) for r in self.display]
        return [[_fmt_cell(v) for v in row] for row in self.rows]

    def __str__(self) -> str:
        from topologies.experiments.render.console import render_table

        return "\n".join(render_table(self))

    def __repr__(self) -> str:
        return str(self)

    def to_markdown(self) -> str:
        from topologies.experiments.render.markdown import table_markdown

        return table_markdown(self)

    def to_html(self) -> str:
        from topologies.experiments.render.html import table_html

        return table_html(self)

    def _repr_html_(self) -> str:
        return self.to_html()

    def to_csv(self, path: str | Path | None = None) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(self.columns)
        for row in self.rows:
            w.writerow(["" if v is None else v for v in row])
        text = buf.getvalue()
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def to_dicts(self) -> list[dict[str, Any]]:
        return [dict(zip(self.columns, r)) for r in self.rows]

    def to_json(self) -> dict[str, Any]:
        from topologies.experiments.store import jsonable

        payload = self.meta.get("json")
        return jsonable(payload if payload is not None else
                        {"title": self.meta.get("title"), "columns": self.columns,
                         "rows": self.rows})


def _fmt_cell(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if not math.isfinite(v):
            return str(v)
        a = abs(v)
        if a != 0 and (a < 1e-3 or a >= 1e6):
            return f"{v:.3g}"
        return f"{v:.4g}" if a >= 100 else f"{v:.3f}"
    return str(v)


def fmt_params(n: Any) -> str:
    if n is None:
        return "—"
    from topologies.build.params import format_params

    try:
        return format_params(int(n))
    except (TypeError, ValueError):
        return str(n)


def fmt_seconds(s: Any) -> str:
    if s is None:
        return "—"
    try:
        s = float(s)
    except (TypeError, ValueError):
        return str(s)
    if s < 60:
        return f"{s:.1f}s" if s < 10 else f"{s:.0f}s"
    m, sec = divmod(round(s), 60)
    if m < 60:
        return f"{m}:{sec:02d}"
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{sec:02d}"


def fmt_rate(v: Any) -> str:
    if v is None:
        return "—"
    v = float(v)
    if v >= 1e6:
        return f"{v / 1e6:.2f}M"
    if v >= 1e3:
        return f"{v / 1e3:.1f}k"
    return f"{v:.1f}"


__all__ = ["Table", "fmt_params", "fmt_rate", "fmt_seconds"]
