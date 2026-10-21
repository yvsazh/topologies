"""Self-contained HTML for introspection reports.

The page shell, CSS and sortable tables come from ``experiments.render.html``.
Figures are rendered with the Matplotlib view renderers into base64 PNGs when
Matplotlib is installed. Without it, the page still contains every table plus a
note. There are no JS dependencies.
"""
from __future__ import annotations

import base64
import io
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from topologies.experiments.render.html import CSS, JS, _section, esc
from topologies.visualization.views import ViewSpec


def _table(columns: list[str], rows: list[Any]) -> str:
    head = "".join(f"<th>{esc(c)}</th>" for c in columns)
    body = []
    for row in rows:
        values = [row.get(c, "") for c in columns] if isinstance(row, Mapping) else list(row)
        cells = []
        for v in values:
            if isinstance(v, float):
                cells.append(f"<td class=\"num\" data-v=\"{v!r}\">{v:.4g}</td>")
            elif isinstance(v, int) and not isinstance(v, bool):
                cells.append(f"<td class=\"num\" data-v=\"{v}\">{v}</td>")
            else:
                cells.append(f"<td style=\"white-space:normal\">{esc(v)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (f"<div class=\"scroll\"><table class=\"sortable\"><thead><tr>{head}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div>")


def _matplotlib_available() -> bool:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True


def figure_png(spec: ViewSpec) -> str | None:
    """One view rendered to a base64 PNG data URI (``None`` without Matplotlib)."""
    if not _matplotlib_available():
        return None
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from topologies.visualization.matplotlib_views import render_views

    fig = Figure(figsize=(6.5, 4.2))
    FigureCanvasAgg(fig)
    try:
        render_views(fig, [spec])
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=90, bbox_inches="tight")
    finally:
        fig.clear()
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def view_html(spec: ViewSpec, *, figures: bool) -> str:
    if spec.kind == "table":
        return f"<p class=\"muted\">{esc(spec.title)}</p>" + _table(
            [str(c) for c in spec.data.get("columns", [])], list(spec.data.get("rows", [])))
    if spec.kind == "text":
        return f"<pre>{esc(spec.data.get('text', ''))}</pre>"
    caption = f"<figcaption class=\"muted\">{esc(spec.caption)}</figcaption>" if spec.caption else ""
    if not figures:
        return f"<p class=\"muted\">figure “{esc(spec.title)}” ({esc(spec.kind)}) needs Matplotlib</p>"
    uri = figure_png(spec)
    return f"<figure><img alt=\"{esc(spec.title)}\" src=\"{uri}\">{caption}</figure>"


def report_html(report: Any, *, max_figures_per_section: int = 6) -> str:
    figures = _matplotlib_available()
    subject = report.subject
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    adapter = (subject.get("adapter") or {}).get("adapter")
    cards = "".join(f"<div class=\"card\"><span class=\"muted\">{esc(k)}</span><b>{esc(v)}</b></div>"
                    for k, v in (("model", str(subject.get("model", "?")).rsplit(".", 1)[-1]),
                                 ("parameters", subject.get("parameters")), ("adapter", adapter),
                                 ("preset", report.preset), ("findings", len(report.findings))))
    head = (f"<h1>Introspection report</h1><p class=\"muted\">{esc(subject.get('model'))} · "
            f"generated {stamp}</p><div class=\"cards\">{cards}</div>")
    if not figures:
        head += "<p class=\"warn\">Matplotlib is not installed: figures are omitted, tables only.</p>"
    findings = [{"severity": f.severity, "basis": f.basis, "finding": f.title,
                 "sites": ", ".join(f.sites)[:80], "explanation": f.explanation,
                 "evidence": "; ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                       for k, v in list(f.evidence.items())[:5])}
                for f in report._sorted()]
    sections = [head,
                _section("Findings", _table(["severity", "basis", "finding", "sites", "evidence",
                                             "explanation"], findings) if findings else
                         "<p class=\"muted\">no findings</p>", "findings"),
                _section("Capabilities", _table(["technique", "tier", "available", "reason", "fix"],
                                                report.capabilities.to_records()), "capabilities")]
    for name, result in report.results.items():
        producer = getattr(getattr(result, "payload", None), "views", None)
        views = producer() if callable(producer) else []
        body = []
        if result.assumptions:
            body.append("<details><summary>assumptions</summary><ul class=\"ins\">" + "".join(
                f"<li>{esc(a)}</li>" for a in result.assumptions) + "</ul></details>")
        shown = 0
        for spec in views:
            if spec.kind not in ("table", "text"):
                if shown >= max_figures_per_section:
                    continue
                shown += 1
            body.append(view_html(spec, figures=figures))
        sections.append(_section(name, "\n".join(body) or "<p class=\"muted\">no views</p>", name))
    return ("<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            "<title>Introspection report</title>"
            f"<style>{CSS}</style></head><body><main>\n" + "\n".join(sections)
            + f"\n</main><script>{JS}</script></body></html>\n")
