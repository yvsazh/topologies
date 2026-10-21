"""Markdown renderers: tables and a pasteable report (leaderboard, pivot, insights)."""

from __future__ import annotations

from typing import Any


def _esc(s: Any) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def table_markdown(table: Any, *, title: bool = True) -> str:
    lines = []
    meta = table.meta or {}
    if title and meta.get("title"):
        lines.append(f"**{_esc(meta['title'])}**")
        lines.append("")
    cols = list(table.columns)
    lines.append("| " + " | ".join(_esc(c) for c in cols) + " |")
    aligns = []
    for j in range(len(cols)):
        vals = [r[j] for r in table.rows if j < len(r) and r[j] is not None]
        num = j > 0 and bool(vals) and all(isinstance(v, (int, float)) and
                                           not isinstance(v, bool) for v in vals)
        aligns.append("---:" if num else "---")
    lines.append("| " + " | ".join(aligns) + " |")
    for row in table.cells():
        lines.append("| " + " | ".join(_esc(c) for c in row) + " |")
    for n in meta.get("notes") or []:
        lines.append("")
        lines.append(f"_{_esc(n)}_")
    return "\n".join(lines)


def report_markdown(rs: Any) -> str:
    """Leaderboard, pivot, marginals, importance, insights and failures as Markdown."""
    out = [f"# {rs.name or 'Experiment'}", ""]
    spec = rs.meta.get("spec") or {}
    if spec.get("description"):
        out += [spec["description"], ""]
    counts: dict[str, int] = {}
    for r in rs.records:
        counts[r.status] = counts.get(r.status, 0) + 1
    out += [f"Goal: `{rs.goal[0]}` ({rs.goal[1]}) · trials: "
            + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())), ""]

    def section(title: str, fn: Any) -> None:
        try:
            body = fn()
        except Exception as e:  # noqa: BLE001
            body = f"_section failed: {type(e).__name__}: {e}_"
        if body:
            out.extend([f"## {title}", "", body, ""])

    section("Leaderboard", lambda: table_markdown(rs.table(), title=False))
    varying = [a for a in rs.axes if rs.varies(a)]
    if len(varying) >= 2:
        from topologies.experiments.analysis import pivot_axes

        row, col = pivot_axes(rs)  # type: ignore[misc]
        section("Pivot", lambda: table_markdown(rs.pivot(row, col)))
    if varying:
        section("Marginal means", lambda: "\n\n".join(
            table_markdown(t) for t in rs.marginals().values()))
        section("Importance", lambda: table_markdown(rs.importance(), title=False))
    section("Insights", lambda: "\n".join(
        f"- {'⚠ ' if i.severity == 'warn' else ''}{_esc(i.text)}" for i in rs.insights()))
    if any(r.status not in ("completed", "pruned") for r in rs.records):
        section("Failures", lambda: table_markdown(rs.failures(), title=False))
    return "\n".join(out).rstrip() + "\n"


__all__ = ["report_markdown", "table_markdown"]
