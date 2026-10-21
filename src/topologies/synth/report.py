"""Console views of synthetic datasets: records, funnel, coverage, run progress.

Every view returns (or prints) ``experiments.table.Table`` objects so they
render in the terminal, as Markdown and as HTML the same way experiment
results do. ``report(ds, html=...)`` builds the full report (``quality/``).
Rendering failures never break a run: callers swallow them.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from topologies.experiments.table import Table, fmt_seconds


def _short(v: Any, width: int = 90) -> str:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)
    s = s.replace("\n", " ⏎ ")
    return s if len(s) <= width else s[: width - 1] + "…"


def format_records(records: Sequence[dict], *, fields: Sequence[str] | None = None,
                   title: str | None = None, width: int = 100) -> str:
    """Readable block per record: fields first, then attributes (dimmed)."""
    lines = [title] if title else []
    for r in records:
        rid = r.get("_id", "?")
        lines.append(f"── {rid} " + "─" * max(0, 40 - len(str(rid))))
        keys = list(fields) if fields else [k for k in r if not str(k).startswith("_")]
        for k in keys:
            lines.append(f"  {k}: {_short(r.get(k), width)}")
        attrs = r.get("_attrs")
        if attrs:
            shown = ", ".join(f"{k}={_short(v, 40)}" for k, v in attrs.items()
                              if k not in keys)
            if shown:
                lines.append(f"  · {_short(shown, width)}")
        if r.get("_op"):
            v = r.get("_verdict") or {}
            lines.append(f"  ✗ {r['_op']}: {_short(v.get('reason', ''), width)}")
    return "\n".join(lines)


def funnel_table(ds: Any) -> Table:
    """Per-op funnel: in, out, pass rate, time."""
    m = ds.manifest
    funnel = m.get("funnel") or {}
    times = (m.get("time") or {}).get("ops") or {}
    rows = []
    gen = funnel.get("generator", {})
    rows.append(["generator", gen.get("calls"), gen.get("out"),
                 (gen.get("out", 0) / gen["calls"]) if gen.get("calls") else None,
                 gen.get("errors"), None])
    for name, d in funnel.items():
        if name in ("generator", "select"):
            continue
        if "in" in d:
            rate = d["out"] / d["in"] if d.get("in") else None
            rows.append([name, d.get("in"), d.get("out"), rate, d["in"] - d["out"],
                         times.get(name)])
        elif "dropped" in d:
            rows.append([name, None, None, None, d["dropped"], None])
    sel = funnel.get("select") or {}
    rows.append(["selected", sel.get("target"), sel.get("selected"), None, None, None])
    display = [[r[0], _n(r[1]), _n(r[2]), f"{r[3]:.1%}" if r[3] is not None else "—",
                _n(r[4]), fmt_seconds(r[5]) if r[5] else "—"] for r in rows]
    return Table(["stage", "in", "out", "pass", "rejected", "time"], rows,
                 meta={"title": f"funnel  {ds.name} ({ds.version_id})"}, display=display)


def _n(v: Any) -> str:
    return "—" if v is None else f"{v:,}"


def coverage_table(ds: Any, *, attrs: Sequence[str] | None = None) -> Table:
    """Per attribute value: generated, accepted, acceptance rate, share vs. target."""
    m = ds.manifest
    cells = m.get("cells") or {}
    spec_attrs = (m.get("attributes") or {}).get("attrs") or {}
    rows = []
    for name, d in cells.items():
        if attrs is not None and name not in attrs:
            continue
        target = _target_probs(spec_attrs.get(name) or {})
        tot_acc = sum(a for _, a in d.values()) or 1
        for val, (g, a) in sorted(d.items(), key=lambda kv: -kv[1][0]):
            rows.append([name, val, g, a, a / g if g else None, a / tot_acc, target.get(val)])
    display = [[r[0], _short(r[1], 30), _n(r[2]), _n(r[3]),
                f"{r[4]:.0%}" if r[4] is not None else "—", f"{r[5]:.1%}",
                f"{r[6]:.1%}" if r[6] is not None else "—"] for r in rows]
    return Table(["attribute", "value", "generated", "accepted", "acceptance", "share",
                  "target"], rows, meta={"title": "attribute coverage"}, display=display)


def _target_probs(desc: dict) -> dict[str, float]:
    vals = desc.get("values")
    if not isinstance(vals, list) or not vals:
        return {}
    probs = desc.get("probs") or [1.0 / len(vals)] * len(vals)
    return {(v if isinstance(v, str) else json.dumps(v, sort_keys=True)): p
            for v, p in zip(vals, probs)}


def spark(values: Sequence[float], width: int = 16) -> str:
    """Acceptance-rate sparkline over the last ``width`` chunks (fixed 0–1 scale)."""
    from topologies.visualization.terminal.text import to_plain
    from topologies.visualization.terminal.widgets import sparkline

    vals = list(values)[-width:]
    if not vals:
        return ""
    return to_plain(sparkline(vals, len(vals), lo=0.0, hi=1.0))


def live_line(spec: Any, res: Any, selection: Any, elapsed: float, clock: Any,
              rates: Sequence[float], recent: Sequence[str], *, tty: bool = False) -> str:
    """One progress line: selected/target, acceptance sparkline, records/s, ETA."""
    from topologies.synth.pipeline import eta_seconds
    from topologies.visualization.format import fmt_duration, fmt_rate

    done, target = len(selection.chosen), selection.target
    eta = eta_seconds(done, target, elapsed)
    acc = rates[-1] if rates else 0.0
    rate = done / elapsed if elapsed > 0 else 0.0
    limit = getattr(clock, "max_time", None)
    parts = [f"{spec.name}: {done:,}/{target:,}", f"chunk {res.index}",
             f"accept {acc:.0%} {spark(rates)}", fmt_rate(rate, "rec"),
             f"{fmt_duration(elapsed)}" + (f"/{fmt_duration(limit)}" if limit else "")
             + (f" eta {fmt_duration(eta)}" if eta else "")]
    line = " · ".join(parts)
    if recent and not tty:
        line += f"\n  recent rejections: {recent[0]}"
    return line


def progress_line(spec: Any, res: Any, selection: Any, elapsed: float, clock: Any,
                  rates: Sequence[float], width: int) -> Any:
    """Styled live line (bar, percent, counts, acceptance, rate, time < ETA)."""
    from topologies.synth.pipeline import eta_seconds
    from topologies.visualization.format import fmt_clock, fmt_rate
    from topologies.visualization.terminal.text import line
    from topologies.visualization.terminal.widgets import fmt_percent, progress_bar

    done, target = len(selection.chosen), selection.target
    frac = done / target if target else None
    eta = eta_seconds(done, target, elapsed)
    rate = done / elapsed if elapsed > 0 else 0.0
    acc = rates[-1] if rates else 0.0
    tail = line(f"  {done:,}/{target:,}", ("  ·  ", "muted"), f"chunk {res.index}",
                ("  ·  ", "muted"), f"accept {acc:.0%} ", spark(rates), ("  ·  ", "muted"),
                fmt_rate(rate, "rec"), ("  ·  ", "muted"), fmt_clock(elapsed),
                (f" < {fmt_clock(eta)}" if eta is not None else "", "muted"))
    head = line((spec.name, "title"), "  ")
    bar_w = max(0, min(30, width - head.width - tail.width - 5))
    bar = line(progress_bar(frac, bar_w), " ") if bar_w >= 8 else line()
    return line(head, bar, fmt_percent(frac), tail).clip(width)


def run_summary(ds: Any, result: dict[str, Any]) -> str:
    m = ds.manifest
    stop = result.get("stop_reason")
    secs = (m.get("time") or {}).get("seconds")
    head = (f"{ds.name}: {len(ds):,} records ({', '.join(f'{k} {v:,}' for k, v in ds.splits.items())})"
            f" · version {ds.version_id}" + ("" if ds.complete else f" · PARTIAL ({stop})"))
    rate = f" · {m.get('raw', 0) / secs:,.0f} raw rec/s" if secs else ""
    lines = [head, f"  raw {m.get('raw', 0):,} · {fmt_seconds(secs)}{rate}"]
    if not ds.complete:
        if stop == "max_time":
            lines.append("  max_time reached: the partial dataset is valid. Resume with more "
                         "time: spec.with_(max_time=...).run()")
        elif stop == "limit":
            lines.append("  limit reached: run again to continue (finished chunks are kept).")
        elif stop == "raw_limit":
            lines.append("  raw limit reached before the target: acceptance is too low. "
                         "Check ds.rejected() and loosen filters, or raise max_raw.")
    rej = m.get("rejected_by") or {}
    if rej:
        top = sorted(rej.items(), key=lambda kv: -kv[1])[:4]
        lines.append("  rejected: " + ", ".join(f"{k} {v:,}" for k, v in top))
    return "\n".join(lines)


def report(ds: Any, *, html: str | Path | None = None, reference: Any = None,
           open_browser: bool = False) -> Any:
    """Console report (and a self-contained HTML file when ``html`` is given)."""
    from topologies.synth.quality.report import full_report

    return full_report(ds, html=html, reference=reference, open_browser=open_browser)


__all__ = ["coverage_table", "format_records", "funnel_table", "live_line", "progress_line",
           "report", "run_summary", "spark"]
