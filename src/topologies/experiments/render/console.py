"""Console renderers (stdlib only): tables, heatmaps, bars, braille line charts and
the final summary, built on the shared terminal toolkit
(:mod:`topologies.visualization.terminal`).

ANSI colors are used when the stream supports them (shared capability probe:
``NO_COLOR``, ``TOPOLOGIES_COLOR``, ``FORCE_COLOR``, TTY); otherwise output is plain
text and heatmap cells are shaded with ``░▒▓█``. Every public renderer returns a list
of lines and never raises: a rendering problem degrades the display, never a run.
"""

from __future__ import annotations

import sys
import textwrap
from collections.abc import Sequence
from typing import Any

from topologies.visualization.terminal.text import Line, as_line, line, styled, to_ansi, to_plain

SHADES = "░▒▓█"


def use_color(stream: Any = None) -> bool:
    """True when ``stream`` (default stdout) gets ANSI colors (shared capability probe)."""
    try:
        from topologies.visualization.terminal.caps import probe

        return probe(sys.stdout if stream is None else stream).color > 0
    except Exception:  # noqa: BLE001
        return False


def _depth() -> int:
    try:
        from topologies.visualization.terminal.caps import probe

        return max(16, probe(sys.stdout).color)
    except Exception:  # noqa: BLE001
        return 16


def term_width(default: int = 100) -> int:
    try:
        from topologies.visualization.terminal.caps import terminal_size

        return max(40, min(200, terminal_size(sys.stdout, (default, 24))[0]))
    except Exception:  # noqa: BLE001
        return default


class Style:
    """Compatibility wrapper: ANSI styling through the shared theme tokens."""

    def __init__(self, color: bool) -> None:
        self.color = color

    def _tok(self, token: str, s: str) -> str:
        return to_ansi(styled(s, token), "default", _depth()) if self.color and s else s

    def bold(self, s: str) -> str:
        return self._tok("bold", s)

    def dim(self, s: str) -> str:
        return self._tok("muted", s)

    def green(self, s: str) -> str:
        return self._tok("good", s)

    def red(self, s: str) -> str:
        return self._tok("bad", s)

    def yellow(self, s: str) -> str:
        return self._tok("warn", s)

    def cyan(self, s: str) -> str:
        return self._tok("accent", s)

    def fg(self, code: int, s: str) -> str:
        return f"\x1b[{code}m{s}\x1b[0m" if self.color and s else s


def _encode(ln: Any, color: bool) -> str:
    return to_ansi(as_line(ln), "default", _depth()) if color else to_plain(as_line(ln))


def _encode_all(lines: Sequence[Any], color: bool) -> list[str]:
    return [_encode(x, color) for x in lines]


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def section_title(title: str, width: int) -> Line:
    """``── title ─────`` (short rule, ``title`` token)."""
    from topologies.visualization.terminal.widgets import rule

    return rule(min(width, max(len(title) + 8, 40)), title)


# ---------------------------------------------------------------- tables


def _shade(value: float | None, lo: float, hi: float, mode: str) -> str:
    if value is None or hi == lo:
        return SHADES[-1] if value is not None else " "
    good = (hi - value) / (hi - lo) if mode == "min" else (value - lo) / (hi - lo)
    return SHADES[max(0, min(3, round(good * 3)))]


def render_table(table: Any, *, width: int | None = None, color: bool | None = None,
                 indent: str = " ") -> list[str]:
    try:
        color = use_color() if color is None else color
        return _encode_all(table_lines(table, width=width, color=color, indent=indent), color)
    except Exception as e:  # noqa: BLE001
        return [f"<table rendering failed: {type(e).__name__}: {e}>"]


def table_lines(table: Any, *, width: int | None = None, color: bool = False,
                indent: str = " ", title: bool = True) -> list[Line]:
    """A results :class:`Table` as toolkit lines (title rule, header + rule, notes)."""
    from topologies.visualization.terminal.widgets import Column
    from topologies.visualization.terminal.widgets import table as tk_table

    width = width or term_width()
    cols = list(table.columns)
    cells: list[list[Any]] = [list(r) for r in table.cells()]
    meta = table.meta or {}
    heat = meta.get("heatmap")
    if heat:
        vals = [v for row in heat["values"] for v in row if v is not None]
        lo, hi = (min(vals), max(vals)) if vals else (0.0, 0.0)
        mode = heat.get("mode", "min")
        best = (lo if mode == "min" else hi) if vals else None
        r0, c0 = heat.get("row_offset", 0), heat.get("col_offset", 1)
        for i, row in enumerate(heat["values"]):
            for j, v in enumerate(row):
                ri, cj = i + r0, j + c0
                if ri >= len(cells) or cj >= len(cells[ri]) or v is None:
                    continue
                text = str(cells[ri][cj])
                if color:
                    cells[ri][cj] = styled(text, "best" if v == best else None)
                else:
                    cells[ri][cj] = f"{_shade(v, lo, hi, mode)} {text}"
    numeric = []
    for j in range(len(cols)):
        col_vals = [r[j] for r in table.rows if j < len(r) and r[j] is not None]
        numeric.append(bool(col_vals) and all(_is_num(v) for v in col_vals) and j > 0)
    align = dict(meta.get("align") or {})
    columns = [Column(c, align=align.get(c, ">" if numeric[j] else "<"), numeric=False,
                      priority=100 if j == 0 else (90 if numeric[j] else 50))
               for j, c in enumerate(cols)]
    if meta.get("separator_before_last_col") and len(columns) > 1:
        columns.insert(len(columns) - 1, Column("│", align="<", numeric=False, priority=100,
                                                min_width=1))
        cells = [[*r[:-1], ("│", "border"), r[-1]] if len(r) == len(cols) else r for r in cells]
    rows = cells
    out: list[Line] = []
    if title and meta.get("title"):
        out.append(section_title(str(meta["title"]), width))
    out.extend(tk_table(columns, rows, width=width, indent=len(indent), color=color,
                        hide_empty=True))
    for n in meta.get("notes") or []:
        token = "warn" if n.startswith(("⚠", "?")) else "muted"
        text = ("▲" + n[1:]) if n.startswith("⚠") else n
        out.append(line(indent, styled(text, token)))
    return out


# ---------------------------------------------------------------- bars


def bar(fraction: float, width: int) -> str:
    """Horizontal bar with eighth-block resolution."""
    eighths = " ▏▎▍▌▋▊▉█"
    f = max(0.0, min(1.0, float(fraction))) * width
    full = int(f)
    rest = round((f - full) * 8)
    if rest == 8:
        full, rest = full + 1, 0
    s = "█" * full + (eighths[rest] if rest and full < width else "")
    return s or "▏"


def render_bars(items: Sequence[tuple[str, float]], *, width: int = 60, inline: bool = True,
                color: bool | None = None, title: str = "importance") -> list[str]:
    """``[(label, share 0..1), ...]`` -> one aligned bar per row (``inline`` = one line)."""
    try:
        color = use_color() if color is None else color
        if not items:
            return []
        if inline:
            parts: list[Any] = [(title, "title"), "  "]
            for k, (lab, v) in enumerate(items):
                if k:
                    parts.append("   ")
                parts += [f"{lab} ", (bar(v, 16), "accent"), f" {v * 100:.0f}%"]
            return [_encode(line(*parts), color)]
        lw = max(len(lab) for lab, _ in items)
        bw = max(10, width - lw - 10)
        return [_encode(line(" ", lab.ljust(lw), " ", (bar(v, bw).ljust(bw), "accent"),
                             f" {v * 100:5.1f}%"), color) for lab, v in items]
    except Exception as e:  # noqa: BLE001
        return [f"<bars failed: {e}>"]


# ---------------------------------------------------------------- line chart


def line_chart(series: dict[str, Sequence[Sequence[float]]], width: int = 60, height: int = 12,
               *, color: bool = False, band: bool = True, legend: bool = True,
               x_label: str | None = None, y_label: str | None = None, log_y: bool = False,
               highlight: str | None = None) -> list[str]:
    """Multi-series braille line chart; points are ``(x, y)`` or ``(x, mean, std)``.

    Without color each series gets its own marker glyph; the ± std band is drawn only
    in color. Returns lines (y axis, x axis, legend). Never raises.
    """
    from topologies.visualization.terminal.widgets import line_chart as tk_line_chart

    try:
        if not isinstance(series, dict):
            raise TypeError("series must be a mapping")
        for value in series.values():
            iter(value)
        lines = tk_line_chart(series, width, height, color=color, band=band if color else None,
                              legend=legend, x_label=x_label, y_label=y_label, log_y=log_y,
                              highlight=highlight)
        return _encode_all(lines, color)
    except Exception as e:  # noqa: BLE001
        return [f"<chart failed: {type(e).__name__}: {e}>"]


def render_curves(curves: Any, *, width: int = 72, height: int = 14,
                  color: bool | None = None, log_y: bool = False) -> list[str]:
    from topologies.experiments.metrics import metric_info

    color = use_color() if color is None else color
    mode = metric_info(curves.metric).mode
    arrow = "↓ lower is better" if mode == "min" else "↑ higher is better"
    title = (f"curves  {curves.metric} by {curves.by or 'all'}  ·  {arrow}  ·  "
             f"mean ± std over seeds")
    series = {k: [(x, m, s) for x, m, s, _n in v] for k, v in curves.series.items()}
    chart = line_chart(series, width=width, height=height, color=color,
                       x_label=str(curves.x), y_label=curves.metric, log_y=log_y)
    return [_encode(section_title(title, width), color), *chart]


# ---------------------------------------------------------------- summary


def render_insights(insights: Sequence[Any], *, color: bool | None = None,
                    width: int | None = None) -> list[str]:
    color = use_color() if color is None else color
    width = width or term_width()
    out = [_encode(section_title("insights", width), color)] if insights else []
    for ins in insights:
        warn = getattr(ins, "severity", "info") == "warn"
        mark = line(("▲ warning", "warn"), " ") if warn else line("• ")
        body = textwrap.wrap(str(ins.text), width=max(20, width - mark.width - 2)) or [""]
        out.append(_encode(line(" ", mark, body[0]), color))
        pad = " " * (mark.width + 1)
        out.extend(_encode(line(pad, rest), color) for rest in body[1:])
    return out


def _marginal_lines(rs: Any, width: int, color: bool) -> list[str]:
    from topologies.visualization.terminal.widgets import table as tk_table

    lines: list[Any] = [section_title("marginal means", width)]
    for axis, t in rs.marginals().items():
        cells = t.cells()
        rows = [[str(row[0]), disp[1]] for row, disp in zip(t.rows, cells)]
        header = t.columns[1] if len(t.columns) > 1 else "mean"
        lines.extend(tk_table([axis, header], rows, width=width, color=color))
    return _encode_all(lines, color)


def render_summary(rs: Any, *, width: int | None = None, color: bool | None = None,
                   top: int = 15) -> list[str]:
    """Leaderboard, failures, pivot (2+ axes), marginal means, importance, insights, paths."""
    color = use_color() if color is None else color
    width = width or term_width()
    out: list[str] = []

    def section(fn: Any) -> None:
        try:
            lines = fn()
        except Exception as e:  # noqa: BLE001
            lines = [_encode(styled(f"<section failed: {type(e).__name__}: {e}>", "muted"),
                             color)]
        if lines:
            if out:
                out.append("")
            out.extend(lines)

    counts: dict[str, int] = {}
    for r in rs.records:
        counts[r.status] = counts.get(r.status, 0) + 1
    from topologies.visualization.terminal.glyphs import UNICODE as G

    status_bits = [f"{G.status(k)} {k} {v}" for k, v in sorted(counts.items())]
    head = line((rs.name or "experiment", "title"), (" · ", "muted"),
                f"{len(rs.records)} trials", (" · ", "muted"), "  ".join(status_bits))
    out.append(_encode(head, color))
    if any(r.status == "completed" for r in rs.records):
        section(lambda: render_table(rs.table(top=top), width=width, color=color))
    else:
        section(lambda: [_encode(styled(f"{G.failed} no completed trials", "bad"), color)])
    if any(r.status not in ("completed", "pruned") for r in rs.records):
        section(lambda: render_table(rs.failures(), width=width, color=color))
    varying = [a for a in rs.axes if rs.varies(a)]
    if len(varying) >= 2:
        def _pivot() -> list[str]:
            from topologies.experiments.analysis import pivot_axes

            row, col = pivot_axes(rs)  # type: ignore[misc]
            return render_table(rs.pivot(row, col), width=width, color=color)

        section(_pivot)
    if varying:
        section(lambda: _marginal_lines(rs, width, color))

        def _imp() -> list[str]:
            from topologies.experiments.analysis import importance_data

            d = importance_data(rs)
            items = sorted((d.get("shares") or {}).items(), key=lambda kv: -kv[1])
            if not items:
                return []
            title = "importance" + ("  (approximate)" if d.get("approximate") else "")
            return [_encode(section_title(title, width), color),
                    *render_bars(items, color=color, width=min(width, 72), inline=False)]

        section(_imp)
    section(lambda: render_insights(rs.insights(), color=color, width=width))
    exp_dir = rs.meta.get("dir")
    if exp_dir:
        section(lambda: _encode_all([styled(f" spec    {exp_dir}/experiment.json", "muted"),
                                     styled(f" trials  {exp_dir}/trials/", "muted"),
                                     styled(f" report  topologies exp report {rs.name}",
                                            "muted")], color))
    return out


def render_comparison(cmp: Any) -> list[str]:
    from topologies.experiments.metrics import metric_info

    out = [f"compare  A = {cmp.a_label}   B = {cmp.b_label}   ({cmp.metric})"]
    if cmp.config_diff:
        out.append(" config differences")
        for k, (a, b) in cmp.config_diff.items():
            out.append(f"   {k}: {a} → {b}")
    va, vb = cmp.values("a"), cmp.values("b")
    from topologies.experiments.stats import mean, std

    info = metric_info(cmp.metric)
    out.append(f" {cmp.metric}: A {info.format(mean(va))} ± {info.format(std(va))} (n={len(va)})"
               f"   B {info.format(mean(vb))} ± {info.format(std(vb))} (n={len(vb)})")
    w = cmp.welch
    d = cmp.cohens_d
    if w is None:
        out.append(" significance: n/a (needs ≥2 seeds per side)")
    else:
        verdict = "significant" if w[2] < 0.05 else "not significant"
        need = cmp.seeds_needed
        out.append(f" Welch t={w[0]:.2f}, df={w[1]:.1f}, p={w[2]:.3g} → {verdict}"
                   + (f"; Cohen's d={d:.2f}" if d is not None else "")
                   + (f"; ≈{need} seeds per config to resolve" if need and w[2] >= 0.05
                      else ""))
    if cmp.metrics:
        out.append(" metrics (mean)")
        for k, (a, b) in cmp.metrics.items():
            mi = metric_info(k)
            out.append(f"   {k:<16} {mi.format(a):>12}  {mi.format(b):>12}")
    return out


def render_trial(rec: Any, *, color: bool | None = None) -> list[str]:
    from topologies.experiments.metrics import metric_info

    color = use_color() if color is None else color
    sty = Style(color)
    out = [sty.bold(f"trial {rec.trial_id}  ·  {rec.describe()}"),
           f" status   {rec.status}" + (f" ({rec.status_reason})" if rec.status_reason else "")
           + (f"   stop: {rec.stop_reason}" if rec.stop_reason else "")]
    goal = rec.goal or {}
    if goal:
        out.append(f" goal     {goal.get('metric')} = "
                   f"{metric_info(goal.get('metric') or '').format(goal.get('value'))} "
                   f"({goal.get('select')}, {goal.get('mode')})")
    out.append(sty.bold(" config"))
    for k, v in sorted((rec.config or {}).items()):
        out.append(f"   {k} = {v}")
    if rec.final:
        out.append(sty.bold(" final"))
        for k, v in sorted(rec.final.items()):
            out.append(f"   {k:<18} {metric_info(k).format(v) if _is_num(v) else v}")
    if rec.cost:
        out.append(sty.bold(" cost"))
        for k, v in rec.cost.items():
            if v is not None:
                out.append(f"   {k:<18} {v:.4g}" if isinstance(v, float) else f"   {k:<18} {v}")
    if rec.build:
        out.append(f" build    {rec.build}")
    m = goal.get("metric")
    if m and rec.series(m):
        out.append(sty.bold(f" curve ({m})"))
        out.extend("   " + ln for ln in line_chart({m: rec.series(m)}, width=60, height=8,
                                                     color=color, legend=False))
    if rec.error:
        out.append(sty.red(f" error    {rec.error.get('type')}: {rec.error.get('message')}"))
        tb = (rec.error.get("traceback") or "").rstrip().splitlines()
        out.extend("   " + sty.dim(t) for t in tb[-25:])
    return out


__all__ = ["SHADES", "Style", "bar", "line_chart", "render_bars", "render_comparison",
           "render_curves", "render_insights", "render_plan", "render_summary", "render_table",
           "render_trial", "term_width", "use_color"]


# ---------------------------------------------------------------- plan


def render_plan(plan: Any, *, width: int | None = None, color: bool | None = None) -> list[str]:
    """Trial matrix, per-model parameters after matching, compatibility, estimate."""
    try:
        return _render_plan(plan, width=width, color=color)
    except Exception as e:  # noqa: BLE001
        return [f"<plan rendering failed: {type(e).__name__}: {e}>"]


def _render_plan(plan: Any, *, width: int | None, color: bool | None) -> list[str]:
    from topologies.experiments.results import Table, fmt_params

    color = use_color() if color is None else color
    sty = Style(color)
    configs = len({tuple(sorted((k, str(v)) for k, v in t.config_labels.items()))
                   for t in plan.trials})
    cached = sum(1 for s in plan.statuses.values() if s == "completed")
    eta = "?" if plan.estimate_s is None else f"≈{_eta(plan.estimate_s)}"
    out = [sty.bold(f"plan  {plan.name} · {len(plan.trials)} trials = {configs} configs × "
                    f"{len(plan.seeds)} seed{'s' if len(plan.seeds) != 1 else ''} · cached "
                    f"{cached} · to run {len(plan.trials) - cached} · ETA {eta}")]
    if plan.rows:
        keys = list(plan.rows[0].labels)
        cols = [*keys, "params", "target", "Δ", "knobs", "emb", "check"]
        rows, disp = [], []
        for r in plan.rows:
            check = "ok" if not r.error else "✗ " + r.error.split(" Hint:")[0]
            if r.note and not r.error:
                check = f"ok ({r.note})"
            knobs = ",".join(f"{k}={v}" for k, v in (r.knobs or {}).items())
            rel = None if r.rel_error is None or r.params is None or r.target is None else \
                (r.params - r.target) / r.target
            rows.append([*[r.labels.get(k) for k in keys], r.params, r.target, rel, knobs,
                         r.embedding_share, check])
            disp.append([*[str(r.labels.get(k, "—")) for k in keys], fmt_params(r.params),
                         fmt_params(r.target) if r.target else "—",
                         "—" if rel is None else f"{rel * 100:+.1f}%", knobs or "—",
                         "—" if r.embedding_share is None else f"{r.embedding_share * 100:.0f}%",
                         check])
        out.extend(render_table(Table(cols, rows, display=disp), width=width, color=color))
    if plan.errors:
        out.append(sty.red(f"errors ({len(plan.errors)} trials will be recorded as failed)"))
        grouped: dict[str, list[Any]] = {}
        by_id = {t.trial_id: t for t in plan.trials}
        for tid, msg in plan.errors.items():
            grouped.setdefault(msg, []).append(by_id[tid])
        for msg, ts in grouped.items():
            cfgs = sorted({" × ".join(map(str, t.config_labels.values())) for t in ts})
            seeds = sorted({t.seed for t in ts})
            out.append(f" {', '.join(cfgs[:3])}{' …' if len(cfgs) > 3 else ''} · seeds "
                       f"{','.join(map(str, seeds))}: {msg}")
    if plan.smoke is not None:
        ok = sum(1 for v in plan.smoke.values() if v == "ok")
        out.append(f"smoke  ✓ {ok}/{len(plan.smoke)} configs (2 steps each)")
    for n in plan.notes:
        out.append(sty.dim(n) if not n.startswith("embeddings") else sty.yellow("⚠ " + n))
    return out


def _eta(seconds: float) -> str:
    from topologies.visualization.format import fmt_eta

    return fmt_eta(seconds).lstrip("≈~")
