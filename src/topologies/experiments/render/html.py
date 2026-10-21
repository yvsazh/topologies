"""Self-contained HTML report: inline CSS + SVG generated here, a few lines of
vanilla JS (sortable tables, series toggles, metric selector), no CDN, no
external fonts, light/dark via ``prefers-color-scheme``.
"""

from __future__ import annotations

import html
import json
import math
from datetime import datetime, timezone
from typing import Any

from topologies.visualization.theme import SERIES_LIGHT, css_variables

# Shared categorical palette (theme.py): slot i is the same entity in the terminal,
# Matplotlib and this report. SVG marks use ``var(--sN)`` so dark mode gets dark steps.
PALETTE = SERIES_LIGHT


def series_color(i: int) -> str:
    """CSS color of categorical slot ``i`` (follows light/dark via custom properties)."""
    return f"var(--s{i % len(PALETTE) + 1})"


CSS = (":root{" + css_variables("light") + """--bg:var(--surface);--fg:var(--ink);
--muted:var(--ink2);--line:var(--grid);--card:rgba(127,127,127,.07);--accent:var(--s1);
--good:12,163,12;--warn:#a86f00;--bad:var(--critical)}
@media (prefers-color-scheme:dark){:root{""" + css_variables("dark") + """--warn:var(--warning)}}
""" + """*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}h1{font-size:24px;margin:0 0 4px}
h2{font-size:17px;margin:32px 0 10px;padding-bottom:4px;border-bottom:1px solid var(--line)}
.muted{color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(auto-fit,
minmax(200px,1fr));gap:12px;margin:16px 0}.card{background:var(--card);border:1px solid
var(--line);border-radius:8px;padding:12px}.card b{display:block;font-size:18px}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:
tabular-nums}th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:left;
white-space:nowrap}th{font-weight:600;color:var(--muted)}table.sortable th{cursor:pointer}
td.num{text-align:right}tr:hover td{background:var(--card)}.warn{color:var(--warn)}
.bad{color:var(--bad)}ul.ins{padding-left:18px}ul.ins li{margin:4px 0}svg{max-width:100%;
height:auto}svg text{fill:var(--muted);font-size:11px}svg .grid{stroke:var(--line)}
svg .off,.legend .off{opacity:.12}.legend span{cursor:pointer;margin-right:14px;
white-space:nowrap}.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;
margin-right:4px}figure{margin:8px 0 20px}pre{white-space:pre-wrap;margin:0;font-size:12px}
details summary{cursor:pointer;color:var(--accent)}select{background:var(--card);
color:var(--fg);border:1px solid var(--line);border-radius:4px;padding:2px 6px}
.bar{height:10px;background:var(--accent);border-radius:2px;display:inline-block}
""")

JS = """
document.querySelectorAll('table.sortable th').forEach(function(th,i){
  th.addEventListener('click',function(){
    var tb=th.closest('table').tBodies[0],rows=Array.from(tb.rows),asc=th.dataset.asc!=='1';
    th.dataset.asc=asc?'1':'0';
    rows.sort(function(a,b){
      var x=a.cells[i],y=b.cells[i];
      x=x?(x.dataset.v!==undefined?x.dataset.v:x.textContent):'';
      y=y?(y.dataset.v!==undefined?y.dataset.v:y.textContent):'';
      var nx=parseFloat(x),ny=parseFloat(y);
      var r=(!isNaN(nx)&&!isNaN(ny))?nx-ny:String(x).localeCompare(String(y));
      return asc?r:-r;});
    rows.forEach(function(r){tb.appendChild(r);});});});
document.querySelectorAll('.legend [data-s]').forEach(function(el){
  el.addEventListener('click',function(){
    var fig=el.closest('figure');el.classList.toggle('off');
    fig.querySelectorAll('svg [data-s="'+el.dataset.s+'"]').forEach(function(p){
      p.classList.toggle('off');});});});
var sel=document.getElementById('curve-metric');
if(sel){sel.addEventListener('change',function(){
  document.querySelectorAll('figure.curve').forEach(function(f){
    f.hidden=f.dataset.m!==sel.value;});});}
"""


def esc(v: Any) -> str:
    return html.escape("" if v is None else str(v), quote=True)


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def table_html(table: Any, *, sortable: bool = True, heat: bool = True) -> str:
    """One ``Table`` as an HTML table (raw values in ``data-v`` for sorting)."""
    meta = table.meta or {}
    cells = table.cells()
    hm = meta.get("heatmap") if heat else None
    colors: dict[tuple[int, int], str] = {}
    if hm:
        vals = [v for row in hm["values"] for v in row if v is not None]
        if vals:
            lo, hi = min(vals), max(vals)
            for i, row in enumerate(hm["values"]):
                for j, v in enumerate(row):
                    if v is None or hi == lo:
                        continue
                    g = (hi - v) / (hi - lo) if hm.get("mode", "min") == "min" else \
                        (v - lo) / (hi - lo)
                    colors[(i + hm.get("row_offset", 0), j + hm.get("col_offset", 1))] = \
                        f"background:rgba(var(--good),{0.08 + 0.5 * g:.2f})"
    out = []
    if meta.get("title"):
        out.append(f"<p class=\"muted\">{esc(meta['title'])}</p>")
    out.append(f"<div class=\"scroll\"><table class=\"{'sortable' if sortable else ''}\">"
               "<thead><tr>" + "".join(f"<th>{esc(c)}</th>" for c in table.columns)
               + "</tr></thead><tbody>")
    for i, (raw, disp) in enumerate(zip(table.rows, cells)):
        tds = []
        for j, d in enumerate(disp):
            v = raw[j] if j < len(raw) else None
            attrs = f" data-v=\"{esc(v)}\"" if _num(v) else ""
            cls = " class=\"num\"" if _num(v) else ""
            style = f" style=\"{colors[(i, j)]}\"" if (i, j) in colors else ""
            tds.append(f"<td{cls}{attrs}{style}>{esc(d)}</td>")
        out.append("<tr>" + "".join(tds) + "</tr>")
    out.append("</tbody></table></div>")
    for n in meta.get("notes") or []:
        cls = "warn" if str(n).startswith("⚠") else "muted"
        out.append(f"<p class=\"{cls}\">{esc(n)}</p>")
    return "\n".join(out)


# ---------------------------------------------------------------- SVG charts


def _ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    if hi == lo:
        return [lo]
    return [lo + (hi - lo) * i / n for i in range(n + 1)]


def _fmt(v: float) -> str:
    a = abs(v)
    if a != 0 and (a < 1e-2 or a >= 1e5):
        return f"{v:.2g}"
    return f"{v:.3g}"


def svg_lines(series: dict[str, list[tuple[float, float, float | None]]], *,
              width: int = 680, height: int = 260, xlabel: str = "", ylabel: str = "") -> str:
    pts = [(x, y) for s in series.values() for x, y, _ in s if _num(x) and _num(y)]
    if not pts:
        return "<p class=\"muted\">no data</p>"
    ys = [y for _, y in pts] + [y + sg * (sd or 0) for s in series.values() for _x, y, sd in s
                                if sd for sg in (-1, 1)]
    x0, x1 = min(p[0] for p in pts), max(p[0] for p in pts)
    y0, y1 = min(ys), max(ys)
    x1, y1 = (x1 + 1 if x1 == x0 else x1), (y1 + 1e-9 if y1 == y0 else y1)
    L, R, T, B = 56, 12, 10, 34
    w, h = width - L - R, height - T - B

    def sx(x: float) -> float:
        return L + (x - x0) / (x1 - x0) * w

    def sy(y: float) -> float:
        return T + (y1 - y) / (y1 - y0) * h

    parts = [f"<svg viewBox=\"0 0 {width} {height}\" role=\"img\">"]
    for t in _ticks(y0, y1):
        parts.append(f"<line class=\"grid\" x1=\"{L}\" x2=\"{L + w}\" y1=\"{sy(t):.1f}\" "
                     f"y2=\"{sy(t):.1f}\"></line><text x=\"{L - 6}\" y=\"{sy(t) + 4:.1f}\" "
                     f"text-anchor=\"end\">{esc(_fmt(t))}</text>")
    for t in _ticks(x0, x1, 4):
        parts.append(f"<text x=\"{sx(t):.1f}\" y=\"{T + h + 16}\" text-anchor=\"middle\">"
                     f"{esc(_fmt(t))}</text>")
    if xlabel:
        parts.append(f"<text x=\"{L + w / 2:.0f}\" y=\"{height - 2}\" text-anchor=\"middle\">"
                     f"{esc(xlabel)}</text>")
    if ylabel:
        parts.append(f"<text x=\"12\" y=\"{T + h / 2:.0f}\" transform=\"rotate(-90 12 "
                     f"{T + h / 2:.0f})\" text-anchor=\"middle\">{esc(ylabel)}</text>")
    for i, (_name, s) in enumerate(series.items()):
        col = series_color(i)
        s = [(x, y, sd) for x, y, sd in s if _num(x) and _num(y)]
        band = [(x, y, sd) for x, y, sd in s if sd]
        if len(band) >= 2:
            up = " ".join(f"{sx(x):.1f},{sy(y + sd):.1f}" for x, y, sd in band)
            dn = " ".join(f"{sx(x):.1f},{sy(y - sd):.1f}" for x, y, sd in reversed(band))
            parts.append(f"<polygon data-s=\"{i}\" points=\"{up} {dn}\" style=\"fill:{col}\" "
                         "fill-opacity=\"0.12\" stroke=\"none\"></polygon>")
        line = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y, _ in s)
        parts.append(f"<polyline data-s=\"{i}\" points=\"{line}\" fill=\"none\" "
                     f"style=\"stroke:{col}\" stroke-width=\"2\"></polyline>")
        for x, y, _ in s:
            parts.append(f"<circle data-s=\"{i}\" cx=\"{sx(x):.1f}\" cy=\"{sy(y):.1f}\" "
                         f"r=\"2.5\" style=\"fill:{col}\"></circle>")
    parts.append("</svg>")
    legend = "".join(f"<span data-s=\"{i}\"><i style=\"background:{series_color(i)}"
                     f"\"></i>{esc(n)}</span>" for i, n in enumerate(series))
    return "".join(parts) + f"<div class=\"legend\">{legend}</div>"


def svg_scatter(points: list[tuple[float, float, str, bool]], *, width: int = 680,
                height: int = 280, xlabel: str = "", ylabel: str = "", logx: bool = False) -> str:
    pts = [p for p in points if _num(p[0]) and _num(p[1]) and (not logx or p[0] > 0)]
    if not pts:
        return "<p class=\"muted\">no data</p>"
    tx = (lambda v: math.log10(v)) if logx else (lambda v: v)
    xs, ys = [tx(p[0]) for p in pts], [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    x1, y1 = (x1 + 1 if x1 == x0 else x1), (y1 + 1e-9 if y1 == y0 else y1)
    L, R, T, B = 56, 12, 10, 34
    w, h = width - L - R, height - T - B

    def sx(x: float) -> float:
        return L + (tx(x) - x0) / (x1 - x0) * w

    def sy(y: float) -> float:
        return T + (y1 - y) / (y1 - y0) * h

    parts = [f"<svg viewBox=\"0 0 {width} {height}\" role=\"img\">"]
    for t in _ticks(y0, y1):
        parts.append(f"<line class=\"grid\" x1=\"{L}\" x2=\"{L + w}\" y1=\"{sy(t):.1f}\" "
                     f"y2=\"{sy(t):.1f}\"></line><text x=\"{L - 6}\" y=\"{sy(t) + 4:.1f}\" "
                     f"text-anchor=\"end\">{esc(_fmt(t))}</text>")
    for t in _ticks(x0, x1, 4):
        val = 10 ** t if logx else t
        xpos = L + (t - x0) / (x1 - x0) * w
        parts.append(f"<text x=\"{xpos:.1f}\" y=\"{T + h + 16}\" text-anchor=\"middle\">"
                     f"{esc(_fmt(val))}</text>")
    parts.append(f"<text x=\"{L + w / 2:.0f}\" y=\"{height - 2}\" text-anchor=\"middle\">"
                 f"{esc(xlabel)}{' (log)' if logx else ''}</text>")
    if ylabel:
        parts.append(f"<text x=\"12\" y=\"{T + h / 2:.0f}\" transform=\"rotate(-90 12 "
                     f"{T + h / 2:.0f})\" text-anchor=\"middle\">{esc(ylabel)}</text>")
    front = sorted([p for p in pts if p[3]], key=lambda p: p[0])
    if len(front) >= 2:
        parts.append("<polyline points=\"" + " ".join(f"{sx(p[0]):.1f},{sy(p[1]):.1f}"
                                                      for p in front)
                     + f"\" fill=\"none\" style=\"stroke:{series_color(1)}\" "
                     "stroke-dasharray=\"4 3\">"
                     "</polyline>")
    for x, y, label, on in pts:
        col = series_color(1) if on else series_color(0)
        parts.append(f"<circle cx=\"{sx(x):.1f}\" cy=\"{sy(y):.1f}\" r=\"{5 if on else 4}\" "
                     f"style=\"fill:{col}\" fill-opacity=\"0.85\"><title>{esc(label)}: "
                     f"{esc(_fmt(y))}</title></circle>")
    parts.append("</svg>")
    return "".join(parts)


def svg_hbars(items: list[tuple[str, float, float | None, float | None]], *,
              width: int = 680, fmt: Any = None) -> str:
    """Horizontal bars with optional whiskers ``(label, value, low, high)``."""
    items = [it for it in items if _num(it[1])]
    if not items:
        return "<p class=\"muted\">no data</p>"
    fmt = fmt or _fmt
    vals = [v for _, v, lo, hi in items for v in (v, lo, hi) if _num(v)]
    lo_all, hi_all = min(vals + [0.0]), max(vals)
    if hi_all == lo_all:
        hi_all = lo_all + 1
    L, R, row = 170, 70, 24
    w, height = width - L - R, row * len(items) + 8

    def sx(v: float) -> float:
        return L + (v - lo_all) / (hi_all - lo_all) * w

    parts = [f"<svg viewBox=\"0 0 {width} {height}\" role=\"img\">"]
    for i, (lab, v, lo, hi) in enumerate(items):
        y = 4 + i * row
        x0, x1 = sx(min(0.0, v) if lo_all <= 0 else lo_all), sx(v)
        parts.append(f"<text x=\"{L - 8}\" y=\"{y + 14}\" text-anchor=\"end\">{esc(lab)}</text>"
                     f"<rect x=\"{min(x0, x1):.1f}\" y=\"{y + 4}\" width=\""
                     f"{max(1.0, abs(x1 - x0)):.1f}\" height=\"12\" "
                     f"style=\"fill:{series_color(0)}\" "
                     "fill-opacity=\"0.8\"></rect>")
        if _num(lo) and _num(hi):
            parts.append(f"<line x1=\"{sx(lo):.1f}\" x2=\"{sx(hi):.1f}\" y1=\"{y + 10}\" "
                         f"y2=\"{y + 10}\" stroke=\"currentColor\"></line>"
                         f"<line x1=\"{sx(lo):.1f}\" x2=\"{sx(lo):.1f}\" y1=\"{y + 5}\" "
                         f"y2=\"{y + 15}\" stroke=\"currentColor\"></line>"
                         f"<line x1=\"{sx(hi):.1f}\" x2=\"{sx(hi):.1f}\" y1=\"{y + 5}\" "
                         f"y2=\"{y + 15}\" stroke=\"currentColor\"></line>")
        parts.append(f"<text x=\"{L + w + 6}\" y=\"{y + 14}\">{esc(fmt(v))}</text>")
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------- sections


def _section(title: str, body: str, anchor: str) -> str:
    return f"<section id=\"{anchor}\"><h2>{esc(title)}</h2>\n{body}\n</section>"


def _safe(fn: Any) -> str:
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return f"<p class=\"bad\">section failed: {esc(type(e).__name__)}: {esc(e)}</p>"


def _header(rs: Any) -> str:
    spec = rs.meta.get("spec") or {}
    task = spec.get("task") or {}
    desc = task.get("describe") or {}
    env = next((r.env for r in rs.records if r.env), {}) or {}
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rows = [("goal", f"{rs.goal[0]} ({'lower' if rs.goal[1] == 'min' else 'higher'} is better)"),
            ("task", " · ".join(f"{k}={v}" for k, v in desc.items()
                                if not isinstance(v, (dict, list)))[:300] or "—"),
            ("axes", ", ".join(f"{a} ({len(v)})" for a, v in (spec.get("axes") or {}).items())
             or ", ".join(rs.axes)),
            ("budget", json.dumps(spec.get("train_for")) if spec.get("train_for") else "—"),
            ("environment", " · ".join(f"{k} {v}" for k, v in env.items() if v) or "—"),
            ("generated", stamp)]
    body = "".join(f"<tr><th>{esc(k)}</th><td style=\"white-space:normal\">{esc(v)}</td></tr>"
                   for k, v in rows)
    d = f"<p>{esc(spec.get('description'))}</p>" if spec.get("description") else ""
    return (f"<h1>{esc(rs.name or 'Experiment')}</h1>{d}"
            f"<table style=\"width:auto\"><tbody>{body}</tbody></table>")


def _cards(rs: Any) -> str:
    from topologies.experiments.results import fmt_seconds

    counts: dict[str, int] = {}
    for r in rs.records:
        counts[r.status] = counts.get(r.status, 0) + 1
    best = rs.best(1)
    best_s = "—"
    if best:
        b = best[0]
        best_s = (f"{esc(b.label(rs.axes))}<br><span class=\"muted\">{esc(rs.goal[0])} "
                  f"{esc(rs.info().format(b.mean))} (n={b.n})</span>")
    total = sum(float((r.cost or {}).get("wall_s") or 0.0) for r in rs.records)
    status = " · ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    return ("<div class=\"cards\">"
            f"<div class=\"card\"><span class=\"muted\">best config</span><b>{best_s}</b></div>"
            f"<div class=\"card\"><span class=\"muted\">trials</span><b>{len(rs.records)}</b>"
            f"<span class=\"muted\">{esc(status)}</span></div>"
            f"<div class=\"card\"><span class=\"muted\">total compute</span>"
            f"<b>{esc(fmt_seconds(total))}</b><span class=\"muted\">sum of trial wall time"
            "</span></div></div>")


def _pivots(rs: Any) -> str:
    varying = [a for a in rs.axes if rs.varies(a)]
    if len(varying) < 2:
        return ""
    from topologies.experiments.analysis import pivot_axes

    pairs = [pivot_axes(rs)]
    if len(varying) == 3:
        pairs = [(varying[0], varying[1]), (varying[0], varying[2]), (varying[1], varying[2])]
    return "\n".join(table_html(rs.pivot(a, b), sortable=False) for a, b in pairs)


def _marginal_bars(rs: Any) -> str:
    from topologies.experiments import stats as st

    out = []
    sums = [s for s in rs.summaries() if s.mean is not None]
    info = rs.info()
    for a in [x for x in rs.axes if rs.varies(x)]:
        by: dict[str, list[float]] = {}
        for s in sums:
            by.setdefault(str(s.labels.get(a)), []).append(s.mean)
        items = []
        for v, vals in by.items():
            m = st.mean(vals)
            ci = st.ci95(vals)
            items.append((v, m, ci[0] if ci else None, ci[1] if ci else None))
        sign = 1 if rs.goal[1] == "min" else -1
        items.sort(key=lambda it: sign * it[1])
        out.append(f"<p class=\"muted\">{esc(a)} · mean of config means, 95 % CI whiskers</p>"
                   + svg_hbars(items, fmt=info.format))
    return "\n".join(out)


def _importance(rs: Any) -> str:
    from topologies.experiments.analysis import importance_data

    d = importance_data(rs)
    items = [(k, v * 100.0, None, None) for k, v in sorted((d.get("shares") or {}).items(),
                                                            key=lambda kv: -kv[1])]
    note = "<p class=\"muted\">approximate (unbalanced or random design)</p>" \
        if d.get("approximate") else ""
    return svg_hbars(items, fmt=lambda v: f"{v:.1f}%") + note


def _curves(rs: Any) -> str:
    metrics = []
    for m in (rs.goal[0], "val_loss", "train_loss", "val_acc"):
        if m not in metrics and any(r.series(m) for r in rs.records):
            metrics.append(m)
    if not metrics:
        return "<p class=\"muted\">no per-epoch history</p>"
    opts = "".join(f"<option value=\"{esc(m)}\">{esc(m)}</option>" for m in metrics)
    out = [(f"<p>metric <select id=\"curve-metric\">{opts}</select> "
            "<span class=\"muted\">click a legend entry to toggle a series</span></p>")]
    for i, m in enumerate(metrics):
        c = rs.curves(metric=m)
        series = {k: [(x, mean, sd) for x, mean, sd, _n in v] for k, v in c.series.items()}
        hidden = "" if i == 0 else " hidden"
        out.append(f"<figure class=\"curve\" data-m=\"{esc(m)}\"{hidden}>"
                   + svg_lines(series, xlabel=c.x, ylabel=m) + "</figure>")
    return "\n".join(out)


def _pareto(rs: Any) -> str:
    out = []
    sums = [s for s in rs.summaries() if s.mean is not None]
    for key, label in (("wall_s", "wall time (s)"), ("params", "parameters")):
        if not any(s.cost(key) for s in sums):
            continue
        front = {id(s) for s in rs.pareto(x=key)}
        pts = [(s.cost(key), s.mean, s.label(rs.axes), id(s) in front) for s in sums]
        out.append(f"<figure>{svg_scatter(pts, xlabel=label, ylabel=rs.goal[0], logx=True)}"
                   "</figure>")
    return "\n".join(out) or "<p class=\"muted\">no cost data</p>"


def _insights(rs: Any) -> str:
    ins = rs.insights()
    if not ins:
        return "<p class=\"muted\">no insights</p>"
    return "<ul class=\"ins\">" + "".join(
        f"<li class=\"{'warn' if i.severity == 'warn' else ''}\">{esc(i.text)}</li>"
        for i in ins) + "</ul>"


def _samples(rs: Any) -> str:
    rows = []
    seen = set()
    for r in rs.records:
        smp = (r.extras or {}).get("samples")
        if not smp or r.status != "completed" or r.config_key in seen:
            continue
        seen.add(r.config_key)
        for prompt, gens in smp.items():
            rows.append(f"<tr><td>{esc(' × '.join(map(str, r.config_labels.values())))}</td>"
                        f"<td style=\"white-space:normal\"><pre>{esc(gens.get('greedy'))}</pre>"
                        f"</td><td style=\"white-space:normal\"><pre>"
                        f"{esc(gens.get('sampled'))}</pre></td></tr>")
            del prompt
    if not rows:
        return ""
    return ("<div class=\"scroll\"><table><thead><tr><th>config</th><th>greedy</th>"
            "<th>sampled (T=0.8)</th></tr></thead><tbody>" + "".join(rows)
            + "</tbody></table></div>")


def _trials(rs: Any) -> str:
    from topologies.experiments.results import fmt_params, fmt_seconds

    head = ["trial", *rs.axes, "seed", "status", rs.goal[0], "params", "time", "config"]
    rows = []
    info = rs.info()
    for r in rs.records:
        v = r.value
        cfg = json.dumps(r.config, indent=1, sort_keys=True, default=str)
        err = ""
        if r.error:
            err = f"<pre class=\"bad\">{esc(r.error.get('type'))}: {esc(r.error.get('message'))}</pre>"
        rows.append(
            "<tr>" + f"<td>{esc(r.trial_id)}</td>"
            + "".join(f"<td>{esc(r.config_labels.get(a, '—'))}</td>" for a in rs.axes)
            + f"<td>{esc(r.seed)}</td><td>{esc(r.status)}</td>"
            + (f"<td class=\"num\" data-v=\"{v}\">{esc(info.format(v))}</td>" if v is not None
               else "<td>—</td>")
            + f"<td class=\"num\" data-v=\"{(r.cost or {}).get('params') or 0}\">"
              f"{esc(fmt_params((r.cost or {}).get('params')))}</td>"
            + f"<td class=\"num\" data-v=\"{(r.cost or {}).get('wall_s') or 0}\">"
              f"{esc(fmt_seconds((r.cost or {}).get('wall_s')))}</td>"
            + f"<td style=\"white-space:normal\"><details><summary>config</summary><pre>"
              f"{esc(cfg)}</pre>{err}</details></td></tr>")
    return ("<div class=\"scroll\"><table class=\"sortable\"><thead><tr>"
            + "".join(f"<th>{esc(h)}</th>" for h in head) + "</tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></div>")


def report_html(rs: Any) -> str:
    """The full report (sections 1-12) as one self-contained HTML document."""
    has_fail = any(r.status not in ("completed", "pruned") for r in rs.records)
    samples = _safe(lambda: _samples(rs))
    sections = [
        _safe(lambda: _header(rs)),
        _safe(lambda: _cards(rs)),
        _section("Leaderboard", _safe(lambda: table_html(rs.table())), "leaderboard"),
        _section("Pivot", _safe(lambda: _pivots(rs)) or
                 "<p class=\"muted\">needs two varying axes</p>", "pivot"),
        _section("Marginal effects", _safe(lambda: _marginal_bars(rs)) or
                 "<p class=\"muted\">no varying axes</p>", "marginals"),
        _section("Importance", _safe(lambda: _importance(rs)), "importance"),
        _section("Learning curves", _safe(lambda: _curves(rs)), "curves"),
        _section("Pareto", _safe(lambda: _pareto(rs)), "pareto"),
        _section("Insights", _safe(lambda: _insights(rs)), "insights"),
        _section("Failures", _safe(lambda: table_html(rs.failures())) if has_fail else
                 "<p class=\"muted\">no failed trials</p>", "failures"),
        _section("Samples", samples or "<p class=\"muted\">no generated samples "
                 "(LanguageModeling(sample_prompts=[...]))</p>", "samples"),
        _section("All trials", _safe(lambda: _trials(rs)), "trials"),
    ]
    title = esc(rs.name or "Experiment")
    return ("<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>{title} · experiment report</title><style>{CSS}</style></head>"
            f"<body><main>\n" + "\n".join(sections) + f"\n</main><script>{JS}</script>"
            "</body></html>\n")


def summary_html(rs: Any) -> str:
    """Compact rich display for notebooks (leaderboard + insights)."""
    return (f"<div><b>{esc(rs.name or 'experiment')}</b> · {len(rs.records)} trials"
            + table_html(rs.table(top=15), sortable=False) + _insights(rs) + "</div>")


__all__ = ["report_html", "summary_html", "svg_hbars", "svg_lines", "svg_scatter",
           "table_html"]
