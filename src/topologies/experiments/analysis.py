"""Analysis views over a ``ResultSet``; every view returns a ``Table`` (or a small
value object with ``__str__``) so the console, Markdown, HTML and ``--json``
renderers share one implementation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from topologies.experiments import stats as st
from topologies.experiments.results import (
    ConfigSummary,
    ResultSet,
    Table,
    TrialRecord,
    fmt_params,
    fmt_rate,
    fmt_seconds,
)

FAILED = ("failed", "diverged", "oom", "timeout", "interrupted")


def _fmt(rs: ResultSet, metric: str, v: float | None) -> str:
    return rs.info(metric).format(v)


def _sparkline(values: Sequence[float], width: int = 8) -> str:
    try:
        from topologies.training.monitoring.dashboard.charts import sparkline

        return sparkline(list(values), width=width)
    except Exception:  # noqa: BLE001
        return ""


def _order(values: Sequence[Any]) -> list[str]:
    """Distinct labels in first-seen order (numeric labels sorted by value)."""
    seen: dict[str, None] = {}
    for v in values:
        seen.setdefault(str(v), None)
    labels = list(seen)
    try:
        nums = [float(x) for x in labels]
    except ValueError:
        return labels
    return [lab for _, lab in sorted(zip(nums, labels))]


def _check_axis(rs: ResultSet, axis: str) -> None:
    if axis not in rs.axes:
        from topologies.experiments.errors import ExperimentError

        raise ExperimentError(f"Unknown axis {axis!r}.", stage="results", available=rs.axes)


def pivot_axes(rs: ResultSet) -> tuple[str, str] | None:
    """The two varying axes with the most values, in declaration order (rows, cols)."""
    varying = [a for a in rs.axes if rs.varies(a)]
    if len(varying) < 2:
        return None
    size = {a: len({str(r.config_labels.get(a)) for r in rs.records}) for a in varying}
    top = sorted(varying, key=lambda a: -size[a])[:2]
    row, col = sorted(top, key=varying.index)
    return row, col


def tied_with_best(best: ConfigSummary, other: ConfigSummary) -> bool | None:
    """Welch p >= 0.05 between ``other`` and the best config (None: n < 2)."""
    res = st.welch_ttest(best.values, other.values)
    if res is None:
        return None
    return res[2] >= 0.05


# ---------------------------------------------------------------- leaderboard


def leaderboard(rs: ResultSet, *, metric: str | None = None, top: int | None = None,
                group_seeds: bool = True, columns: Sequence[str] | None = None,
                include_failed: bool = False, ci: bool = False) -> Table:
    metric = metric or rs.goal[0]
    header = rs.metric_header(metric)
    if not group_seeds:
        return _trial_table(rs, metric, header, top=top, include_failed=include_failed)
    sums = rs.summaries(metric, include_failed=include_failed)
    ranked = [s for s in sums if s.mean is not None]
    best = ranked[0] if ranked else None
    base = rs.baseline_summary(metric)
    ref = next((x for x in sums if base is not None and x.labels == base.labels), None) or best
    use_tokens = any((r.cost or {}).get("tokens_per_s") for r in rs.records)
    rate_key, rate_col = ("tokens_per_s", "tok/s") if use_tokens else ("samples_per_s",
                                                                        "samples/s")
    show_status = any(set(s.statuses) - {"completed"} for s in sums)
    show_synth = any((r.cost or {}).get("synth_seconds") is not None for r in rs.records)
    cols = ["#", *rs.axes, header, "n", "Δ", "params", "time", rate_col]
    if show_synth:
        cols.append("gen s")
    if show_status:
        cols.append("status")
    cols.append("curve")
    rows, disp, payload = [], [], []
    rank = 0
    for s in sums:
        if top is not None and rank >= top:
            break
        mean, sd = s.mean, s.std
        if mean is None:
            rank_s, val_s = "—", "✗ " + ", ".join(sorted(set(s.statuses)))
        else:
            rank += 1
            rank_s = str(rank)
            val_s = _fmt(rs, metric, mean) + (f" ± {_fmt(rs, metric, sd)}" if sd is not None
                                              else "")
            if ci and s.ci is not None:
                val_s += f" [{_fmt(rs, metric, s.ci[0])}, {_fmt(rs, metric, s.ci[1])}]"
            if best is not None and s is not best and tied_with_best(best, s):
                val_s += " ≈"
        delta = None
        if ref is not None and ref.mean not in (None, 0) and mean is not None and s is not ref:
            delta = (mean - ref.mean) / abs(ref.mean) * 100.0
        if s is ref:
            delta_s = "base" if base is not None else "—"
        else:
            from topologies.visualization.format import fmt_delta

            delta_s = "—" if delta is None else fmt_delta(mean, ref.mean)
        stat_s = " ".join(f"{_STATUS_ICON.get(k, '?')} {k} {v}"
                          for k, v in sorted(s.statuses.items()))
        curve = s.curve(metric) or s.curve()
        row = [rank if mean is not None else None, *[s.labels.get(a, "—") for a in rs.axes],
               mean, s.n, delta, s.cost("params"), s.cost("wall_s"), s.cost(rate_key)]
        d = [rank_s, *[str(s.labels.get(a, "—")) for a in rs.axes], val_s, str(s.n), delta_s,
             fmt_params(s.cost("params")), fmt_seconds(s.cost("wall_s")),
             fmt_rate(s.cost(rate_key))]
        if show_synth:
            gen_s = s.cost("synth_seconds")
            row.append(gen_s)
            d.append("—" if gen_s is None else fmt_seconds(gen_s))
        if show_status:
            row.append(stat_s)
            d.append(stat_s)
        row.append(_sparkline(curve))
        d.append(row[-1])
        rows.append(row)
        disp.append(d)
        payload.append({"rank": row[0], "labels": s.labels, "mean": mean, "std": sd,
                        "ci": list(s.ci) if s.ci else None, "n": s.n, "delta_pct": delta,
                        "params": s.cost("params"), "wall_s": s.cost("wall_s"),
                        rate_key: s.cost(rate_key), "statuses": dict(s.statuses),
                        "synth_seconds": s.cost("synth_seconds"),
                        "curve": curve, "trials": [t.trial_id for t in s.trials]})
    notes = []
    if any("≈" in d[len(rs.axes) + 1] for d in disp):
        notes.append("≈ statistically indistinguishable from #1 (Welch p ≥ 0.05)")
    if rs.not_comparable(metric):
        notes.append(f"⚠ {metric} is per-token and the tokenizer varies: values are not "
                     "comparable across tokenizers (use val_bpb)")
    if rs.info(metric).inferred:
        notes.append(f"? direction of {metric} inferred from its name; set goal=(metric, mode)")
    table = Table(cols, rows, display=disp, meta={
        "title": f"leaderboard  {header} (mean over seeds)", "notes": notes,
        "json": {"metric": metric, "mode": rs.mode(metric), "rows": payload}})
    if columns:
        keep = [i for i, c in enumerate(cols) if c in columns or c == "#" or c in rs.axes]
        table = Table([cols[i] for i in keep], [[r[i] for i in keep] for r in rows],
                      display=[[d[i] for i in keep] for d in disp], meta=table.meta)
    return table


_STATUS_ICON = {"completed": "✓", "failed": "✗", "diverged": "↯", "oom": "▲", "timeout": "◔",
                "pruned": "✂", "interrupted": "‖", "queued": "·", "running": "▸"}


def _trial_table(rs: ResultSet, metric: str, header: str, *, top: int | None,
                 include_failed: bool) -> Table:
    recs = [r for r in rs.records if include_failed or r.status == "completed"]
    sign = 1.0 if rs.mode(metric) == "min" else -1.0
    recs.sort(key=lambda r: (r.metric(metric) is None, sign * (r.metric(metric) or 0.0)))
    if top is not None:
        recs = recs[:top]
    cols = ["#", *rs.axes, "seed", header, "status", "params", "time", "trial"]
    rows, disp = [], []
    for i, r in enumerate(recs, 1):
        v = r.metric(metric)
        rows.append([i, *[r.config_labels.get(a, "—") for a in rs.axes], r.seed, v, r.status,
                     (r.cost or {}).get("params"), (r.cost or {}).get("wall_s"), r.trial_id])
        disp.append([str(i), *[str(r.config_labels.get(a, "—")) for a in rs.axes],
                     str(r.seed), _fmt(rs, metric, v), r.status,
                     fmt_params((r.cost or {}).get("params")),
                     fmt_seconds((r.cost or {}).get("wall_s")), str(r.trial_id)])
    return Table(cols, rows, display=disp, meta={"title": f"trials  {header}"})


# ---------------------------------------------------------------- pivot


def _agg(values: list[float], how: str) -> float | None:
    if not values:
        return None
    if how == "mean":
        return sum(values) / len(values)
    if how == "min":
        return min(values)
    if how == "max":
        return max(values)
    if how == "median":
        s = sorted(values)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    raise ValueError(f"agg must be mean|min|max|median; got {how!r}")


def pivot(rs: ResultSet, row: str, col: str, *, metric: str | None = None,
          agg: str = "mean") -> Table:
    _check_axis(rs, row)
    _check_axis(rs, col)
    metric = metric or rs.goal[0]
    mode = rs.mode(metric)
    better = (lambda a, b: a < b) if mode == "min" else (lambda a, b: a > b)
    rlabels = _order(r.config_labels.get(row) for r in rs.records)
    clabels = _order(r.config_labels.get(col) for r in rs.records)
    vals: dict[tuple[str, str], list[float]] = defaultdict(list)
    present: set[tuple[str, str]] = set()
    for r in rs.records:
        k = (str(r.config_labels.get(row)), str(r.config_labels.get(col)))
        present.add(k)
        if r.status == "completed":
            v = r.metric(metric)
            if v is not None:
                vals[k].append(v)
    grid = [[_agg(vals.get((rl, cl), []), agg) for cl in clabels] for rl in rlabels]
    failed = [[(rl, cl) in present and not vals.get((rl, cl)) for cl in clabels]
              for rl in rlabels]
    row_means = [st.mean([v for v in g if v is not None]) for g in grid]
    col_means = [st.mean([grid[i][j] for i in range(len(rlabels)) if grid[i][j] is not None])
                 for j in range(len(clabels))]
    star_cells = set()
    for j in range(len(clabels)):
        best_i = None
        for i in range(len(rlabels)):
            v = grid[i][j]
            if v is not None and (best_i is None or better(v, grid[best_i][j])):
                best_i = i
        if best_i is not None and sum(grid[i][j] is not None for i in range(len(rlabels))) > 1:
            star_cells.add((best_i, j))
    best_col = None
    for j, v in enumerate(col_means):
        if v is not None and (best_col is None or better(v, col_means[best_col])):
            best_col = j
    best_row = None
    for i, v in enumerate(row_means):
        if v is not None and (best_row is None or better(v, row_means[best_row])):
            best_row = i
    cols = [f"{row} \\ {col}", *clabels, "mean"]
    rows_raw, disp = [], []
    any_partial = False
    for i, rl in enumerate(rlabels):
        cells = []
        for j in range(len(clabels)):
            v = grid[i][j]
            if v is None:
                cells.append("✗" if failed[i][j] else "—")
            else:
                cells.append(_fmt(rs, metric, v) + ("★" if (i, j) in star_cells else ""))
        partial = any(failed[i])
        any_partial = any_partial or partial
        m = row_means[i]
        m_s = "—" if m is None else _fmt(rs, metric, m) + ("*" if partial else "") + (
            "★" if i == best_row and len(rlabels) > 1 else "")
        rows_raw.append([rl, *grid[i], m])
        disp.append([rl, *cells, m_s])
    rows_raw.append(["mean", *col_means, None])
    disp.append(["mean", *[("—" if v is None else _fmt(rs, metric, v) + (
        "★" if j == best_col and len(clabels) > 1 else "")) for j, v in enumerate(col_means)],
        ""])
    others = [a for a in rs.axes if a not in (row, col) and rs.varies(a)]
    title = (f"pivot  {rs.metric_header(metric)}   rows: {row} · cols: {col} · {agg} over seeds"
             + (f" · averaged over {', '.join(others)}" if others else ""))
    notes = ["* excludes failed cells"] if any_partial else []
    return Table(cols, rows_raw, display=disp, meta={
        "title": title, "notes": notes, "separator_before_last_col": True,
        "heatmap": {"values": grid, "mode": mode, "row_offset": 0, "col_offset": 1},
        "json": {"metric": metric, "row": row, "col": col, "agg": agg, "rows": rlabels,
                 "cols": clabels, "values": grid, "row_means": row_means,
                 "col_means": col_means, "failed": failed}})


# ---------------------------------------------------------------- marginals


def _contexts(sums: list[ConfigSummary], axis: str) -> dict[tuple, dict[str, ConfigSummary]]:
    ctx: dict[tuple, dict[str, ConfigSummary]] = defaultdict(dict)
    for s in sums:
        key = tuple(sorted((k, str(v)) for k, v in s.labels.items() if k != axis))
        ctx[key][str(s.labels.get(axis))] = s
    return ctx


def marginals(rs: ResultSet, axis: str | None = None, *,
              metric: str | None = None) -> dict[str, Table]:
    metric = metric or rs.goal[0]
    mode = rs.mode(metric)
    sums = [s for s in rs.summaries(metric) if s.mean is not None]
    axes = [axis] if axis else [a for a in rs.axes if rs.varies(a)]
    out: dict[str, Table] = {}
    for a in axes:
        _check_axis(rs, a)
        labels = _order(s.labels.get(a) for s in sums)
        by_value: dict[str, list[float]] = defaultdict(list)
        for s in sums:
            by_value[str(s.labels.get(a))].append(s.mean)  # type: ignore[arg-type]
        wins: Counter = Counter()
        n_ctx = 0
        for group in _contexts(sums, a).values():
            if len(group) < 2:
                continue
            n_ctx += 1
            w = min(group.items(), key=lambda kv: kv[1].mean if mode == "min"
                    else -kv[1].mean)  # type: ignore[operator]
            wins[w[0]] += 1
        means = {v: st.mean(by_value[v]) for v in labels}
        order = sorted(labels, key=lambda v: (means[v] is None,
                                              (means[v] or 0.0) * (1 if mode == "min" else -1)))
        rank = {v: i + 1 for i, v in enumerate(order)}
        cols = [a, rs.metric_header(metric), "spread", "configs", "wins", "rank"]
        rows, disp = [], []
        for v in order:
            sd = st.std(by_value[v])
            rows.append([v, means[v], sd, len(by_value[v]), wins.get(v, 0), rank[v]])
            disp.append([v, _fmt(rs, metric, means[v]), "—" if sd is None else
                         f"±{_fmt(rs, metric, sd)}", str(len(by_value[v])),
                         f"{wins.get(v, 0)}/{n_ctx}" if n_ctx else "—", str(rank[v])])
        out[a] = Table(cols, rows, display=disp, meta={
            "title": f"marginal  {a}  ({metric}, mean over other axes)",
            "json": {"axis": a, "metric": metric, "values": [
                {"value": v, "mean": means[v], "std": st.std(by_value[v]),
                 "configs": len(by_value[v]), "wins": wins.get(v, 0), "contexts": n_ctx,
                 "rank": rank[v]} for v in order]}})
    return out


# ---------------------------------------------------------------- importance


def _is_numeric(labels: Sequence[str]) -> bool:
    try:
        [float(x) for x in labels]
        return True
    except (TypeError, ValueError):
        return False


def importance_data(rs: ResultSet, metric: str | None = None) -> dict[str, Any]:
    metric = metric or rs.goal[0]
    recs = [r for r in rs.records if r.status == "completed" and r.metric(metric) is not None]
    axes = [a for a in rs.axes if len({str(r.config_labels.get(a)) for r in recs}) > 1]
    rows = [{a: str(r.config_labels.get(a)) for a in axes} for r in recs]
    binned = []
    for a in axes:
        labs = [row[a] for row in rows]
        if len(set(labs)) > 5 and _is_numeric(labs):
            bins = st.quantile_bins([float(x) for x in labs], 5)
            for row, b in zip(rows, bins):
                row[a] = f"bin{b}"
            binned.append(a)
    for row, r in zip(rows, recs):
        row["__config"] = r.config_key
    res = st.importance(rows, [r.metric(metric) for r in recs], axes,
                        config_key=lambda row: row["__config"])
    spec = rs.meta.get("spec") or {}
    res["approximate"] = (not res.get("balanced", True)) or bool(binned) or \
        spec.get("space") == "random"
    res["binned"] = binned
    res["n"] = len(recs)
    res["metric"] = metric
    return res


def importance_table(rs: ResultSet, *, metric: str | None = None) -> Table:
    res = importance_data(rs, metric)
    shares = res.get("shares") or {}
    items = sorted(shares.items(), key=lambda kv: -kv[1])
    rows = [[k, v * 100.0, res["axes"].get(k, res["interaction"] if k == "interaction"
                                           else res["seed"])] for k, v in items]
    disp = [[k, f"{v:.1f}%", f"{ss:.4g}"] for k, v, ss in rows]  # v is already a percentage
    notes = []
    if res.get("approximate"):
        notes.append("approximate: unbalanced or random design"
                     + (f"; binned {', '.join(res['binned'])}" if res["binned"] else ""))
    if res.get("n", 0) < 2:
        notes.append("needs at least two completed trials")
    return Table(["factor", "share", "SS"], rows, display=disp, meta={
        "title": f"importance  ({res.get('metric')} variance share)", "notes": notes,
        "bars": True, "json": {k: v for k, v in res.items() if k != "axes"} | {
            "axes": res.get("axes")}})


# ---------------------------------------------------------------- consistency


def consistency_data(rs: ResultSet, axis: str, metric: str | None = None) -> dict[str, Any]:
    _check_axis(rs, axis)
    metric = metric or rs.goal[0]
    mode = rs.mode(metric)
    sums = [s for s in rs.summaries(metric) if s.mean is not None]
    values = _order(s.labels.get(axis) for s in sums)
    contexts = {k: g for k, g in _contexts(sums, axis).items()
                if len(g) == len(values) and len(values) >= 2}
    rank_rows, winners = [], {}
    for key, group in contexts.items():
        means = [group[v].mean for v in values]
        rank_rows.append(st.ranks(means, higher_is_better=(mode == "max")))  # type: ignore[arg-type]
        winners[key] = values[min(range(len(values)), key=lambda i: means[i]
                                  if mode == "min" else -means[i])]  # type: ignore[operator]
    w = st.kendall_w(rank_rows)
    overall = None
    marg = {v: st.mean([s.mean for s in sums if str(s.labels.get(axis)) == v]) for v in values}
    cands = [v for v in values if marg[v] is not None]
    if cands:
        overall = min(cands, key=lambda v: marg[v] if mode == "min" else -marg[v])
    flips = [dict(k) for k, win in winners.items() if win != overall]
    top_share = (sum(1 for win in winners.values() if win == overall) / len(winners)
                 if winners else None)
    cause = None
    if flips:
        others = sorted({k for key in winners for k, _ in key})
        for b in others:
            by_b: dict[str, set] = defaultdict(set)
            for key, win in winners.items():
                by_b[dict(key)[b]].add(win)
            if all(len(s) == 1 for s in by_b.values()) and len(
                    {next(iter(s)) for s in by_b.values()}) > 1:
                cause = b
                break
    winner_by: dict[str, str] = {}
    if cause is not None:
        for key, win in winners.items():
            winner_by[dict(key)[cause]] = win
    return {"axis": axis, "metric": metric, "values": values, "contexts": len(contexts),
            "kendall_w": w, "top": overall, "top_share": top_share,
            "winners": [{"context": dict(k), "winner": v} for k, v in winners.items()],
            "flips": flips, "cause": cause, "winner_by": winner_by}


def consistency(rs: ResultSet, axis: str, *, metric: str | None = None) -> Table:
    d = consistency_data(rs, axis, metric)
    rows = [[", ".join(f"{k}={v}" for k, v in w["context"].items()) or "(all)", w["winner"],
             "" if w["winner"] == d["top"] else "flip"] for w in d["winners"]]
    w_s = "n/a" if d["kendall_w"] is None else f"{d['kendall_w']:.2f}"
    share = "n/a" if d["top_share"] is None else f"{d['top_share'] * 100:.0f}%"
    notes = [(f"overall best {axis}: {d['top']} (wins {share} of {d['contexts']} contexts); "
              f"Kendall W {w_s}")]
    if d["cause"]:
        notes.append(f"rank reversals follow {d['cause']}: " + ", ".join(
            f"{v} wins with {d['cause']}={k}" for k, v in d["winner_by"].items()))
    return Table(["context", "winner", ""], rows, meta={
        "title": f"consistency  {axis}  ({d['metric']})", "notes": notes, "json": d})


# ---------------------------------------------------------------- compare


@dataclass
class Comparison:
    metric: str
    mode: str
    a: list[TrialRecord]
    b: list[TrialRecord]
    a_label: str
    b_label: str
    config_diff: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    metrics: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)

    def values(self, side: str) -> list[float]:
        recs = self.a if side == "a" else self.b
        return [v for v in (r.metric(self.metric) for r in recs if r.status == "completed")
                if v is not None]

    @property
    def welch(self) -> tuple[float, float, float] | None:
        return st.welch_ttest(self.values("a"), self.values("b"))

    @property
    def cohens_d(self) -> float | None:
        return st.cohens_d(self.values("a"), self.values("b"))

    @property
    def diff(self) -> float | None:
        ma, mb = st.mean(self.values("a")), st.mean(self.values("b"))
        return None if ma is None or mb is None else mb - ma

    @property
    def seeds_needed(self) -> int | None:
        return st.seeds_needed(st.pooled_std([self.values("a"), self.values("b")]), self.diff)

    def to_json(self) -> dict[str, Any]:
        w = self.welch
        return {"metric": self.metric, "mode": self.mode, "a": self.a_label, "b": self.b_label,
                "a_values": self.values("a"), "b_values": self.values("b"),
                "diff": self.diff, "welch": None if w is None else
                {"t": w[0], "df": w[1], "p": w[2]}, "cohens_d": self.cohens_d,
                "seeds_needed": self.seeds_needed,
                "config_diff": {k: list(v) for k, v in self.config_diff.items()},
                "metrics": {k: list(v) for k, v in self.metrics.items()}}

    def __str__(self) -> str:
        from topologies.experiments.render.console import render_comparison

        return "\n".join(render_comparison(self))


def compare(rs: ResultSet, a: Any, b: Any, *, metric: str | None = None) -> Comparison:
    from topologies.experiments.errors import ExperimentError

    metric = metric or rs.goal[0]
    ra, rb = rs.find(a), rs.find(b)
    for sel, recs in ((a, ra), (b, rb)):
        if not recs:
            raise ExperimentError(f"No trials match {sel!r}.", stage="results",
                                  hint="Use a trial id prefix or 'axis=label,axis=label'.")
    ca, cb = ra[0].config or {}, rb[0].config or {}
    diff = {k: (ca.get(k), cb.get(k)) for k in sorted(set(ca) | set(cb))
            if ca.get(k) != cb.get(k)}
    keys = sorted({k for r in ra + rb for k, v in (r.final or {}).items()
                   if isinstance(v, (int, float))})
    keys += ["wall_s", "params", "tokens_per_s", "samples_per_s", "peak_mem_mb"]
    mets = {}
    for k in keys:
        va = st.mean([r.metric(k) for r in ra if r.status == "completed"])
        vb = st.mean([r.metric(k) for r in rb if r.status == "completed"])
        if va is not None or vb is not None:
            mets[k] = (va, vb)
    return Comparison(metric=metric, mode=rs.mode(metric), a=ra, b=rb,
                      a_label=_sel_label(a, ra), b_label=_sel_label(b, rb),
                      config_diff=diff, metrics=mets)


def _sel_label(sel: Any, recs: list[TrialRecord]) -> str:
    if isinstance(sel, dict):
        return ",".join(f"{k}={v}" for k, v in sel.items())
    if isinstance(sel, str) and "=" not in sel and len(recs) == 1:
        return recs[0].describe()
    return str(sel)


# ---------------------------------------------------------------- pareto / curves


def pareto(rs: ResultSet, *, x: str = "wall_s", y: str | None = None) -> list[ConfigSummary]:
    y = y or rs.goal[0]
    sums = [s for s in rs.summaries(y) if s.mean is not None]
    pts = []
    for s in sums:
        xv = s.cost(x)
        if xv is None:
            xv = st.mean([t.metric(x) for t in s.completed])
        pts.append((xv, s.mean))
    idx = st.pareto_front(pts, y_mode=rs.mode(y))
    return [sums[i] for i in idx]


@dataclass
class Curves:
    metric: str
    by: str | None
    x: str
    series: dict[str, list[tuple[float, float, float | None, int]]]

    def to_json(self) -> dict[str, Any]:
        return {"metric": self.metric, "by": self.by, "x": self.x,
                "series": {k: [{"x": a, "mean": m, "std": s, "n": n} for a, m, s, n in v]
                           for k, v in self.series.items()}}

    def lines(self, width: int = 72, height: int = 14, color: bool | None = None) -> list[str]:
        from topologies.experiments.render.console import render_curves

        return render_curves(self, width=width, height=height, color=color)

    def __str__(self) -> str:
        return "\n".join(self.lines())


def curves(rs: ResultSet, *, metric: str = "val_loss", by: str | None = None,
           x: str = "epoch") -> Curves:
    if by is None:
        varying = [a for a in rs.axes if rs.varies(a)]
        by = varying[0] if varying else None
    if by is not None:
        _check_axis(rs, by)
    groups: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rs.records:
        if r.status not in ("completed", "pruned", "timeout"):
            continue
        key = str(r.config_labels.get(by)) if by else "all"
        hist = r.history or []
        spe = None
        if x == "step":
            steps = (r.cost or {}).get("steps")
            spe = (steps / len(hist)) if steps and hist else None
        for e, v in r.series(metric):
            xv = float(e + 1) if x == "epoch" else ((e + 1) * spe if spe else float(e + 1))
            groups[key][xv].append(v)
    series = {}
    for k in _order(groups):
        pts = []
        for xv in sorted(groups[k]):
            vals = groups[k][xv]
            pts.append((xv, sum(vals) / len(vals), st.std(vals), len(vals)))
        series[k] = pts
    return Curves(metric=metric, by=by, x=x, series=series)


# ---------------------------------------------------------------- failures


def failures(rs: ResultSet) -> Table:
    from topologies.experiments.trial import short_error

    bad = [r for r in rs.records if r.status in FAILED]
    groups: dict[tuple[str, str, str], list[TrialRecord]] = defaultdict(list)
    for r in bad:
        etype = (r.error or {}).get("type") or r.status
        for a in rs.axes or ["(all)"]:
            groups[(a, str(r.config_labels.get(a, "—")), f"{r.status}/{etype}")].append(r)
    rows, disp = [], []
    for (a, v, kind), recs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        total = sum(1 for r in rs.records if str(r.config_labels.get(a, "—")) == v)
        msg = Counter(short_error(r.error) or (r.status_reason or "") for r in recs)
        common = msg.most_common(1)[0][0] if msg else ""
        rows.append([a, v, kind, len(recs), total, common])
        disp.append([a, v, kind, str(len(recs)), str(total), common[:90]])
    pruned = sum(1 for r in rs.records if r.status == "pruned")
    notes = [f"{pruned} pruned trial(s) are excluded from ranking"] if pruned else []
    return Table(["axis", "value", "status/error", "failed", "of", "most common message"],
                 rows, display=disp, meta={"title": f"failures  ({len(bad)} trials)",
                                           "notes": notes})


__all__ = ["Comparison", "Curves", "compare", "consistency", "consistency_data", "curves",
           "failures", "importance_data", "importance_table", "leaderboard", "marginals",
           "pareto", "pivot", "tied_with_best"]
