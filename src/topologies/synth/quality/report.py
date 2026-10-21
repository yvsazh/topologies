"""``ds.quality()`` / ``ds.report()``: every diagnostic as Tables + deterministic insights.

Sections: summary (size, splits, generation time, evidence mix, TSTR when known), funnel,
attribute coverage, intrinsic text / label / tabular statistics, fidelity vs a
real reference (C2ST AUC with giveaways, MMD / Fréchet / JS, marginals),
contamination, privacy (memorized reference records), and samples per label plus rejected
samples per op. ``report(html="r.html")`` writes one self-contained HTML file
(no CDN; the experiments report's CSS / JS). Rendering failures degrade a
section, never the report.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from topologies.experiments.insights import Insight
from topologies.experiments.table import Table
from topologies.synth.records import as_records


class QualityReport:
    def __init__(self, ds: Any, sections: dict[str, Any], insights: list[Insight]) -> None:
        self.ds, self.sections, self.insights = ds, sections, insights

    def __getitem__(self, key: str) -> Any:
        return self.sections[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.sections.get(key, default)

    # -- tables --------------------------------------------------------------------
    def tables(self) -> list[Table]:
        out = []
        for fn in (self._summary_table, self._text_table, self._label_table,
                   self._fidelity_table, self._giveaway_table, self._tabular_table,
                   self._privacy_table):
            try:
                t = fn()
            except Exception as e:  # noqa: BLE001 - a section never breaks the report
                t = Table(["error"], [[f"{type(e).__name__}: {e}"]],
                          meta={"title": fn.__name__})
            if t is not None:
                out.append(t)
        return out

    def _summary_table(self) -> Table:
        s = self.sections["summary"]
        rows = [[k, v] for k, v in s.items() if not isinstance(v, (dict, list))]
        ev = s.get("evidence") or {}
        total = sum(ev.values()) or 1
        rows += [[f"evidence: {k}", f"{v / total:.0%}"] for k, v in sorted(
            ev.items(), key=lambda kv: -kv[1])]
        return Table(["", "value"], rows, meta={"title": f"summary  {self.ds.name} "
                                                         f"({self.ds.version_id})"})

    def _text_table(self) -> Table | None:
        t = (self.sections.get("intrinsic") or {}).get("text")
        if not t:
            return None
        keys = ["chars_mean", "chars_p5", "chars_p95", "words_mean", "distinct1", "distinct2",
                "distinct3", "ttr", "compression", "self_bleu"]
        rows = [[k, t[k]] for k in keys if k in t]
        emb = (self.sections.get("intrinsic") or {}).get("embedding") or {}
        if "dispersion" in emb:
            rows += [["embedding dispersion", emb["dispersion"]],
                     ["effective clusters", emb["clusters"]]]
        nd = t.get("near_dups") or {}
        rows.append(["near-duplicate share", nd.get("in_clusters")])
        rows += [[f"opening: {o!r}", f"{share:.1%}"] for o, _, share in t["openings"][:5]]
        return Table(["text", "value"], rows, meta={"title": "diversity"})

    def _label_table(self) -> Table | None:
        lab = (self.sections.get("intrinsic") or {}).get("labels")
        if not lab:
            return None
        total = sum(lab["counts"].values()) or 1
        rows = [[k, v, v / total, lab["distinct2_by_class"].get(k)]
                for k, v in sorted(lab["counts"].items(), key=lambda kv: -kv[1])]
        t = Table(["label", "count", "share", "distinct-2"], rows,
                  meta={"title": f"labels (balance {lab['balance']:.2f})"})
        return t

    def _fidelity_table(self) -> Table | None:
        f = self.sections.get("fidelity")
        if not f:
            return None
        rows = []
        c = f.get("c2st") or {}
        if c.get("auc") is not None:
            lo, hi = c["ci"]
            rows.append(["C2ST AUC (0.5 = indistinguishable)",
                         f"{c['auc']:.3f} [{lo:.3f}, {hi:.3f}]"])
        for k in ("mmd", "frechet", "js_length", "js_vocab"):
            if (f.get("text") or {}).get(k) is not None:
                rows.append([k, f["text"][k]])
        for m in f.get("marginals") or []:
            rows.append([f"{m['field']} ({m['kind']})", m.get("ks", m.get("tv"))])
        return Table(["metric", "value"], rows, meta={"title": "fidelity vs real reference"})

    def _giveaway_table(self) -> Table | None:
        c = (self.sections.get("fidelity") or {}).get("c2st") or {}
        g = c.get("giveaway_ngrams")
        if not g:
            return None
        rows = [[d["ngram"], d["z"], d["synthetic"], d["real"]] for d in g]
        return Table(["n-gram", "z", "synthetic", "real"], rows,
                     meta={"title": "what gives synthetic data away"})

    def _tabular_table(self) -> Table | None:
        tab = (self.sections.get("intrinsic") or {}).get("tabular")
        if not tab:
            return None
        rows = [[k, "numeric", v["mean"], v["std"], v["missing"]] for k, v in
                tab.get("numeric", {}).items()]
        rows += [[k, "categorical", json.dumps(dict(list(v["freq"].items())[:3])), None,
                  v["missing"]] for k, v in tab.get("categorical", {}).items()]
        return Table(["column", "kind", "mean / top", "std", "missing"], rows,
                     meta={"title": "columns"})

    def _privacy_table(self) -> Table | None:
        p = self.sections.get("privacy")
        if not p or "nearest_similarity" not in p:
            return None
        rows = [["nearest seed similarity (mean)", p["nearest_similarity"].get("mean")],
                ["baseline (holdout real, mean)", (p["baseline_similarity"] or {}).get("mean")],
                ["closer than baseline p99", p["closer_than_baseline_p99"]],
                ["memorized (>= 50 shared chars)", p["memorized"]],
                ["flagged", p["flagged_count"]]]
        return Table(["privacy", "value"], rows, meta={"title": "privacy vs private real records"})

    # -- display ---------------------------------------------------------------------
    def __str__(self) -> str:
        parts = [str(t) for t in self.tables()]
        if self.insights:
            parts.append("insights\n" + "\n".join(f" {i}" for i in self.insights))
        return "\n\n".join(parts)

    def __repr__(self) -> str:
        return str(self)

    def show(self) -> None:
        print(self)

    def to_json(self) -> dict[str, Any]:
        from topologies.experiments.store import jsonable

        return jsonable({"sections": self.sections,
                         "insights": [i.to_json() for i in self.insights]})

    def to_html(self) -> str:
        return render_html(self)


# ---------------------------------------------------------------- building


def quality_report(ds: Any, *, reference: Any = None, embed: Any = None, sample: int = 2000,
                   seed: int = 0, private: Any = None, label_issues: bool = False,
                   eval_result: Any = None) -> QualityReport:
    from topologies.synth.quality import fidelity as fid
    from topologies.synth.quality.intrinsic import intrinsic

    m = ds.manifest
    recs = ds.to_records()
    schema = ds.schema
    secs = (m.get("time") or {}).get("seconds")
    summary: dict[str, Any] = {
        "records": len(ds), "splits": ", ".join(f"{k} {v}" for k, v in ds.splits.items()),
        "complete": ds.complete, "raw generated": m.get("raw"),
        "acceptance": (m.get("total", 0) / m["raw"]) if m.get("raw") else None,
        "generation seconds": secs,
        "raw records/s": (m["raw"] / secs) if secs and m.get("raw") else None,
        "generator": m.get("generator"), "evidence": m.get("evidence") or {}}
    if eval_result is not None and getattr(eval_result, "tstr", None):
        summary["TSTR"] = eval_result.tstr.get("mean")
    sections: dict[str, Any] = {"summary": summary}
    sections["intrinsic"] = intrinsic(recs, schema=schema, embed=embed, sample=sample,
                                      seed=seed)
    sections["cells"] = m.get("cells") or {}
    sections["rejected_by"] = m.get("rejected_by") or {}
    if reference is not None:
        ref = as_records(reference, what="reference")
        is_text = bool(sections["intrinsic"].get("text"))
        lf = schema.label_field() if schema is not None else None
        key = [f for f in schema.text_fields() if f != lf] if (schema is not None and
                                                               is_text) else None
        if is_text:
            sections["fidelity"] = {"c2st": fid.c2st(recs, ref, key=key or None, seed=seed,
                                                     embed=embed, schema=schema),
                                    "text": fid.text_fidelity(recs, ref, key=key or None,
                                                              embed=embed, seed=seed),
                                    "marginals": fid.marginals(recs, ref)}
        else:
            tf = fid.tabular_fidelity(recs, ref, seed=seed)
            sections["fidelity"] = {"c2st": fid.c2st(recs, ref, schema=schema, seed=seed),
                                    "type_features": fid.type_fidelity(recs, ref, schema, seed=seed),
                                    "marginals": tf["columns"],
                                    "correlation_diff": tf["correlation_diff"]}
        from topologies.synth.ops.contamination import overlap

        found = overlap(recs, ref)
        sections["contamination"] = {"overlapping": len(found), "examples": found[:10]}
    seeds = private
    if seeds is not None:
        from topologies.synth.quality.privacy import privacy

        sections["privacy"] = privacy(recs, as_records(seeds), seed=seed)
    if label_issues and schema is not None and schema.label_field():
        try:
            li = ds.label_issues()
            sections["label_issues"] = {"count": len(li), "share": len(li) / max(1, len(ds)),
                                        "top": li.rows[:10]}
        except Exception as e:  # noqa: BLE001
            sections["label_issues"] = {"error": str(e)}
    from topologies.synth.quality.insights import generate

    rep = QualityReport(ds, sections, [])
    rep.insights = generate(rep)
    return rep


def full_report(ds: Any, *, html: str | Path | None = None, reference: Any = None,
                open_browser: bool = False, **kw: Any) -> QualityReport:
    rep = quality_report(ds, reference=reference, **kw)
    try:
        print(rep)
    except Exception:  # noqa: BLE001, S110 - display never breaks
        pass
    if html is not None:
        p = Path(html)
        p.write_text(render_html(rep), encoding="utf-8")
        try:
            from topologies.experiments.store import write_json_atomic

            write_json_atomic(Path(ds.dir) / "quality" / "report.json", rep.to_json())
            (Path(ds.dir) / "quality" / "report.html").write_text(p.read_text(encoding="utf-8"),
                                                                 encoding="utf-8")
        except OSError:
            pass
        if open_browser:
            import webbrowser

            webbrowser.open(p.resolve().as_uri())
    return rep


# ---------------------------------------------------------------- HTML


def _samples_html(ds: Any, n: int = 3) -> str:
    from topologies.experiments.render.html import esc

    schema = ds.schema
    lf = schema.label_field() if schema is not None else None
    view = ds.sample(n, by=lf) if lf else ds.sample(n * 3)
    rows = []
    for r in view:
        shown = {k: v for k, v in r.items() if not k.startswith("_") and k != lf}
        rows.append(f"<tr><td>{esc(r.get(lf, ''))}</td><td style=\"white-space:normal\">"
                    f"<pre>{esc(json.dumps(shown, ensure_ascii=False)[:600])}</pre></td>"
                    f"<td class=\"muted\" style=\"white-space:normal\">"
                    f"{esc(', '.join(f'{k}={v}' for k, v in (r.get('_attrs') or {}).items()))[:300]}"
                    "</td></tr>")
    return ("<div class=\"scroll\"><table><thead><tr><th>label</th><th>record</th>"
            "<th>attributes</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


def _rejected_html(ds: Any, n: int = 3) -> str:
    from topologies.experiments.render.html import esc

    by: dict[str, list[dict]] = {}
    for r in ds.rejected():
        lst = by.setdefault(str(r.get("_op")), [])
        if len(lst) < n:
            lst.append(r)
    if not by:
        return "<p class=\"muted\">nothing was rejected</p>"
    rows = []
    for op, lst in by.items():
        for r in lst:
            shown = {k: v for k, v in r.items() if not k.startswith("_")}
            rows.append(f"<tr><td>{esc(op)}</td><td style=\"white-space:normal\">"
                        f"{esc((r.get('_verdict') or {}).get('reason', ''))}</td>"
                        f"<td style=\"white-space:normal\"><pre>"
                        f"{esc(json.dumps(shown, ensure_ascii=False)[:400])}</pre></td></tr>")
    return ("<div class=\"scroll\"><table><thead><tr><th>op</th><th>reason</th>"
            "<th>record</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


def _coverage_html(ds: Any) -> str:
    from topologies.experiments.render.html import svg_hbars, table_html

    parts = []
    for attr, cells in (ds.manifest.get("cells") or {}).items():
        items = [(v, a / g if g else 0.0, None, None) for v, (g, a) in
                 sorted(cells.items(), key=lambda kv: -kv[1][0])[:20]]
        parts.append(f"<h3>{attr}: acceptance per value</h3>" +
                     svg_hbars(items, fmt=lambda x: f"{x:.0%}"))
    parts.append(table_html(ds.funnel()))
    return "".join(parts)


def render_html(rep: QualityReport) -> str:
    from topologies.experiments.render.html import CSS, JS, esc, table_html

    ds = rep.ds
    s = rep.sections["summary"]
    cards = "".join(f"<div class=\"card\"><span class=\"muted\">{esc(k)}</span><b>"
                    f"{esc(_fmt(v))}</b></div>" for k, v in s.items()
                    if not isinstance(v, (dict, list)) and v is not None)
    ins = "".join(f"<li class=\"{'warn' if i.severity == 'warn' else ''}\">{esc(i.text)}</li>"
                  for i in rep.insights) or "<li class=\"muted\">no findings</li>"

    def safe(fn: Any) -> str:
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            return f"<p class=\"bad\">section failed: {esc(type(e).__name__)}: {esc(e)}</p>"

    tables = "".join(f"<h2>{esc(t.meta.get('title', ''))}</h2>" + safe(lambda t=t: table_html(t))
                     for t in rep.tables())
    giveaways = ""
    c = (rep.sections.get("fidelity") or {}).get("c2st") or {}
    if c.get("most_synthetic"):
        items = "".join(f"<li><code>{esc(_short(r))}</code> <span class=\"muted\">"
                        f"p={r.get('_p_synthetic')}</span></li>" for r in c["most_synthetic"][:6])
        giveaways = f"<h2>most synthetic-looking records</h2><ul>{items}</ul>"
    body = (f"<h1>{esc(ds.name)}</h1><p class=\"muted\">version {esc(ds.version_id)} · "
            f"synthetic data quality report</p><div class=\"cards\">{cards}</div>"
            f"<h2>insights</h2><ul class=\"ins\">{ins}</ul>"
            f"<h2>funnel and coverage</h2>{safe(lambda: _coverage_html(ds))}{tables}{giveaways}"
            f"<h2>samples per label</h2>{safe(lambda: _samples_html(ds))}"
            f"<h2>rejected samples per op</h2>{safe(lambda: _rejected_html(ds))}")
    return ("<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>{esc(ds.name)} · synth report</title><style>{CSS}</style></head><body>"
            f"<main>{body}</main><script>{JS}</script></body></html>\n")


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _short(r: dict, n: int = 160) -> str:
    s = json.dumps({k: v for k, v in r.items() if not k.startswith("_")}, ensure_ascii=False)
    return s if len(s) <= n else s[: n - 1] + "…"


__all__ = ["QualityReport", "full_report", "quality_report", "render_html"]

_ = Counter
