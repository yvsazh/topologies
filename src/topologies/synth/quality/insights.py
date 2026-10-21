"""Deterministic insight rules over a ``QualityReport``; every insight states its numbers.

Rules: repeated openings per label, C2ST separability with its giveaways,
acceptance collapse in one attribute value (generator/filter mismatch), label
predictable from a non-label attribute or from length (shortcuts), low
per-class diversity, near-duplicate clusters, high self-BLEU, weak evidence
mix, contamination with the reference, privacy flags, label issues and a
partial dataset. Curve rules (plateau, size for a target) live on ``Curve``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from topologies.experiments.insights import Insight
from topologies.synth.textutil import normalize, text_of


def _openings_by_label(rep: Any) -> list[Insight]:
    ds = rep.ds
    schema = ds.schema
    lf = schema.label_field() if schema is not None else None
    if lf is None or not schema.text_fields():
        return []
    tf = [f for f in schema.text_fields() if f != lf]
    by: dict[str, Counter] = defaultdict(Counter)
    totals: Counter = Counter()
    for r in ds.view():
        lab = str(r.get(lf))
        opening = " ".join(normalize(text_of(r, tf[0] if tf else None)).split()[:2])
        by[lab][opening] += 1
        totals[lab] += 1
    out = []
    for lab, c in by.items():
        if totals[lab] < 20:
            continue
        op, n = c.most_common(1)[0]
        share = n / totals[lab]
        if share >= 0.3 and op:
            out.append(Insight("openings", "warn",
                               f"{share:.0%} of `{lab}` samples start with '{op}'. Add an "
                               "`opening` attribute or a dedup on the first 5 tokens "
                               "(synth.ops.dedup.prefix(5)).",
                               {"label": lab, "opening": op, "share": share}))
    return out


def _c2st(rep: Any) -> list[Insight]:
    c = (rep.sections.get("fidelity") or {}).get("c2st") or {}
    auc = c.get("auc")
    if auc is None:
        return []
    if auc >= 0.8:
        tells = []
        g = c.get("giveaway_ngrams") or []
        if g:
            tells.append(", ".join(f"'{d['ngram']}'" for d in g[:2]))
        t = c.get("tells") or {}
        lr = t.get("length_ratio")
        if lr and (lr > 1.3 or lr < 0.77):
            tells.append(f"average length {lr:.1f}× real")
        dr = t.get("digit_ratio")
        if dr and (dr > 2 or dr < 0.5):
            tells.append(f"digits {dr:.1f}× real")
        return [Insight("c2st", "warn",
                        f"C2ST AUC {auc:.2f}: synthetic data is easy to tell apart." +
                        (f" The top giveaways are {'; '.join(tells)}." if tells else ""),
                        {"auc": auc, "tells": t})]
    if auc <= 0.6:
        return [Insight("c2st", "info", f"C2ST AUC {auc:.2f}: synthetic and real data are hard "
                                        "to tell apart.", {"auc": auc})]
    return []


def _acceptance(rep: Any) -> list[Insight]:
    cells = rep.sections.get("cells") or {}
    rej = rep.sections.get("rejected_by") or {}
    top_op = max(rej, key=rej.get) if rej else None
    out = []
    for attr, d in cells.items():
        g = sum(x for x, _ in d.values())
        a = sum(y for _, y in d.values())
        if g < 30:
            continue
        overall = a / g
        for val, (gv, av) in d.items():
            if gv < 15:
                continue
            rate = av / gv
            if rate < 0.5 * overall and overall - rate > 0.2:
                out.append(Insight("acceptance", "warn",
                                   f"{top_op or 'The pipeline'} rejects {1 - rate:.0%} of "
                                   f"`{attr}={val}` (vs {1 - overall:.0%} overall). Check the "
                                   "generator-filter mismatch: ds.rejected().filter("
                                   f"{attr}={val!r}).sample(5).",
                                   {"attribute": attr, "value": val, "acceptance": rate,
                                    "overall": overall}))
            if gv == 0:
                out.append(Insight("coverage", "warn", f"`{attr}={val}` was never generated.",
                                   {"attribute": attr, "value": val}))
    return out


def _shortcuts(rep: Any) -> list[Insight]:
    intr = rep.sections.get("intrinsic") or {}
    out = []
    for d in intr.get("leakage") or []:
        if d["lift"] >= 0.2 and d["acc"] >= 0.5:
            out.append(Insight("leakage", "warn",
                               f"Label is predictable from `{d['attribute']}` alone (acc "
                               f"{d['acc']:.2f} vs {d['baseline']:.2f} majority): a spurious "
                               "shortcut; decouple the attribute from the label.", d))
    ls = intr.get("length_shortcut") or {}
    if ls.get("lift", 0) >= 0.2 and ls.get("acc", 0) >= 0.5:
        out.append(Insight("leakage", "warn",
                           f"Label is predictable from `length` alone (acc {ls['acc']:.2f} vs "
                           f"{ls['baseline']:.2f}): spurious shortcut; vary length per label.",
                           ls))
    lab = intr.get("labels") or {}
    for k in lab.get("low_diversity_classes") or []:
        out.append(Insight("diversity", "warn",
                           f"Class `{k}` has low diversity (distinct-2 "
                           f"{lab['distinct2_by_class'][k]:.2f}, under half the median); "
                           "add attributes for it or raise temperature.", {"label": k}))
    if lab and lab.get("balance", 1) < 0.8:
        out.append(Insight("balance", "info", f"Label balance {lab['balance']:.2f} "
                                              "(entropy ratio); synth.ops.balance('label') "
                                              "equalizes classes.", {}))
    return out


def _repetition(rep: Any) -> list[Insight]:
    t = (rep.sections.get("intrinsic") or {}).get("text") or {}
    out = []
    nd = t.get("near_dups") or {}
    if nd.get("in_clusters", 0) > 0.1:
        out.append(Insight("near_dups", "warn",
                           f"{nd['in_clusters']:.0%} of records sit in near-duplicate clusters "
                           f"(largest {nd['largest']}); add synth.ops.dedup.near(0.85).", nd))
    sb = t.get("self_bleu")
    if sb is not None and sb > 0.5:
        out.append(Insight("self_bleu", "warn", f"Self-BLEU {sb:.2f}: records reuse phrasing; "
                                                "diversify attributes, templates or grammars.", {}))
    return out


def _evidence(rep: Any) -> list[Insight]:
    ev = (rep.sections.get("summary") or {}).get("evidence") or {}
    total = sum(ev.values())
    if not total:
        return []
    strong = (ev.get("construction", 0) + ev.get("verifier", 0)) / total
    weak = ev.get("none", 0) / total
    if weak > 0.5:
        return [Insight("evidence", "warn",
                        f"{weak:.0%} of records passed no check; derive labels from latents "
                        "(construction) or add a verifier.", {"strong": strong, "none": weak})]
    return [Insight("evidence", "info",
                    f"{strong:.0%} of records are correct by construction or verified.",
                    {"strong": strong})]


def _misc(rep: Any) -> list[Insight]:
    out = []
    cont = rep.sections.get("contamination") or {}
    if cont.get("overlapping"):
        out.append(Insight("contamination", "warn",
                           f"{cont['overlapping']} synthetic records overlap the reference; "
                           "add synth.ops.contamination.check(reference, action='drop').",
                           {"n": cont["overlapping"]}))
    priv = rep.sections.get("privacy") or {}
    if priv.get("flagged_count"):
        out.append(Insight("privacy", "warn",
                           f"{priv['flagged_count']} records are suspiciously close to private "
                           f"real records ({priv.get('memorized', 0)} share 50+ characters).", {}))
    li = rep.sections.get("label_issues") or {}
    if li.get("share", 0) > 0.05:
        out.append(Insight("label_issues", "warn",
                           f"{li['share']:.1%} of records look mislabeled (confident learning); "
                           "ds.label_issues().drop() or relabel them.", {"count": li["count"]}))
    if not rep.ds.complete:
        out.append(Insight("partial", "warn",
                           f"This version is partial (stop: {rep.ds.manifest.get('stop_reason')})"
                           "; resume the run to complete it.", {}))
    return out


RULES = (_c2st, _openings_by_label, _acceptance, _shortcuts, _repetition, _evidence, _misc)


def generate(rep: Any) -> list[Insight]:
    out: list[Insight] = []
    for rule in RULES:
        try:
            out += rule(rep)
        except Exception:  # noqa: BLE001, S112 - a broken rule never breaks the report
            continue
    return sorted(out, key=lambda i: i.severity != "warn")


__all__ = ["RULES", "generate"]
