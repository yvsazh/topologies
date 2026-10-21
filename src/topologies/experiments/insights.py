"""Insights: deterministic rules over a ``ResultSet``; every insight shows its numbers.

Rules: winner (gap, Welch p, seeds needed), importance (variance shares),
consistent winner (Kendall W), interaction / rank reversal, failure pattern,
efficiency (Pareto), under-training, overfitting, comparability (per-token
metrics across tokenizers, mixed versions, parameter matching, embedding
share) and noise (overlapping confidence intervals).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from topologies.experiments import stats as st


@dataclass
class Insight:
    kind: str
    severity: str  # "info" | "warn"
    text: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        from topologies.experiments.store import jsonable

        return jsonable({"kind": self.kind, "severity": self.severity, "text": self.text,
                         "evidence": self.evidence})

    def __str__(self) -> str:
        return ("⚠ " if self.severity == "warn" else "• ") + self.text


def _pct(x: float) -> str:
    return f"{x:+.1f} %"


def fmt_p(p: float) -> str:
    return "p<0.001" if p < 0.001 else f"p={p:.2g}"


def _plural(word: str, n: int) -> str:
    return word if n == 1 else (word + "es" if word.endswith(("s", "x")) else word + "s")


def _compress(labels: list[str], universe: list[str]) -> str:
    """``["bpe:1k", "bpe:4k"]`` -> ``"bpe:*"`` when that prefix covers exactly them."""
    if len(labels) > 1:
        prefixes = {lab.split(":", 1)[0] for lab in labels}
        if len(prefixes) == 1 and all(":" in lab for lab in labels):
            p = prefixes.pop()
            same = [u for u in universe if u.split(":", 1)[0] == p and ":" in u]
            if sorted(same) == sorted(labels):
                return f"{p}:*"
    return ", ".join(labels)


def winner(rs: Any) -> list[Insight]:
    sums = [s for s in rs.summaries() if s.mean is not None]
    if not sums:
        return []
    metric, info = rs.goal[0], rs.info()
    best = sums[0]
    sd = f" ±{info.format(best.std)}" if best.std is not None else ""
    text = (f"Best: {best.label(rs.axes)}, {metric} {info.format(best.mean)}{sd} "
            f"(n={best.n}).")
    ev: dict[str, Any] = {"best": best.labels, "mean": best.mean, "std": best.std, "n": best.n}
    if len(sums) > 1:
        ru = sums[1]
        gap = (ru.mean - best.mean) if best.mean is not None and ru.mean is not None else None
        rel = gap / abs(best.mean) * 100.0 if gap is not None and best.mean else None
        w = st.welch_ttest(best.values, ru.values)
        text += f" Runner-up {ru.label(rs.axes)} {info.format(ru.mean)}"
        if rel is not None:
            text += f": {_pct(rel)}"
        ev.update({"runner_up": ru.labels, "runner_up_mean": ru.mean, "gap": gap,
                   "gap_pct": rel})
        if w is None:
            text += ", significance n/a (needs ≥2 seeds per config)."
        else:
            ev["p"] = w[2]
            if w[2] < 0.05:
                text += f", {fmt_p(w[2])}, significant."
            else:
                need = st.seeds_needed(st.pooled_std([s.values for s in sums]), gap)
                ev["seeds_needed"] = need
                text += f", {fmt_p(w[2])}, not significant." + (
                    f" ≈{need} seeds per config needed to resolve." if need else "")
    if best.n < 2:
        text += " With 1 seed per config, significance cannot be assessed."
    return [Insight("winner", "info", text, ev)]


def importance(rs: Any) -> list[Insight]:
    from topologies.experiments.analysis import importance_data

    d = importance_data(rs)
    shares = d.get("shares") or {}
    axes = {k: v for k, v in shares.items() if k not in ("interaction", "seed noise")}
    if not axes:
        return []
    order = sorted(axes.items(), key=lambda kv: -kv[1])
    parts = [f"{k} {v * 100:.0f}%" + (" (negligible)" if v < 0.02 else "") for k, v in order]
    text = "Variance explained: " + ", ".join(parts)
    if "interaction" in shares:
        text += f"; interaction {shares['interaction'] * 100:.0f}%"
    if "seed noise" in shares:
        text += f"; seed noise {shares['seed noise'] * 100:.0f}%"
    text += "." + (" (approximate: unbalanced design)" if d.get("approximate") else "")
    return [Insight("importance", "info", text, {"shares": shares,
                                                 "approximate": d.get("approximate")})]


def consistency(rs: Any) -> list[Insight]:
    from topologies.experiments.analysis import consistency_data

    out = []
    varying = [a for a in rs.axes if rs.varies(a)]
    for a in varying:
        d = consistency_data(rs, a)
        if d["contexts"] < 2 or d["top"] is None:
            continue
        others = [b for b in varying if b != a]
        what = f"{_plural(others[0], d['contexts'])}" if len(others) == 1 else "contexts"
        wins = sum(1 for w in d["winners"] if w["winner"] == d["top"])
        w_s = "" if d["kendall_w"] is None else f" (Kendall W {d['kendall_w']:.2f})"
        if d["top_share"] is not None and d["top_share"] >= 0.8:
            out.append(Insight("consistent winner", "info",
                               f"{d['top']} is the best {a} for {wins}/{d['contexts']} "
                               f"{what}{w_s}.", d))
        if d["flips"] and d["cause"]:
            universe = sorted({str(k) for k in d["winner_by"]})
            groups: dict[str, list[str]] = defaultdict(list)
            for bval, win in d["winner_by"].items():
                groups[win].append(str(bval))
            parts = [f"{win} wins with {_compress(sorted(v), universe)}"
                     for win, v in groups.items()]
            out.append(Insight("interaction", "info",
                               f"{a.capitalize()} ranking depends on {d['cause']}: "
                               + ", ".join(parts) + ".", d))
        elif d["flips"]:
            out.append(Insight("interaction", "info",
                               f"Best {a} changes in {len(d['flips'])}/{d['contexts']} "
                               f"contexts (overall best {d['top']}).", d))
    return out


def failure_pattern(rs: Any) -> list[Insight]:
    from topologies.experiments.analysis import FAILED
    from topologies.experiments.trial import short_error

    out = []
    groups: dict[tuple, list[Any]] = defaultdict(list)
    for r in rs.records:
        for a in rs.axes:
            groups[((a, str(r.config_labels.get(a))),)].append(r)
        for i, a in enumerate(rs.axes):
            for b in rs.axes[i + 1:]:
                groups[((a, str(r.config_labels.get(a))), (b, str(r.config_labels.get(b))))
                       ].append(r)
    flagged: list[tuple] = []
    for key, recs in sorted(groups.items(), key=lambda kv: (len(kv[0]), str(kv[0]))):
        bad = [r for r in recs if r.status in FAILED]
        if len(recs) < 1 or not bad or len(bad) / len(recs) < 0.75:
            continue
        if any(set(f) <= set(key) for f in flagged):
            continue  # already explained by a smaller group
        if len(rs.axes) > 1 and len(key) == 1 and all(
                r.status in FAILED for r in rs.records):
            continue
        flagged.append(key)
        msg = Counter(short_error(r.error) or str(r.status_reason or r.status) for r in bad)
        common = msg.most_common(1)[0][0]
        label = " × ".join(v for _, v in key)
        n = len(recs)
        if n == 1:
            head = f"The {label} trial"
        else:
            head = f"All {n} {label} trials" if len(bad) == n else \
                f"{len(bad)}/{n} {label} trials"
        kinds = Counter(r.status for r in bad)
        verb = "failed" if set(kinds) == {"failed"} else "/".join(sorted(kinds))
        out.append(Insight("failure pattern", "warn", f'{head} {verb}: "{common}".',
                           {"group": dict(key), "failed": len(bad), "total": n,
                            "message": common}))
    return out[:5]


def efficiency(rs: Any) -> list[Insight]:
    sums = [s for s in rs.summaries() if s.mean is not None]
    if len(sums) < 2:
        return []
    best = sums[0]
    mode = rs.mode()
    out = []
    for cost_key, what in (("wall_s", "time"), ("params", "parameters"),
                           ("flops_est", "compute")):
        bc = best.cost(cost_key)
        if not bc:
            continue
        front = rs.pareto(x=cost_key)
        for s in front:
            if s is best or s.mean is None:
                continue
            c = s.cost(cost_key)
            if c is None:
                continue
            if mode == "min":
                quality = best.mean / s.mean if s.mean else 0.0
            else:
                quality = s.mean / best.mean if best.mean else 0.0
            if quality >= 0.95 and c / bc <= 0.60:
                out.append(Insight(
                    "efficiency", "info",
                    f"{s.label(rs.axes)} reaches {quality * 100:.0f} % of best in "
                    f"{c / bc * 100:.0f} % of the {what} (Pareto).",
                    {"config": s.labels, "quality": quality, "cost_ratio": c / bc,
                     "cost": cost_key}))
                break
    return out[:2]


def under_training(rs: Any) -> list[Insight]:
    sums = [s for s in rs.summaries() if s.mean is not None][:3]
    mode = rs.mode()
    improving = []
    for s in sums:
        c = s.curve()
        if len(c) >= 2 and c[-2]:
            rel = (c[-1] - c[-2]) / abs(c[-2])
            gain = -rel if mode == "min" else rel
            if gain > 0.01:
                improving.append((s, rel))
    if not improving or len(improving) < min(2, len(sums)):
        return []
    avg = sum(r for _, r in improving) / len(improving)
    epochs = None
    spec = rs.meta.get("spec") or {}
    tf = (spec.get("train_for") or {}).get("epochs")
    hint = f'consider train_for="{int(tf) * 2} epochs"' if tf else "consider a longer train_for"
    epochs = tf
    return [Insight("under-training", "warn",
                    f"Top configs still improving in the last epoch ({avg * 100:+.1f} %): "
                    f"{hint}.", {"configs": [s.labels for s, _ in improving],
                                 "last_epoch_change": avg, "epochs": epochs})]


def overfitting(rs: Any) -> list[Insight]:
    flagged = []
    metric = "val_loss"
    for s in rs.summaries():
        v, tr = s.curve(metric), s.curve("train_loss")
        if len(v) >= 3 and len(tr) >= 3 and v[-1] > v[-2] > v[-3] and tr[-1] < tr[-3]:
            flagged.append(s)
    if not flagged:
        return []
    names = ", ".join(s.label(rs.axes) for s in flagged[:3])
    return [Insight("overfitting", "warn",
                    f"{names}: val_loss rose for 2+ epochs while train_loss kept falling "
                    "(overfitting); select='best' or more data/regularization.",
                    {"configs": [s.labels for s in flagged]})]


def comparability(rs: Any) -> list[Insight]:
    out = []
    metric = rs.goal[0]
    if rs.not_comparable(metric):
        out.append(Insight("comparability", "warn",
                           f"{metric} is a per-token metric but the tokenizer varies: tokenizers "
                           "split text into different numbers of tokens, so values are not "
                           "comparable. Rank by val_bpb (bits per byte) instead.",
                           {"metric": metric}))
    for key in ("torch", "topologies", "device"):
        vals = {str((r.env or {}).get(key)) for r in rs.records if r.status == "completed"
                and r.env}
        if len(vals) > 1:
            out.append(Insight("comparability", "warn",
                               f"Results mix {key} {', '.join(sorted(vals))}; timings and "
                               "possibly metrics are not directly comparable.",
                               {key: sorted(vals)}))
    off = [r for r in rs.records if r.build and isinstance(r.build.get("rel_error"), float)
           and r.build["rel_error"] > 0.15]
    if off:
        worst = max(off, key=lambda r: r.build["rel_error"])
        out.append(Insight("comparability", "warn",
                           f"Parameter matching is off by {worst.build['rel_error'] * 100:.0f} % "
                           f"for {worst.describe()} (target {worst.build.get('target')}).",
                           {"trials": [r.trial_id for r in off]}))
    shares = []
    for s in rs.summaries():
        p, e = s.cost("params"), s.cost("embedding_params")
        if p and e is not None and any(t.build for t in s.trials):
            shares.append((e / p, s))
    big = [x for x in shares if x[0] > 0.30]
    if big:
        share, s = max(big, key=lambda x: x[0])
        out.append(Insight("comparability", "warn",
                           f"Embeddings are {share * 100:.0f} % of the parameters for "
                           f"{s.label(rs.axes)}: matched totals hide very different compute; "
                           "consider param_count='non_embedding'.",
                           {"embedding_share": share, "config": s.labels}))
    skipped = [r for r in rs.records if (r.extras or {}).get("param_matching")]
    if skipped:
        out.append(Insight("comparability", "warn",
                           f"{len(skipped)} trial(s) were not parameter-matched "
                           "(factory(ctx) models).", {"trials": [r.trial_id for r in skipped]}))
    return out


def noise(rs: Any) -> list[Insight]:
    sums = [s for s in rs.summaries() if s.mean is not None and s.ci is not None]
    if len(sums) < 2:
        return []
    lo0, hi0 = sums[0].ci  # type: ignore[misc]
    k = 1
    for s in sums[1:]:
        lo, hi = s.ci  # type: ignore[misc]
        if lo <= hi0 and hi >= lo0:
            k += 1
        else:
            break
    if k < 2:
        return []
    return [Insight("noise", "warn" if k >= 3 else "info",
                    f"Top {k} configs are statistically tied (95 % CIs overlap): "
                    + ", ".join(s.label(rs.axes) for s in sums[:k]) + ".",
                    {"tied": [s.labels for s in sums[:k]]})]


RULES = (winner, importance, consistency, failure_pattern, efficiency, under_training,
         overfitting, comparability, noise)


def generate(rs: Any) -> list[Insight]:
    """All rules, in order; a failing rule is skipped (never breaks reporting)."""
    out: list[Insight] = []
    if not rs.records:
        return out
    for rule in RULES:
        try:
            out.extend(rule(rs))
        except Exception:  # noqa: BLE001, S112 - one failing insight rule must not hide the others
            continue
    return [i for i in out if i.text and not any(
        isinstance(v, float) and math.isnan(v) for v in i.evidence.values()
        if isinstance(v, float))]


__all__ = ["RULES", "Insight", "generate"]
