"""Intrinsic diagnostics: what a dataset looks like on its own (no reference data).

Text: length distributions, distinct-n, type-token ratio, self-BLEU (sample),
gzip compression ratio, repeated n-grams and openings, near-duplicate
clusters, embedding dispersion and an effective cluster count. Labels: balance,
per-class diversity, label-attribute leakage (can the label be predicted from
the *other* attributes?) and length-label shortcuts. Tabular: marginals,
correlations, prevalence and missingness. Every function returns plain dicts;
``tables()`` turns them into ``experiments.table.Table`` views.

These metrics *explain* downstream utility (TSTR); they are not substitutes for it.
"""

from __future__ import annotations

import gzip
import math
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

import torch

from topologies.synth.textutil import ngrams, normalize, words


def _pct(xs: Sequence[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return float(s[min(len(s) - 1, max(0, round(q * (len(s) - 1))))])


def length_stats(texts: Sequence[str]) -> dict[str, Any]:
    chars = [len(t) for t in texts]
    w = [len(words(t)) for t in texts]
    return {"chars_mean": sum(chars) / max(1, len(chars)), "chars_p5": _pct(chars, 0.05),
            "chars_p50": _pct(chars, 0.5), "chars_p95": _pct(chars, 0.95),
            "words_mean": sum(w) / max(1, len(w)), "words_p50": _pct(w, 0.5)}


def distinct_n(texts: Sequence[str], n: int) -> float:
    total, uniq = 0, set()
    for t in texts:
        g = ngrams(words(t), n)
        total += len(g)
        uniq.update(g)
    return len(uniq) / max(1, total)


def type_token_ratio(texts: Sequence[str]) -> float:
    toks = [w for t in texts for w in words(t)]
    return len(set(toks)) / max(1, len(toks))


def compression_ratio(texts: Sequence[str]) -> float:
    """``len(gzip(all)) / len(all)``: lower means more redundant (a cheap diversity proxy)."""
    data = "\n".join(texts).encode("utf-8")
    if not data:
        return 0.0
    return len(gzip.compress(data, mtime=0)) / len(data)


def bleu(hyp: Sequence[str], refs: Sequence[Sequence[str]], max_n: int = 4) -> float:
    """Sentence BLEU with add-one smoothing (for self-BLEU)."""
    if not hyp:
        return 0.0
    logs = 0.0
    for n in range(1, max_n + 1):
        h = Counter(ngrams(hyp, n))
        if not h:
            logs += math.log(1 / (len(hyp) + 1))
            continue
        maxref: Counter = Counter()
        for r in refs:
            for g, c in Counter(ngrams(r, n)).items():
                maxref[g] = max(maxref[g], c)
        clipped = sum(min(c, maxref[g]) for g, c in h.items())
        logs += math.log((clipped + 1) / (sum(h.values()) + 1))
    ref_len = min((len(r) for r in refs), key=lambda L: abs(L - len(hyp)), default=len(hyp))
    bp = 1.0 if len(hyp) > ref_len else math.exp(1 - ref_len / max(1, len(hyp)))
    return bp * math.exp(logs / max_n)


def self_bleu(texts: Sequence[str], *, sample: int = 100, refs: int = 50, seed: int = 0
              ) -> float:
    """Mean BLEU of a sample against other records (higher = more repetitive)."""
    toks = [words(t) for t in texts if t]
    if len(toks) < 3:
        return 0.0
    rng = random.Random(seed)
    idx = rng.sample(range(len(toks)), min(sample, len(toks)))
    scores = []
    for i in idx:
        others = [toks[j] for j in rng.sample(range(len(toks)), min(refs + 1, len(toks)))
                  if j != i][:refs]
        scores.append(bleu(toks[i], others))
    return sum(scores) / len(scores)


def top_ngrams(texts: Sequence[str], n: int = 3, k: int = 10) -> list[tuple[str, int, float]]:
    c: Counter = Counter()
    docs: Counter = Counter()
    for t in texts:
        g = ngrams(words(t), n)
        c.update(g)
        docs.update(set(g))
    return [(" ".join(g), cnt, docs[g] / max(1, len(texts))) for g, cnt in c.most_common(k)]


def top_openings(texts: Sequence[str], n_words: int = 3, k: int = 10
                 ) -> list[tuple[str, int, float]]:
    c = Counter(" ".join(normalize(t).split()[:n_words]) for t in texts if t)
    return [(o, cnt, cnt / max(1, len(texts))) for o, cnt in c.most_common(k)]


def near_dup_clusters(texts: Sequence[str], threshold: float = 0.7) -> dict[str, Any]:
    from topologies.synth.textutil import LSHIndex, MinHasher

    if not texts:
        return {"clusters": 0, "largest": 0, "in_clusters": 0.0}
    h = MinHasher(64)
    idx = LSHIndex(64, threshold)
    sigs = h.signatures(list(texts))
    keys = idx.band_keys(sigs)
    parent = list(range(len(texts)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(texts)):
        cands = idx.candidates(keys[i:i + 1])[0]
        slot, sim = idx.best_match(sigs[i], cands)
        if slot is not None and sim >= threshold:
            parent[find(i)] = find(slot)
        idx.add(sigs[i], keys[i])
    sizes = Counter(find(i) for i in range(len(texts)))
    multi = [s for s in sizes.values() if s > 1]
    return {"clusters": len(multi), "largest": max(multi, default=1),
            "in_clusters": sum(multi) / len(texts)}


def feature_near_dup_clusters(features: torch.Tensor, threshold: float = 0.95) -> dict[str, Any]:
    """Cluster records by cosine similarity of their declared type features."""
    raw = torch.as_tensor(features, dtype=torch.float32)
    if raw.ndim != 2 or raw.shape[1] == 0:
        raise ValueError("type features must be a 2D tensor with at least one feature column")
    raw = torch.nan_to_num(raw)
    norms = raw.norm(dim=1)
    x = torch.nn.functional.normalize(raw, dim=1)
    n = len(x)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i):
            similarity = float(x[i] @ x[j])
            if norms[i] == 0 and norms[j] == 0:
                similarity = 1.0
            if similarity >= threshold:
                parent[find(i)] = find(j)
    sizes = Counter(find(i) for i in range(n))
    multi = [size for size in sizes.values() if size > 1]
    return {"clusters": len(multi), "largest": max(multi, default=1),
            "in_clusters": sum(multi) / max(1, n)}


def kmeans(x: torch.Tensor, k: int, *, iters: int = 25, seed: int = 0) -> tuple[torch.Tensor,
                                                                                 float]:
    g = torch.Generator().manual_seed(seed)
    c = x[torch.randperm(x.shape[0], generator=g)[:k]].clone()
    for _ in range(iters):
        d = torch.cdist(x, c)
        a = d.argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = x[m].mean(0)
    d = torch.cdist(x, c)
    return d.argmin(1), float(d.min(1).values.pow(2).mean())


def embedding_stats(texts: Sequence[str], *, embed: Any = None, sample: int = 1000,
                    seed: int = 0) -> dict[str, Any]:
    """Mean pairwise cosine distance and an effective cluster count (silhouette-picked k)."""
    from topologies.synth.quality.embed import get_embedder

    rng = random.Random(seed)
    pick = rng.sample(list(texts), min(sample, len(texts))) if texts else []
    if len(pick) < 4:
        return {"dispersion": 0.0, "clusters": len(pick), "embedder": None}
    emb = get_embedder(embed)
    x = torch.as_tensor(emb.encode(pick), dtype=torch.float32)
    x = torch.nn.functional.normalize(x, dim=-1)
    if x.shape[1] > 256:
        x = _pca(x, 64)
    sims = x @ x.T
    n = x.shape[0]
    disp = float(1 - (sims.sum() - n) / (n * (n - 1)))
    best_k, best_s = 1, -1.0
    for k in (2, 3, 4, 6, 8, 12, 16, 24):
        if k >= n:
            break
        a, _ = kmeans(x, k, seed=seed)
        s = _silhouette(x, a, k)
        if s > best_s:
            best_k, best_s = k, s
    return {"dispersion": disp, "clusters": best_k, "silhouette": best_s,
            "embedder": getattr(emb, "name", type(emb).__name__)}


def _pca(x: torch.Tensor, k: int) -> torch.Tensor:
    xc = x - x.mean(0)
    q = min(k, min(xc.shape) - 1)
    u, s, _ = torch.pca_lowrank(xc, q=q, center=False)
    return torch.nn.functional.normalize(u * s, dim=-1)


def _silhouette(x: torch.Tensor, a: torch.Tensor, k: int, sample: int = 400) -> float:
    idx = torch.arange(min(sample, x.shape[0]))
    d = torch.cdist(x[idx], x)
    out = []
    for i in idx.tolist():
        same = a == a[i]
        same[i] = False
        if not same.any():
            continue
        ai = float(d[i][same].mean())
        bi = min((float(d[i][a == j].mean()) for j in range(k) if j != int(a[i]) and
                  (a == j).any()), default=ai)
        out.append((bi - ai) / max(ai, bi, 1e-9))
    return sum(out) / max(1, len(out))


# ---------------------------------------------------------------- labels


def label_stats(records: Sequence[dict], field: str, text_key: Any = None, *,
                diversity: bool = True) -> dict[str, Any]:
    """Counts, balance and (text data) per-class distinct-2 diversity."""
    from topologies.synth.textutil import text_of

    counts = Counter(str(r.get(field)) for r in records)
    n = sum(counts.values()) or 1
    probs = [c / n for c in counts.values()]
    ent = -sum(p * math.log(p) for p in probs if p > 0)
    balance = ent / math.log(len(counts)) if len(counts) > 1 else 1.0
    by: dict[str, list[str]] = defaultdict(list)
    for r in records if diversity else ():
        by[str(r.get(field))].append(text_of(r, text_key))
    div = {k: distinct_n(v, 2) for k, v in by.items()}
    med = sorted(div.values())[len(div) // 2] if div else 0.0
    low = [k for k, v in div.items() if v < 0.5 * med]
    return {"counts": dict(counts), "balance": balance, "distinct2_by_class": div,
            "low_diversity_classes": low}


def _predictability(features: Sequence[tuple], labels: Sequence[str]) -> tuple[float, float]:
    """Holdout accuracy of a count-table predictor ``features -> label`` vs. majority."""
    n = len(labels)
    if n < 10:
        return 0.0, 0.0
    idx = list(range(n))
    random.Random(0).shuffle(idx)
    cut = int(n * 0.7)
    tr, te = idx[:cut], idx[cut:]
    table: dict[tuple, Counter] = defaultdict(Counter)
    for i in tr:
        table[features[i]][labels[i]] += 1
    majority = Counter(labels[i] for i in tr).most_common(1)[0][0]
    hit = sum(1 for i in te if (table[features[i]].most_common(1)[0][0] if table[features[i]]
                                else majority) == labels[i])
    base = sum(1 for i in te if labels[i] == majority)
    return hit / max(1, len(te)), base / max(1, len(te))


def leakage(records: Sequence[dict], field: str, *, exclude: Sequence[str] = ()) -> list[dict]:
    """Per attribute: how well it alone predicts the label (spurious shortcuts)."""
    labels = [str(r.get(field)) for r in records]
    attrs = sorted({k for r in records for k in (r.get("_attrs") or {})} - {field, *exclude})
    out = []
    for a in attrs:
        vals = [(r.get("_attrs") or {}).get(a) for r in records]
        if not all(isinstance(v, (str, int, bool, type(None))) for v in vals):
            continue
        if len(set(vals)) > max(50, len(vals) // 5):
            continue
        acc, base = _predictability([(v,) for v in vals], labels)
        out.append({"attribute": a, "acc": acc, "baseline": base, "lift": acc - base})
    return sorted(out, key=lambda d: -d["lift"])


def length_shortcut(records: Sequence[dict], field: str, text_key: Any = None,
                    bins: int = 8) -> dict[str, float]:
    """Accuracy of predicting the label from text length alone (binned)."""
    from topologies.synth.textutil import text_of

    lens = [len(text_of(r, text_key)) for r in records]
    if not lens:
        return {"acc": 0.0, "baseline": 0.0}
    qs = sorted(lens)
    edges = [qs[min(len(qs) - 1, int(len(qs) * k / bins))] for k in range(1, bins)]
    feats = [(sum(1 for e in edges if L > e),) for L in lens]
    acc, base = _predictability(feats, [str(r.get(field)) for r in records])
    return {"acc": acc, "baseline": base, "lift": acc - base}


# ---------------------------------------------------------------- tabular


def tabular_stats(records: Sequence[dict], schema: Any = None) -> dict[str, Any]:
    if not records:
        return {}
    keys = [k for k in records[0] if not str(k).startswith("_")]
    num, cat = {}, {}
    for k in keys:
        vals = [r.get(k) for r in records]
        present = [v for v in vals if v is not None and not (isinstance(v, float) and
                                                              math.isnan(v))]
        miss = 1 - len(present) / len(vals)
        if present and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                           for v in present):
            t = torch.tensor([float(v) for v in present])
            num[k] = {"mean": float(t.mean()), "std": float(t.std()) if t.numel() > 1 else 0.0,
                      "min": float(t.min()), "max": float(t.max()), "missing": miss}
        elif present and all(isinstance(v, (str, bool, int)) for v in present):
            c = Counter(map(str, present))
            if len(c) <= 64:
                cat[k] = {"freq": {a: b / len(present) for a, b in c.most_common(20)},
                          "missing": miss}
    corr = None
    cols = list(num)
    if len(cols) >= 2:
        m = torch.tensor([[float(r.get(c)) if isinstance(r.get(c), (int, float)) and
                           r.get(c) == r.get(c) else float("nan") for c in cols]
                          for r in records])
        m = torch.where(torch.isnan(m), torch.nanmean(m, 0, keepdim=True), m)
        corr = {"columns": cols, "matrix": torch.corrcoef(m.T).nan_to_num().tolist()}
    return {"numeric": num, "categorical": cat, "correlation": corr}


# ---------------------------------------------------------------- one call


def intrinsic(records: Sequence[dict], *, schema: Any = None, text_key: Any = None,
              label_field: str | None = None, embed: Any = None, sample: int = 2000,
              seed: int = 0) -> dict[str, Any]:
    """All intrinsic metrics that apply to the records' schema."""
    from topologies.synth.textutil import text_of

    rng = random.Random(seed)
    recs = list(records)
    if len(recs) > sample:
        recs = rng.sample(recs, sample)
    out: dict[str, Any] = {"n": len(records), "sampled": len(recs)}
    lf = label_field or (schema.label_field() if schema is not None else None)
    if lf is None and recs and "label" in recs[0]:
        lf = "label"
    text_fields = [f for f in (schema.text_fields() if schema is not None else []) if f != lf]
    if text_key is None and text_fields:
        text_key = text_fields if len(text_fields) > 1 else text_fields[0]
    is_text = bool(text_fields) or (schema is None and recs and any(
        isinstance(v, str) and len(v) > 20 for k, v in recs[0].items() if k != lf))
    if is_text:
        texts = [text_of(r, text_key) for r in recs]
        out["text"] = {**length_stats(texts), "distinct1": distinct_n(texts, 1),
                       "distinct2": distinct_n(texts, 2), "distinct3": distinct_n(texts, 3),
                       "ttr": type_token_ratio(texts), "compression": compression_ratio(texts),
                       "self_bleu": self_bleu(texts, seed=seed),
                       "top_ngrams": top_ngrams(texts), "openings": top_openings(texts),
                       "near_dups": near_dup_clusters(texts)}
        try:
            out["embedding"] = embedding_stats(texts, embed=embed, seed=seed)
        except Exception as e:  # noqa: BLE001 - optional diagnostics
            out["embedding"] = {"error": str(e)}
    if lf is not None and recs and lf in recs[0]:
        out["labels"] = label_stats(recs, lf, text_key if is_text else None,
                                    diversity=bool(is_text))
        out["leakage"] = leakage(recs, lf)
        if is_text:
            out["length_shortcut"] = length_shortcut(recs, lf, text_key)
    if not is_text:
        if schema is not None:
            from topologies.synth.quality.fidelity import _record_features

            features = _record_features(recs, schema)
            if features is not None:
                out["features"] = {"near_dups": feature_near_dup_clusters(features),
                                   "dimensions": features.shape[1]}
        out["tabular"] = tabular_stats(recs, schema)
    return out


__all__ = ["bleu", "compression_ratio", "distinct_n", "embedding_stats", "intrinsic",
           "kmeans", "label_stats", "leakage", "length_shortcut", "length_stats",
           "near_dup_clusters", "self_bleu", "tabular_stats", "top_ngrams", "top_openings",
           "type_token_ratio"]
