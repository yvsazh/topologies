"""Fidelity: how distinguishable synthetic data is from real data.

- ``c2st``: classifier two-sample test. A cross-validated linear model on hashed
  n-gram features (or embeddings / tabular features) tries to tell real from
  synthetic; ROC AUC ~0.5 = indistinguishable, ~1.0 = trivially separable. It also
  returns the **giveaways**: the most synthetic-looking records and n-grams (log-odds
  with an informative prior), and simple tells (length ratio, punctuation, digits).
- ``mmd`` (RBF, median heuristic) and ``frechet`` distance on (PCA-reduced) embeddings.
- ``js_divergence`` of length and unigram distributions.
- ``marginals``: per-attribute / per-column comparison (TV for categories, KS for numbers).
- ``tabular_fidelity``: per-column KS / TV, correlation-matrix difference, C2ST.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Sequence
from typing import Any

import torch

from topologies.synth.textutil import ngrams, text_of, words


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Mann-Whitney AUC (ties averaged)."""
    pairs = sorted(zip(scores, labels))
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    pos = sum(1 for _, y in pairs if y == 1)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return 0.5
    rsum = sum(r for r, (_, y) in zip(ranks, pairs) if y == 1)
    return (rsum - pos * (pos + 1) / 2) / (pos * neg)


def _bootstrap_ci(scores: Sequence[float], labels: Sequence[int], reps: int = 200,
                  seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(scores)
    vals = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(roc_auc([scores[i] for i in idx], [labels[i] for i in idx]))
    vals.sort()
    return vals[int(0.025 * reps)], vals[int(0.975 * reps) - 1]


def log_odds_ngrams(syn: Sequence[str], real: Sequence[str], *, n_max: int = 2, k: int = 10,
                    min_count: int = 3) -> list[dict[str, Any]]:
    """n-grams over-represented in synthetic text (z-scored log-odds, Dirichlet prior)."""
    def counts(texts: Sequence[str]) -> Counter:
        c: Counter = Counter()
        for t in texts:
            w = words(t)
            for n in range(1, n_max + 1):
                c.update(" ".join(g) for g in ngrams(w, n))
        return c

    cs, cr = counts(syn), counts(real)
    prior = cs + cr
    a0 = sum(prior.values()) or 1
    ns, nr = sum(cs.values()) or 1, sum(cr.values()) or 1
    out = []
    for g, ap in prior.items():
        if cs[g] + cr[g] < min_count:
            continue
        a = ap * 0.01 + 0.01
        ls = math.log((cs[g] + a) / (ns + a0 * 0.01 - cs[g] - a + 1e-9))
        lr = math.log((cr[g] + a) / (nr + a0 * 0.01 - cr[g] - a + 1e-9))
        var = 1 / (cs[g] + a) + 1 / (cr[g] + a)
        out.append({"ngram": g, "z": (ls - lr) / math.sqrt(var), "synthetic": cs[g],
                    "real": cr[g]})
    out.sort(key=lambda d: -d["z"])
    return out[:k]


def _tells(syn: Sequence[str], real: Sequence[str]) -> dict[str, float]:
    def rate(texts: Sequence[str], pred: Any) -> float:
        chars = sum(len(t) for t in texts) or 1
        return sum(sum(1 for ch in t if pred(ch)) for t in texts) / chars

    ls = sum(len(t) for t in syn) / max(1, len(syn))
    lr = sum(len(t) for t in real) / max(1, len(real))
    return {"length_ratio": ls / max(lr, 1e-9),
            "punct_ratio": rate(syn, lambda c: c in ".,;:!?") /
            max(rate(real, lambda c: c in ".,;:!?"), 1e-9),
            "upper_ratio": rate(syn, str.isupper) / max(rate(real, str.isupper), 1e-9),
            "digit_ratio": (rate(syn, str.isdigit) + 1e-6) / (rate(real, str.isdigit) + 1e-6)}


def tabular_features(records: Sequence[dict], columns: Sequence[str] | None = None,
                     *, stats: dict[str, Any] | None = None) -> tuple[torch.Tensor, dict]:
    """Standardized numeric columns + one-hot categoricals (``stats`` reuse across sets)."""
    if stats is None:
        cols = list(columns) if columns is not None else [
            k for k in records[0] if not str(k).startswith("_")]
        num, cat = [], {}
        for c in cols:
            vals = [r.get(c) for r in records if r.get(c) is not None]
            if vals and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                            for v in vals):
                t = torch.tensor([float(v) for v in vals])
                num.append((c, float(t.nanmean()), float(t[~torch.isnan(t)].std()) or 1.0))
            elif vals:
                cat[c] = sorted({str(v) for v in vals})[:50]
        stats = {"num": num, "cat": cat}
    rows = []
    for r in records:
        f = []
        for c, mu, sd in stats["num"]:
            v = r.get(c)
            ok = isinstance(v, (int, float)) and not math.isnan(float(v))
            x = float(v) if ok else mu
            f.append((x - mu) / (sd or 1.0))
            f.append(0.0 if ok else 1.0)
        for c, vals in stats["cat"].items():
            v = str(r.get(c))
            f += [1.0 if v == x else 0.0 for x in vals]
        rows.append(f)
    return torch.tensor(rows, dtype=torch.float32), stats


def c2st(synthetic: Sequence[dict], real: Sequence[dict], *, key: Any = None,
         embed: Any = None, tabular: bool = False, folds: int = 5, max_per_side: int = 2000,
         seed: int = 0, k: int = 8, schema: Any = None) -> dict[str, Any]:
    """Classifier two-sample test (see module docstring)."""
    from topologies.synth.ops.label import out_of_fold_probs

    rng = random.Random(seed)
    n = min(len(synthetic), len(real), max_per_side)
    if n < 10:
        return {"auc": None, "n": n, "note": "too few records for C2ST"}
    syn = rng.sample(list(synthetic), n)
    rea = rng.sample(list(real), n)
    all_features = (_record_features([*syn, *rea], schema, key) if schema is not None else None)
    type_features = all_features[:n] if all_features is not None else None
    real_type_features = all_features[n:] if all_features is not None else None
    if type_features is not None:
        xs, xr = type_features, real_type_features
        text_names = _input_names(schema, key)
        text_names = [name for name in text_names if schema[name].kind == "text"]
        syn_txt = [text_of(r, text_names) for r in syn] if text_names else None
        rea_txt = [text_of(r, text_names) for r in rea] if text_names else None
    elif tabular:
        xs, st = tabular_features(syn)
        xr, _ = tabular_features(rea, stats=st)
        syn_txt = rea_txt = None
    else:
        syn_txt = [text_of(r, key) for r in syn]
        rea_txt = [text_of(r, key) for r in rea]
        if embed is None:
            from topologies.synth.quality.embed import HashedEmbedder

            e = HashedEmbedder(2048)
        else:
            from topologies.synth.quality.embed import get_embedder

            e = get_embedder(embed)
        xs, xr = e.encode(syn_txt), e.encode(rea_txt)
    x = torch.cat([torch.as_tensor(xs, dtype=torch.float32),
                   torch.as_tensor(xr, dtype=torch.float32)])
    y = torch.tensor([1] * n + [0] * n)
    probs = out_of_fold_probs(x, y, 2, folds=folds, seed=seed, epochs=100)[:, 1]
    scores = probs.tolist()
    labels = y.tolist()
    auc = roc_auc(scores, labels)
    lo, hi = _bootstrap_ci(scores, labels, seed=seed)
    order = sorted(range(n), key=lambda i: -scores[i])
    out: dict[str, Any] = {"auc": auc, "ci": (lo, hi), "n": n,
                           "most_synthetic": [{**syn[i], "_p_synthetic": round(scores[i], 4)}
                                              for i in order[:k]],
                           "most_real_looking": [{**syn[i], "_p_synthetic": round(scores[i], 4)}
                                                 for i in order[-k:][::-1]]}
    if syn_txt is not None:
        out["giveaway_ngrams"] = log_odds_ngrams(syn_txt, rea_txt, k=k)
        out["tells"] = _tells(syn_txt, rea_txt)
    if type_features is not None:
        out["giveaway_features"] = feature_giveaways(xs, xr, k=k,
                                                      names=_feature_names(schema, key, xs.shape[1],
                                                                           [*syn, *rea]))
    return out


def _input_names(schema: Any, key: Any = None) -> list[str]:
    names = [key] if isinstance(key, str) else list(key) if isinstance(key, (tuple, list)) else None
    names = names or [name for name in schema.fields if name != schema.label_field()]
    return [name for name in names if name in schema.fields and name != schema.label_field()]


def _record_features(records: Sequence[dict], schema: Any, key: Any = None) -> torch.Tensor | None:
    """Concatenate declared input-field features, excluding labels and metadata."""
    if schema is None or not records:
        return None
    names = _input_names(schema, key)
    parts = []
    for name in names:
        field = schema[name]
        present = [field.unwrap(record[name]) if record.get(name) is not None else None
                   for record in records]
        available = [v for v in present if v is not None]
        if available:
            encoded = torch.as_tensor(field.features(available), dtype=torch.float32)
            if encoded.ndim == 1:
                encoded = encoded[:, None]
            if encoded.shape[1] == 0:
                raise ValueError(f"{name!r} Type.features returned zero columns; provide at "
                                 "least one feature")
            width = encoded.shape[1]
            part = torch.zeros((len(records), width + 1), dtype=torch.float32)
            row = 0
            for i, value in enumerate(present):
                if value is None:
                    part[i, width] = 1.0
                else:
                    part[i, :width] = torch.nan_to_num(encoded[row])
                    row += 1
        else:
            part = torch.ones((len(records), 1), dtype=torch.float32)
        if part.shape[0] != len(records):
            raise ValueError(f"{name!r} features must have one row per record")
        parts.append(part)
    return torch.cat(parts, dim=1) if parts else None


def _feature_names(schema: Any, key: Any, width: int,
                   records: Sequence[dict]) -> list[str]:
    names = []
    for field_name in _input_names(schema, key):
        typ = schema[field_name]
        values = [typ.unwrap(r[field_name]) for r in records if r.get(field_name) is not None]
        encoded = torch.as_tensor(typ.features(values), dtype=torch.float32) if values else None
        field_width = encoded.shape[-1] if encoded is not None and encoded.ndim > 1 else 1
        names.extend(f"{field_name}:{typ.kind}:feature_{i}" for i in range(field_width))
        names.append(f"{field_name}:{typ.kind}:missing")
    return names + [f"feature_{i}" for i in range(len(names), width)]


def feature_giveaways(xs: torch.Tensor, xr: torch.Tensor, *, k: int = 8,
                      names: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Rank standardized feature coordinates by synthetic-vs-real mean difference."""
    delta = xs.mean(0) - xr.mean(0)
    scale = torch.cat((xs, xr)).std(0, unbiased=False).clamp_min(1e-6)
    z = delta / scale
    return [{"feature": int(i), "name": names[i] if names else f"feature_{i}",
             "z": float(z[i]), "synthetic_mean": float(xs[:, i].mean()),
             "real_mean": float(xr[:, i].mean())}
            for i in sorted(range(z.numel()), key=lambda j: -abs(float(z[j])))[:k]]


def _reduce(xs: torch.Tensor, xr: torch.Tensor, dim: int = 32) -> tuple[torch.Tensor,
                                                                        torch.Tensor]:
    x = torch.cat([xs, xr]).float()
    x = x - x.mean(0)
    q = min(dim, min(x.shape) - 1)
    if q < 1:
        return xs, xr
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        _, _, v = torch.pca_lowrank(x, q=q, center=False)
    z = x @ v[:, :q]
    return z[: xs.shape[0]], z[xs.shape[0]:]


def mmd(xs: torch.Tensor, xr: torch.Tensor) -> float:
    """Unbiased MMD^2 with an RBF kernel (median-distance bandwidth)."""
    z = torch.cat([xs, xr])
    d = torch.cdist(z, z)
    sigma = float(d[d > 0].median()) if (d > 0).any() else 1.0
    k = torch.exp(-(d ** 2) / (2 * sigma ** 2))
    n, m = xs.shape[0], xr.shape[0]
    kxx, kyy, kxy = k[:n, :n], k[n:, n:], k[:n, n:]
    return float((kxx.sum() - kxx.diag().sum()) / (n * (n - 1)) +
                 (kyy.sum() - kyy.diag().sum()) / (m * (m - 1)) - 2 * kxy.mean())


def _sqrtm_psd(a: torch.Tensor) -> torch.Tensor:
    w, v = torch.linalg.eigh((a + a.T) / 2)
    return (v * w.clamp(min=0).sqrt()) @ v.T


def frechet(xs: torch.Tensor, xr: torch.Tensor) -> float:
    mu_s, mu_r = xs.mean(0), xr.mean(0)
    cs = torch.cov(xs.T.double())
    cr = torch.cov(xr.T.double())
    s = _sqrtm_psd(cs)
    cross = _sqrtm_psd(s @ cr @ s)
    return float((mu_s - mu_r).double().pow(2).sum() + torch.trace(cs + cr - 2 * cross))


def js_divergence(p: dict[Any, float], q: dict[Any, float]) -> float:
    keys = set(p) | set(q)
    ps, qs = sum(p.values()) or 1.0, sum(q.values()) or 1.0
    out = 0.0
    for k in keys:
        a, b = p.get(k, 0.0) / ps, q.get(k, 0.0) / qs
        m = (a + b) / 2
        if a > 0:
            out += 0.5 * a * math.log(a / m)
        if b > 0:
            out += 0.5 * b * math.log(b / m)
    return out / math.log(2)


def ks_statistic(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (max ECDF gap, ties handled)."""
    import bisect

    a, b = sorted(a), sorted(b)
    if not a or not b:
        return 0.0
    return max(abs(bisect.bisect_right(a, x) / len(a) - bisect.bisect_right(b, x) / len(b))
               for x in set(a) | set(b))


def tv_distance(a: Sequence[Any], b: Sequence[Any]) -> float:
    ca, cb = Counter(map(str, a)), Counter(map(str, b))
    na, nb = sum(ca.values()) or 1, sum(cb.values()) or 1
    return 0.5 * sum(abs(ca[k] / na - cb[k] / nb) for k in set(ca) | set(cb))


def marginals(synthetic: Sequence[dict], real: Sequence[dict]) -> list[dict[str, Any]]:
    """Per shared field / attribute: KS (numbers) or TV distance (categories)."""
    def val(r: dict, k: str) -> Any:
        return r[k] if k in r else (r.get("_attrs") or {}).get(k)

    keys = sorted(({k for r in synthetic[:50] for k in r} | {k for r in synthetic[:50]
                                                             for k in r.get("_attrs") or {}})
                  & ({k for r in real[:50] for k in r} | {k for r in real[:50]
                                                         for k in r.get("_attrs") or {}}))
    out = []
    for k in keys:
        if str(k).startswith("_"):
            continue
        a = [val(r, k) for r in synthetic if val(r, k) is not None]
        b = [val(r, k) for r in real if val(r, k) is not None]
        if not a or not b:
            continue
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in a + b):
            out.append({"field": k, "kind": "numeric", "ks": ks_statistic(a, b)})
        elif all(isinstance(v, (str, bool, int)) for v in a + b) and \
                len(set(map(str, a + b))) <= 100:
            out.append({"field": k, "kind": "categorical", "tv": tv_distance(a, b)})
        elif all(isinstance(v, str) for v in a + b):
            la, lb = [len(x) for x in a], [len(x) for x in b]
            out.append({"field": k, "kind": "text-length", "ks": ks_statistic(la, lb)})
    return out


def text_fidelity(synthetic: Sequence[dict], real: Sequence[dict], *, key: Any = None,
                  embed: Any = None, sample: int = 1000, seed: int = 0) -> dict[str, Any]:
    rng = random.Random(seed)
    syn = [text_of(r, key) for r in rng.sample(list(synthetic), min(sample, len(synthetic)))]
    rea = [text_of(r, key) for r in rng.sample(list(real), min(sample, len(real)))]
    from topologies.synth.quality.embed import get_embedder

    e = get_embedder(embed)
    xs = torch.as_tensor(e.encode(syn), dtype=torch.float32)
    xr = torch.as_tensor(e.encode(rea), dtype=torch.float32)
    zs, zr = _reduce(xs, xr)
    def hist(texts: Sequence[str]) -> dict[int, float]:
        return Counter(min(len(words(t)) // 4, 40) for t in texts)

    def vocab(texts: Sequence[str]) -> dict[str, float]:
        return Counter(w for t in texts for w in words(t))

    return {"mmd": mmd(zs, zr) if len(syn) > 1 and len(rea) > 1 else None,
            "frechet": frechet(zs, zr) if len(syn) > 2 and len(rea) > 2 else None,
            "js_length": js_divergence(hist(syn), hist(rea)),
            "js_vocab": js_divergence(vocab(syn), vocab(rea)),
             "embedder": getattr(e, "name", type(e).__name__)}


def type_fidelity(synthetic: Sequence[dict], real: Sequence[dict], schema: Any, *,
                  key: Any = None, seed: int = 0) -> dict[str, Any]:
    """Distribution distances over declared value-type features."""
    n = len(synthetic)
    features = _record_features([*synthetic, *real], schema, key)
    if features is None:
        return {"mmd": None, "frechet": None, "features": 0}
    xs, xr = features[:n], features[n:]
    xs, xr = _reduce(xs, xr)
    return {"mmd": mmd(xs, xr) if len(xs) > 1 and len(xr) > 1 else None,
            "frechet": frechet(xs, xr) if len(xs) > 2 and len(xr) > 2 else None,
            "features": xs.shape[1], "seed": seed}


def tabular_fidelity(synthetic: Sequence[dict], real: Sequence[dict], *, seed: int = 0
                     ) -> dict[str, Any]:
    cols = marginals(synthetic, real)
    num = [c["field"] for c in cols if c["kind"] == "numeric"]
    corr_diff = None
    if len(num) >= 2:
        def corr(recs: Sequence[dict]) -> torch.Tensor:
            m = torch.tensor([[float(r.get(c)) if isinstance(r.get(c), (int, float)) and
                               r.get(c) == r.get(c) else float("nan") for c in num]
                              for r in recs])
            m = torch.where(torch.isnan(m), torch.nanmean(m, 0, keepdim=True), m)
            return torch.corrcoef(m.T).nan_to_num()

        d = corr(synthetic) - corr(real)
        corr_diff = {"max_abs": float(d.abs().max()), "frobenius": float(d.norm()),
                     "columns": num}
    return {"columns": cols, "correlation_diff": corr_diff,
            "c2st": c2st(synthetic, real, tabular=True, seed=seed)}


__all__ = ["c2st", "frechet", "js_divergence", "ks_statistic", "log_odds_ngrams", "marginals",
           "mmd", "roc_auc", "tabular_features", "tabular_fidelity", "text_fidelity",
           "tv_distance"]
