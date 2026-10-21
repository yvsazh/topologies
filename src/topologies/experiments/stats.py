"""Statistics for comparing trials (pure Python, no numpy/scipy).

- ``mean``, sample ``std`` (n-1), ``ci95`` = mean ± t(0.975, n-1) · s / √n;
- Student-t CDF through the regularized incomplete beta function
  (continued fraction, modified Lentz), t-quantiles by bisection;
- ``welch_ttest`` (Welch-Satterthwaite df, two-sided p), ``cohens_d``;
- ``seeds_needed`` ≈ ceil(15.7 σ² / δ²) (α=0.05, power 0.8, two groups);
- ``importance``: variance decomposition (axes, interaction, seed noise);
- ``kendall_w``: agreement of rankings across contexts;
- ``pareto_front``: non-dominated points.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Hashable, Sequence
from typing import Any


def _finite(xs: Sequence[float | None]) -> list[float]:
    out = []
    for x in xs:
        if x is None or isinstance(x, bool):
            continue
        try:
            f = float(x)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def mean(xs: Sequence[float]) -> float | None:
    v = _finite(xs)
    return sum(v) / len(v) if v else None


def std(xs: Sequence[float]) -> float | None:
    """Sample standard deviation (n-1); ``None`` for fewer than two values."""
    v = _finite(xs)
    if len(v) < 2:
        return None
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


# ---------------------------------------------------------------- Student t


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    tiny, eps = 1e-300, 3e-16
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta ``I_x(a, b)``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_bt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    bt = math.exp(ln_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    """Student-t cumulative distribution function."""
    if df <= 0:
        raise ValueError("df must be > 0")
    if math.isinf(t):
        return 1.0 if t > 0 else 0.0
    x = df / (df + t * t)
    tail = 0.5 * betainc(df / 2.0, 0.5, x)
    return 1.0 - tail if t >= 0 else tail


def t_two_sided_p(t: float, df: float) -> float:
    return max(0.0, min(1.0, 2.0 * (1.0 - t_cdf(abs(t), df))))


def t_ppf(q: float, df: float) -> float:
    """Quantile of the Student-t distribution (bisection on ``t_cdf``)."""
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    if q < 0.5:
        return -t_ppf(1.0 - q, df)
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < q:
        hi *= 2.0
        if hi > 1e8:
            return hi
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return (lo + hi) / 2.0


def ci95(xs: Sequence[float]) -> tuple[float, float] | None:
    """``(low, high)`` 95 % confidence interval of the mean; None for n < 2."""
    v = _finite(xs)
    if len(v) < 2:
        return None
    m, s = sum(v) / len(v), std(v)
    half = t_ppf(0.975, len(v) - 1) * (s or 0.0) / math.sqrt(len(v))
    return m - half, m + half


def welch_ttest(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float] | None:
    """``(t, df, two-sided p)``; ``None`` when either side has fewer than two values."""
    x, y = _finite(a), _finite(b)
    if len(x) < 2 or len(y) < 2:
        return None
    mx, my = sum(x) / len(x), sum(y) / len(y)
    vx, vy = std(x) ** 2, std(y) ** 2  # type: ignore[operator]
    se2 = vx / len(x) + vy / len(y)
    if se2 <= 0:
        if mx == my:
            return 0.0, float(len(x) + len(y) - 2), 1.0
        return math.copysign(math.inf, mx - my), float(len(x) + len(y) - 2), 0.0
    t = (mx - my) / math.sqrt(se2)
    denom = (vx / len(x)) ** 2 / (len(x) - 1) + (vy / len(y)) ** 2 / (len(y) - 1)
    df = se2 ** 2 / denom if denom > 0 else float(len(x) + len(y) - 2)
    return t, df, t_two_sided_p(t, df)


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Standardized mean difference with the pooled standard deviation."""
    x, y = _finite(a), _finite(b)
    if len(x) < 2 or len(y) < 2:
        return None
    sx, sy = std(x) or 0.0, std(y) or 0.0
    pooled = math.sqrt(((len(x) - 1) * sx ** 2 + (len(y) - 1) * sy ** 2) / (len(x) + len(y) - 2))
    if pooled == 0:
        return None
    return (sum(x) / len(x) - sum(y) / len(y)) / pooled


def seeds_needed(sigma: float | None, delta: float | None) -> int | None:
    """Seeds per config to detect a gap ``delta`` (α=0.05, power 0.8): ceil(15.7 σ²/δ²)."""
    if sigma is None or delta is None or delta == 0 or not math.isfinite(sigma):
        return None
    return max(2, math.ceil(15.7 * sigma ** 2 / delta ** 2))


def pooled_std(groups: Sequence[Sequence[float]]) -> float | None:
    """Within-group standard deviation pooled over groups with n >= 2."""
    num, den = 0.0, 0
    for g in groups:
        v = _finite(g)
        if len(v) >= 2:
            m = sum(v) / len(v)
            num += sum((x - m) ** 2 for x in v)
            den += len(v) - 1
    return math.sqrt(num / den) if den > 0 else None


# ---------------------------------------------------------------- importance


def quantile_bins(values: Sequence[float], k: int = 5) -> list[int]:
    """Bin index per value into at most ``k`` quantile bins."""
    v = list(values)
    if not v:
        return []
    order = sorted(range(len(v)), key=lambda i: v[i])
    bins = [0] * len(v)
    k = max(1, min(k, len(set(v))))
    for rank, i in enumerate(order):
        bins[i] = min(k - 1, rank * k // len(v))
    return bins


def importance(rows: Sequence[dict[str, Hashable]], values: Sequence[float],
               axes: Sequence[str], *, config_key: Callable[[dict], Hashable] | None = None
               ) -> dict[str, Any]:
    """Variance decomposition of trial-level goal values.

    ``rows[i]`` holds the axis labels of trial ``i`` (continuous labels should
    be binned first). Returns ``{"total", "axes": {axis: SS}, "seed": SS,
    "interaction": SS, "shares": {...}, "balanced": bool}``.
    """
    ys = [float(y) for y in values]
    n = len(ys)
    if n < 2:
        return {"total": 0.0, "axes": {}, "seed": 0.0, "interaction": 0.0, "shares": {},
                "balanced": True}
    grand = sum(ys) / n
    ss_total = sum((y - grand) ** 2 for y in ys)
    ss_axes: dict[str, float] = {}
    balanced = True
    for a in axes:
        groups: dict[Hashable, list[float]] = defaultdict(list)
        for r, y in zip(rows, ys):
            groups[r.get(a)].append(y)
        sizes = {len(g) for g in groups.values()}
        balanced = balanced and len(sizes) <= 1
        ss_axes[a] = sum(len(g) * (sum(g) / len(g) - grand) ** 2 for g in groups.values())
    key = config_key or (lambda r: tuple(r.get(a) for a in axes))
    configs: dict[Hashable, list[float]] = defaultdict(list)
    for r, y in zip(rows, ys):
        configs[key(r)].append(y)
    ss_seed = sum(sum((y - sum(g) / len(g)) ** 2 for y in g) for g in configs.values())
    ss_inter = max(0.0, ss_total - sum(ss_axes.values()) - ss_seed)
    shares: dict[str, float] = {}
    if ss_total > 0:
        shares = {a: s / ss_total for a, s in ss_axes.items()}
        shares["interaction"] = ss_inter / ss_total
        shares["seed noise"] = ss_seed / ss_total
        norm = sum(shares.values())
        if norm > 1.0 + 1e-9:  # unbalanced designs can over-attribute: renormalize
            shares = {k: v / norm for k, v in shares.items()}
    return {"total": ss_total, "axes": ss_axes, "seed": ss_seed, "interaction": ss_inter,
            "shares": shares, "balanced": balanced}


# ---------------------------------------------------------------- consistency


def ranks(values: Sequence[float], *, higher_is_better: bool = False) -> list[float]:
    """1-based ranks (best = 1), ties get the average rank."""
    order = sorted(range(len(values)), key=lambda i: (-values[i] if higher_is_better
                                                      else values[i]))
    r = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def kendall_w(rank_rows: Sequence[Sequence[float]]) -> float | None:
    """Kendall's coefficient of concordance for ``m`` rankings of ``n`` items."""
    m = len(rank_rows)
    if m < 2:
        return None
    n = len(rank_rows[0])
    if n < 2 or any(len(r) != n for r in rank_rows):
        return None
    totals = [sum(r[i] for r in rank_rows) for i in range(n)]
    mean_t = sum(totals) / n
    s = sum((t - mean_t) ** 2 for t in totals)
    return max(0.0, min(1.0, 12.0 * s / (m * m * (n ** 3 - n))))


# ---------------------------------------------------------------- pareto


def pareto_front(points: Sequence[tuple[float, float]], *, y_mode: str = "min") -> list[int]:
    """Indices of non-dominated ``(cost, goal)`` points (cost lower is better)."""
    idx = [i for i, (x, y) in enumerate(points)
           if x is not None and y is not None and math.isfinite(x) and math.isfinite(y)]
    sign = 1.0 if y_mode == "min" else -1.0
    out = []
    for i in idx:
        xi, yi = points[i][0], sign * points[i][1]
        dominated = False
        for j in idx:
            if j == i:
                continue
            xj, yj = points[j][0], sign * points[j][1]
            if xj <= xi and yj <= yi and (xj < xi or yj < yi):
                dominated = True
                break
        if not dominated:
            out.append(i)
    return sorted(out, key=lambda i: points[i][0])


def loglog_slope(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Least-squares slope of log(y) on log(x) (scaling exponent)."""
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys)
           if x and y and x > 0 and y > 0 and math.isfinite(x) and math.isfinite(y)]
    if len(pts) < 2:
        return None
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    if sxx == 0:
        return None
    return sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx


__all__ = ["betainc", "ci95", "cohens_d", "importance", "kendall_w", "loglog_slope", "mean",
           "pareto_front", "pooled_std", "quantile_bins", "ranks", "seeds_needed", "std",
           "t_cdf", "t_ppf", "t_two_sided_p", "welch_ttest"]
