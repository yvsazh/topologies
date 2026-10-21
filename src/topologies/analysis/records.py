"""Generic record collection comparisons and multi-objective helpers."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

from topologies.analysis.core import RecordSet, field_value


def near_optimal(records: Sequence[Mapping[str, Any]], objective: str, *, mode: str = "min",
                 relative: float = 0.0, absolute: float = 0.0) -> RecordSet:
    """Retain rows within either tolerance of the best finite objective.

    Relative tolerance is scaled by ``abs(best)``; absolute tolerance therefore
    remains meaningful around zero and negative objectives.
    """
    if mode not in ("min", "max"):
        raise ValueError("mode must be 'min' or 'max'")
    for value, label in ((relative, "relative"), (absolute, "absolute")):
        if isinstance(value, bool) or not isinstance(value, Real) or not __import__("math").isfinite(value) or value < 0:
            raise ValueError(f"{label} tolerance must be a finite non-negative real number")
    valid = []
    for row in records:
        value = field_value(row, objective)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            import math
            if math.isfinite(value):
                valid.append((row, float(value)))
    if not valid:
        return RecordSet()
    best = (min if mode == "min" else max)(v for _, v in valid)
    tolerance = absolute + relative * abs(best)
    return RecordSet(row for row, value in valid if abs(value - best) <= tolerance)


def pareto_front(records: Sequence[Mapping[str, Any]], objectives: Mapping[str, str]
                 ) -> RecordSet:
    """Return nondominated finite rows; objective modes are ``min`` or ``max``."""
    if not objectives:
        raise ValueError("at least one objective is required")
    if any(mode not in ("min", "max") for mode in objectives.values()):
        raise ValueError("objective modes must be 'min' or 'max'")
    valid = []
    for row in records:
        vals = [field_value(row, key) for key in objectives]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) and
               __import__("math").isfinite(v) for v in vals):
            valid.append((row, vals))
    front = []
    for i, (row, vals) in enumerate(valid):
        dominated = False
        for j, (_, other) in enumerate(valid):
            if i == j:
                continue
            at_least_as_good = all((b <= a if mode == "min" else b >= a)
                                   for a, b, mode in zip(vals, other, objectives.values()))
            strictly_better = any((b < a if mode == "min" else b > a)
                                  for a, b, mode in zip(vals, other, objectives.values()))
            if at_least_as_good and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(row)
    return RecordSet(front)


def correlation(records: Sequence[Mapping[str, Any]], x: str, y: str) -> float | None:
    """Pearson correlation of paired finite numeric fields; None when undefined."""
    import math
    pairs = []
    for row in records:
        a, b = field_value(row, x), field_value(row, y)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and \
                not isinstance(a, bool) and not isinstance(b, bool) and \
                math.isfinite(a) and math.isfinite(b):
            pairs.append((float(a), float(b)))
    if len(pairs) < 2:
        return None
    mx = sum(a for a, _ in pairs) / len(pairs)
    my = sum(b for _, b in pairs) / len(pairs)
    xx = sum((a-mx)**2 for a, _ in pairs)
    yy = sum((b-my)**2 for _, b in pairs)
    if xx == 0 or yy == 0:
        return None
    return sum((a-mx)*(b-my) for a, b in pairs) / (xx*yy)**.5


def normalize(records: Sequence[Mapping[str, Any]], field: str, *, mode: str = "minmax",
              name: str | None = None) -> RecordSet:
    """Return copied rows with finite numeric field normalized to [0,1] or z scores."""
    import math
    if mode not in ("minmax", "zscore"):
        raise ValueError("mode must be 'minmax' or 'zscore'")
    vals = [float(r[field]) for r in records]
    if not vals or not all(math.isfinite(v) for v in vals):
        raise ValueError("normalization requires non-empty finite numeric values")
    mean = sum(vals)/len(vals)
    scale = max(vals)-min(vals) if mode == "minmax" else (
        sum((v-mean)**2 for v in vals)/len(vals))**.5
    key = name or f"{field}_{mode}"
    return RecordSet({**row, key: ((float(row[field])-min(vals))/scale if mode == "minmax"
                                  else (float(row[field])-mean)/scale) if scale else 0.0}
                     for row in records)


def compare(baseline: Mapping[str, Any], candidate: Mapping[str, Any],
            metrics: Sequence[str], *, provenance: Mapping[str, Any] | None = None) -> RecordSet:
    """Return one comparison row per metric with absolute and safe relative deltas."""
    rows = []
    for metric in metrics:
        before, after = field_value(baseline, metric), field_value(candidate, metric)
        delta = float(after) - float(before) if _numeric(before) and _numeric(after) else None
        relative = delta / abs(float(before)) if delta is not None and float(before) != 0 else None
        rows.append({"metric": metric, "baseline": before, "candidate": after,
                     "absolute_delta": delta, "relative_delta": relative,
                     "provenance": dict(provenance or {})})
    return RecordSet(rows)


def _numeric(value: Any) -> bool:
    import math
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
