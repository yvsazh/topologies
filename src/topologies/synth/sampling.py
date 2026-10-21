"""Coverage-aware samplers: which attribute assignment record ``i`` gets.

Every sampler builds a *design* in blocks: block ``b`` covers raw indices
``[b*B, (b+1)*B)`` and is constructed from ``(seed, b)`` only, so record
``i``'s attributes never depend on the dataset size or on how many records
were generated before (resumes, top-ups and scaling curves see nested
prefixes of one infinite design).

- ``random``: iid draws (respects weights).
- ``stratified(axes)``: equal counts per cell of the chosen categorical axes.
- ``quota(axis, counts)``: exact per-cell counts (default: proportional to weights).
- ``pairwise(k=1)``: every pair of values of the categorical attributes appears at
  least ``k`` times per block (greedy all-pairs covering design); the rest of the
  block is drawn by weight.
- ``lhs``: Latin hypercube over the continuous attributes.
- ``adaptive(factors)``: weights multiplied per value (loop / pilot acceptance).
- ``auto``: quota on the label attribute + pairwise on the other categoricals (>= 4
  of them, else stratified-by-weight random) + LHS on continuous attributes.

Constraints: rows violating ``where`` are redrawn (attributes outside the
covering columns first); ``require=[...]`` rows are placed at the start of
every block.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any

from topologies.synth.attributes import AttributeSpace
from topologies.synth.ctor import Recorded
from topologies.synth.errors import SynthError, unknown
from topologies.synth.rng import SynthRandom, derive_seed

LABEL_NAMES = ("label", "labels", "intent", "class", "target", "category")


def _largest_remainder(probs: Sequence[float], n: int) -> list[int]:
    exact = [p * n for p in probs]
    counts = [math.floor(x) for x in exact]
    rest = n - sum(counts)
    order = sorted(range(len(probs)), key=lambda i: (exact[i] - counts[i], -i), reverse=True)
    for i in order[:rest]:
        counts[i] += 1
    return counts


class Sampler(Recorded):
    """Base sampler: ``assign(i)`` -> sampled (categorical + continuous) attributes."""

    name = "random"

    def __init__(self, *, block: int | None = None) -> None:
        self.block_arg = block
        self.space: AttributeSpace | None = None
        self.seed = 0
        self._blocks: dict[int, list[dict[str, Any]]] = {}

    # -- binding -----------------------------------------------------------
    def bind(self, space: AttributeSpace, seed: int) -> Sampler:
        """A copy bound to ``space`` and ``seed`` (samplers are reusable templates)."""
        out = type(self).__new__(type(self))
        out.__dict__.update(self.__dict__)
        out.space, out.seed, out._blocks = space, int(seed), {}
        out.validate()
        return out

    def validate(self) -> None:
        return None

    def block_size(self) -> int:
        return int(self.block_arg or 256)

    def describe(self) -> dict[str, Any]:
        return {"sampler": self.name, "block": self.block_arg}

    # -- design ----------------------------------------------------------------
    def assign(self, i: int) -> dict[str, Any]:
        b, j = divmod(int(i), self.block_size())
        rows = self._blocks.get(b)
        if rows is None:
            rows = self._build_checked(b)
            self._blocks[b] = rows
            while len(self._blocks) > 8:
                self._blocks.pop(next(iter(self._blocks)))
        return dict(rows[j])

    def _build_checked(self, b: int) -> list[dict[str, Any]]:
        space = self.space
        assert space is not None
        rows = self.build_block(b)
        n = self.block_size()
        required = list(space.require)
        for k, req in enumerate(required[: n // 2 or 1]):
            rows[k] = {**rows[k], **req}
        if space.where is not None:
            fixed_cols = self.fixed_columns()
            for j, row in enumerate(rows):
                if self._ok(row):
                    continue
                rng = SynthRandom(derive_seed(self.seed, "where", b, j))
                for attempt in range(200):
                    free = space.sample_free(rng)
                    keep = fixed_cols if attempt < 100 else set()
                    cand = {**free, **{k: row[k] for k in keep if k in row}}
                    if self._ok(cand):
                        rows[j] = cand
                        break
                else:
                    raise SynthError(
                        "No attribute combination satisfies where=... after 200 draws.",
                        stage="sampling", hint="Relax the where= constraint or remove "
                                               "attribute values it always rejects.")
        return rows

    def _ok(self, row: Mapping[str, Any]) -> bool:
        assert self.space is not None
        return self.space.allowed(row)

    def fixed_columns(self) -> set[str]:
        """Columns with coverage guarantees (kept when redrawing ``where`` violations)."""
        return set()

    def build_block(self, b: int) -> list[dict[str, Any]]:
        assert self.space is not None
        rng = SynthRandom(derive_seed(self.seed, "random", b))
        return [self.space.sample_free(rng) for _ in range(self.block_size())]


class RandomSampler(Sampler):
    name = "random"


class StratifiedSampler(Sampler):
    """Equal counts per cell of ``axes`` (default: every categorical attribute)."""

    name = "stratified"

    def __init__(self, axes: Sequence[str] | None = None, *, block: int | None = None) -> None:
        super().__init__(block=block)
        self.axes = list(axes) if axes is not None else None

    def validate(self) -> None:
        assert self.space is not None
        axes = self.axes if self.axes is not None else self.space.categorical
        for a in axes:
            if a not in self.space:
                raise unknown("attribute", a, self.space.names, stage="sampling")
            if self.space[a].kind != "categorical":
                raise SynthError(f"stratified axis {a!r} is not categorical.", stage="sampling")
        self._axes = list(axes)
        self._cells = list(itertools.product(*(self.space.values(a) for a in self._axes)))
        if len(self._cells) > 100_000:
            raise SynthError(f"stratified sampling over {len(self._cells)} cells is too many.",
                             stage="sampling", hint="Stratify fewer axes, or use "
                                                    "sampler='pairwise'.")

    def block_size(self) -> int:
        n = len(self._cells) or 1
        base = int(self.block_arg or max(256, n))
        return max(n, (base // n) * n)

    def fixed_columns(self) -> set[str]:
        return set(self._axes)

    def build_block(self, b: int) -> list[dict[str, Any]]:
        assert self.space is not None
        rng = SynthRandom(derive_seed(self.seed, "stratified", b))
        n = self.block_size()
        cells = [self._cells[k % len(self._cells)] for k in range(n)]
        rng.shuffle(cells)
        rows = []
        for cell in cells:
            row = self.space.sample_free(rng)
            row.update(dict(zip(self._axes, cell)))
            rows.append(row)
        return rows

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "axes": self.axes}


class QuotaSampler(Sampler):
    """Exact counts per value of ``axis`` in every block (``counts`` or the weights)."""

    name = "quota"

    def __init__(self, axis: str | None = None, counts: Mapping[Any, float] | None = None, *,
                 block: int | None = None) -> None:
        super().__init__(block=block)
        self.axis = axis
        self.counts = dict(counts) if counts is not None else None

    def validate(self) -> None:
        assert self.space is not None
        axis = self.axis or _label_axis(self.space)
        if axis is None:
            raise SynthError("quota sampling needs an axis.", stage="sampling",
                             hint="sampler=synth.quota('label') or name an attribute 'label'.")
        if axis not in self.space:
            raise unknown("attribute", axis, self.space.names, stage="sampling")
        if self.space[axis].kind != "categorical":
            raise SynthError(f"quota axis {axis!r} is not categorical.", stage="sampling")
        self._axis = axis
        values = self.space.values(axis)
        if self.counts is not None:
            unknown_keys = [k for k in self.counts if k not in values]
            if unknown_keys:
                raise unknown("quota value", unknown_keys[0], [str(v) for v in values],
                              stage="sampling")
            weights = [float(self.counts.get(v, 0)) for v in values]
        else:
            weights = self.space[axis].probs  # type: ignore[attr-defined]
        total = sum(weights) or 1.0
        self._values, self._probs = values, [w / total for w in weights]

    def fixed_columns(self) -> set[str]:
        return {self._axis}

    def column(self, b: int, n: int) -> list[Any]:
        rng = SynthRandom(derive_seed(self.seed, "quota", self._axis, b))
        counts = _largest_remainder(self._probs, n)
        col = [v for v, c in zip(self._values, counts) for _ in range(c)]
        rng.shuffle(col)
        return col

    def build_block(self, b: int) -> list[dict[str, Any]]:
        assert self.space is not None
        rng = SynthRandom(derive_seed(self.seed, "quota-rows", b))
        n = self.block_size()
        col = self.column(b, n)
        rows = []
        for v in col:
            row = self.space.sample_free(rng)
            row[self._axis] = v
            rows.append(row)
        return rows

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "axis": self.axis, "counts": self.counts}


def covering_design(levels: Sequence[int], *, strength: int = 1, rng: SynthRandom,
                    candidates: int = 30) -> list[tuple[int, ...]]:
    """Greedy all-pairs covering array: rows over ``levels`` (value counts per factor)
    such that every pair of values of every two factors appears >= ``strength`` times."""
    k = len(levels)
    if k < 2:
        return [tuple(rng.randrange(n) for n in levels)]
    need: dict[tuple[int, int, int, int], int] = {}
    for a, b in itertools.combinations(range(k), 2):
        for va in range(levels[a]):
            for vb in range(levels[b]):
                need[(a, b, va, vb)] = strength
    rows: list[tuple[int, ...]] = []
    pairs = list(itertools.combinations(range(k), 2))
    remaining = len(need) * strength
    while remaining > 0:
        best_row, best_gain = None, -1
        # Seed candidates from an uncovered pair so every row makes progress.
        uncovered = next(key for key, c in need.items() if c > 0)
        for _ in range(candidates):
            row = [rng.randrange(n) for n in levels]
            row[uncovered[0]], row[uncovered[1]] = uncovered[2], uncovered[3]
            order = [f for f in range(k) if f not in uncovered[:2]]
            rng.shuffle(order)
            for f in order:  # greedy value choice per factor
                best_v, best_g = row[f], -1
                start = rng.randrange(levels[f])
                for dv in range(levels[f]):
                    v = (start + dv) % levels[f]
                    g = 0
                    for g2 in range(k):
                        if g2 == f:
                            continue
                        a, b = (f, g2) if f < g2 else (g2, f)
                        va, vb = (v, row[g2]) if f < g2 else (row[g2], v)
                        if need[(a, b, va, vb)] > 0:
                            g += 1
                    if g > best_g:
                        best_v, best_g = v, g
                row[f] = best_v
            gain = sum(1 for a, b in pairs if need[(a, b, row[a], row[b])] > 0)
            if gain > best_gain:
                best_row, best_gain = tuple(row), gain
        assert best_row is not None
        for a, b in pairs:
            key = (a, b, best_row[a], best_row[b])
            if need[key] > 0:
                need[key] -= 1
                remaining -= 1
        rows.append(best_row)
    return rows


class PairwiseSampler(Sampler):
    """All-pairs coverage (``k`` times) over ``axes`` (default: all categoricals)."""

    name = "pairwise"

    def __init__(self, axes: Sequence[str] | None = None, *, k: int = 1,
                 block: int | None = None) -> None:
        super().__init__(block=block)
        self.axes = list(axes) if axes is not None else None
        self.k = int(k)

    def validate(self) -> None:
        assert self.space is not None
        axes = self.axes if self.axes is not None else self.space.categorical
        for a in axes:
            if a not in self.space:
                raise unknown("attribute", a, self.space.names, stage="sampling")
            if self.space[a].kind != "categorical":
                raise SynthError(f"pairwise axis {a!r} is not categorical.", stage="sampling")
        self._axes = list(axes)
        self._design_cache: dict[int, list[tuple[int, ...]]] = {}

    def design(self, b: int) -> list[tuple[int, ...]]:
        if b not in self._design_cache:
            assert self.space is not None
            levels = [len(self.space.values(a)) for a in self._axes]
            rng = SynthRandom(derive_seed(self.seed, "pairwise", b))
            self._design_cache = {b: covering_design(levels, strength=self.k, rng=rng)}
        return self._design_cache[b]

    def block_size(self) -> int:
        rows = len(self.design(0)) if self._axes else 0
        return max(int(self.block_arg or 256), rows)

    def fixed_columns(self) -> set[str]:
        return set(self._axes)

    def rows_for(self, b: int, n: int, rng: SynthRandom) -> list[dict[str, Any]]:
        """``n`` rows for the covering columns: the design first, weighted draws after."""
        assert self.space is not None
        values = [self.space.values(a) for a in self._axes]
        out: list[dict[str, Any]] = []
        for row in self.design(b)[:n]:
            out.append({a: values[f][row[f]] for f, a in enumerate(self._axes)})
        while len(out) < n:
            out.append({a: self.space[a].sample(rng, {}) for a in self._axes})
        rng.shuffle(out)
        return out

    def build_block(self, b: int) -> list[dict[str, Any]]:
        assert self.space is not None
        rng = SynthRandom(derive_seed(self.seed, "pairwise-rows", b))
        n = self.block_size()
        cols = self.rows_for(b, n, rng)
        rows = []
        for c in cols:
            row = self.space.sample_free(rng)
            row.update(c)
            rows.append(row)
        return rows

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "axes": self.axes, "k": self.k}


class LHSSampler(Sampler):
    """Latin hypercube over ``axes`` (default: every continuous attribute)."""

    name = "lhs"

    def __init__(self, axes: Sequence[str] | None = None, *, block: int | None = None) -> None:
        super().__init__(block=block)
        self.axes = list(axes) if axes is not None else None

    def validate(self) -> None:
        assert self.space is not None
        axes = self.axes if self.axes is not None else self.space.continuous
        for a in axes:
            if a not in self.space:
                raise unknown("attribute", a, self.space.names, stage="sampling")
            if self.space[a].kind != "continuous":
                raise SynthError(f"lhs axis {a!r} is not continuous.", stage="sampling")
        self._axes = list(axes)

    def fixed_columns(self) -> set[str]:
        return set(self._axes)

    def columns(self, b: int, n: int) -> dict[str, list[Any]]:
        assert self.space is not None
        out = {}
        for a in self._axes:
            rng = SynthRandom(derive_seed(self.seed, "lhs", a, b))
            strata = list(range(n))
            rng.shuffle(strata)
            attr = self.space[a]
            out[a] = [attr.ppf((s + rng.random()) / n) for s in strata]  # type: ignore
        return out

    def build_block(self, b: int) -> list[dict[str, Any]]:
        assert self.space is not None
        rng = SynthRandom(derive_seed(self.seed, "lhs-rows", b))
        n = self.block_size()
        cols = self.columns(b, n)
        rows = []
        for j in range(n):
            row = self.space.sample_free(rng)
            for a, col in cols.items():
                row[a] = col[j]
            rows.append(row)
        return rows

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "axes": self.axes}


class AdaptiveSampler(Sampler):
    """Weights multiplied per value (``{"tone": {"angry": 3.0}}``) on top of ``base``."""

    name = "adaptive"

    def __init__(self, factors: Mapping[str, Mapping[Any, float]] | None = None,
                 base: Sampler | str = "auto", *, block: int | None = None) -> None:
        super().__init__(block=block)
        self.factors = {k: dict(v) for k, v in (factors or {}).items()}
        self.base = base

    def bind(self, space: AttributeSpace, seed: int) -> Sampler:
        for a in self.factors:
            if a not in space:
                raise unknown("attribute", a, space.names, stage="sampling")
        inner = make_sampler(self.base, space)
        bound = inner.bind(space.reweighted(self.factors), seed)
        return _Wrapped(bound, self.describe())

    def describe(self) -> dict[str, Any]:
        base = self.base if isinstance(self.base, str) else self.base.describe()
        return {"sampler": "adaptive", "factors": {k: {str(v): w for v, w in f.items()}
                                                   for k, f in self.factors.items()},
                "base": base}


class _Wrapped(Sampler):
    def __init__(self, inner: Sampler, desc: dict[str, Any]) -> None:
        super().__init__()
        self.inner, self._desc = inner, desc
        self.space, self.seed = inner.space, inner.seed
        self.name = desc.get("sampler", "adaptive")

    def assign(self, i: int) -> dict[str, Any]:
        return self.inner.assign(i)

    def block_size(self) -> int:
        return self.inner.block_size()

    def describe(self) -> dict[str, Any]:
        return self._desc


class AutoSampler(Sampler):
    """Quota on the label + pairwise on the other categoricals + LHS on continuous."""

    name = "auto"

    def __init__(self, label: str | None = None, *, pairwise_min: int = 4, k: int = 1,
                 max_levels: int = 24, block: int | None = None) -> None:
        super().__init__(block=block)
        self.label, self.pairwise_min, self.k = label, int(pairwise_min), int(k)
        self.max_levels = int(max_levels)

    def validate(self) -> None:
        assert self.space is not None
        space = self.space
        label = self.label or _label_axis(space)
        if label is not None and (label not in space or space[label].kind != "categorical"
                                  or not getattr(space[label], "ready", True)):
            label = None
        self._label = label
        # High-cardinality attributes (pools, long lists) are drawn by weight: covering
        # every pair of 200 personas x 12 labels would dominate the block.
        rest = [a for a in space.categorical if a != label and getattr(space[a], "ready", True)
                and len(space.values(a)) <= self.max_levels]
        self._pair_axes = rest if len(rest) >= max(2, self.pairwise_min) else []
        self._quota = QuotaSampler(label).bind(space, self.seed) if label else None
        self._pair = PairwiseSampler(self._pair_axes, k=self.k).bind(space, self.seed) \
            if self._pair_axes else None
        self._lhs = LHSSampler().bind(space, self.seed) if space.continuous else None

    def block_size(self) -> int:
        n = int(self.block_arg or 256)
        if self._pair is not None:
            n = max(n, len(self._pair.design(0)))
        if self._label is not None:
            n = max(n, len(self.space.values(self._label)))  # type: ignore[union-attr]
        return n

    def fixed_columns(self) -> set[str]:
        out = set(self._pair_axes) | set(self.space.continuous)  # type: ignore[union-attr]
        if self._label:
            out.add(self._label)
        return out

    def build_block(self, b: int) -> list[dict[str, Any]]:
        assert self.space is not None
        n = self.block_size()
        rng = SynthRandom(derive_seed(self.seed, "auto-rows", b))
        label_col = self._quota.column(b, n) if self._quota is not None else None
        pair_rows = self._pair.rows_for(b, n, rng) if self._pair is not None else None
        lhs_cols = self._lhs.columns(b, n) if self._lhs is not None else {}
        rows = []
        for j in range(n):
            row = self.space.sample_free(rng)
            if label_col is not None:
                row[self._label] = label_col[j]  # type: ignore[index]
            if pair_rows is not None:
                row.update(pair_rows[j])
            for a, col in lhs_cols.items():
                row[a] = col[j]
            rows.append(row)
        return rows

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "label": self.label, "pairwise_min": self.pairwise_min,
                "k": self.k, "max_levels": self.max_levels}


def _label_axis(space: AttributeSpace) -> str | None:
    for name in LABEL_NAMES:
        if name in space and space[name].kind == "categorical":
            return name
    return None


SAMPLERS: dict[str, type[Sampler]] = {
    "random": RandomSampler, "stratified": StratifiedSampler, "quota": QuotaSampler,
    "pairwise": PairwiseSampler, "lhs": LHSSampler, "auto": AutoSampler,
    "adaptive": AdaptiveSampler,
}


def make_sampler(spec: str | Sampler | None, space: AttributeSpace | None = None) -> Sampler:
    """``"auto"`` / ``"pairwise"`` / a ``Sampler`` instance -> an (unbound) sampler."""
    if spec is None:
        return AutoSampler()
    if isinstance(spec, Sampler):
        return spec
    if isinstance(spec, str):
        cls = SAMPLERS.get(spec)
        if cls is None:
            raise unknown("sampler", spec, SAMPLERS, stage="sampling")
        return cls()
    raise SynthError(f"Invalid sampler {spec!r}.", stage="sampling",
                     hint="Use 'auto', 'random', 'stratified', 'pairwise', 'quota', 'lhs' or a "
                          "Sampler instance such as synth.quota('label').")


# Convenience constructors (``sampler=synth.quota("label")``).
def random_sampler(**kw: Any) -> Sampler:
    return RandomSampler(**kw)


def stratified(axes: Sequence[str] | None = None, **kw: Any) -> Sampler:
    return StratifiedSampler(axes, **kw)


def quota(axis: str | None = None, counts: Mapping[Any, float] | None = None,
          **kw: Any) -> Sampler:
    return QuotaSampler(axis, counts, **kw)


def pairwise(axes: Sequence[str] | None = None, *, k: int = 1, **kw: Any) -> Sampler:
    return PairwiseSampler(axes, k=k, **kw)


def lhs(axes: Sequence[str] | None = None, **kw: Any) -> Sampler:
    return LHSSampler(axes, **kw)


def adaptive(factors: Mapping[str, Mapping[Any, float]] | None = None,
             base: Sampler | str = "auto", **kw: Any) -> Sampler:
    return AdaptiveSampler(factors, base, **kw)


def pair_coverage(rows: Sequence[Mapping[str, Any]], space: AttributeSpace,
                  axes: Sequence[str] | None = None) -> dict[str, Any]:
    """``{"pairs": total, "covered": n, "min_count": m, "missing": [...]}`` over ``rows``."""
    axes = list(axes) if axes is not None else space.categorical
    counts: dict[tuple, int] = {}
    for r in rows:
        for a, b in itertools.combinations(axes, 2):
            key = (a, b, _k(r.get(a)), _k(r.get(b)))
            counts[key] = counts.get(key, 0) + 1
    total, missing, min_count = 0, [], None
    for a, b in itertools.combinations(axes, 2):
        for va in space.values(a):
            for vb in space.values(b):
                total += 1
                c = counts.get((a, b, _k(va), _k(vb)), 0)
                min_count = c if min_count is None else min(min_count, c)
                if c == 0:
                    missing.append({a: va, b: vb})
    return {"pairs": total, "covered": total - len(missing), "min_count": min_count or 0,
            "missing": missing}


def _k(v: Any) -> Any:
    try:
        hash(v)
        return v
    except TypeError:
        return repr(v)


__all__ = ["SAMPLERS", "AdaptiveSampler", "AutoSampler", "LHSSampler", "PairwiseSampler",
           "QuotaSampler", "RandomSampler", "Sampler", "StratifiedSampler", "adaptive",
           "covering_design", "lhs", "make_sampler", "pair_coverage", "pairwise", "quota",
           "random_sampler", "stratified"]
