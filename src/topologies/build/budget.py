"""Generic parameter-budget search: "give me ~15k parameters of this architecture".

Works for any constructor (library or user-defined): the caller (or the
architecture, via ``register_budget_knobs``) names integer knobs and their
candidate values; we measure real modules (built on the ``meta`` device, so
no memory is allocated) and return the closest valid configuration.
Architecture constraints stay in the constructor: candidates whose
construction raises ``ValueError`` are simply skipped.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import torch

from topologies.build.params import count_params_as, format_params, parse_param_count
from topologies.build.validation import ConfigError


@dataclass
class BudgetResult:
    kwargs: dict[str, Any]
    params: int
    target: int
    evaluated: int
    candidates: list[tuple[dict[str, Any], int]] = field(default_factory=list, repr=False)
    count: str = "total"

    @property
    def rel_error(self) -> float:
        return abs(self.params - self.target) / max(1, self.target)

    def __str__(self) -> str:
        knobs = ", ".join(f"{k}={v}" for k, v in self.kwargs.items())
        return (f"{format_params(self.params)} params (target {format_params(self.target)}, "
                f"{self.rel_error:+.1%} off) with {knobs}")


def measure_params(factory: Callable[..., Any], kwargs: dict[str, Any],
                   count: str | Callable[..., int] = "total") -> int | None:
    """Parameter count of ``factory(**kwargs)``; ``None`` for invalid configs.

    ``count`` is a counting rule (``"total"``, ``"trainable"``,
    ``"non_embedding"`` or ``fn(module) -> int``; see ``count_params_as``).
    ``TypeError`` (wrong argument names) is re-raised: that is a usage error,
    not an invalid point of the search space.
    """
    try:
        with torch.device("meta"):
            return count_params_as(factory(**kwargs), count)
    except (ValueError, AssertionError):
        return None
    except TypeError:
        raise
    except Exception:  # noqa: BLE001, S110 - constructor not meta-friendly: build on CPU
        pass
    try:
        return count_params_as(factory(**kwargs), count)
    except (ValueError, AssertionError):
        return None


def _search_sorted(
    values: list[int], evaluate: Callable[[int], int | None], target: int
) -> list[tuple[int, int]]:
    """Binary-search ``values`` (params assumed non-decreasing); returns probes."""
    probes: list[tuple[int, int]] = []
    lo, hi = 0, len(values) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        j, p = None, None
        for off in range(hi - lo + 1):
            for cand in (mid + off, mid - off) if off else (mid,):
                if lo <= cand <= hi:
                    got = evaluate(values[cand])
                    if got is not None:
                        j, p = cand, got
                        break
            if j is not None:
                break
        if j is None:
            break
        probes.append((values[j], p))
        if p < target:
            lo = max(j, mid) + 1
        elif p > target:
            hi = min(j, mid) - 1
        else:
            break
    return probes


def fit_budget(
    factory: Callable[..., Any],
    target_params: float | str,
    knobs: dict[str, Iterable[int]],
    *,
    fixed: dict[str, Any] | None = None,
    tolerance: float | None = 0.15,
    anchor: dict[str, Any] | None = None,
    max_grid: int = 4096,
    count: str | Callable[..., int] = "total",
) -> BudgetResult:
    """Find knob values whose built module is closest to ``target_params``.

    The knob with the most candidates is binary-searched (parameters are
    assumed to grow with it). ``anchor`` holds the preferred values of the
    other knobs (preset/default shape): they are tried first, and the full
    grid (capped by ``max_grid``) is swept only if that misses the
    tolerance; among in-tolerance results the one closest to the anchor
    shape wins, so budgets scale a model without distorting it. Raises ``ConfigError`` if the
    best candidate is further than ``tolerance`` (relative) from the target;
    pass ``tolerance=None`` to accept the closest regardless. ``count`` picks
    what is matched: ``"total"`` (default), ``"trainable"``, ``"non_embedding"``
    or a callable ``fn(module) -> int``.
    """
    target = parse_param_count(target_params)
    fixed = dict(fixed or {})
    space = {k: sorted({int(v) for v in vals}) for k, vals in knobs.items()}
    space = {k: v for k, v in space.items() if k not in fixed}
    name = getattr(factory, "__name__", repr(factory))
    if not space:
        raise ConfigError(
            f"No free knobs to search for {name}.",
            hint="pass search={'width': range(16, 1025, 16)} (or leave knobs out of the "
                 "explicit kwargs), or declare defaults with register_budget_knobs().",
        )
    empty = [k for k, v in space.items() if not v]
    if empty:
        raise ConfigError(f"Knobs {empty} have no candidate values.")
    primary = max(space, key=lambda k: len(space[k]))
    others = [k for k in space if k != primary]
    grid = list(itertools.product(*(space[k] for k in others)))
    if len(grid) > max_grid:
        raise ConfigError(
            f"Budget search grid too large ({len(grid)} combinations of {others}).",
            hint="narrow the candidate ranges of the secondary knobs.",
        )
    cache: dict[tuple, int | None] = {}
    results: list[tuple[dict[str, Any], int]] = []

    def sweep(combos: list[tuple]) -> None:
        for combo in combos:
            base = dict(zip(others, combo))

            def evaluate(v: int, base: dict = base) -> int | None:
                kw = {**fixed, **base, primary: v}
                key = tuple(sorted((k, repr(x)) for k, x in kw.items()))
                if key not in cache:
                    cache[key] = measure_params(factory, kw, count)
                return cache[key]

            for v, p in _search_sorted(space[primary], evaluate, target):
                results.append(({**base, primary: v}, p))

    anchor = {k: v for k, v in (anchor or {}).items() if k in others}

    def distance(kw: dict[str, Any]) -> float:
        import math

        return sum(abs(math.log(max(1, kw[k]) / max(1, anchor[k]))) for k in anchor)

    def within(r: tuple[dict[str, Any], int]) -> bool:
        return tolerance is None or abs(r[1] - target) / target <= tolerance

    if anchor:
        # Sweep combos in order of distance from the anchor shape; stop at the
        # first distance level that yields an in-tolerance configuration.
        def combo_distance(combo: tuple) -> float:
            return distance({**dict(zip(others, combo)), **{
                k: anchor[k] for k in anchor if k not in others}})

        levels: dict[float, list[tuple]] = {}
        for combo in grid:
            levels.setdefault(round(combo_distance(combo), 9), []).append(combo)
        for d in sorted(levels):
            sweep(levels[d])
            if any(within(r) for r in results):
                break
    else:
        sweep(grid)
    if not results:
        raise ConfigError(
            f"No valid configuration of {name} in the search space {space}.",
            hint="check fixed kwargs against the architecture's constraints "
                 "(e.g. dim divisible by heads).",
        )
    ok = [r for r in results if within(r)]
    if ok and anchor:
        best_kw, best_p = min(ok, key=lambda r: (distance(r[0]), abs(r[1] - target)))
    else:
        best_kw, best_p = min(results, key=lambda r: (abs(r[1] - target), distance(r[0])
                                                      if anchor else 0.0))
    res = BudgetResult(
        kwargs=best_kw, params=best_p, target=target, evaluated=len(cache),
        candidates=results,
        count=count if isinstance(count, str) else getattr(count, "__name__", "custom"),
    )
    if tolerance is not None and res.rel_error > tolerance:
        lo = max((r for r in results if r[1] <= target), key=lambda r: r[1], default=None)
        hi = min((r for r in results if r[1] >= target), key=lambda r: r[1], default=None)
        span = ", ".join(f"{format_params(r[1])} ({r[0]})" for r in (lo, hi) if r)
        raise ConfigError(
            f"Closest {name} to {format_params(target)} params is {res}.",
            hint=f"nearest candidates: {span}. Widen the knob ranges, change fixed kwargs, "
                 f"or raise tolerance (now {tolerance:.0%}).",
        )
    return res
