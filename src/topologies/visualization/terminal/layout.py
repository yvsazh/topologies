"""Tiny constraint-free layout: rectangles, row/column splits and a panel grid.

Panels declare ``min_width``, ``min_height``, ``preferred_height`` and ``priority``.
:func:`grid_layout` picks one, two or three columns by width (≥ 100 two, ≥ 140 three),
drops panels by ascending priority until the rest meet their minimum heights, and
never returns a rectangle outside the given area.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    def inset(self, dx: int = 0, dy: int = 0) -> Rect:
        return Rect(self.x + dx, self.y + dy, max(0, self.w - 2 * dx), max(0, self.h - 2 * dy))


@dataclass(frozen=True)
class Fixed:
    size: int


@dataclass(frozen=True)
class Min:
    size: int


@dataclass(frozen=True)
class Ratio:
    weight: float = 1.0


Constraint = Fixed | Min | Ratio


def _split(total: int, constraints: Sequence[Constraint]) -> list[int]:
    sizes = [0] * len(constraints)
    remaining = total
    for i, c in enumerate(constraints):
        if isinstance(c, (Fixed, Min)):
            sizes[i] = min(max(0, c.size), max(0, remaining))
            remaining -= sizes[i]
    flex = [i for i, c in enumerate(constraints) if isinstance(c, (Ratio, Min))]
    weights = [constraints[i].weight if isinstance(constraints[i], Ratio) else 1.0
               for i in flex]
    total_w = sum(weights) or 1.0
    if remaining > 0 and flex:
        given = 0
        for k, i in enumerate(flex):
            share = int(remaining * weights[k] / total_w) if k < len(flex) - 1 \
                else remaining - given
            sizes[i] += share
            given += share
    return sizes


def split_rows(rect: Rect, constraints: Sequence[Constraint]) -> list[Rect]:
    """Split ``rect`` vertically; Fixed/Min first, Ratio/Min share the remainder."""
    out, y = [], rect.y
    for size in _split(rect.h, constraints):
        out.append(Rect(rect.x, y, rect.w, size))
        y += size
    return out


def split_cols(rect: Rect, constraints: Sequence[Constraint]) -> list[Rect]:
    """Split ``rect`` horizontally; Fixed/Min first, Ratio/Min share the remainder."""
    out, x = [], rect.x
    for size in _split(rect.w, constraints):
        out.append(Rect(x, rect.y, size, rect.h))
        x += size
    return out


def columns_for_width(width: int) -> int:
    """Responsive column count: ≥ 140 → 3, ≥ 100 → 2, else 1."""
    return 3 if width >= 140 else (2 if width >= 100 else 1)


def _attr(panel: Any, name: str, default: int) -> int:
    try:
        return int(getattr(panel, name, default) or default)
    except (TypeError, ValueError):
        return default


def grid_layout(panels: Sequence[Any], area: Rect, *, columns: int | None = None
                ) -> list[tuple[Any, Rect]]:
    """Assign rectangles to ``panels`` (in order) filling ``area`` column by column.

    Panels are distributed round-robin over the columns; within a column the height is
    shared by ``preferred_height`` weights. Lowest-priority panels are dropped until
    every remaining panel gets at least its ``min_height`` and ``min_width``.
    """
    if area.w <= 0 or area.h <= 0:
        return []
    ncols = columns or columns_for_width(area.w)
    active = list(panels)
    while active:
        ncols_eff = max(1, min(ncols, len(active)))
        col_rects = split_cols(area, [Ratio(1)] * ncols_eff)
        buckets: list[list[Any]] = [[] for _ in range(ncols_eff)]
        for i, p in enumerate(active):
            buckets[i % ncols_eff].append(p)
        ok = True
        placed: list[tuple[Any, Rect]] = []
        for rect, bucket in zip(col_rects, buckets):
            need = sum(_attr(p, "min_height", 3) for p in bucket)
            if need > rect.h or any(_attr(p, "min_width", 10) > rect.w for p in bucket):
                ok = False
                break
            # Guarantee minimums, then share the rest by preferred height.
            mins = [_attr(p, "min_height", 3) for p in bucket]
            extra = rect.h - sum(mins)
            prefs = [max(0, _attr(p, "preferred_height", m) - m) for p, m in zip(bucket, mins)]
            total_pref = sum(prefs)
            heights = list(mins)
            if extra > 0:
                if total_pref:
                    given = 0
                    for k in range(len(bucket)):
                        add = min(prefs[k], extra * prefs[k] // total_pref)
                        heights[k] += add
                        given += add
                    left = extra - given
                else:
                    left = extra
                heights[-1] += left
            y = rect.y
            for p, hgt in zip(bucket, heights):
                placed.append((p, Rect(rect.x, y, rect.w, hgt)))
                y += hgt
        if ok:
            order = {id(p): i for i, p in enumerate(active)}
            return sorted(placed, key=lambda item: order[id(item[0])])
        if ncols > 1 and ncols_eff > 1 and len(active) <= ncols_eff:
            ncols -= 1
            continue
        drop = min(range(len(active)), key=lambda i: (_attr(active[i], "priority", 50), -i))
        active.pop(drop)
    return []


__all__ = ["Constraint", "Fixed", "Min", "Ratio", "Rect", "columns_for_width", "grid_layout",
           "split_cols", "split_rows"]
