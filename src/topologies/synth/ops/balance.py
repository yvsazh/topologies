"""Class / attribute balancing of the final selection.

    synth.ops.balance("label")                              # equal count per label
    synth.ops.balance("format", "language")                 # equal per (format, language) cell
    synth.ops.balance("label", counts={"fraud": 500, "ok": 9500})
    synth.ops.balance("label", mode="downsample", max_ratio=3)   # cells may differ up to 3x
    synth.ops.balance("label", mode="upsample")             # under-filled cells repeat records

Balance is a *final* operator: it sees the accepted stream in raw-index order
and keeps a record while its cell is under quota. Generation continues until
every cell is filled; when the raw limit is reached first, ``quota`` /
``downsample`` fill the remaining slots from other cells and ``upsample``
repeats records of the under-filled cells (with a parent link).
Axes are record fields or attributes; their values come from the attribute
space / schema (``values={...}`` overrides) so quotas are known up front.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from topologies.synth.errors import SynthError, unknown
from topologies.synth.ops.base import FinalAggregate

MODES = ("quota", "downsample", "upsample")


def _k(v: Any) -> str:
    return v if isinstance(v, str) else json.dumps(v, sort_keys=True, default=str)


class Balance(FinalAggregate):
    def __init__(self, *axes: str, mode: str = "quota", max_ratio: float | None = None,
                 counts: Mapping[Any, int] | None = None,
                 values: Mapping[str, Sequence[Any]] | None = None,
                 name: str = "balance") -> None:
        if not axes:
            raise SynthError("balance() needs at least one axis.", stage="balance",
                             hint='synth.ops.balance("label")')
        if mode not in MODES:
            raise unknown("balance mode", mode, MODES, stage="balance")
        if mode == "downsample" and max_ratio is None:
            max_ratio = 1.0
        self.axes, self.mode, self.max_ratio = list(axes), mode, max_ratio
        self.counts = {(_k(k) if not isinstance(k, tuple) else tuple(_k(x) for x in k)): int(v)
                       for k, v in (counts or {}).items()} or None
        self.values = {k: list(v) for k, v in (values or {}).items()}
        self.name = name
        self._known: list[tuple] | None = None

    def setup(self, runtime: Any) -> None:
        spec = getattr(runtime, "spec", None) if runtime is not None else None
        known: list[list[str]] = []
        for a in self.axes:
            vals = self.values.get(a)
            if vals is None and spec is not None:
                space = getattr(spec, "attributes", None)
                if space is not None and a in space and space[a].kind == "categorical":
                    vals = space.values(a)
                schema = getattr(spec, "schema", None)
                if vals is None and schema is not None and a in schema and \
                        hasattr(schema[a], "names"):
                    vals = schema[a].names
            if vals is None:
                self._known = None
                return
            known.append([_k(v) for v in vals])
        import itertools

        self._known = list(itertools.product(*known))

    def cell(self, record: Mapping[str, Any], meta: Mapping[str, Any]) -> tuple:
        attrs = meta.get("attrs") or record.get("_attrs") or {}
        out = []
        for a in self.axes:
            v = record.get(a, attrs.get(a))
            out.append(_k(v))
        return tuple(out)

    def selector(self, target: int) -> _Selector:
        return _Selector(self, int(target))

    def config(self) -> dict[str, Any]:
        return {"op": self.name, "axes": self.axes, "mode": self.mode,
                "max_ratio": self.max_ratio,
                "counts": {str(k): v for k, v in (self.counts or {}).items()},
                "values": {k: [_k(x) for x in v] for k, v in self.values.items()}}

    def describe(self) -> str:
        return f"balance({', '.join(self.axes)}, {self.mode})"


class _Selector:
    def __init__(self, op: Balance, target: int) -> None:
        self.op, self.target = op, target
        self.counts: Counter = Counter()
        self.dropped: dict[tuple, list[tuple[int, str]]] = defaultdict(list)
        self.kept: dict[tuple, list[tuple[int, str]]] = defaultdict(list)
        self.seen_cells: list[tuple] = list(op._known or [])
        self.dups: list[tuple[str, int]] = []

    def quota(self, cell: tuple) -> float:
        op = self.op
        if op.counts:
            key = cell if len(cell) > 1 else cell[0]
            return float(op.counts.get(key, 0))
        n = max(1, len(self.seen_cells))
        q = math.ceil(self.target / n)
        if op.mode == "downsample" and op.max_ratio:
            q = math.ceil(q * float(op.max_ratio))
        return q

    def offer(self, record: dict, meta: dict) -> bool:
        cell = self.op.cell(record, meta)
        if cell not in self.seen_cells:
            self.seen_cells.append(cell)
        ref = (int(meta.get("chunk", 0)), str(meta.get("id")))
        if self.counts[cell] < self.quota(cell):
            self.counts[cell] += 1
            self.kept[cell].append(ref)
            return True
        self.dropped[cell].append(ref)
        return False

    def relax(self) -> list[tuple[int, str]]:
        """Raw budget exhausted before every cell filled (see module docstring)."""
        if self.op.mode == "upsample":
            missing = {c: int(self.quota(c) - self.counts[c]) for c in self.seen_cells
                       if self.counts[c] < self.quota(c) and self.kept[c]}
            for c, k in missing.items():
                src = self.kept[c]
                for j in range(k):
                    self.dups.append((src[j % len(src)][1], j // len(src) + 1))
            return []
        out: list[tuple[int, str]] = []
        # Fill from the largest surplus cells first, round robin, in raw order.
        queues = {c: list(v) for c, v in self.dropped.items() if v}
        while queues:
            for c in sorted(queues, key=lambda c: -len(queues[c])):
                out.append(queues[c].pop(0))
                if not queues[c]:
                    del queues[c]
        return out


def balance(*axes: str, **kw: Any) -> Balance:
    return Balance(*axes, **kw)


__all__ = ["MODES", "Balance", "balance"]
