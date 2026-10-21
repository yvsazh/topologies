"""Train/val/test splits: random, stratified, group-aware, attribute-held-out, length.

    splits={"train": 0.9, "val": 0.1}                               # random, exact counts
    splits=synth.ops.split({"train": 0.8, "test": 0.2}, stratify="label")
    splits=synth.ops.split({"train": 0.9, "test": 0.1}, group="parent")   # augmentations of one
                                                                          # source stay together
    splits=synth.ops.split({"train": 1.0}, holdout={"persona": 0.2})      # unseen personas -> test
    splits=synth.ops.split({"train": 1.0}, holdout={"topic": ["billing"]})
    splits=synth.ops.split(by_length={"field": "tokens", "train_max": 64})  # length generalization

Assignment is by a stable hash rank of the group key (the record id when no
group is set), so it is deterministic, independent of record order, and a
group never straddles two splits. Exact counts come from largest-remainder
rounding over the selected records.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from topologies.synth.errors import SynthError
from topologies.synth.ops.base import FinalAggregate


def _rank(key: str, salt: str) -> float:
    h = hashlib.blake2b(f"{salt}:{key}".encode("utf-8", "surrogatepass"), digest_size=8)
    return int.from_bytes(h.digest(), "little") / 2**64


def _k(v: Any) -> str:
    return v if isinstance(v, str) else json.dumps(v, sort_keys=True, default=str)


class SplitSpec(FinalAggregate):
    """How accepted records are divided into named splits (see module docstring)."""

    name = "split"

    def __init__(self, fractions: Mapping[str, float] | None = None, *,
                 stratify: str | None = None, group: str | None = None,
                 holdout: Mapping[str, Any] | None = None, holdout_split: str = "test",
                 by_length: Mapping[str, Any] | None = None, seed: int = 0) -> None:
        fr = dict(fractions or {"train": 1.0})
        if not fr:
            raise SynthError("splits needs at least one split.", stage="split")
        if any((not isinstance(v, (int, float))) or v < 0 for v in fr.values()):
            raise SynthError(f"Split fractions must be numbers >= 0; got {fr}.", stage="split")
        total = sum(fr.values())
        if total <= 0:
            raise SynthError("Split fractions sum to 0.", stage="split")
        if abs(total - 1.0) > 1e-6:
            if total > 1.0 + 1e-6 and all(v >= 1 for v in fr.values()):
                pass  # counts (normalized below)
            elif total > 1.0 + 1e-6:
                raise SynthError(f"Split fractions sum to {total:g} > 1.", stage="split",
                                 hint="Use fractions that sum to 1, e.g. "
                                      "{'train': 0.9, 'val': 0.1}.")
        self.fractions = {str(k): float(v) / total for k, v in fr.items()}
        self.stratify, self.group = stratify, group
        self.holdout = dict(holdout or {})
        self.holdout_split = holdout_split
        self.by_length = dict(by_length) if by_length else None
        self.seed = int(seed)
        if self.by_length is not None and "field" not in self.by_length:
            raise SynthError("by_length needs a 'field'.", stage="split",
                             hint="split(by_length={'field': 'tokens', 'train_max': 64})")

    @property
    def names(self) -> list[str]:
        names = list(self.fractions)
        if (self.holdout or self.by_length) and self.holdout_split not in names:
            names.append(self.holdout_split)
        return names

    # -- helpers -------------------------------------------------------------------
    def group_key(self, rec: Mapping[str, Any], meta: Mapping[str, Any]) -> str:
        if self.group is None:
            return str(meta.get("group") or meta.get("id") or rec.get("_id"))
        if self.group in ("parent", "group"):
            return str(meta.get("group") or meta.get("id"))
        v = rec.get(self.group, (rec.get("_attrs") or {}).get(self.group, meta.get("id")))
        return _k(v)

    def _holdout_values(self, attr: str, spec: Any, values: Sequence[Any]) -> set[str]:
        distinct = sorted({_k(v) for v in values})
        if isinstance(spec, (int, float)) and not isinstance(spec, bool):
            k = max(1, round(len(distinct) * float(spec))) if spec < 1 else int(spec)
            ranked = sorted(distinct, key=lambda v: _rank(v, f"holdout:{attr}:{self.seed}"))
            return set(ranked[:k])
        if isinstance(spec, (list, tuple, set)):
            return {_k(v) for v in spec}
        return {_k(spec)}

    def _length(self, rec: Mapping[str, Any]) -> int:
        v = rec.get(self.by_length["field"])  # type: ignore[index]
        try:
            return len(v)  # type: ignore[arg-type]
        except TypeError:
            return int(v or 0)

    def assign(self, items: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> list[str]:
        """Split name per ``(record, meta)`` item (meta: ``id``, ``group``)."""
        out: list[str | None] = [None] * len(items)
        rest: list[int] = []
        held_values: dict[str, set[str]] = {}
        for attr, spec in self.holdout.items():
            vals = [(r.get("_attrs") or {}).get(attr, r.get(attr)) for r, _ in items]
            held_values[attr] = self._holdout_values(attr, spec, vals)
        for i, (r, _m) in enumerate(items):
            if self.by_length is not None:
                n = self._length(r)
                lo = self.by_length.get("train_min", 0)
                hi = self.by_length.get("train_max")
                if hi is not None and n > hi or n < lo:
                    out[i] = self.holdout_split
                    continue
            held = False
            for attr, vals in held_values.items():
                v = (r.get("_attrs") or {}).get(attr, r.get(attr))
                if _k(v) in vals:
                    held = True
                    break
            if held:
                out[i] = self.holdout_split
            else:
                rest.append(i)
        # Groups -> strata -> hash rank -> exact counts per split.
        groups: dict[str, list[int]] = defaultdict(list)
        for i in rest:
            groups[self.group_key(*items[i])].append(i)
        strata: dict[str, list[str]] = defaultdict(list)
        for g, idx in groups.items():
            s = "" if self.stratify is None else _k(
                items[idx[0]][0].get(self.stratify,
                                     (items[idx[0]][0].get("_attrs") or {}).get(self.stratify)))
            strata[s].append(g)
        names = list(self.fractions)
        for s in sorted(strata):
            gs = sorted(strata[s], key=lambda g: (_rank(g, f"split:{self.seed}"), g))
            sizes = [len(groups[g]) for g in gs]
            total = sum(sizes)
            targets = _counts([self.fractions[n] for n in names], total)
            k, filled = 0, 0
            bounds = []
            acc = 0
            for t in targets:
                acc += t
                bounds.append(acc)
            for g, size in zip(gs, sizes):
                while k < len(names) - 1 and filled >= bounds[k]:
                    k += 1
                for i in groups[g]:
                    out[i] = names[k]
                filled += size
        return [o or names[0] for o in out]

    def config(self) -> dict[str, Any]:
        return {"op": "split", "fractions": self.fractions, "stratify": self.stratify,
                "group": self.group, "holdout": {k: v if not isinstance(v, set) else sorted(v)
                                                 for k, v in self.holdout.items()},
                "holdout_split": self.holdout_split, "by_length": self.by_length,
                "seed": self.seed}

    def describe(self) -> str:
        parts = [", ".join(f"{k} {v:.0%}" for k, v in self.fractions.items())]
        if self.stratify:
            parts.append(f"stratified by {self.stratify}")
        if self.group:
            parts.append(f"grouped by {self.group}")
        if self.holdout:
            parts.append(f"held out {self.holdout} -> {self.holdout_split}")
        if self.by_length:
            parts.append(f"length split on {self.by_length['field']}")
        return "; ".join(parts)


def _counts(probs: Sequence[float], n: int) -> list[int]:
    exact = [p * n for p in probs]
    counts = [int(x) for x in exact]
    rest = n - sum(counts)
    order = sorted(range(len(probs)), key=lambda i: (exact[i] - counts[i], -i), reverse=True)
    for i in order[:rest]:
        counts[i] += 1
    return counts


def split(fractions: Mapping[str, float] | None = None, **kw: Any) -> SplitSpec:
    """``synth.ops.split({"train": 0.9, "val": 0.1}, stratify=..., group=..., holdout=...)``."""
    return SplitSpec(fractions, **kw)


def as_split(value: Any) -> SplitSpec:
    if value is None:
        return SplitSpec({"train": 1.0})
    if isinstance(value, SplitSpec):
        return value
    if isinstance(value, Mapping):
        return SplitSpec(dict(value))
    if isinstance(value, (list, tuple)):
        return SplitSpec({str(n): 1.0 / len(value) for n in value})
    raise SynthError(f"Invalid splits={value!r}.", stage="split",
                     hint="splits={'train': 0.9, 'val': 0.1} or synth.ops.split(...).")


def leakage(assignments: Sequence[str], keys: Sequence[str]) -> list[str]:
    """Group keys that appear in more than one split (should be empty)."""
    seen: dict[str, set[str]] = defaultdict(set)
    for s, k in zip(assignments, keys):
        seen[k].add(s)
    return sorted(k for k, v in seen.items() if len(v) > 1)


__all__ = ["SplitSpec", "as_split", "leakage", "split"]
