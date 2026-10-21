"""Pipeline operators: ``map`` / ``filter`` / ``flat_map`` / ``aggregate``.

Per-record ops (map, filter, flat_map) run inside a chunk; *streaming*
aggregates (dedup, contamination) see records in raw-index order with
incremental state; *final* aggregates (balance, split) always run last, over
the accepted stream. Every op has a ``name`` and a ``fingerprint()``.

Level 2: wrap a function as an op:

    synth.ops.map(lambda r, ctx: {**r, "text": r["text"].strip()})
    synth.ops.filter(lambda r, ctx: len(r["text"]) > 20, name="long-enough")
    synth.ops.flat_map(lambda r, ctx: [r, {**r, "text": r["text"].lower()}])
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any

from topologies.synth.ctor import Recorded
from topologies.synth.errors import SynthError
from topologies.synth.fingerprint import callable_fingerprint, fp_canonical
from topologies.synth.verdict import Verdict, as_verdict


class Reject(Exception):
    """Raised by a map op to reject the record (``raise Reject("reason")``)."""

    def __init__(self, reason: str = "", score: float | None = None, **evidence: Any) -> None:
        super().__init__(reason)
        self.verdict = Verdict(False, score=score, reason=reason, evidence=evidence)


class Op(Recorded):
    """Base operator (see module docstring)."""

    kind: str = "map"
    name: str = "op"
    #: Evidence kind a record gains by passing this op (``verifier``, ``heuristic``, ...).
    evidence: str | None = None

    def setup(self, runtime: Any) -> None:
        """Called once per run (load data, build indexes)."""

    def config(self) -> dict[str, Any]:
        state = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        return {"op": type(self).__name__, **fp_canonical(state)}

    def fingerprint(self) -> str:
        from topologies.experiments.cache import stable_hash

        return stable_hash(self.config())

    def describe(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"<op {self.kind}:{self.name}>"


class MapOp(Op):
    kind = "map"

    def apply(self, record: dict[str, Any], ctx: Any) -> dict[str, Any]:
        raise NotImplementedError


class FilterOp(Op):
    kind = "filter"

    def check(self, record: dict[str, Any], ctx: Any) -> Verdict | bool:
        raise NotImplementedError

    def check_sync(self, record: dict[str, Any], ctx: Any) -> Verdict:
        return as_verdict(self.check(record, ctx))


class FlatMapOp(Op):
    kind = "flat_map"
    #: Children inherit the parent's labels; set ``relabel`` ops after to change that.
    inherit: bool = True

    def expand(self, record: dict[str, Any], ctx: Any) -> list[dict[str, Any]]:
        raise NotImplementedError


class Aggregate(Op):
    """Set-level op. ``streaming=True``: ``offer`` decides per record in index order and
    ``observe`` replays accepted records (resume). ``streaming=False`` (final):
    ``select`` runs over the accepted stream at the end."""

    kind = "aggregate"
    streaming: bool = True

    def start(self) -> Any:
        """Fresh incremental state."""
        return None

    def offer(self, state: Any, record: dict[str, Any], meta: dict[str, Any]) -> Verdict:
        """Accept/reject ``record`` and update ``state`` when accepted."""
        raise NotImplementedError

    def observe(self, state: Any, record: dict[str, Any], meta: dict[str, Any]) -> None:
        """Add an already-accepted record to ``state`` (replaying finished chunks)."""
        self.offer(state, record, meta)


class FinalAggregate(Aggregate):
    """Runs after every per-record op (balance, split): sees the accepted stream."""

    streaming = False

    def selector(self, target: int) -> Any:
        """Object with ``offer(record, meta) -> bool`` (streaming quota selection) and
        ``relax()`` -> records to add when the raw budget is exhausted. Default: keep all."""
        return _KeepAll()


class _KeepAll:
    def offer(self, record: dict, meta: dict) -> bool:
        return True

    def done(self) -> bool:
        return False

    def relax(self) -> list:
        return []


# ---------------------------------------------------------------- function ops


def _arity(fn: Callable[..., Any]) -> int:
    try:
        return len([p for p in inspect.signature(fn).parameters.values()
                    if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
    except (TypeError, ValueError):
        return 2


class FnMap(MapOp):
    def __init__(self, fn: Callable[..., Any], *, name: str | None = None) -> None:
        self.fn, self.name = fn, name or getattr(fn, "__name__", "map")
        self._two = _arity(fn) >= 2

    def apply(self, record: dict[str, Any], ctx: Any) -> Any:
        return self.fn(record, ctx) if self._two else self.fn(record)

    def config(self) -> dict[str, Any]:
        return {"op": "map", "fn": callable_fingerprint(self.fn), "name": self.name}


class FnFilter(FilterOp):
    def __init__(self, fn: Callable[..., Any], *, name: str | None = None,
                 evidence: str | None = None) -> None:
        self.fn, self.name = fn, name or getattr(fn, "__name__", "filter")
        if self.name == "<lambda>":
            self.name = "filter"
        self.evidence = evidence
        self._two = _arity(fn) >= 2

    def check(self, record: dict[str, Any], ctx: Any) -> Any:
        return self.fn(record, ctx) if self._two else self.fn(record)

    def config(self) -> dict[str, Any]:
        return {"op": "filter", "fn": callable_fingerprint(self.fn), "name": self.name,
                "evidence": self.evidence}


class FnFlatMap(FlatMapOp):
    def __init__(self, fn: Callable[..., Any], *, name: str | None = None,
                 inherit: bool = True) -> None:
        self.fn, self.name = fn, name or getattr(fn, "__name__", "flat_map")
        self.inherit = bool(inherit)
        self._two = _arity(fn) >= 2

    def expand(self, record: dict[str, Any], ctx: Any) -> Any:
        return self.fn(record, ctx) if self._two else self.fn(record)

    def config(self) -> dict[str, Any]:
        return {"op": "flat_map", "fn": callable_fingerprint(self.fn), "name": self.name}


def map(fn: Callable[..., Any], *, name: str | None = None) -> MapOp:
    """``fn(record[, ctx]) -> record`` (raise ``Reject`` to drop it)."""
    return FnMap(fn, name=name)


def filter(fn: Callable[..., Any], *, name: str | None = None,
           evidence: str | None = None) -> FilterOp:
    """``fn(record[, ctx]) -> bool | Verdict``."""
    return FnFilter(fn, name=name, evidence=evidence)


def flat_map(fn: Callable[..., Any], *, name: str | None = None,
             inherit: bool = True) -> FlatMapOp:
    """``fn(record[, ctx]) -> list[record]`` (children keep a parent link)."""
    return FnFlatMap(fn, name=name, inherit=inherit)


def as_op(obj: Any) -> Op:
    """An ``Op``; a plain function becomes a filter (``fn(record[, ctx]) -> bool|Verdict``)."""
    if isinstance(obj, Op):
        return obj
    to_op = getattr(obj, "as_op", None)
    if callable(to_op) and not isinstance(obj, type):
        return to_op()
    if callable(obj) and not isinstance(obj, type):
        return FnFilter(obj)
    raise SynthError(f"Cannot use {obj!r} as a pipeline op.", stage="pipeline",
                     hint="Use synth.ops.* operators, synth.ops.filter(fn), synth.ops.map(fn) "
                          "or a function fn(record, ctx) -> bool | Verdict.")


def unique_names(ops: list[Op]) -> list[Op]:
    """Make op names unique (``dedup.near``, ``dedup.near#2``) for funnel stats."""
    seen: dict[str, int] = {}
    for op in ops:
        base = op.name
        if base in seen:
            seen[base] += 1
            op.name = f"{base}#{seen[base]}"
        else:
            seen[base] = 1
    return ops


def describe_ops(ops: list[Op]) -> list[Mapping[str, Any]]:
    return [{"name": op.name, "kind": op.kind, "evidence": op.evidence,
             "streaming": getattr(op, "streaming", None)} for op in ops]


__all__ = ["Aggregate", "FilterOp", "FinalAggregate", "FlatMapOp", "MapOp", "Op", "Reject",
           "as_op", "describe_ops", "filter", "flat_map", "map", "unique_names"]
