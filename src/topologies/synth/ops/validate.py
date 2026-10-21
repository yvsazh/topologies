"""Explicit schema + constraint validation.

    synth.ops.validate()                                    # the spec schema (coerced)
    synth.ops.validate(schema, constraints={"total": (0, 10_000),
                                            "currency": ["EUR", "USD"],
                                            "text": lambda v: "http" not in v})
    synth.ops.validate(constraints=[lambda r: r["total"] == sum(r["items"])])

Every spec already validates against its schema first (op ``schema``); add
``validate`` for extra constraints or a stricter schema later in the pipeline.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from topologies.synth.fingerprint import fp_canonical
from topologies.synth.ops.base import MapOp, Reject
from topologies.synth.verdict import as_verdict


class Validate(MapOp):
    """Coerce + validate (a map op so coercions stick); rejects with the first problems."""

    evidence = "heuristic"

    def __init__(self, schema: Any = None, *, constraints: Any = None,
                 name: str = "validate") -> None:
        self.schema = schema
        self.constraints = constraints
        self.name = name
        self._schema = schema

    def setup(self, runtime: Any) -> None:
        if self._schema is None and runtime is not None:
            self._schema = getattr(runtime.spec, "schema", None)

    def _field_check(self, field: str, rule: Any, record: dict) -> str | None:
        if field not in record:
            return f"missing {field!r}"
        v = record[field]
        if isinstance(rule, tuple) and len(rule) == 2:
            lo, hi = rule
            try:
                if (lo is not None and v < lo) or (hi is not None and v > hi):
                    return f"{field}={v!r} outside [{lo}, {hi}]"
            except TypeError:
                return f"{field}={v!r} is not comparable"
        elif isinstance(rule, (list, set, frozenset)):
            if v not in rule:
                return f"{field}={v!r} not in {sorted(map(str, rule))[:10]}"
        elif callable(rule):
            ver = as_verdict(rule(v))
            if not ver.ok:
                return ver.reason or f"{field} fails its constraint"
        return None

    def apply(self, record: dict, ctx: Any) -> dict:
        rec = {k: v for k, v in record.items() if not str(k).startswith("_")}
        schema = self._schema
        if schema is not None:
            rec = schema.coerce(rec)
            errs = schema.validate(rec)
            if errs:
                raise Reject("; ".join(errs[:3]))
        c = self.constraints
        if isinstance(c, Mapping):
            for f, rule in c.items():
                why = self._field_check(f, rule, rec)
                if why:
                    raise Reject(why)
        elif isinstance(c, Sequence):
            for rule in c:
                ver = as_verdict(rule(rec))
                if not ver.ok:
                    raise Reject(ver.reason or f"constraint {getattr(rule, '__name__', '?')} "
                                               "failed")
        elif callable(c):
            ver = as_verdict(c(rec))
            if not ver.ok:
                raise Reject(ver.reason or "constraint failed")
        return rec

    def config(self) -> dict[str, Any]:
        return {"op": self.name, "schema": self.schema.to_config() if self.schema is not None
                else None, "constraints": fp_canonical(self.constraints)}


def validate(schema: Any = None, *, constraints: Mapping[str, Any] | Sequence[Callable] |
             Callable | None = None, name: str = "validate") -> Validate:
    return Validate(schema, constraints=constraints, name=name)


__all__ = ["Validate", "validate"]
