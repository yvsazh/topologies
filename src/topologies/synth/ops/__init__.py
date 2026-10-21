"""Pipeline operators (see ``ops.base``).

    synth.ops.filters.length(min_chars=40)        # also synth.ops.filter.length(...)
    synth.ops.filter(fn)                          # fn(record, ctx) -> bool | Verdict
    synth.ops.map(fn); synth.ops.flat_map(fn)
    synth.ops.parse.json(); synth.ops.validate(schema)
    synth.ops.dedup.exact("text"); synth.ops.dedup.near(threshold=0.85)
    synth.ops.verify(fn); synth.ops.verify.round_trip(parser, tolerance="fields>=0.95")
    synth.ops.label.program(fn); synth.ops.label.vote([...])
    synth.ops.balance("format", "language"); synth.ops.split({...}, group="parent")
    synth.ops.contamination.check(reference, ngram=13, action="drop")

Names that are both callable and namespaces (``filter``, ``verify``, ``label``,
``dedup``, ``parse``, ``augment``) dispatch calls to
their default operator and attribute access to the submodule.
"""

from __future__ import annotations

import importlib
from typing import Any

from topologies.synth.ops.base import (
    Aggregate,
    FilterOp,
    FinalAggregate,
    FlatMapOp,
    MapOp,
    Op,
    Reject,
    as_op,
    flat_map,
)
from topologies.synth.ops.base import map as map
from topologies.synth.ops.split import SplitSpec, split

_P = "topologies.synth.ops."
#: Callable submodules (``ops.verify(fn)`` / ``ops.verify.round_trip(...)``).
_MODULES = {"filter": _P + "filters", "filters": _P + "filters", "verify": _P + "verify",
            "label": _P + "label", "dedup": _P + "dedup", "parse": _P + "parse",
            "contamination": _P + "contamination",
            "augment": _P + "augment"}

_LAZY = {
    "validate": (_P + "validate", "validate"),
    "balance": (_P + "balance", "balance"),
}


def __getattr__(name: str) -> Any:
    if name in _MODULES:
        value = importlib.import_module(_MODULES[name])
        globals()[name] = value
        return value
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'topologies.synth.ops' has no attribute {name!r}")
    value = getattr(importlib.import_module(target[0]), target[1])
    globals()[name] = value
    return value


__all__ = ["Aggregate", "FilterOp", "FinalAggregate", "FlatMapOp", "MapOp",
           "Op", "Reject", "SplitSpec", "as_op", "augment", "balance", "contamination", "dedup",
           "filter", "filters", "flat_map", "label", "map", "parse", "split", "validate",
           "verify"]
