"""Class-based generators (``Generator``), the ops context and raw outputs.

``Generator`` is a ``synth.Node`` whose ``generate(ctx, attrs)`` returns a
record, a list of records (fan-out), ``None`` or yields records;
``Raw(value, parents=[...], **lineage)`` attaches lineage to one output.
Function generators are ``@synth.generator`` (``synth.node``); records from
outputs: dicts pass through, attribute values fill schema fields the output
does not set (``default_assemble``).

``GenContext`` is what pipeline ops see (``ctx.rng`` seeded per record and op).
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from topologies.synth.core import Node
from topologies.synth.errors import SynthError
from topologies.synth.fingerprint import fp_canonical
from topologies.synth.rng import SynthRandom, derive_seed
from topologies.synth.schema import Schema

# ---------------------------------------------------------------- outputs


class Raw:
    """One generator output plus lineage (``parents``, ``group``, ``evidence`` and free
    keys)."""

    __slots__ = ("evidence", "group", "lineage", "parents", "value")

    def __init__(self, value: Any, *, parents: Sequence[str] | None = None,
                 group: str | None = None, evidence: str | None = None, **lineage: Any) -> None:
        self.value = value
        self.parents = list(parents or [])
        self.group = group
        self.evidence = evidence
        self.lineage = dict(lineage)

    def __repr__(self) -> str:
        return f"Raw({self.value!r})"


# ---------------------------------------------------------------- context


class GenContext:
    """What a generator call sees (see module docstring).

    ``rng`` is a ``SynthRandom`` seeded from ``(spec seed, record index, stage)``
    (``rng.torch`` is a matching ``torch.Generator``).
    """

    def __init__(self, *, runtime: Any = None, index: int = 0,
                 attrs: Mapping[str, Any] | None = None, seed: int = 0, stage: str = "gen",
                 rng: SynthRandom | None = None) -> None:
        self.runtime = runtime
        self.record_index = int(index)
        self.attrs: dict[str, Any] = dict(attrs or {})
        self.stage = stage
        self.spec_seed = int(seed)
        self.rng = rng or SynthRandom(derive_seed(seed, index, stage))
        self.logs: dict[str, Any] = {}

    @property
    def dir(self) -> Path | None:
        return getattr(self.runtime, "dir", None)

    def child(self, stage: str) -> GenContext:
        """Context for a sub-step with its own reproducible RNG."""
        c = GenContext(runtime=self.runtime, index=self.record_index, attrs=self.attrs,
                       seed=self.spec_seed, stage=f"{self.stage}/{stage}")
        c.logs = self.logs
        return c

    def log(self, **values: Any) -> None:
        """Free-form values stored in the record's lineage (``ds.explain(id)``)."""
        self.logs.update(values)

    def verify(self, record: Mapping[str, Any], verifier: Any) -> Any:
        """Run a verifier (callable or ``ops.verify`` op) on a record -> ``Verdict``."""
        from topologies.synth.verdict import as_verdict

        check = getattr(verifier, "check_sync", None)
        if callable(check):
            return check(dict(record), self)
        return as_verdict(verifier(dict(record), self))


# ---------------------------------------------------------------- generators


class Generator(Node):
    """Base class: implement ``generate(ctx, attrs)`` (``ctx.rng``, ``ctx.log``, ...)."""

    name: str = "generator"
    produces: Schema | None = None

    def generate(self, ctx: Any, attrs: dict[str, Any]) -> Any:
        raise NotImplementedError(f"{type(self).__name__}.generate(ctx, attrs)")

    def sample(self, ctx: Any) -> Any:
        out = self.generate(ctx, dict(ctx.attrs))
        if isinstance(out, Raw) and not out.parents and out.group is None:
            # simple lineage travels as logs, so the value is usable as a plain field too
            if out.lineage:
                ctx.log(**out.lineage)
            return out.value
        return out

    def node_kind(self) -> str:
        return self.name if isinstance(self.name, str) else super().node_kind()

    def config(self) -> dict[str, Any]:
        """JSON-able description (fingerprint + reports)."""
        state = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        return {"type": type(self).__name__, **fp_canonical(state)}

    def fingerprint_config(self) -> Any:
        return self.config()

    def generators(self) -> list[Generator]:
        return [self]


class Legacy(Generator):
    """A generator object that is not a node (its own ``sample`` / ``label`` methods mean
    something else, e.g. the SCM) used as a generator through ``generate(ctx, attrs)``."""

    def __init__(self, obj: Any) -> None:
        self.obj = obj
        self.name = getattr(obj, "name", None) or type(obj).__name__.lower()
        self.produces = getattr(obj, "produces", None)
        self.evidence = getattr(obj, "evidence", None)
        rows = getattr(obj, "rows", 1)
        self.rows = rows if isinstance(rows, int) else 1

    def params(self) -> dict[str, Any]:
        return {}

    def generate(self, ctx: Any, attrs: dict[str, Any]) -> Any:
        return self.obj.generate(ctx, attrs)

    def setup(self, env: Any) -> None:
        setup = getattr(self.obj, "setup", None)
        if callable(setup):
            setup(env)

    def fingerprint(self) -> str:
        return str(self.obj.fingerprint())

    def fingerprint_config(self) -> Any:
        return self.obj.fingerprint()

    def config(self) -> dict[str, Any]:
        return self.obj.config()


# ---------------------------------------------------------------- output handling


def collect(out: Any) -> list[Any]:
    """Outputs of one generator call (drains iterators)."""
    if out is None:
        return []
    if isinstance(out, (dict, Raw, str)):
        return [out]
    if isinstance(out, (list, tuple)):
        return list(out)
    if inspect.isgenerator(out) or isinstance(out, Iterable):
        return list(out)
    return [out]


def default_assemble(attrs: Mapping[str, Any], raw: Any, schema: Schema | None) -> dict:
    """Records from raw outputs: dicts pass through; a string fills the single text field
    (or ``text``); attribute values fill schema fields the output does not set."""
    if isinstance(raw, Mapping):
        rec = dict(raw)
    elif isinstance(raw, str):
        texts = schema.text_fields() if schema is not None else []
        rec = {texts[0] if texts else "text": raw}
    elif hasattr(raw, "model_dump"):
        rec = raw.model_dump(mode="json")
    else:
        raise SynthError(f"Generator output {type(raw).__name__} is not a record.",
                         stage="generator",
                         hint="A generator's value must be a dict of fields: wrap fields in "
                              "synth.record(...), or return {field: value} from the function.")
    if schema is not None:
        for name in schema.names:
            if name not in rec and name in attrs:
                rec[name] = attrs[name]
    return rec


__all__ = ["GenContext", "Generator", "Legacy", "Raw", "collect", "default_assemble"]
