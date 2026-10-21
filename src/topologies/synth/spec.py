"""``Spec``: what a sample is, what varies, how it is produced and checked.

    spec = synth.Spec(
        "invoice-extraction",
        schema=synth.Schema(input=Text(), output=Json(model=Invoice)),
        attributes={"format": ["email", "ocr-scan", "chat"]},
        generator=synth.record(input=render_invoice(invoice), output=invoice),
        pipeline=[synth.ops.filters.length(min_chars=40), synth.ops.dedup.near(0.85)],
        size=50_000, splits={"train": 0.95, "val": 0.05}, seed=0, max_time="30m",
    )
    spec.plan(pilot=200)    # validate, preview, throughput / disk projection
    ds = spec.run()         # chunked, resumable; returns a SynthDataset

The generator is any record node (``synth.record(...)``, a ``{field: node}`` dict,
a ``@synth.generator`` function or a node whose value is a dict).

Fingerprints: ``gen_fingerprint()`` identifies raw generation (seed,
attributes, sampler, generator graph, schema); record ids derive from it.
``fingerprint()`` (the spec hash) adds the pipeline and splits. The size,
limits, root, workers and labels do not change record content.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from topologies.experiments.cache import stable_hash
from topologies.synth.attributes import AttributeSpace
from topologies.synth.core import Node, as_generator, iter_nodes
from topologies.synth.errors import SynthError, unknown
from topologies.synth.fingerprint import fp_canonical
from topologies.synth.limits import parse_count, parse_time
from topologies.synth.ops.base import Op, as_op, unique_names
from topologies.synth.ops.split import SplitSpec, as_split
from topologies.synth.records import as_records, fingerprint_records
from topologies.synth.sampling import Sampler, make_sampler
from topologies.synth.schema import Schema, schema_of

_CHOICES = {"on_error": ("skip", "raise"), "monitor": ("auto", "live", "lines", "none")}


def _check(name: str, value: Any) -> None:
    allowed = _CHOICES[name]
    if value not in allowed:
        raise unknown(f"{name} value", str(value), allowed, stage="spec")


class Spec:
    """A synthetic dataset definition (see module docstring and §4.5 of docs/synth.md)."""

    def __init__(
        self,
        name: str,
        *,
        schema: Schema | Mapping[str, Any] | None = None,
        attributes: Mapping[str, Any] | AttributeSpace | None = None,
        sampler: str | Sampler = "auto",
        generator: Any = None,
        blueprint: Any = None,
        pipeline: Sequence[Any] = (),
        size: int | Mapping[str, int] | None = None,
        oversample: float | str = "auto",
        splits: Mapping[str, float] | SplitSpec | None = None,
        reference: Any = None,
        seed: int = 0,
        root: str | Path | None = None,
        workers: int | str = "auto",
        on_error: str = "skip",
        version: str | None = None,
        meta: Mapping[str, Any] | None = None,
        chunk_size: int | str = "auto",
        compress: bool = False,
        max_raw: int | None = None,
        max_time: Any = None,
        monitor: str = "auto",
        where: Callable[[dict], bool] | None = None,
        require: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        self._kwargs = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        if not isinstance(name, str) or not name.strip():
            raise SynthError("Spec needs a non-empty name.", stage="spec",
                             hint='synth.Spec("support-intents", generator=..., size=1000)')
        if blueprint is not None:
            base = _blueprint_spec(blueprint, name)
            explicit = {k: v for k, v in self._kwargs.items()
                        if v is not None and k not in ("blueprint", "name") and _differs(k, v)}
            merged = {**base._kwargs, **explicit, "name": name}
            merged["blueprint"] = None
            self.__init__(**merged)  # type: ignore[misc]
            self._kwargs["blueprint"] = blueprint
            if getattr(base, "_from_blueprint", None) is not None:
                self._from_blueprint = base._from_blueprint
                self._bp_overrides = {**getattr(base, "_bp_overrides", {}), **explicit}
            return
        if generator is None:
            raise SynthError(f"Spec {name!r} has no generator.", stage="spec",
                             hint="Pass generator=<a @synth.generator function>, or start "
                                  "from a blueprint: synth.blueprints.tabular(...).")
        for k, v in (("on_error", on_error), ("monitor", monitor)):
            _check(k, v)
        self.name = name.strip()
        self.generator: Node = as_generator(generator)
        self.attributes = attributes if isinstance(attributes, AttributeSpace) and \
            where is None and require is None else AttributeSpace(attributes, where=where,
                                                                 require=require)
        self.sampler_spec = make_sampler(sampler, self.attributes)
        self.seed = int(seed)
        self.sampler: Sampler = self.sampler_spec.bind(self.attributes, self.seed)
        if schema is not None and not isinstance(schema, Schema):
            schema = Schema.from_config(schema) if isinstance(schema, Mapping) else None
            if schema is None:
                raise SynthError("schema must be a synth.Schema or a {field: config} dict.",
                                 stage="spec")
        self.schema: Schema | None = schema if schema is not None else \
            schema_of(self.generator)
        self.schema_given = self.schema is not None
        self.pipeline: list[Op] = unique_names([as_op(o) for o in (pipeline or ())])
        self._check_pipeline()
        self.size_spec = size
        self.size, split_from_size = _resolve_size(size)
        if split_from_size is not None:
            if splits is not None:
                raise SynthError("Give the split sizes either in size={...} or in splits=..., "
                                 "not both.", stage="spec")
            splits = split_from_size
        self.splits: SplitSpec = as_split(splits)
        if oversample != "auto" and not (isinstance(oversample, (int, float))
                                         and oversample >= 1):
            raise SynthError(f"oversample={oversample!r} must be 'auto' or a number >= 1.",
                             stage="spec")
        self.oversample = oversample
        self.max_raw = int(parse_count(max_raw)) if max_raw else None
        self.max_time = parse_time(max_time)
        self.reference = as_records(reference, what="reference") if reference is not None \
            else None
        self.root = root
        if workers != "auto" and not (isinstance(workers, int) and workers >= 1):
            raise SynthError(f"workers={workers!r} must be 'auto' or an int >= 1.", stage="spec")
        self.workers = workers
        self.on_error = on_error
        self.version_label = version
        self.meta = dict(meta or {})
        self.chunk_size = self._resolve_chunk(chunk_size)
        self.compress = bool(compress)
        self.monitor = monitor

    # -- validation ------------------------------------------------------------
    def _resolve_chunk(self, chunk_size: int | str) -> int:
        """``"auto"``: 256 records per chunk; batch generators (``rows`` records per call,
        e.g. the SCM) get ~size/(8*rows)."""
        if chunk_size == "auto":
            per_call = max((getattr(g, "rows", 1) if isinstance(getattr(g, "rows", 1), int)
                            else 1 for g in iter_nodes(self.generator)), default=1)
            return 256 if per_call <= 1 else max(1, min(256, -(-self.size // (per_call * 8))))
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise SynthError("chunk_size must be a positive int or 'auto'.", stage="spec")
        return chunk_size

    def _check_pipeline(self) -> None:
        finals = [i for i, op in enumerate(self.pipeline)
                  if op.kind == "aggregate" and not getattr(op, "streaming", True)]
        for i in finals:
            if any(j > i for j, op in enumerate(self.pipeline)
                   if not (op.kind == "aggregate" and not getattr(op, "streaming", True))):
                op = self.pipeline[i]
                raise SynthError(f"{op.name} is a final operator (it needs the whole accepted "
                                 "set) but per-record ops follow it.", stage="spec",
                                 hint=f"Move {op.name} to the end of the pipeline.")
        if any(isinstance(op, SplitSpec) for op in self.pipeline):
            raise SynthError("Splits are configured with Spec(splits=...), not as a pipeline op.",
                             stage="spec", hint="splits=synth.ops.split({...}, group=...)")

    # -- fingerprints ------------------------------------------------------------
    def gen_fingerprint(self) -> str:
        """Identity of raw generation (record ids derive from it)."""
        return stable_hash({
            "seed": self.seed, "attributes": self.attributes.describe(),
            "sampler": self.sampler_spec.describe(),
            "generator": self.generator.fingerprint(),
            "schema": self.schema.to_config() if self.schema is not None else None})

    def stream_fingerprint(self) -> str:
        """Generation + streaming (per-record / incremental) ops: identity of chunk files."""
        return stable_hash({"gen": self.gen_fingerprint(),
                            "ops": [op.fingerprint() for op in self.stream_ops()],
                            "chunk": self.chunk_size,
                            "reference": fingerprint_records(self.reference)})

    def fingerprint(self) -> str:
        """The spec hash: generation + pipeline + splits."""
        return stable_hash({"stream": self.stream_fingerprint(),
                            "final": [op.fingerprint() for op in self.final_ops()],
                            "splits": self.splits.fingerprint()})

    def version_id(self) -> str:
        """Content-addressed id of the version a complete ``run()`` produces."""
        return stable_hash({"spec": self.fingerprint(), "size": self.size,
                            "max_raw": self.max_raw}, n=12)

    def stream_ops(self) -> list[Op]:
        return [op for op in self.pipeline
                if not (op.kind == "aggregate" and not getattr(op, "streaming", True))]

    def final_ops(self) -> list[Op]:
        return [op for op in self.pipeline
                if op.kind == "aggregate" and not getattr(op, "streaming", True)]

    # -- derived settings -----------------------------------------------------------
    def raw_limit(self) -> int:
        if self.max_raw:
            return self.max_raw
        ratio = float(self.oversample) if self.oversample != "auto" else 1.0
        return max(int(self.size * ratio * 20), self.size + 10 * self.chunk_size, 1000)

    def store(self) -> Any:
        from topologies.synth.store import SynthStore

        return SynthStore(self.root, self.name)

    def describe(self) -> dict[str, Any]:
        """JSON-able description (what ``spec.json`` stores)."""
        return {
            "spec_hash": self.fingerprint(), "gen_fingerprint": self.gen_fingerprint(),
            "seed": self.seed, "size": self.size_spec if self.size_spec is not None else
            self.size, "attributes": self.attributes.describe(),
            "sampler": self.sampler_spec.describe(), "generator": self.generator.node_kind(),
            "generator_tree": self.generator.describe(),
            "schema": self.schema.to_config() if self.schema is not None else None,
            "pipeline": [{"name": op.name, "kind": op.kind, "fp": op.fingerprint()}
                         for op in self.pipeline],
            "splits": self.splits.config(), "max_time": self.max_time,
            "max_raw": self.max_raw, "reference": len(self.reference or []),
            "version_label": self.version_label, "meta": fp_canonical(self.meta),
            "chunk_size": self.chunk_size}

    # -- copies --------------------------------------------------------------------------
    def with_(self, **changes: Any) -> Spec:
        """Copy with overrides; dotted keys reach inside parts (``generator.depth``,
        ``ops.dedup.near.threshold``, ``attributes.tone``, ``mix.real``)."""
        from topologies.synth.overrides import apply_overrides

        return apply_overrides(self, changes)

    def _copy_with(self, **kwargs: Any) -> Spec:
        bp = getattr(self, "_from_blueprint", None)
        if bp is not None:  # stay a blueprint + overrides (compact YAML, same content)
            over = {**getattr(self, "_bp_overrides", {}), **kwargs}
            name = over.pop("name", self.name)
            return type(self)(name, blueprint=dict(bp), **over)
        args = dict(self._kwargs)
        args.update(kwargs)
        return type(self)(**args)

    def copy(self) -> Spec:
        return copy.copy(self)

    # -- actions ---------------------------------------------------------------------------
    def plan(self, pilot: int = 200, *, preview: int = 3, verbose: bool | None = None) -> Any:
        """Validate, preview and measure before generating (see ``synth.planning.Plan``)."""
        from topologies.synth.planning import make_plan

        plan = make_plan(self, pilot=pilot, preview=preview)
        if verbose if verbose is not None else self.monitor != "none":
            try:
                print(plan)
            except Exception:  # noqa: BLE001, S110 - display never breaks planning
                pass
        return plan

    def run(self, *, limit: int | None = None, resume: bool = True, rerun: str | None = None,
            monitor: str | None = None) -> Any:
        """Generate (resumably) and return the ``SynthDataset``.

        ``limit`` caps raw records generated by this call (a partial run);
        ``resume=False`` / ``rerun="all"`` discards finished chunks.
        """
        from topologies.synth.pipeline import run_spec

        if rerun not in (None, "all", "failed"):
            raise SynthError(f"rerun={rerun!r} must be None, 'failed' or 'all'.", stage="spec")
        return run_spec(self, limit=limit, rerun="all" if not resume else rerun,
                        monitor=monitor)

    def load(self, version: str = "latest") -> Any:
        from topologies.synth.dataset import load

        return load(self.name, version=version, root=self.root)

    def to_yaml(self, path: str | Path | None = None) -> str:
        from topologies.synth.yaml_spec import spec_to_yaml

        return spec_to_yaml(self, path)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Spec:
        from topologies.synth.yaml_spec import spec_from_yaml

        return spec_from_yaml(path)

    def __repr__(self) -> str:
        return (f"Spec({self.name!r}, generator={self.generator.node_kind()}, size={self.size}, "
                f"attributes={self.attributes.names}, ops={[op.name for op in self.pipeline]})")


def _differs(key: str, value: Any) -> bool:
    """Was ``key`` given explicitly (not the constructor default)?"""
    import inspect

    default = inspect.signature(Spec.__init__).parameters[key].default
    try:
        return bool(value != default)
    except Exception:  # noqa: BLE001
        return True


def _blueprint_spec(blueprint: Any, name: str) -> Spec:
    if isinstance(blueprint, Spec):
        return blueprint
    if isinstance(blueprint, Mapping):
        d = dict(blueprint)
        kind = d.pop("name", d.pop("type", None))
        if kind is None:
            raise SynthError("blueprint dict needs a 'name'.", stage="spec")
        from topologies.synth import blueprints

        fn = blueprints.get(kind)
        return fn(name, **d)
    if isinstance(blueprint, str):
        from topologies.synth import blueprints

        raise SynthError(f"Blueprint {blueprint!r} needs its arguments.", stage="spec",
                         hint=f"Use synth.blueprints.{blueprint}(name, ...) or "
                              f"blueprint={{'name': {blueprint!r}, ...}}.",
                         available=blueprints.names())
    raise SynthError(f"Invalid blueprint {blueprint!r}.", stage="spec")


def _resolve_size(size: Any) -> tuple[int, dict[str, float] | None]:
    if size is None:
        return 1000, None
    if isinstance(size, bool):
        raise SynthError(f"Invalid size {size!r}.", stage="spec")
    if isinstance(size, (int, float)):
        if size < 1:
            raise SynthError("size must be >= 1.", stage="spec")
        return int(size), None
    if isinstance(size, str):
        try:
            return int(parse_count(size)), None
        except SynthError:
            raise SynthError(f"Invalid size {size!r}.", stage="spec",
                             hint="size=20_000, '20k' or {'train': 18000, 'val': 2000}") from None
    if isinstance(size, Mapping):
        counts = {str(k): int(v) for k, v in size.items()}
        if not counts or any(v < 0 for v in counts.values()) or sum(counts.values()) < 1:
            raise SynthError(f"Invalid per-split size {size!r}.", stage="spec")
        total = sum(counts.values())
        return total, {k: v / total for k, v in counts.items()}
    raise SynthError(f"Invalid size {size!r}.", stage="spec",
                     hint="size=20_000 or {'train': 18000, 'val': 2000}")


__all__ = ["Spec"]
