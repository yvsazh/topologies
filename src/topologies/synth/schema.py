"""Record schemas: what one synthetic record is.

    schema = Schema(text=Text(min_chars=10), label=Label({"refund": "wants money back",
                                                           "shipping": "asks about delivery"}))
    schema.validate({"text": "where is my parcel?", "label": "shipping"})   # -> [] (no errors)

Fields are value types (``synth.valuetypes``): ``Text``, ``Label`` (one of N),
``Labels`` (multi-label), ``Category``, ``Number``, ``Tokens``, ``Tensor``,
``Audio``, ``Image``, ``Video``, ``Series``, ``Mask``, ``PointCloud``,
``Boxes``, ``Keypoints``, ``Graph``, ``Spans`` (offsets into a text field) and
``Json``. Every field validates values, renders a JSON schema, produces an
example, and round-trips through ``to_config()`` / ``field_from_config()``
(YAML specs, fingerprints). ``schema_of(generator)`` reads the schema a node
graph declares through its value types.

Record keys starting with ``_`` (``_id``, ``_attrs``, ...) are metadata and
are never validated.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from typing import Any

from topologies.synth.ctor import Recorded
from topologies.synth.errors import SynthError, unknown
from topologies.synth.valuetypes import (
    TYPES,
    Audio,
    Boxes,
    Category,
    Field,
    Graph,
    Image,
    Json,
    Keypoints,
    Label,
    Labels,
    Mask,
    Number,
    PointCloud,
    Series,
    Spans,
    Tensor,
    Text,
    Tokens,
    Type,
    Video,
    as_type,
    infer_type,
)

FIELD_TYPES = TYPES


def field_from_config(cfg: Mapping[str, Any] | Type) -> Type:
    """Inverse of ``Type.to_config()`` (pydantic models come back as JSON schemas)."""
    if isinstance(cfg, Type):
        return cfg
    if isinstance(cfg, str):
        cfg = {"type": cfg}
    d = dict(cfg)
    kind = str(d.pop("type", "text"))
    cls = TYPES.get(kind)
    if cls is None:
        raise unknown("field type", kind, TYPES, stage="schema")
    if cls is Json:
        d.pop("model", None)
        return Json(schema=d.pop("schema", None), **d)
    d = {k: v for k, v in d.items() if v is not None}
    try:
        return cls(**d)
    except TypeError as e:
        raise SynthError(f"Invalid {kind} field config {cfg!r}: {e}", stage="schema") from e


# ---------------------------------------------------------------- Schema


class Schema(Recorded):
    """Named fields of one record: ``Schema(text=Text(), label=Label([...]))``.

    ``validate(record)`` returns a list of problems; ``json_schema()`` the
    JSON schema of a record (documentation, validation); ``example()``
    a valid example record.
    """

    def __init__(self, fields: Mapping[str, Any] | None = None, /, **kw: Any) -> None:
        merged = {**dict(fields or {}), **kw}
        self.fields: dict[str, Type] = {}
        for name, f in merged.items():
            if isinstance(f, type) and issubclass(f, Type):
                f = as_type(f)
            if isinstance(f, Type):
                f = f.spec()
            if not isinstance(f, Type):
                f = field_from_config(f) if isinstance(f, (Mapping, str)) else None
                if f is None:
                    raise SynthError(f"Schema field {name!r} must be a Field (Text(), Label(...),"
                                     f" ...); got {merged[name]!r}.", stage="schema")
            if name.startswith("_"):
                raise SynthError(f"Field names cannot start with '_' ({name!r}): that prefix is "
                                 "reserved for record metadata.", stage="schema")
            self.fields[name] = f

    # -- access ----------------------------------------------------------
    def __getitem__(self, name: str) -> Field:
        try:
            return self.fields[name]
        except KeyError:
            raise unknown("schema field", name, self.fields, stage="schema") from None

    def __contains__(self, name: object) -> bool:
        return name in self.fields

    def __iter__(self):
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    @property
    def names(self) -> list[str]:
        return list(self.fields)

    def of_kind(self, *kinds: str) -> list[str]:
        return [n for n, f in self.fields.items() if f.kind in kinds]

    def label_field(self) -> str | None:
        """Name of the (first) classification target field, if any."""
        found = self.of_kind("label", "labels")
        return found[0] if found else None

    def text_fields(self) -> list[str]:
        return self.of_kind("text")

    # -- behaviour -------------------------------------------------------
    def coerce(self, record: dict[str, Any]) -> dict[str, Any]:
        out = dict(record)
        for name, f in self.fields.items():
            if name in out:
                out[name] = f.coerce(out[name])
        return out

    def validate(self, record: Mapping[str, Any]) -> list[str]:
        if not isinstance(record, Mapping):
            return [f"record must be a dict, got {type(record).__name__}"]
        errors: list[str] = []
        for name, f in self.fields.items():
            if name not in record or record[name] is None:
                if not f.optional:
                    errors.append(f"missing field {name!r}")
                continue
            errors += [f"{name}: {e}" for e in f.validate(record[name], record)]
        return errors

    def json_schema(self) -> dict[str, Any]:
        return {"type": "object",
                "properties": {n: f.json_schema() for n, f in self.fields.items()},
                "required": [n for n, f in self.fields.items() if not f.optional]}

    def example(self, rng: random.Random | None = None) -> dict[str, Any]:
        rng = rng or random.Random(0)
        return {n: f.example(rng) for n, f in self.fields.items()}

    # -- config ----------------------------------------------------------
    def to_config(self) -> dict[str, Any]:
        return {n: f.to_config() for n, f in self.fields.items()}

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> Schema:
        return cls({n: field_from_config(c) for n, c in cfg.items()})

    def fingerprint(self) -> str:
        from topologies.experiments.cache import stable_hash

        return stable_hash(self.to_config())

    @classmethod
    def infer(cls, records: Iterable[Mapping[str, Any]], *, max_labels: int = 50) -> Schema:
        """Best-effort schema from sample records (strings with few distinct values on a
        key named like a label become ``Label``)."""
        rows = [r for _, r in zip(range(500), records)]
        if not rows:
            raise SynthError("Cannot infer a schema from zero records.", stage="schema")
        keys = [k for k in rows[0] if not str(k).startswith("_")]
        fields: dict[str, Field] = {}
        for k in keys:
            vals = [r.get(k) for r in rows if r.get(k) is not None]
            optional = len(vals) < len(rows)
            first = vals[0] if vals else None
            if isinstance(first, bool):
                f: Field = Category(["False", "True"])
            elif isinstance(first, (int, float)):
                f = Number(integer=all(isinstance(v, int) for v in vals))
            elif isinstance(first, str):
                distinct = sorted({str(v) for v in vals})
                labelish = k.lower() in ("label", "labels", "intent", "class", "category",
                                         "target", "y")
                f = Label(distinct) if labelish and len(distinct) <= max_labels else Text()
            elif isinstance(first, list) and first and all(isinstance(x, int) for x in first):
                f = Tokens()
            elif isinstance(first, list) and first and all(
                    isinstance(x, (int, float)) or x is None for x in first):
                f = Series()
            elif isinstance(first, Mapping) and "$blob" in first:
                kind = first.get("kind") or {"png": "image", "wav": "audio"}.get(
                    first.get("type"), "tensor")
                f = TYPES.get(kind, Tensor)()  # type: ignore[call-arg]
            elif isinstance(first, Mapping) and "$tensor" in first:
                f = Tensor()
            elif isinstance(first, Mapping) and first.get("$type") in TYPES:
                f = TYPES[first["$type"]]()  # type: ignore[call-arg]
            elif isinstance(first, list) and first and isinstance(first[0], str):
                f = Labels(sorted({str(x) for v in vals for x in v}))
            else:
                f = Json()
            f.optional = optional
            fields[k] = f
        return cls(fields)

    def __repr__(self) -> str:
        inner = ", ".join(f"{n}={f!r}" for n, f in self.fields.items())
        return f"Schema({inner})"

    def merged(self, other: Schema | None) -> Schema:
        """This schema plus the fields of ``other`` it does not declare."""
        if other is None:
            return self
        return Schema({**other.fields, **self.fields})

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Schema) and self.to_config() == other.to_config()

    def __hash__(self) -> int:
        return hash(self.fingerprint())


__all__ = ["FIELD_TYPES", "Audio", "Boxes", "Category", "Field", "Graph", "Image", "Json",
           "Keypoints", "Label", "Labels", "Mask", "Number", "PointCloud", "Schema", "Series",
           "Spans", "Tensor", "Text", "Tokens", "Type", "Video", "field_from_config",
           "infer_type", "schema_of"]


def schema_of(root: Any) -> Schema | None:
    """The schema declared by a generator: ``produces=`` / ``schema=`` or the value types of
    a record's field nodes (``@synth.node(type=synth.Audio)``); ``None`` when nothing is
    declared."""
    from topologies.synth.core import Record

    produces = getattr(root, "produces", None)
    if isinstance(produces, Schema):
        return produces
    if not isinstance(root, Record):
        return None
    fields: dict[str, Type] = {}
    for name, v in root.fields.items():
        t = getattr(v, "type", None)
        if t is None:
            continue
        try:
            spec = as_type(t)
        except SynthError:
            spec = None
        if spec is not None:
            fields[name] = spec
    return Schema(fields) if fields else None
