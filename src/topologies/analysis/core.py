"""Analysis invocation, result lineage, pipelines, and explicit JSON support."""
from __future__ import annotations

import csv
import inspect
import io
import json
import math
import sys
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable


@dataclass
class AnalysisContext:
    """Caller-supplied parameters and metadata; analyzers may not mutate input data."""

    parameters: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """An in-memory payload with explicit semantics, lineage, and caveats."""

    payload: Any
    kind: str = "result"
    name: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    stages: list[AnalysisResult] = field(default_factory=list)

    def summary(self) -> str:
        title = self.name or self.kind
        return f"{title}: {self._payload_summary()}" + (
            f" ({len(self.warnings)} warning(s))" if self.warnings else ""
        )

    def _payload_summary(self) -> str:
        if isinstance(self.payload, Mapping):
            return ", ".join(f"{k}={v}" for k, v in list(self.payload.items())[:5])
        if isinstance(self.payload, (list, tuple, RecordSet)):
            return f"{len(self.payload)} item(s)"
        return str(self.payload)

    def to_json(self) -> str:
        return json.dumps(_encode(self), indent=2, allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> AnalysisResult:
        decoded = _decode(json.loads(text))
        if not isinstance(decoded, cls):
            raise TypeError("JSON does not contain an AnalysisResult")
        return decoded

    def to_csv(self) -> str:
        rows = _rows(self.payload)
        columns = list(dict.fromkeys(k for row in rows for k in row))
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=columns)
        if columns:
            writer.writeheader()
            writer.writerows(rows)
        return output.getvalue()

    def to_markdown(self) -> str:
        rows = _rows(self.payload)
        cols = list(dict.fromkeys(k for row in rows for k in row))
        if not cols:
            return self.summary()
        fmt = lambda row: "| " + " | ".join(str(row.get(c, "")) for c in cols) + " |"
        return "\n".join([fmt(dict(zip(cols, cols))), fmt(dict.fromkeys(cols, "---")),
                           *(fmt(r) for r in rows)])

    def visualize(self, **options: Any) -> Any:
        """Explicitly render this result with optional Matplotlib support."""
        from topologies.analysis.visualization import visualize
        return visualize(self, **options)


@runtime_checkable
class Analysis(Protocol):
    """Reusable analyzer contract; implement ``run(data, context)``."""

    def run(self, data: Any, context: AnalysisContext) -> Any: ...


@dataclass
class Stage:
    name: str
    analysis: Any
    kind: str | None = None
    assumptions: list[str] = field(default_factory=list)


class Pipeline:
    """Sequential named analyses retaining inspectable stage results and caveats."""

    def __init__(self, *stages: Stage | tuple[str, Any] | Any, name: str = "pipeline"):
        self.name = name
        self.stages = [s if isinstance(s, Stage) else
                       Stage(s[0], s[1]) if isinstance(s, tuple) and len(s) == 2 else
                       Stage(getattr(s, "name", None) or getattr(s, "__name__", type(s).__name__), s)
                       for s in stages]

    def run(self, data: Any, context: AnalysisContext | None = None) -> AnalysisResult:
        ctx = context or AnalysisContext()
        current: Any = data
        results: list[AnalysisResult] = []
        for stage in self.stages:
            result = analyze(current, analysis=stage.analysis, context=ctx,
                             kind=stage.kind or "derived")
            # The stage's name wins: built-in stages wrap anonymous functions.
            result = replace(result, name=stage.name or result.name,
                             kind=stage.kind or result.kind,
                             assumptions=list(dict.fromkeys(
                                 [*stage.assumptions, *result.assumptions])),
                             provenance={**result.provenance, "stage": stage.name},
                             warnings=list(result.warnings), metadata=dict(result.metadata),
                             stages=list(result.stages))
            results.append(result)
            current = result
        warnings = list(dict.fromkeys([
            *(data.warnings if isinstance(data, AnalysisResult) else []),
            *(w for r in results for w in r.warnings),
        ]))
        assumptions = list(dict.fromkeys([
            *(data.assumptions if isinstance(data, AnalysisResult) else []),
            *(a for r in results for a in r.assumptions),
        ]))
        payload = current.payload if isinstance(current, AnalysisResult) else current
        source = data if isinstance(data, AnalysisResult) else None
        source_metadata = getattr(data, "source_provenance", None)
        if source_metadata is None and source:
            source_metadata = source.provenance.get("source")
        if not isinstance(source_metadata, Mapping):
            source_metadata = None
        return AnalysisResult(payload, kind="pipeline", name=self.name,
                              provenance={"analyzer": self.name,
                                          **({"source": dict(source_metadata)} if source_metadata else {}),
                                          "upstream": [source.provenance] if source else [],
                                          "stages": [r.provenance for r in results]},
                              warnings=warnings, assumptions=assumptions,
                              metadata={**(dict(source.metadata) if source else {}),
                                        "context": dict(ctx.metadata)}, stages=results)


def _invoke(analysis: Any, data: Any, context: AnalysisContext) -> Any:
    if isinstance(analysis, Analysis):
        return analysis.run(data, context)
    if not callable(analysis):
        raise TypeError("analysis must be callable or implement run(data, context)")
    signature = inspect.signature(analysis)
    if "context" in signature.parameters:
        return analysis(data, context=context)
    return analysis(data)


def analyze(data: Any, analysis: Any = None, *, context: AnalysisContext | None = None,
            name: str | None = None, kind: str = "result") -> AnalysisResult:
    """Run an analyzer without hiding its exceptions.

    Default dispatch is deliberately limited to PyTorch modules; all other
    values need an explicit analyzer. Torch is imported only for this branch.
    """
    ctx = context or AnalysisContext()
    payload = data.payload if isinstance(data, AnalysisResult) else data
    if analysis is not None:
        # Never import experiment/training dependencies from generic analysis.
        loaded_results = sys.modules.get("topologies.experiments.results")
        result_type = getattr(loaded_results, "ResultSet", None)
        if isinstance(result_type, type) and isinstance(payload, result_type):
            from topologies.analysis.integrations import from_result_set
            payload = from_result_set(payload)
    if analysis is None:
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None and isinstance(payload, torch.nn.Module):
            from topologies.analysis.engineering import ModelAnalysis
            analysis = ModelAnalysis()
        else:
            raise ValueError("analysis is required for non-model data; pass a callable or analyzer")
    raw = _invoke(analysis, payload, ctx)
    analyzer_name = name or getattr(analysis, "name", None) or getattr(
        analysis, "__name__", type(analysis).__name__)
    if isinstance(raw, AnalysisResult):
        result = replace(raw, provenance=dict(raw.provenance), warnings=list(raw.warnings),
                         assumptions=list(raw.assumptions), metadata=dict(raw.metadata),
                         stages=list(raw.stages))
        result.name = result.name or analyzer_name
        result.provenance.setdefault("analyzer", analyzer_name)
        result.provenance.setdefault("method", analyzer_name)
        result.provenance.setdefault("source", type(payload).__name__)
        result.provenance.setdefault("parameters", dict(ctx.parameters))
        result.provenance.setdefault("context_metadata", dict(ctx.metadata))
        result.provenance.setdefault("sample_count", _sample_count(payload))
    else:
        result = AnalysisResult(raw, kind=kind, name=analyzer_name,
                                provenance={"analyzer": analyzer_name,
                                            "method": analyzer_name,
                                            "source": data.name if isinstance(data, AnalysisResult)
                                            else type(data).__name__,
                                            "sample_count": _sample_count(payload),
                                            "parameters": dict(ctx.parameters),
                                             "context_metadata": dict(ctx.metadata)})
    source_provenance = getattr(payload, "source_provenance", None)
    if source_provenance:
        result.provenance["source"] = dict(source_provenance)
    if isinstance(data, AnalysisResult):
        result.provenance["upstream"] = [*result.provenance.get("upstream", []),
                                          dict(data.provenance)]
        result.warnings = list(dict.fromkeys([*data.warnings, *result.warnings]))
        result.assumptions = list(dict.fromkeys([*data.assumptions, *result.assumptions]))
    return result


def _sample_count(data: Any) -> int | None:
    try:
        return len(data)
    except (TypeError, AttributeError):
        return None


class RecordSet(list):
    """Small immutable-by-convention mapping-record collection with fluent methods."""

    def __init__(self, records: Iterable[Mapping[str, Any]] = (), *,
                 source_provenance: Mapping[str, Any] | None = None):
        if source_provenance is None and isinstance(records, RecordSet):
            source_provenance = records.source_provenance
        super().__init__(dict(row) for row in records)
        self.source_provenance = deepcopy(dict(source_provenance)) if source_provenance is not None else None

    @classmethod
    def from_data(cls, data: Any) -> RecordSet:
        """Adapt mappings or a DataFrame explicitly; never flatten arbitrary iterables."""
        if isinstance(data, Mapping):
            if all(isinstance(v, Mapping) for v in data.values()):
                return cls(data.values())
            return cls([data])
        if isinstance(data, (list, tuple)) and all(isinstance(row, Mapping) for row in data):
            return cls(data)
        if type(data).__module__.startswith("pandas") and hasattr(data, "to_dict"):
            return cls(data.to_dict(orient="records"))
        raise TypeError("from_data expects a mapping, mapping records, or pandas DataFrame")

    @classmethod
    def from_experiments(cls, results: Any, *, include_history: bool = False) -> RecordSet:
        """Lazy bridge to the optional experiment adapter."""
        from topologies.analysis.integrations import from_result_set

        return from_result_set(results, include_history=include_history)

    def select(self, selector: str | Callable[[Mapping[str, Any]], Any]) -> list[Any]:
        return [field_value(r, selector) for r in self]

    def filter(self, predicate: Callable[[Mapping[str, Any]], bool]) -> RecordSet:
        return RecordSet((r for r in self if predicate(r)), source_provenance=self.source_provenance)

    def derive(self, name: str, fn: Callable[[Mapping[str, Any]], Any]) -> RecordSet:
        return RecordSet(({**r, name: fn(r)} for r in self), source_provenance=self.source_provenance)

    def group_by(self, *keys: str) -> GroupedRecords:
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in self:
            groups.setdefault(tuple(field_value(row, k) for k in keys), []).append(row)
        return GroupedRecords(keys, groups, self.source_provenance)

    def sort(self, key: str | Callable[[Mapping[str, Any]], Any], *, reverse: bool = False
             ) -> RecordSet:
        return RecordSet(sorted(self, key=lambda r: field_value(r, key), reverse=reverse),
                         source_provenance=self.source_provenance)

    def rank(self, key: str, *, mode: str = "min") -> RecordSet:
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        ordered = sorted(self, key=lambda r: field_value(r, key), reverse=mode == "max")
        return RecordSet(({**r, "rank": i} for i, r in enumerate(ordered, 1)),
                         source_provenance=self.source_provenance)

    def aggregate(self, key: str, reducer: Callable[[list[Any]], Any]) -> Any:
        return reducer(self.select(key))

    def describe(self, key: str) -> dict[str, Any]:
        vals = sorted(float(v) for v in self.select(key) if _finite_number(v))
        if not vals:
            return {"n": 0, "mean": None, "min": None, "max": None}
        return {"n": len(vals), "mean": sum(vals) / len(vals), "min": vals[0],
                "max": vals[-1], "q25": quantile(vals, .25), "median": quantile(vals, .5),
                "q75": quantile(vals, .75)}


@dataclass
class GroupedRecords:
    keys: tuple[str, ...]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]]
    source_provenance: dict[str, Any] | None = None

    def aggregate(self, name: str, field: str, reducer: Callable[[list[Any]], Any]
                  ) -> RecordSet:
        return RecordSet([{**dict(zip(self.keys, key)), name: reducer(
            [field_value(row, field) for row in rows])} for key, rows in self.groups.items()],
            source_provenance=self.source_provenance)


def field_value(record: Mapping[str, Any], selector: str | Callable) -> Any:
    if callable(selector):
        return selector(record)
    value: Any = record
    for part in selector.split("."):
        value = value[part]
    return value


def quantile(values: list[float], q: float) -> float:
    if not 0 <= q <= 1 or not values:
        raise ValueError("quantile requires non-empty values and q in [0, 1]")
    pos = (len(values) - 1) * q
    lo = int(pos)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (pos - lo)


def _finite_number(value: Any) -> bool:
    try:
        import math
        return math.isfinite(float(value)) and not isinstance(value, bool)
    except (TypeError, ValueError, OverflowError):
        return False


_BUILTIN_TAGS = frozenset({"result", "recordset", "grouped-records", "tuple", "mapping",
                           "serializer"})
_SERIALIZERS: dict[type, tuple[str, Callable, Callable]] = {}


def register_serializer(typ: type, tag: str, encode: Callable[[Any], Any],
                        decode: Callable[[Any], Any]) -> None:
    """Register explicit lossless conversion for a user-defined payload type."""
    if not tag or tag in _BUILTIN_TAGS:
        raise ValueError("serializer tag must be a unique non-reserved string")
    if any(existing[0] == tag and cls is not typ for cls, existing in _SERIALIZERS.items()):
        raise ValueError(f"serializer tag {tag!r} is already registered")
    _SERIALIZERS[typ] = (tag, encode, decode)


def _encode(value: Any) -> Any:
    if isinstance(value, AnalysisResult):
        return {"$analysis": "result", "payload": _encode(value.payload), "kind": value.kind,
                "name": value.name, "provenance": _encode(value.provenance),
                "warnings": value.warnings, "assumptions": value.assumptions,
                "metadata": _encode(value.metadata), "stages": [_encode(s) for s in value.stages]}
    if isinstance(value, RecordSet):
        return {"$analysis": "recordset", "records": _encode(list(value)),
                "source_provenance": _encode(value.source_provenance)}
    if isinstance(value, GroupedRecords):
        return {"$analysis": "grouped-records", "keys": _encode(value.keys),
                "groups": _encode([[key, rows] for key, rows in value.groups.items()]),
                "source_provenance": _encode(value.source_provenance)}
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("non-finite floats cannot be serialized")
        return value
    if isinstance(value, list):
        return [_encode(v) for v in value]
    if isinstance(value, tuple):
        return {"$analysis": "tuple", "items": [_encode(v) for v in value]}
    if isinstance(value, Mapping):
        if not all(isinstance(k, str) for k in value):
            raise TypeError("JSON object keys must be strings")
        return {"$analysis": "mapping",
                "items": [[k, _encode(v)] for k, v in value.items()]}
    for typ, (tag, encode, _) in _SERIALIZERS.items():
        if isinstance(value, typ):
            return {"$analysis": "serializer", "tag": tag, "value": _encode(encode(value))}
    raise TypeError(f"No analysis JSON serializer registered for {type(value).__name__}")


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    tag = value.get("$analysis")
    if tag == "result":
        return AnalysisResult(_decode(value["payload"]), value["kind"], value.get("name"),
                              _decode(value["provenance"]), list(value["warnings"]),
                              list(value["assumptions"]), _decode(value["metadata"]),
                              [_decode(s) for s in value["stages"]])
    if tag == "recordset":
        return RecordSet(_decode(value["records"]),
                         source_provenance=_decode(value.get("source_provenance")))
    if tag == "grouped-records":
        return GroupedRecords(_decode(value["keys"]),
                              {tuple(key): rows for key, rows in _decode(value["groups"])},
                              _decode(value.get("source_provenance")))
    if tag == "tuple":
        return tuple(_decode(v) for v in value["items"])
    if tag == "mapping":
        return {key: _decode(item) for key, item in value["items"]}
    if tag == "serializer":
        serializer_tag = value["tag"]
        for registered_tag, _, decode in _SERIALIZERS.values():
            if registered_tag == serializer_tag:
                return decode(_decode(value["value"]))
        raise ValueError(f"Unknown analysis serializer tag {serializer_tag!r}")
    raise ValueError(f"Unknown analysis JSON type tag {tag!r}")


def _rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, RecordSet):
        return list(payload)
    to_records = getattr(payload, "to_records", None)
    if callable(to_records) and not isinstance(payload, (Mapping, list, tuple)):
        return [dict(r) for r in to_records()]
    if isinstance(payload, list) and all(isinstance(r, Mapping) for r in payload):
        return [dict(r) for r in payload]
    if isinstance(payload, Mapping):
        return [dict(payload)]
    return []


def Filter(predicate: Callable[[Mapping[str, Any]], bool]) -> Stage:
    return Stage("filter", lambda rows: RecordSet(rows).filter(predicate))


def Select(*fields: str) -> Stage:
    return Stage("select", lambda rows: RecordSet(
        {field: field_value(row, field) for field in fields} for row in rows))


def Derive(name: str, fn: Callable[[Mapping[str, Any]], Any]) -> Stage:
    return Stage(f"derive:{name}", lambda rows: RecordSet(rows).derive(name, fn))


def GroupBy(*keys: str) -> Stage:
    return Stage("group_by", lambda rows: RecordSet(rows).group_by(*keys))


def Aggregate(name: str, field: str, reducer: Callable[[list[Any]], Any]) -> Stage:
    return Stage(f"aggregate:{name}", lambda groups: groups.aggregate(name, field, reducer))
