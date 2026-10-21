"""Results of one session run: call tree, per-site records and the captured tensors."""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from topologies.introspection.errors import IntrospectionError
from topologies.introspection.payloads import Payload
from topologies.introspection.semantics import Axes
from topologies.introspection.sites import ROOT, SiteKey, parse_term


@dataclass
class SiteRecord:
    """One captured value: identity, layout, cheap summary and (optionally) the tensor."""

    site: str
    port: str
    occurrence: int
    shape: tuple[int, ...]
    dtype: str
    device: str
    axes: Axes | None
    value: torch.Tensor | None = None
    summary: dict[str, Any] | None = None
    grad: torch.Tensor | None = None
    key: SiteKey | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    loader: Any = None

    @property
    def tensor(self) -> torch.Tensor:
        if self.value is None and self.loader is not None:
            self.value = self.loader()
        if self.value is None:
            raise IntrospectionError(
                f"{self.site}@{self.occurrence}: only a summary was kept; re-run with "
                "keep='tensors' to keep the tensor")
        return self.value

    def to_dict(self) -> dict[str, Any]:
        return {"site": self.site, "port": self.port, "occurrence": self.occurrence,
                "shape": list(self.shape), "dtype": self.dtype, "device": self.device,
                "axes": self.axes.to_dict() if self.axes is not None else None,
                "summary": self.summary, "meta": _jsonable(self.meta),
                "has_value": self.value is not None or self.loader is not None}


@dataclass
class CallRecord:
    index: int
    path: str
    occurrence: int
    parent: int | None
    depth: int
    module_type: str
    inputs: Any = None
    outputs: Any = None
    dtypes: tuple[str, ...] = ()
    devices: tuple[str, ...] = ()
    duration: float | None = None
    error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "path": self.path or ROOT, "occurrence": self.occurrence,
                "parent": self.parent, "depth": self.depth, "type": self.module_type,
                "inputs": _jsonable(self.inputs), "outputs": _jsonable(self.outputs),
                "dtypes": list(self.dtypes), "devices": list(self.devices),
                "duration": self.duration, "error": self.error}


@dataclass
class CallTree:
    """Every observed module call in execution order (occurrence = k-th call of that path)."""

    calls: list[CallRecord] = field(default_factory=list)
    all_paths: tuple[str, ...] = ()
    tracked: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.calls)

    def __iter__(self) -> Iterator[CallRecord]:
        return iter(self.calls)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for call in self.calls:
            out[call.path] = out.get(call.path, 0) + 1
        return out

    def never_called(self) -> list[str]:
        """Tracked modules that never ran (dead code paths, unused heads)."""
        seen = self.counts()
        return [p for p in self.tracked if p not in seen]

    def multi_called(self) -> dict[str, int]:
        return {p: n for p, n in self.counts().items() if n > 1}

    def children(self, index: int | None) -> list[CallRecord]:
        return [c for c in self.calls if c.parent == index]

    def render(self, *, max_lines: int = 200, summaries: Mapping[str, Mapping[str, Any]] | None = None
               ) -> str:
        lines = []
        for call in self.calls[:max_lines]:
            name = (call.path.rsplit(".", 1)[-1] if call.path else ROOT)
            occ = f"@{call.occurrence}" if call.occurrence else ""
            shape = _shape_text(call.outputs)
            stats = ""
            if summaries is not None:
                summary = summaries.get(f"{call.path or ROOT}@{call.occurrence}")
                if summary:
                    stats = f"  mean={_num(summary.get('mean'))} std={_num(summary.get('std'))}"
                    if summary.get("nonfinite_frac") not in (0.0, None):
                        stats += f" NONFINITE={_num(summary.get('nonfinite_frac'))}"
            lines.append(f"{'  ' * call.depth}{name}{occ} ({call.module_type}) -> {shape}{stats}")
        if len(self.calls) > max_lines:
            lines.append(f"... {len(self.calls) - max_lines} more call(s)")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"calls": [c.to_dict() for c in self.calls], "tracked": list(self.tracked)}


def _num(value: Any) -> str:
    return f"{value:.4g}" if isinstance(value, (int, float)) else str(value)


def _shape_text(value: Any) -> str:
    if isinstance(value, tuple) and all(isinstance(v, int) for v in value):
        return "[" + ", ".join(map(str, value)) + "]"
    if isinstance(value, list):
        return "(" + ", ".join(_shape_text(v) for v in value) + ")"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {_shape_text(v)}" for k, v in value.items()) + "}"
    return str(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        return value.tolist() if value.numel() <= 64 else f"tensor{tuple(value.shape)}"
    return str(value)


class Trace:
    """Result of one ``Session`` run (or ``insp.trace``)."""

    def __init__(self, output: Any, calls: CallTree, records: dict[SiteKey, SiteRecord], policy: Any,
                 *, warnings: list[str] | None = None, assumptions: list[str] | None = None,
                 metadata: dict[str, Any] | None = None, provenance: dict[str, Any] | None = None):
        self.output = output
        self.calls = calls
        self.records = records
        self.policy = policy
        self.warnings = list(warnings or [])
        self.assumptions = list(assumptions or [])
        self.metadata = dict(metadata or {})
        self.provenance = dict(provenance or {})

    # -- access -------------------------------------------------------------------

    def _site_records(self, site: str) -> list[SiteRecord]:
        """Records matching one site string; ``@k`` picks an occurrence."""
        spec = parse_term(site)
        path = "" if spec.module in ("", ROOT) else spec.module
        port = spec.port or "output"
        out = [r for k, r in self.records.items()
               if (k.path == path or (k.path or ROOT) == spec.module) and k.port == port
               and k.op == spec.op and (spec.op is None or spec.op_index is None
                                        or k.op_index == spec.op_index)]
        if not out and spec.op is None:
            # Allow the display form ``a.b/fn:sdpa#0`` and aliases via exact site string.
            out = [r for r in self.records.values() if r.site == site.split("@")[0]]
        if isinstance(spec.occurrence, int):
            out = [r for r in out if r.occurrence == spec.occurrence]
        return sorted(out, key=lambda r: (r.occurrence, r.key.op_index if r.key else 0))

    def __getitem__(self, site: str) -> SiteRecord | list[SiteRecord]:
        records = self._site_records(site)
        if not records:
            close = [s for s in self.sites() if site.split("@")[0].split(":")[0] in s][:5]
            raise KeyError(f"no record for {site!r}" + (f"; similar: {close}" if close else
                                                         f"; recorded sites: {self.sites()[:8]}"))
        if site.endswith("@*") or len(records) > 1:
            return records
        return records[0]

    def __contains__(self, site: str) -> bool:
        return bool(self._site_records(site))

    def get(self, site: str, default: Any = None) -> Any:
        try:
            return self[site]
        except KeyError:
            return default

    def tensor(self, site: str) -> torch.Tensor:
        """The kept tensor of a single-occurrence site (use ``@k`` or ``stack`` otherwise)."""
        records = self._site_records(site)
        if not records:
            raise KeyError(f"no record for {site!r}; recorded sites: {self.sites()[:8]}")
        if len(records) > 1:
            raise IntrospectionError(
                f"{site!r} was recorded {len(records)} times; use '{site}@k' or trace.stack(site)")
        return records[0].tensor

    def stack(self, site: str, dim: int = 0) -> torch.Tensor:
        """Stack all occurrences of a site (trajectories over steps)."""
        records = self._site_records(site.split("@")[0])
        if not records:
            raise KeyError(f"no record for {site!r}")
        return torch.stack([r.tensor for r in records], dim=dim)

    def sites(self) -> list[str]:
        return list(dict.fromkeys(r.site for r in sorted(
            self.records.values(), key=lambda r: (self._order(r), r.occurrence))))

    def _order(self, record: SiteRecord) -> int:
        return record.meta.get("order", 0)

    def summaries(self) -> dict[str, dict[str, Any]]:
        return {f"{r.site}@{r.occurrence}": r.summary for r in self.records.values()
                if r.summary is not None}

    def summary(self, *, max_lines: int = 200) -> str:
        head = [(f"Trace: {len(self.calls)} call(s), {len(self.records)} record(s), "
                 f"keep={getattr(self.policy, 'keep', '?')}")]
        output_summary = {f"{(r.site.split(':')[0])}@{r.occurrence}": r.summary
                          for r in self.records.values() if r.port == "output" and r.summary}
        head.append(self.calls.render(max_lines=max_lines, summaries=output_summary))
        never = self.calls.never_called()
        if never:
            head.append(f"never called: {', '.join(p or ROOT for p in never[:20])}")
        multi = self.calls.multi_called()
        if multi:
            head.append("called more than once: " + ", ".join(
                f"{p or ROOT} x{n}" for p, n in list(multi.items())[:20]))
        head.extend(f"warning: {w}" for w in self.warnings)
        return "\n".join(head)

    def __repr__(self) -> str:
        return f"Trace(calls={len(self.calls)}, records={len(self.records)})"

    # -- conversion ------------------------------------------------------------------

    def payload(self) -> TracePayload:
        return TracePayload.from_trace(self)

    def to_result(self) -> Any:
        from topologies.analysis.core import AnalysisResult
        from topologies.introspection.payloads import register_payload_serializer

        register_payload_serializer()
        return AnalysisResult(self.payload(), kind="introspection.trace", name="trace",
                              provenance=dict(self.provenance), warnings=list(self.warnings),
                              assumptions=list(self.assumptions))

    def views(self, **options: Any) -> list[Any]:
        return self.payload().views(**options)

    def __visualization__(self, **options: Any) -> list[Any]:
        return self.views(**options)

    def visualize(self, **options: Any) -> Any:
        from topologies.visualization import visualize

        return visualize(self, **options)

    def save(self, path: Any, *, format: str = "torch") -> Any:
        """Write ``manifest.json`` + tensors (format ``topologies-trace`` v1).

        ``format="torch"`` writes ``tensors/*.pt`` shards; ``format="safetensors"`` (needs the
        optional ``safetensors`` package) writes one ``tensors.safetensors`` file.
        """
        from topologies.introspection.store import save_trace

        ref = getattr(self, "_model_ref", None)
        return save_trace(self, path, model=ref() if ref is not None else None, format=format)

    @classmethod
    def load(cls, path: Any) -> Trace:
        from topologies.introspection.store import load_trace

        return load_trace(path)


@dataclass
class TracePayload(Payload):
    """JSON-serializable view of a trace (no tensors): call tree + per-record summaries."""

    calls: list[dict[str, Any]]
    records: list[dict[str, Any]]
    never_called: list[str]
    multi_called: dict[str, int]

    @classmethod
    def from_trace(cls, trace: Trace) -> TracePayload:
        return cls([c.to_dict() for c in trace.calls], [r.to_dict() for r in trace.records.values()],
                   [p or ROOT for p in trace.calls.never_called()],
                   {p or ROOT: n for p, n in trace.calls.multi_called().items()})

    def to_records(self) -> list[dict[str, Any]]:
        rows = []
        for record in self.records:
            row = {"site": record["site"], "occurrence": record["occurrence"],
                   "shape": "x".join(map(str, record["shape"])), "dtype": record["dtype"]}
            row.update(record.get("summary") or {})
            rows.append(row)
        return rows

    def views(self, **options: Any) -> list[Any]:
        from topologies.introspection.views import trace_views

        return trace_views(self, **options)
