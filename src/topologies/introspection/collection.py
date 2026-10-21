"""Dataset-scale collection: stream batches through reducers with persistent hooks.

Hooks are registered once for the whole collection. Progress is reported as
``introspection.*`` visualization events (scalars only), and reducer states can be
checkpointed with ``resume=path`` and merged after a restart.
"""
from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from topologies.introspection import _tree
from topologies.introspection.capture import CapturePolicy, budget_message, default_axes_for
from topologies.introspection.errors import CaptureBudgetExceeded
from topologies.introspection.inputs import normalize_inputs
from topologies.introspection.payloads import Payload, register_payload_serializer
from topologies.introspection.reducers import BatchMeta, Collect, SiteMeta, make_reducer
from topologies.introspection.session import Session
from topologies.introspection.sites import ROOT, SiteKey


class ReducerSink:
    """Recorder replacement: routes captured values into reducer states (no tensors kept)."""

    def __init__(self, session: Session, reducers: list[Any], states: dict | None = None):
        self.session = session
        self.reducers = reducers
        self.states: dict[tuple[str, str], Any] = states if states is not None else {}
        self.meta: dict[str, SiteMeta] = {}
        self.order: dict[str, int] = {}
        self.by_path: dict[str, list[Any]] = {}
        for req in session.requests:
            if req.op is None:
                self.by_path.setdefault(req.path, []).append(req)
        self.batch: BatchMeta | None = None
        self.axes_for = None
        self.warnings: list[str] = []
        self.assumptions: list[str] = []
        self.grad_enabled = False
        self.store = None
        self._live: set[tuple[str, str]] = set()

    def start(self, batch: BatchMeta, size: int | None) -> None:
        self.batch = batch
        self.axes_for = default_axes_for(self.session.subject.declared_axes, size)

    def wants(self, path: str) -> bool:
        return path in self.by_path

    def on_pre(self, path: str, occurrence: int, module: nn.Module, args: tuple, kwargs: dict) -> None:
        for req in self.by_path.get(path, ()):
            if req.port.startswith("input") and _occ_ok(req.occurrence, occurrence):
                found = _tree.first_tensor((args, kwargs)) if req.port == "input" else None
                if req.port.startswith("input."):
                    head = req.port.split(".", 1)[1]
                    value = args[int(head)] if head.isdigit() and int(head) < len(args) else \
                        kwargs.get(head)
                    found = ("", value) if isinstance(value, torch.Tensor) else None
                if found is not None:
                    self._feed(SiteKey(path, req.port, occurrence), module, found[1])

    def on_post(self, path: str, occurrence: int, module: nn.Module, output: Any) -> None:
        for req in self.by_path.get(path, ()):
            if not req.port.startswith("output") or not _occ_ok(req.occurrence, occurrence):
                continue
            value = output
            if req.port.startswith("output."):
                value = _tree.get_path(output, req.port[len("output."):])
            if not isinstance(value, torch.Tensor):
                found = _tree.first_tensor(value)
                if found is None:
                    continue
                value = found[1]
            self._feed(SiteKey(path, req.port, occurrence), module, value)

    def feed_value(self, key: SiteKey, module: nn.Module | None, value: torch.Tensor,
                   axes: Any = None) -> None:
        """Entry point for functional (op-level) captures."""
        self._feed(key, module, value, axes)

    def _feed(self, key: SiteKey, module: nn.Module | None, value: torch.Tensor,
              axes: Any = None) -> None:
        site = key.site
        if site not in self.meta:
            axes = axes or self.axes_for(key.path, key.port, module, value)
            self.meta[site] = SiteMeta(site, axes, value.dtype, value.device, tuple(value.shape),
                                       type(module).__name__ if module is not None else None)
            self.order[site] = len(self.order)
        meta = self.meta[site]
        batch = self.batch
        batch.occurrence = key.occurrence
        batch.extra["order"] = self.order[site]
        for reducer in self.reducers:
            state_key = (site, reducer.name)
            if state_key not in self.states:
                self.states[state_key] = reducer.init(meta)
            elif state_key not in self._live:
                # Resumed state: re-attach fields stripped for weights_only checkpoints.
                state = self.states[state_key]
                if isinstance(state, dict):
                    fresh = reducer.init(meta)
                    self.states[state_key] = {**fresh, **state} if isinstance(fresh, dict) else state
            self._live.add(state_key)
            self.states[state_key] = reducer.update(self.states[state_key], value, batch)

    def protect(self, args: tuple) -> None:
        return

    def remove_grad_hooks(self) -> None:
        return


def _occ_ok(requested: Any, occurrence: int) -> bool:
    return requested == "*" or requested == occurrence


@dataclass
class ExampleResolver:
    """Turns example ids back into displayable inputs (token contexts or tensors)."""

    data: Any = None
    tokens: dict[int, list[str]] = field(default_factory=dict)
    batch_of: dict[int, tuple[int, int]] = field(default_factory=dict)

    def resolve(self, example_id: int, *, position: int | None = None, window: int = 8) -> Any:
        example_id = int(example_id)
        if example_id in self.tokens:
            toks = self.tokens[example_id]
            if position is None:
                return {"tokens": toks}
            lo, hi = max(0, position - window), min(len(toks), position + window + 1)
            return {"tokens": toks[lo:hi], "focus": position - lo, "position": position}
        item = self._dataset_item(example_id)
        if item is not None:
            return _as_input(item)
        where = self.batch_of.get(example_id)
        if where is None or self.data is None:
            return {"example_id": example_id}
        batch_index, row = where
        for i, batch in enumerate(self.data):
            if i == batch_index:
                return _as_input(_row(batch, row))
        return {"example_id": example_id}

    def _dataset_item(self, example_id: int) -> Any:
        """Index the underlying dataset when ids are dataset indices (sequential order)."""
        data = self.data
        try:
            from torch.utils.data import DataLoader, Dataset, SequentialSampler
        except ImportError:  # pragma: no cover
            return None
        dataset = None
        if isinstance(data, DataLoader) and isinstance(data.sampler, SequentialSampler):
            dataset = data.dataset
        elif isinstance(data, Dataset):
            dataset = data
        if dataset is None or not hasattr(dataset, "__getitem__"):
            return None
        try:
            return dataset[example_id]
        except Exception:  # noqa: BLE001 - fall back to batch scanning
            return None


def _row(batch: Any, row: int) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch[row]
    if isinstance(batch, Mapping):
        return {k: (v[row] if isinstance(v, torch.Tensor) and v.ndim else v) for k, v in batch.items()}
    if isinstance(batch, (tuple, list)):
        return type(batch)(_row(v, row) for v in batch)
    return batch


def _as_input(item: Any) -> dict[str, Any]:
    if isinstance(item, (tuple, list)) and item and not isinstance(item[0], str):
        out = {"input": item[0]}
        if len(item) > 1:
            out["target"] = item[1]
        return out
    return {"input": item}


class Collection:
    """Reducer outputs per site (+ provenance); ``result(name)`` gives one ``AnalysisResult``."""

    def __init__(self, results: dict[str, dict[str, Any]], *, reducers: list[Any], batches: int,
                 examples: int, order: dict[str, int], resolver: ExampleResolver,
                 provenance: dict[str, Any], warnings: list[str], assumptions: list[str],
                 seconds: float):
        self.results = results
        self.reducers = reducers
        self.batches = batches
        self.examples = examples
        self.order = order
        self.resolver = resolver
        self.provenance = provenance
        self.warnings = warnings
        self.assumptions = assumptions
        self.seconds = seconds

    def sites(self) -> list[str]:
        return sorted(self.results, key=lambda s: self.order.get(s, 0))

    def __getitem__(self, site: str) -> dict[str, Any]:
        if site in self.results:
            return self.results[site]
        raise KeyError(f"no collected site {site!r}; collected: {self.sites()[:8]}")

    def reducer(self, name: str) -> dict[str, Any]:
        return {s: r[name] for s, r in self.results.items() if name in r}

    def result(self, name: str) -> Any:
        from topologies.analysis.core import AnalysisResult

        register_payload_serializer()
        approximate = any(getattr(r, "approximate", False) for r in self.reducers if r.name == name)
        payload = CollectionPayload(name, {s: self.results[s][name] for s in self.sites()
                                           if name in self.results[s]}, self.examples)
        assumptions = list(self.assumptions)
        if approximate:
            assumptions.append(f"{name} is an approximate streaming estimate")
        return AnalysisResult(payload, kind=f"introspection.collection.{name}", name=name,
                              provenance=dict(self.provenance), warnings=list(self.warnings),
                              assumptions=assumptions)

    def to_result(self) -> Any:
        from topologies.analysis.core import AnalysisResult

        stages = [self.result(r.name) for r in self.reducers]
        payload = CollectionPayload("collection", {}, self.examples, stages=[s.payload for s in stages])
        return AnalysisResult(payload, kind="introspection.collection", name="collection",
                              provenance=dict(self.provenance), warnings=list(self.warnings),
                              assumptions=list(self.assumptions), stages=stages)

    def views(self, **options: Any) -> list[Any]:
        return self.to_result().payload.views(**options)

    def __visualization__(self, **options: Any) -> list[Any]:
        return self.views(**options)

    def visualize(self, **options: Any) -> Any:
        from topologies.visualization import visualize

        return visualize(self, **options)

    def summary(self) -> str:
        lines = [(f"Collection: {self.batches} batch(es), {self.examples} example(s), "
                  f"{len(self.results)} site(s), {self.seconds:.2f}s")]
        for site in self.sites()[:40]:
            parts = []
            for name, value in self.results[site].items():
                if name == "moments" and isinstance(value.get("mean"), float):
                    parts.append(f"mean={value['mean']:.4g} std={value['std']:.4g}")
                elif name == "sparsity" and "dead_fraction" in value:
                    parts.append(f"dead={value['dead_fraction']:.2%}")
                else:
                    parts.append(name)
            lines.append(f"  {site}: {', '.join(parts)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Collection(sites={len(self.results)}, batches={self.batches})"


@dataclass
class CollectionPayload(Payload):
    reducer: str
    sites: dict[str, Any]
    examples: int = 0
    stages: list[Any] = field(default_factory=list)

    def to_records(self) -> list[dict[str, Any]]:
        rows = []
        for site, value in self.sites.items():
            row: dict[str, Any] = {"site": site}
            if isinstance(value, Mapping):
                for k, v in value.items():
                    if isinstance(v, (int, float, str, bool)) or v is None:
                        row[k] = v
                    elif isinstance(v, torch.Tensor) and v.numel() == 1:
                        row[k] = float(v)
            rows.append(row)
        for stage in self.stages:
            rows.extend({"reducer": stage.reducer, **r} for r in stage.to_records())
        return rows

    def views(self, **options: Any) -> list[Any]:
        from topologies.introspection.views import collection_views

        return collection_views(self, **options)


# --------------------------------------------------------------------------- driver


def _emit(group: Any, kind: str, **fields: Any) -> None:
    if group is None:
        return
    from topologies.visualization.core import VisualizationEvent

    group.handle(VisualizationEvent(source="introspection", kind=f"introspection.{kind}", **fields))


def collect(inspector: Any, data: Any, sites: Any, reducers: Any = ("moments",), *,
            max_batches: int | None = None, batch_size: int | None = None, progress: bool = True,
            visualizer: Any = None, resume: str | Path | None = None, checkpoint_every: int = 10,
            policy: CapturePolicy | None = None, call_tree: bool = False) -> Collection:
    """Stream ``data`` through ``reducers`` at ``sites`` (see ``Inspector.collect``)."""
    subject = inspector.subject
    made = [make_reducer(r) for r in ([reducers] if isinstance(reducers, (str, Mapping)) or
                                      hasattr(reducers, "init") else reducers)]
    names = [r.name for r in made]
    if len(set(names)) != len(names):
        raise ValueError(f"reducer names must be unique, got {names}")
    policy = policy or CapturePolicy(keep="none")
    session = Session(subject, sites=sites, policy=policy, call_tree=call_tree,
                      recorder_factory=lambda s, b: sink)
    saved_states, skip = _load_resume(resume)
    # Resumed collections continue from the checkpointed states (merge with an empty state).
    sink = ReducerSink(session, made, states=dict(saved_states))
    total = None
    try:
        total = len(data)
    except TypeError:
        total = None
    if max_batches is not None:
        total = min(total, max_batches) if total is not None else max_batches
    _preflight(inspector, data, made, sites, policy, total, batch_size)
    group = None
    if visualizer is not None and visualizer is not False:
        from topologies.visualization import create_visualizer

        group = create_visualizer(visualizer)
        group.prepare(source="introspection", context={"total": total})
    resolver = ExampleResolver(data=data)
    examples = 0
    start = time.perf_counter()
    batches = 0
    index = 0
    try:
        _emit(group, "collection_started", progress=_progress(0, total),
              data={"sites": len(session.requests), "reducers": names})
        with session.persistent():
            for batch_index, batch in enumerate(normalize_inputs(subject, data,
                                                                 batch_size=batch_size,
                                                                 from_data=True)):
                if max_batches is not None and batch_index >= max_batches:
                    break
                index = batch_index
                size = batch.size
                ids = None
                if size is not None:
                    ids = torch.arange(examples, examples + size)
                    for row in range(size):
                        resolver.batch_of[examples + row] = (batch_index, row)
                    for row, toks in enumerate(batch.meta.get("tokens", [])):
                        resolver.tokens[examples + row] = toks
                if batch_index < skip:
                    examples += size or 0
                    continue
                sink.start(BatchMeta(batch_index, ids, size), size)
                session.run_batch(batch)
                examples += size or 0
                batches += 1
                if resume is not None and batches % checkpoint_every == 0:
                    _save_resume(resume, sink.states, batch_index + 1)
                if progress:
                    _emit(group, "batch_completed", progress=_progress(batch_index + 1, total),
                          metrics={"examples": examples,
                                   "seconds": round(time.perf_counter() - start, 4)})
        if resume is not None:
            _save_resume(resume, sink.states, index + 1)
        results: dict[str, dict[str, Any]] = {}
        for (site, name), state in sink.states.items():
            reducer = next(r for r in made if r.name == name)
            results.setdefault(site, {})[name] = reducer.finalize(state)
        seconds = time.perf_counter() - start
        _emit(group, "collection_completed", progress=_progress(index + 1, total),
              metrics={"examples": examples, "seconds": round(seconds, 4)})
    finally:
        if group is not None:
            group.close()
    provenance = {"subject": subject.describe(), "sites": str(sites), "reducers": names,
                  "batches": batches, "examples": examples, "policy": policy.to_dict()}
    assumptions = list(dict.fromkeys(sink.assumptions + [
        "occurrences of a module within one batch are pooled as extra samples"]))
    return Collection(results, reducers=made, batches=batches + skip, examples=examples,
                      order=sink.order, resolver=resolver, provenance=provenance,
                      warnings=list(dict.fromkeys(session.warnings + sink.warnings)),
                      assumptions=assumptions, seconds=seconds)


def _progress(current: int, total: int | None) -> Any:
    from topologies.visualization.core import Progress

    return Progress(current, total)


def _preflight(inspector: Any, data: Any, reducers: list[Any], sites: Any, policy: CapturePolicy,
               total: int | None, batch_size: int | None) -> None:
    """Raise ``CaptureBudgetExceeded`` before the run when raw collection cannot fit the budget."""
    collectors = [r for r in reducers if isinstance(r, Collect)]
    budget = policy.max_bytes
    if not collectors or budget is None or total is None or policy.spill_dir is not None:
        return
    first = next(iter(normalize_inputs(inspector.subject, data, batch_size=batch_size,
                                       from_data=True)), None)
    if first is None:
        return
    trace = inspector.trace(first, sites=sites, keep="summary", call_tree=False)
    per_batch = 0
    for record in trace.records.values():
        dtype = getattr(torch, record.dtype, torch.float32)
        shape = record.shape
        if collectors[0].pool and len(shape) >= 2:
            shape = (shape[0], shape[-1])
        per_batch += int(torch.tensor(shape).prod()) * torch.empty((), dtype=dtype).element_size()
    needed = per_batch * total
    limit = min(budget, *(c.limit_bytes for c in collectors))
    if needed > limit:
        raise CaptureBudgetExceeded(budget_message(needed, limit, f"{total} batch(es) x "
                                                   f"{per_batch / 2**20:.2f} MiB"))


def _load_resume(path: str | Path | None) -> tuple[dict, int]:
    if path is None or not Path(path).exists():
        return {}, 0
    blob = torch.load(path, weights_only=True)
    states = {tuple(k.split("\x00")): v for k, v in blob["states"].items()}
    return states, int(blob["batches"])


def _strip(state: Any) -> Any:
    if isinstance(state, dict):
        return {k: _strip(v) for k, v in state.items() if k not in ("axes",)}
    if isinstance(state, list):
        return [_strip(v) for v in state]
    return state


def _save_resume(path: str | Path, states: dict, batches: int) -> None:
    blob = {"batches": batches, "states": {"\x00".join(k): _strip(v) for k, v in states.items()}}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(blob, tmp)
    tmp.replace(target)


__all__ = ["ROOT", "Collection", "CollectionPayload", "ExampleResolver", "ReducerSink", "collect"]
