"""Structure: static module tree, dynamic call tree, and module/op dataflow graphs (U, F)."""
from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from topologies.introspection import _tree
from topologies.introspection.payloads import Payload, input_fingerprint, make_result
from topologies.introspection.sites import ROOT
from topologies.introspection.techniques import Technique, builtin
from topologies.introspection.views import bars, finish, table
from topologies.visualization.views import ViewSpec


@dataclass
class StructurePayload(Payload):
    modules: list[dict[str, Any]]
    parameters: dict[str, Any]
    buffers: dict[str, Any] = field(default_factory=dict)

    def to_records(self) -> list[dict[str, Any]]:
        return [{k: v for k, v in m.items() if k != "aliases"} for m in self.modules]

    def views(self, **options: Any) -> list[ViewSpec]:
        nodes = [{"path": m["path"], "depth": m["depth"],
                  "label": f"{m['path'].rsplit('.', 1)[-1]} ({m['type'].rsplit('.', 1)[-1]})",
                  "metrics": {"params": m["params"]}} for m in self.modules]
        return finish([ViewSpec("tree", "Module tree", {"nodes": nodes}),
                       table("Parameters", [self.parameters])], **options)


@builtin
class Structure(Technique):
    name = "structure"
    tier = "U"
    result_kind = "introspection.structure"
    summary = "static module tree (no forward pass)"

    def run(self, subject: Any, inputs: Any = None, **params: Any) -> Any:
        from topologies.analysis import analyze_model

        engineering = analyze_model(subject.model)
        structure = engineering.payload["structure"]
        modules = [info.to_dict() for info in subject.index.modules.values()]
        payload = StructurePayload(modules, dict(structure.get("parameters", {})),
                                   {"count": structure.get("buffers"),
                                    "bytes": structure.get("buffer_bytes")})
        return make_result(self, payload, subject, stages=[engineering],
                           assumptions=[("parameter counts come from analysis.analyze_model "
                                        "(unique parameter objects)")])


@dataclass
class CallTreePayload(Payload):
    calls: list[dict[str, Any]]
    never_called: list[str]
    multi_called: dict[str, int]
    output_summary: dict[str, Any] | None = None

    def to_records(self) -> list[dict[str, Any]]:
        return [{"path": c["path"], "occurrence": c["occurrence"], "type": c["type"],
                 "depth": c["depth"], "outputs": str(c["outputs"])} for c in self.calls]

    def views(self, **options: Any) -> list[ViewSpec]:
        nodes = [{"path": c["path"], "depth": c["depth"],
                  "label": f"{c['path'].rsplit('.', 1)[-1]}@{c['occurrence']} ({c['type']})",
                  "metrics": {"out": str(c["outputs"])[:24]}} for c in self.calls]
        views = [ViewSpec("tree", "Call tree", {"nodes": nodes})]
        if self.never_called:
            views.append(table("Never-called modules", [{"module": p} for p in self.never_called]))
        if self.multi_called:
            views.append(table("Modules called more than once",
                               [{"module": p, "calls": n} for p, n in self.multi_called.items()]))
        return finish(views, **options)


@builtin
class CallTreeTechnique(Technique):
    name = "structure.call_tree"
    tier = "U"
    result_kind = "introspection.call_tree"
    summary = "dynamic call order, occurrences, shapes, never-called modules"

    def run(self, subject: Any, inputs: Any = None, **params: Any) -> Any:
        trace = subject.inspector.trace(inputs, sites=ROOT, keep="summary", call_tree=True)
        tree = trace.calls
        roots = [r for r in trace.records.values() if r.key is not None and r.key.path == ""]
        summary = roots[-1].summary if roots else None
        payload = CallTreePayload([c.to_dict() for c in tree], [p or ROOT for p in tree.never_called()],
                                  {p or ROOT: n for p, n in tree.multi_called().items()}, summary)
        return make_result(self, payload, subject, warnings=trace.warnings,
                           assumptions=["occurrence k = the k-th call of a module within this run",
                                        ("ModuleList/ModuleDict containers have no forward and are "
                                        "not tracked; never-called = tracked modules that did not "
                                        "run for these inputs"), *trace.assumptions],
                           provenance={"inputs": input_fingerprint(inputs)})


@dataclass
class GraphPayload(Payload):
    level: str
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    truncated: bool = False

    def to_records(self) -> list[dict[str, Any]]:
        return self.edges

    def views(self, **options: Any) -> list[ViewSpec]:
        nodes = [{"path": n["id"], "depth": n.get("depth", 0), "label": n["label"],
                  "metrics": {"in": n.get("in_degree", 0), "out": n.get("out_degree", 0)}}
                 for n in self.nodes]
        extra: list[ViewSpec] = []
        if self.level == "op":
            counts: dict[str, int] = {}
            for node in self.nodes:
                if "/fn:" in node["id"]:
                    op = node["id"].split("/fn:", 1)[1].split("#", 1)[0]
                    counts[op] = counts.get(op, 0) + 1
            ordered = sorted(counts.items(), key=lambda kv: -kv[1])[:30]
            extra.append(bars("Ops by type", [k for k, _ in ordered], [v for _, v in ordered]))
        if len(self.nodes) <= 60:
            ids = [n["id"] for n in self.nodes]
            position = {node: i for i, node in enumerate(ids)}
            matrix = [[0] * len(ids) for _ in ids]
            for edge in self.edges:
                if edge["producer"] in position and edge["consumer"] in position:
                    matrix[position[edge["producer"]]][position[edge["consumer"]]] = edge["count"]
            if ids:
                extra.append(ViewSpec("heatmap", "Adjacency (producer -> consumer)",
                                      {"matrix": matrix, "row_labels": ids, "col_labels": ids}))
        return finish([ViewSpec("tree", f"{self.level}-level dataflow", {"nodes": nodes}), *extra,
                       table("Edges", self.edges[:200],
                             caption="edges inferred by tensor identity (views/copies/in-place "
                                     "ops can break identity)")], **options)


class _IdentityRecorder:
    """Records producer/consumer tensor identities per call (no values kept)."""

    def __init__(self, session: Any):
        self.session = session
        self.produced: dict[tuple[int, int, int], tuple[str, Any]] = {}
        self.edges: dict[tuple[str, str], int] = {}
        self.nodes: dict[str, dict[str, Any]] = {}
        self.warnings: list[str] = []
        self.assumptions: list[str] = []
        self.grad_enabled = False
        self.store = None

    @staticmethod
    def _ident(t: torch.Tensor) -> tuple[int, int, int]:
        version = 0 if torch.is_inference(t) else t._version
        return (id(t), t.data_ptr() if t.numel() else 0, version)

    def wants(self, path: str) -> bool:
        return True

    def _producer(self, tensor: torch.Tensor) -> str | None:
        entry = self.produced.get(self._ident(tensor))
        # A dead weakref means the id may have been reused by an unrelated tensor.
        if entry is None or entry[1]() is not tensor:
            return None
        return entry[0]

    def _produce(self, tensor: torch.Tensor, node: str) -> None:
        self.produced[self._ident(tensor)] = (node, weakref.ref(tensor))

    def on_pre(self, path: str, occurrence: int, module: nn.Module, args: tuple, kwargs: dict) -> None:
        node = f"{path or ROOT}@{occurrence}"
        self.nodes.setdefault(node, {"id": node, "label": f"{path or ROOT}@{occurrence} "
                                     f"({type(module).__name__})", "depth": path.count(".") + 1
                                     if path else 0})
        for _, tensor in _tree.leaves((args, kwargs)):
            producer = self._producer(tensor)
            if producer is not None and producer != node:
                self.edges[(producer, node)] = self.edges.get((producer, node), 0) + 1

    def on_post(self, path: str, occurrence: int, module: nn.Module, output: Any) -> None:
        node = f"{path or ROOT}@{occurrence}"
        for _, tensor in _tree.leaves(output):
            self._produce(tensor, node)

    def on_op(self, node: str, label: str, inputs: list[torch.Tensor], outputs: list[torch.Tensor],
              depth: int) -> None:
        self.nodes.setdefault(node, {"id": node, "label": label, "depth": depth})
        for tensor in inputs:
            producer = self._producer(tensor)
            if producer is not None and producer != node:
                self.edges[(producer, node)] = self.edges.get((producer, node), 0) + 1
        for tensor in outputs:
            self._produce(tensor, node)

    def protect(self, args: tuple) -> None:
        return

    def remove_grad_hooks(self) -> None:
        return


@builtin
class Graph(Technique):
    name = "structure.graph"
    tier = "U"
    result_kind = "introspection.graph"
    summary = "module-level dataflow by tensor identity (op level needs functional capture)"

    def run(self, subject: Any, inputs: Any = None, *, level: str = "module", max_nodes: int = 2000,
            **params: Any) -> Any:
        from topologies.introspection.capture import CapturePolicy
        from topologies.introspection.session import Session

        if level not in ("module", "op"):
            raise ValueError("level must be 'module' or 'op'")
        holder: dict[str, Any] = {}

        def factory(session: Any, batch: Any) -> Any:
            holder["recorder"] = _IdentityRecorder(session)
            return holder["recorder"]
        session = Session(subject, sites="*", policy=CapturePolicy(keep="none"), call_tree=True,
                          recorder_factory=factory, ops="*" if level == "op" else None)
        if session.functional is not None:
            session.functional.graph = True
            session.functional.max_ops = max_nodes
        session.run(inputs)
        recorder = holder["recorder"]
        degree_in: dict[str, int] = {}
        degree_out: dict[str, int] = {}
        for producer, consumer in recorder.edges:
            degree_out[producer] = degree_out.get(producer, 0) + 1
            degree_in[consumer] = degree_in.get(consumer, 0) + 1
        nodes = [{**n, "in_degree": degree_in.get(n["id"], 0), "out_degree": degree_out.get(n["id"], 0)}
                 for n in recorder.nodes.values()]
        truncated = len(nodes) > max_nodes
        edges = [{"producer": p, "consumer": c, "count": n, "basis": "inferred_by_identity"}
                 for (p, c), n in recorder.edges.items()]
        payload = GraphPayload(level, nodes[:max_nodes], edges, truncated)
        warnings = ["graph truncated at max_nodes"] if truncated else []
        return make_result(self, payload, subject, warnings=warnings,
                           assumptions=[("edges are inferred by tensor identity (id, data_ptr, "
                                        "version); views, copies and in-place ops can hide edges")],
                           provenance={"inputs": input_fingerprint(inputs), "level": level})
