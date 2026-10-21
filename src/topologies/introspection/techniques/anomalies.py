"""Anomalies: non-finite origin, exploding/vanishing activations, collapse, dead code (U)."""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from topologies.introspection import _tree
from topologies.introspection.capture import CapturePolicy, summary_dict, summary_vector
from topologies.introspection.payloads import Payload, input_fingerprint, make_result
from topologies.introspection.reducers import pool_features
from topologies.introspection.semantics import SemanticSource
from topologies.introspection.session import Session
from topologies.introspection.sites import ROOT
from topologies.introspection.techniques import Technique, builtin
from topologies.introspection.views import finish, lines, table
from topologies.visualization.views import ViewSpec


@dataclass
class AnomaliesPayload(Payload):
    items: list[dict[str, Any]]
    rms: list[dict[str, Any]] = field(default_factory=list)
    nonfinite_origin: dict[str, Any] | None = None

    def to_records(self) -> list[dict[str, Any]]:
        return [{k: (", ".join(v) if isinstance(v, (list, tuple)) else v)
                 for k, v in item.items() if k != "evidence"} for item in self.items]

    def views(self, **options: Any) -> list[ViewSpec]:
        views = []
        if self.items:
            views.append(table("Anomalies", [{"severity": i["severity"], "basis": i["basis"],
                                              "title": i["title"], "sites": ", ".join(i["sites"])[:60]}
                                             for i in self.items]))
        if self.rms:
            views.append(lines("RMS along the compared sites", [{"y": [r["rms"] for r in self.rms],
                                                                 "label": "rms"}],
                               x_ticks=[r["site"] for r in self.rms], y_label="RMS",
                               log_y=True))
        return finish(views, **options)


class _CallRecorder:
    """Per call (completion order): input/output finiteness, output summary, ranges."""

    def __init__(self, session: Any):
        self.session = session
        self.calls: list[dict[str, Any]] = []
        self.inputs_finite: dict[tuple[str, int], tuple[bool, list[float] | None]] = {}
        self.warnings: list[str] = []
        self.assumptions: list[str] = []
        self.grad_enabled = False
        self.store = None

    def wants(self, path: str) -> bool:
        return True

    def on_pre(self, path: str, occurrence: int, module: nn.Module, args: tuple, kwargs: dict) -> None:
        tensors = [t for _, t in _tree.leaves((args, kwargs)) if t.is_floating_point()]
        finite = all(bool(torch.isfinite(t).all()) for t in tensors)
        rng = _range(tensors)
        self.inputs_finite[(path, occurrence)] = (finite, rng)

    def on_post(self, path: str, occurrence: int, module: nn.Module, output: Any) -> None:
        tensors = [t for _, t in _tree.leaves(output) if t.is_floating_point()]
        finite_in, in_range = self.inputs_finite.pop((path, occurrence), (True, None))
        primary = tensors[0] if tensors else None
        self.calls.append({
            "path": path, "occurrence": occurrence, "type": type(module).__name__,
            "inputs_finite": finite_in, "input_range": in_range,
            "output_finite": all(bool(torch.isfinite(t).all()) for t in tensors),
            "output_range": _range(tensors),
            "summary": summary_dict(summary_vector(primary)) if primary is not None else None,
            "leaf": not any(True for _ in module.children())})

    def protect(self, args: tuple) -> None:
        return

    def remove_grad_hooks(self) -> None:
        return


def _range(tensors: list[torch.Tensor]) -> list[float] | None:
    finite = [t[torch.isfinite(t)] for t in tensors if t.numel()]
    finite = [t for t in finite if t.numel()]
    if not finite:
        return None
    return [min(float(t.min()) for t in finite), max(float(t.max()) for t in finite)]


def _rms(summary: dict[str, Any] | None) -> float | None:
    if not summary:
        return None
    try:
        mean, std = float(summary["mean"]), float(summary["std"])
    except (TypeError, ValueError, KeyError):
        return None
    if not (math.isfinite(mean) and math.isfinite(std)):
        return None
    return math.sqrt(mean ** 2 + std ** 2)


def nonfinite_origin(calls: list[dict[str, Any]]) -> dict[str, Any] | None:
    """First call in completion order whose output is non-finite while its inputs are finite."""
    for call in calls:
        if not call["output_finite"] and call["inputs_finite"]:
            return {"site": call["path"] or ROOT, "occurrence": call["occurrence"],
                    "type": call["type"], "input_range": call["input_range"],
                    "output_range": call["output_range"]}
    return None


@builtin
class Anomalies(Technique):
    name = "anomalies"
    tier = "U"
    result_kind = "introspection.anomalies"
    summary = "non-finite origin, exploding/vanishing RMS, collapse, never-called modules"

    def run(self, subject: Any, inputs: Any = None, *, explode_ratio: float = 10.0,
            vanish_ratio: float = 0.1, collapse_cosine: float = 0.99, collapse_sites: Any = None,
            **params: Any) -> Any:
        holder: dict[str, Any] = {}

        def factory(session: Any, batch: Any) -> Any:
            holder["rec"] = _CallRecorder(session)
            return holder["rec"]
        session = Session(subject, sites="*", policy=CapturePolicy(keep="none"), call_tree=True,
                          recorder_factory=factory)
        trace = session.run(inputs)
        calls = holder["rec"].calls
        items: list[dict[str, Any]] = []
        origin = nonfinite_origin(calls)
        if origin is not None:
            items.append(_item("nonfinite_origin", f"Non-finite values first appear at {origin['site']}",
                               "critical", "measured", [origin["site"]], origin,
                               "this module produced NaN/Inf from finite inputs (first in completion "
                               "order); every later NaN may be a consequence",
                               "inspect its parameters (insp.weights) and input range"))
        # exploding / vanishing between consecutive compared sites
        block_paths = subject.index.resolve_modules("role:block", allow_empty=True)
        compared = [c for c in calls if (c["path"] in block_paths if block_paths else c["leaf"])
                    and c["occurrence"] == 0]
        basis = "measured" if block_paths else "heuristic"
        rms = [{"site": c["path"] or ROOT, "rms": r} for c in compared if (r := _rms(c["summary"]))
               is not None]
        for prev, cur in itertools.pairwise(rms):
            if prev["rms"] <= 0:
                continue
            ratio = cur["rms"] / prev["rms"]
            if ratio > explode_ratio or ratio < vanish_ratio:
                kind = "exploding" if ratio > explode_ratio else "vanishing"
                items.append(_item(f"{kind}_activations", f"{kind.capitalize()} RMS at {cur['site']}",
                                   "warning", basis, [prev["site"], cur["site"]],
                                   {"ratio": ratio, "rms_before": prev["rms"], "rms_after": cur["rms"],
                                    "threshold": explode_ratio if kind == "exploding" else vanish_ratio},
                                   f"RMS changed by x{ratio:.3g} between consecutive "
                                   f"{'blocks' if block_paths else 'leaf modules (heuristic)'}; "
                                   "normalisation layers legitimately rescale",
                                   None))
        never = [p or ROOT for p in trace.calls.never_called()]
        if never:
            unused = sum(sum(p.numel() for p in subject.index.module(n if n != ROOT else "")
                             .parameters()) for n in never)
            items.append(_item("never_called", f"{len(never)} module(s) never ran", "notice",
                               "measured", never, {"modules": never, "unused_parameters": unused},
                               "these modules (and their parameters) took no part in this forward "
                               "pass for these inputs; other inputs/branches may use them", None))
        items.extend(_collapse(subject, inputs, collapse_sites, collapse_cosine))
        payload = AnomaliesPayload(items, rms, origin)
        return make_result(self, payload, subject, warnings=trace.warnings,
                           assumptions=[(f"exploding/vanishing thresholds: x{explode_ratio} / "
                                        f"x{vanish_ratio} between consecutive "
                                        f"{'role:block sites' if block_paths else 'leaf calls'}"),
                                        f"collapse: mean pairwise cosine > {collapse_cosine}"],
                           provenance={"inputs": input_fingerprint(inputs)})


def _item(ident: str, title: str, severity: str, basis: str, sites: list[str],
          evidence: dict[str, Any], explanation: str, suggestion: str | None) -> dict[str, Any]:
    return {"id": ident, "title": title, "severity": severity, "basis": basis, "sites": sites,
            "evidence": evidence, "explanation": explanation, "suggestion": suggestion}


def _collapse(subject: Any, inputs: Any, sites: Any, threshold: float) -> list[dict[str, Any]]:
    index = subject.index
    if sites is None:
        sites = index.resolve_modules("role:block", allow_empty=True) or \
            [p for p in index.paths() if p and "." not in p]
    if not sites:
        return []
    trace = subject.inspector.trace(inputs, sites=sites if isinstance(sites, str) else list(sites),
                                    keep="tensors", call_tree=False, max_bytes=256 * 2**20)
    items = []
    for record in trace.records.values():
        value = record.value
        if value is None or value.ndim < 2 or not value.is_floating_point() or value.shape[0] < 2:
            continue
        axes = record.axes
        known = axes is not None and axes.source != SemanticSource.UNKNOWN and \
            axes.feature_axis is not None and axes.batch_axis == 0
        pooled = pool_features(value.float(), axes, "mean") if known and value.ndim > 2 else \
            value.float().reshape(value.shape[0], -1)
        if record.axes is None or (record.axes.batch_axis != 0 and not known):
            continue
        normed = torch.nn.functional.normalize(pooled, dim=-1)
        cos = normed @ normed.T
        n = cos.shape[0]
        mean_cos = float((cos.sum() - cos.diagonal().sum()) / (n * (n - 1)))
        if mean_cos > threshold:
            items.append(_item("representation_collapse",
                               f"Representations nearly identical across the batch at {record.site}",
                               "warning", "heuristic", [record.site],
                               {"mean_pairwise_cosine": mean_cos, "threshold": threshold, "batch": n,
                                "pooling": "mean over positions" if known and value.ndim > 2
                                else "flatten"},
                               "examples map to almost the same vector here; on a tiny or "
                               "homogeneous batch this can be expected", None))
    return items
