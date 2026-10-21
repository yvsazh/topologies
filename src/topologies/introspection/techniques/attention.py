"""Attention weights and derived statistics (F / A).

Sources, in order of preference: model-native weights (Hugging Face
``output_attentions``), adapter-declared attention sites, interception of SDPA or
``nn.MultiheadAttention``, and op-mode capture of the softmax inside a module marked as
attention. The last source is **heuristic**: that softmax is assumed to be the
attention softmax. Attention rollout and flow are heuristics, not explanations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from topologies.introspection.errors import IntrospectionError
from topologies.introspection.payloads import Payload, input_fingerprint, make_result
from topologies.introspection.techniques import AVAILABLE, Availability, Technique, builtin
from topologies.introspection.views import finish, heatmap, lines
from topologies.visualization.views import ViewSpec


@dataclass
class AttentionPayload(Payload):
    weights: dict[str, torch.Tensor]
    sources: dict[str, str]
    entropy: dict[str, torch.Tensor] = field(default_factory=dict)
    distance: dict[str, torch.Tensor] = field(default_factory=dict)
    first_token: dict[str, torch.Tensor] = field(default_factory=dict)
    rollout: torch.Tensor | None = None
    head_similarity: torch.Tensor | None = None
    head_labels: list[str] = field(default_factory=list)
    tokens: list[list[str]] | None = None

    def to_records(self) -> list[dict[str, Any]]:
        rows = []
        for site, ent in self.entropy.items():
            for h, value in enumerate(ent.tolist()):
                row = {"site": site, "head": h, "entropy": value, "source": self.sources.get(site)}
                if site in self.first_token:
                    row["first_token"] = float(self.first_token[site][h])
                if site in self.distance:
                    row["mean_distance"] = float(self.distance[site][h])
                rows.append(row)
        return rows

    def views(self, **options: Any) -> list[ViewSpec]:
        labels = self.tokens[0] if self.tokens else None
        views: list[ViewSpec] = []
        for site, w in list(self.weights.items())[: int(options.get("max_layers", 4))]:
            maps = [{"matrix": w[0, h].tolist()} for h in range(w.shape[1])]
            views.append(ViewSpec("heatmap_grid", f"Attention {site}", {
                "maps": maps, "labels": [f"head {h}" for h in range(w.shape[1])],
                "site": site, "row_labels": labels, "col_labels": labels}, {},
                f"source: {self.sources.get(site)} (example 0)"))
        if self.entropy:
            series = []
            names = list(self.entropy)
            heads = max(e.numel() for e in self.entropy.values())
            for h in range(heads):
                series.append({"y": [float(self.entropy[n][h]) if h < self.entropy[n].numel()
                                     else float("nan") for n in names], "label": f"head {h}"})
            views.append(lines("Mean attention entropy per head", series, x_ticks=names,
                               y_label="nats"))
        if self.rollout is not None:
            views.append(heatmap("Attention rollout (heuristic)", self.rollout[0], rows=labels,
                                 cols=labels, caption="product of head-averaged maps with "
                                 "identity for the residual; a heuristic, not an explanation"))
        if self.head_similarity is not None:
            views.append(heatmap("Head similarity (cosine)", self.head_similarity,
                                 rows=self.head_labels, cols=self.head_labels))
        return finish(views, **options)

    def findings(self, params: dict[str, Any]) -> list[Any]:
        from topologies.introspection.report import Finding

        out = []
        low = float(params.get("attention_entropy_threshold", 0.05))
        sink = float(params.get("attention_sink_threshold", 0.9))
        for site, ent in self.entropy.items():
            w = self.weights[site]
            norm = ent / max(math.log(max(w.shape[-1], 2)), 1e-12)
            collapsed = [h for h, v in enumerate(norm.tolist()) if v < low]
            sinks = [h for h, v in enumerate(self.first_token.get(site, torch.zeros(0)).tolist())
                     if v > sink]
            if collapsed or sinks:
                out.append(Finding("attention_collapse", f"Attention collapse at {site}", "notice",
                                   "measured", (site,),
                                   {"low_entropy_heads": collapsed, "first_token_sink_heads": sinks,
                                    "entropy_threshold_fraction_of_log_s": low,
                                    "sink_threshold": sink},
                                   "these heads attend almost deterministically (low entropy) or "
                                   "mostly to position 0; often benign (e.g. sink heads)", None))
        return out


@builtin
class Attention(Technique):
    name = "attention"
    tier = "F"
    result_kind = "introspection.attention"
    summary = "attention weights (native / SDPA / MHA / softmax op), entropy, rollout"

    def check(self, subject: Any) -> Availability:
        if subject.adapter is not None and subject.adapter.attention_sites(subject.index):
            return AVAILABLE
        return Availability(False, "no SDPA/MHA attention module and no adapter declares attention",
                            "pass sites='<attention module>' with ops='softmax', or an adapter")

    def check_call(self, subject: Any, params: dict[str, Any]) -> Availability:
        return AVAILABLE if params.get("sites") is not None else self.check(subject)

    def run(self, subject: Any, inputs: Any = None, *, sites: Any = None, heads: Any = None,
            queries: Any = None, derived: tuple[str, ...] = ("entropy", "distance", "first_token",
                                                            "rollout", "similarity"),
            method: str | None = None, **params: Any) -> Any:
        weights, sources, assumptions, warnings, tokens = collect_weights(subject, inputs, sites,
                                                                          method)
        if not weights:
            raise IntrospectionError(
                "attention: no attention weights observed (no SDPA/MHA calls, no softmax in the "
                "selected modules, no adapter attention sites); pass sites='<attention module>' "
                "or an adapter")
        for key in list(weights):
            w = weights[key]
            if heads is not None:
                w = w[:, list(heads) if not isinstance(heads, int) else [heads]]
            if queries is not None:
                w = w[:, :, list(queries) if not isinstance(queries, int) else [queries]]
            weights[key] = w
        payload = AttentionPayload(weights, sources, tokens=tokens)
        if "entropy" in derived:
            payload.entropy = {k: entropy(w) for k, w in weights.items()}
        if "distance" in derived:
            payload.distance = {k: distance(w) for k, w in weights.items() if w.shape[-1] == w.shape[-2]}
        if "first_token" in derived:
            payload.first_token = {k: w[..., 0].mean(dim=(0, 2)) for k, w in weights.items()}
        square = [w for w in weights.values() if w.shape[-1] == w.shape[-2]]
        if "rollout" in derived and square and len({w.shape[-1] for w in square}) == 1 \
                and queries is None:
            payload.rollout = rollout(square)
            assumptions.append("rollout: layers in capture order, A' = 0.5 A + 0.5 I (heuristic)")
        if "similarity" in derived:
            payload.head_similarity, payload.head_labels = head_similarity(weights)
        assumptions.append("attention weights describe where information may flow; they are "
                           "not explanations of the output")
        return make_result(self, payload, subject, warnings=warnings, assumptions=assumptions,
                           provenance={"inputs": input_fingerprint(inputs), "sources": sources})


def collect_weights(subject: Any, inputs: Any, sites: Any, method: str | None
                    ) -> tuple[dict[str, torch.Tensor], dict[str, str], list[str], list[str], Any]:
    index = subject.index
    assumptions: list[str] = []
    warnings: list[str] = []
    from topologies.introspection.adapters.huggingface import is_huggingface

    if sites is None and (method in (None, "native")) and is_huggingface(subject.model):
        native = _native(subject, inputs)
        if native is not None:
            return native[0], native[1], ["model-native attention weights (output_attentions)"], \
                [], native[2]
    if sites is None:
        declared = subject.adapter.attention_sites(index) if subject.adapter is not None else []
        plan = [(s.path, method or s.method) for s in declared if s.method != "native"]
    else:
        plan = []
        for path in index.resolve_modules(sites):
            module = index.module(path)
            if method is not None:
                plan.append((path, method))
            elif isinstance(module, nn.MultiheadAttention):
                plan.append((path, "mha"))
            else:
                plan.append((path, "sdpa"))
    if not plan:
        return {}, {}, assumptions, warnings, None
    selectors = []
    for path, how in plan:
        if how == "sdpa":
            selectors.append(f"{path or '<root>'}/fn:sdpa:weights")
        elif how == "mha":
            selectors.append(f"{path or '<root>'}/fn:mha:weights")
        elif how == "softmax":
            selectors.append(f"{path or '<root>'}/fn:softmax")
    trace = subject.inspector.trace(inputs, sites=",".join(selectors), keep="tensors",
                                    call_tree=False, max_bytes=None)
    weights: dict[str, torch.Tensor] = {}
    sources: dict[str, str] = {}
    for record in sorted(trace.records.values(), key=lambda r: (r.meta.get("order", 0),
                                                                r.occurrence)):
        key = record.key
        w = record.value
        if w is None or w.ndim != 4:
            continue
        name = f"{key.path or '<root>'}/fn:{key.op}#{key.op_index}"
        if record.occurrence:
            name += f"@{record.occurrence}"
        weights[name] = w.float()
        sources[name] = {"sdpa": "sdpa-recomputed", "mha": "mha-need_weights",
                         "softmax": "softmax-op (heuristic)"}[key.op]
    if any(s.startswith("softmax") for s in sources.values()):
        assumptions.append("softmax op inside an attention module is assumed to be the attention "
                           "softmax (heuristic); axes assumed (batch, heads, query, key)")
    if any(path and how == "sdpa" for path, how in plan) and not any(
            v == "sdpa-recomputed" for v in sources.values()) and sites is not None:
        # Fall back to op-mode softmax for modules that compute attention by hand.
        retry = [p for p, how in plan if how == "sdpa"]
        if retry:
            fallback = collect_weights(subject, inputs, ",".join(p or "<root>" for p in retry),
                                       "softmax")
            weights.update(fallback[0])
            sources.update(fallback[1])
            assumptions.extend(fallback[2])
    assumptions.extend(trace.assumptions)
    warnings.extend(trace.warnings)
    return weights, sources, assumptions, warnings, trace.metadata.get("tokens")


def _native(subject: Any, inputs: Any) -> Any:
    from topologies._execution import ModelCall
    from topologies.introspection.inputs import Batch, normalize_inputs

    batch = next(iter(normalize_inputs(subject, inputs)))
    if batch.call is None:
        return None
    call = Batch(ModelCall(batch.call.args, {**batch.call.kwargs, "output_attentions": True}),
                 meta=batch.meta)
    trace = subject.inspector.trace(call, sites=None, keep="none", call_tree=False)
    attentions = getattr(trace.output, "attentions", None)
    if not attentions:
        return None
    layers = subject.index.resolve_modules("role:attention", allow_empty=True)
    weights, sources = {}, {}
    for i, w in enumerate(attentions):
        name = f"{layers[i]}/native" if i < len(layers) else f"layer{i}/native"
        weights[name] = w.float()
        sources[name] = "native"
    return weights, sources, batch.meta.get("tokens")


def entropy(w: torch.Tensor) -> torch.Tensor:
    """Mean row entropy (nats) per head: ``[B, H, Q, K] -> [H]``."""
    p = w.clamp_min(1e-12)
    return -(w * p.log()).sum(-1).mean(dim=(0, 2))


def distance(w: torch.Tensor) -> torch.Tensor:
    q = torch.arange(w.shape[-2], dtype=torch.float32)[:, None]
    k = torch.arange(w.shape[-1], dtype=torch.float32)[None]
    return (w * (q - k).abs()).sum(-1).mean(dim=(0, 2))


def rollout(layers: list[torch.Tensor]) -> torch.Tensor:
    size = layers[0].shape[-1]
    eye = torch.eye(size)
    result = eye.expand(layers[0].shape[0], size, size).clone()
    for w in layers:
        a = 0.5 * w.mean(1) + 0.5 * eye
        a = a / a.sum(-1, keepdim=True)
        result = a @ result
    return result


def head_similarity(weights: dict[str, torch.Tensor], limit: int = 64
                    ) -> tuple[torch.Tensor | None, list[str]]:
    shapes = {tuple(w.shape[-2:]) for w in weights.values()}
    if len(shapes) != 1:
        return None, []
    vectors, labels = [], []
    for name, w in weights.items():
        for h in range(w.shape[1]):
            if len(vectors) >= limit:
                break
            vectors.append(w[:, h].mean(0).reshape(-1))
            labels.append(f"{name.split('/fn:')[0].rsplit('.', 2)[-2:][0]}:{h}")
    stacked = torch.nn.functional.normalize(torch.stack(vectors), dim=-1)
    return stacked @ stacked.T, labels
