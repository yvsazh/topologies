"""Units (U): top activating examples, label selectivity, concept overlap (correlational)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from topologies.introspection.errors import SemanticsRequired
from topologies.introspection.payloads import Payload, input_fingerprint, make_result
from topologies.introspection.reducers import TopK
from topologies.introspection.techniques import Technique, builtin
from topologies.introspection.techniques._features import batches, collect_features
from topologies.introspection.views import bars, finish, table
from topologies.visualization.views import ViewSpec

CORRELATIONAL = ("correlational: these statistics describe co-occurrence of activations and "
                 "inputs/labels; they do not show that a unit causes or is responsible for anything")


@dataclass
class TopExamplesPayload(Payload):
    site: str
    units: list[int]
    values: torch.Tensor          # [k, U]
    example_ids: torch.Tensor     # [k, U]
    positions: torch.Tensor       # [k, U]
    examples: dict[int, list[dict[str, Any]]] = field(default_factory=dict)

    def to_records(self) -> list[dict[str, Any]]:
        rows = []
        for u_index, unit in enumerate(self.units):
            for rank in range(self.values.shape[0]):
                entry = self.examples.get(unit, [{}] * self.values.shape[0])[rank]
                rows.append({"unit": unit, "rank": rank, "value": float(self.values[rank, u_index]),
                             "example": int(self.example_ids[rank, u_index]),
                             "position": int(self.positions[rank, u_index]),
                             "context": " ".join(entry.get("tokens", [])) if entry else ""})
        return rows

    def views(self, **options: Any) -> list[ViewSpec]:
        views: list[ViewSpec] = []
        for unit in self.units[:4]:
            entries = self.examples.get(unit, [])
            token_rows = [e for e in entries if "tokens" in e]
            if token_rows:
                tokens = [t for e in token_rows[:1] for t in e["tokens"]]
                scores = [1.0 if i == token_rows[0].get("focus") else 0.0 for i in range(len(tokens))]
                views.append(ViewSpec("token_heat", f"{self.site} unit {unit}: top context",
                                      {"tokens": tokens, "scores": scores, "site": self.site},
                                      {}, CORRELATIONAL))
            images = [e["input"] for e in entries if isinstance(e.get("input"), torch.Tensor)
                      and e["input"].ndim in (2, 3)]
            if images:
                views.append(ViewSpec("image_grid", f"{self.site} unit {unit}: top examples",
                                      {"images": [i.float().tolist() for i in images[:8]],
                                       "labels": [f"#{r}" for r in range(len(images[:8]))],
                                       "site": self.site}, {}, CORRELATIONAL))
        views.append(table(f"Top activations at {self.site}", self.to_records()[:40],
                           caption=CORRELATIONAL))
        return finish(views, **options)


@builtin
class TopExamples(Technique):
    name = "neurons.top_examples"
    tier = "U"
    result_kind = "introspection.top_examples"
    summary = "top-k activating examples per unit with resolved inputs"

    def run(self, subject: Any, inputs: Any = None, *, site: str = "", units: Any = None,
            k: int = 20, per: str = "example", window: int = 8, max_batches: int | None = None,
            **params: Any) -> Any:
        path = subject.index.resolve_modules(site)[0]
        reducer = TopK(k=k, per=per, units=tuple(units) if units is not None else None)
        collection = subject.inspector.collect(inputs, path or "<root>", [reducer],
                                               max_batches=max_batches, progress=False)
        result = collection[path or "<root>"]["topk"]
        if result.get("values") is None:
            raise SemanticsRequired(f"no activations collected at {site!r}")
        unit_list = list(units) if units is not None else list(range(result["values"].shape[1]))
        examples: dict[int, list[dict[str, Any]]] = {}
        for u_index, unit in enumerate(unit_list):
            entries = []
            for rank in range(result["values"].shape[0]):
                ident = int(result["example_ids"][rank, u_index])
                position = int(result["positions"][rank, u_index])
                entries.append(collection.resolver.resolve(ident, position=position, window=window))
            examples[unit] = entries
        payload = TopExamplesPayload(path or "<root>", unit_list, result["values"],
                                     result["example_ids"], result["positions"], examples)
        return make_result(self, payload, subject, assumptions=[CORRELATIONAL, f"per={per!r}"],
                           provenance={"inputs": input_fingerprint(inputs), "k": k})


@dataclass
class SelectivityPayload(Payload):
    site: str
    classes: list[Any]
    class_means: torch.Tensor     # [C, U]
    selectivity: torch.Tensor     # [U]
    mutual_information: torch.Tensor  # [U] (bits)
    concept_iou: dict[str, torch.Tensor] = field(default_factory=dict)

    def to_records(self) -> list[dict[str, Any]]:
        best = self.class_means.argmax(0)
        rows = [{"unit": u, "selectivity": float(self.selectivity[u]),
                 "mutual_information_bits": float(self.mutual_information[u]),
                 "preferred_class": self.classes[int(best[u])]} for u in range(self.selectivity.numel())]
        for concept, iou in self.concept_iou.items():
            for u in range(iou.numel()):
                rows[u][f"iou_{concept}"] = float(iou[u])
        return rows

    def views(self, **options: Any) -> list[ViewSpec]:
        order = self.selectivity.argsort(descending=True)[:30].tolist()
        return finish([bars(f"Selectivity index at {self.site}", [str(u) for u in order],
                            [float(self.selectivity[u]) for u in order], caption=CORRELATIONAL),
                       bars("Mutual information (bits)", [str(u) for u in order],
                            [float(self.mutual_information[u]) for u in order])], **options)


@builtin
class Selectivity(Technique):
    name = "neurons.selectivity"
    tier = "U"
    result_kind = "introspection.selectivity"
    summary = "per-unit class means, selectivity index, binned mutual information, mask IoU"

    def run(self, subject: Any, inputs: Any = None, *, site: str = "", labels: Any = None,
            pool: str = "mean", bins: int = 10, masks: Any = None, quantile: float = 0.995,
            max_batches: int | None = None, **params: Any) -> Any:
        path = subject.index.resolve_modules(site)[0]
        feats, ys, notes = collect_features(subject, inputs, [path], pool=pool, labels=labels,
                                            max_batches=max_batches)
        if ys is None:
            raise ValueError("selectivity needs labels= (tensor, field name or callable) or "
                             "(input, target) data")
        x = feats[path].double()
        classes = sorted(set(ys.tolist()))
        means = torch.stack([x[ys == c].mean(0) for c in classes])
        positive = means.clamp_min(0)
        top = positive.max(0).values
        rest = (positive.sum(0) - top) / max(len(classes) - 1, 1)
        selectivity = ((top - rest) / (top + rest).clamp_min(1e-12)).float()
        mi = mutual_information(x, ys, bins)
        payload = SelectivityPayload(path or "<root>", classes, means.float(), selectivity, mi)
        assumptions = [CORRELATIONAL, ("selectivity index = (mu_max - mu_rest) / (mu_max + mu_rest) "
                       "on ReLU-clipped class means"), f"mutual information: {bins} quantile bins",
                       *notes]
        if masks is not None:
            payload.concept_iou = concept_overlap(subject, inputs, path, masks, quantile,
                                                  max_batches)
            assumptions.append(f"concept IoU: activation maps thresholded at the {quantile} "
                               "quantile, upsampled bilinearly to the mask size")
        return make_result(self, payload, subject, assumptions=assumptions,
                           provenance={"inputs": input_fingerprint(inputs)})


def mutual_information(x: torch.Tensor, labels: torch.Tensor, bins: int = 10) -> torch.Tensor:
    """Binned MI (bits) between each unit's activation (quantile bins) and the labels."""
    classes, label_index = torch.unique(labels, return_inverse=True)
    n = x.shape[0]
    out = torch.zeros(x.shape[1])
    qs = torch.linspace(0, 1, bins + 1, dtype=x.dtype)[1:-1]
    for u in range(x.shape[1]):
        edges = torch.quantile(x[:, u], qs)
        binned = torch.bucketize(x[:, u].contiguous(), edges)
        joint = torch.zeros(bins, len(classes), dtype=torch.float64)
        joint.index_put_((binned, label_index), torch.ones(n, dtype=torch.float64), accumulate=True)
        joint /= n
        px, py = joint.sum(1, keepdim=True), joint.sum(0, keepdim=True)
        mask = joint > 0
        out[u] = float((joint[mask] * (joint[mask] / (px @ py)[mask]).log2()).sum())
    return out


def concept_overlap(subject: Any, data: Any, path: str, masks: Any, quantile: float,
                    max_batches: int | None) -> dict[str, torch.Tensor]:
    """Network-dissection IoU between thresholded unit maps and concept masks ``{name: [N,H,W]}``."""
    maps = []
    for batch in batches(subject, data, max_batches):
        trace = subject.inspector.trace(batch, sites=path or "<root>", keep="tensors",
                                        call_tree=False, max_bytes=None)
        record = next(iter(trace.records.values()))
        axes = record.axes
        if axes is None or axes.names[:2] != ("batch", "channel") or record.value.ndim != 4:
            raise SemanticsRequired(f"concept overlap at {path!r} needs spatial b c h w axes")
        maps.append(record.value.float())
    acts = torch.cat(maps)
    thresholds = torch.quantile(acts.transpose(0, 1).reshape(acts.shape[1], -1), quantile, dim=1)
    out = {}
    for name, mask in masks.items():
        mask = torch.as_tensor(mask).bool()[: acts.shape[0]]
        up = F.interpolate(acts[: mask.shape[0]], size=mask.shape[-2:], mode="bilinear",
                           align_corners=False)
        on = up > thresholds[None, :, None, None]
        m = mask[:, None]
        inter = (on & m).sum(dim=(0, 2, 3)).double()
        union = (on | m).sum(dim=(0, 2, 3)).double().clamp_min(1)
        out[name] = (inter / union).float()
    return out


__all__ = ["Selectivity", "TopExamples", "concept_overlap", "mutual_information"]
