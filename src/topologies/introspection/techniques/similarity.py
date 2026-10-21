"""Representation similarity (U): CKA (linear, minibatch), SVCCA, cosine; model comparison."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from topologies.introspection.payloads import Payload, input_fingerprint, make_result
from topologies.introspection.reducers import CKAAccumulator, linear_cka
from topologies.introspection.techniques import Technique, builtin
from topologies.introspection.techniques._features import batches, collect_features, pooled
from topologies.introspection.views import finish, heatmap, table
from topologies.visualization.views import ViewSpec

METHODS = ("cka_linear", "cka_minibatch", "svcca", "cosine")


@dataclass
class SimilarityPayload(Payload):
    method: str
    rows: list[str]
    cols: list[str]
    matrix: torch.Tensor
    examples: int
    weights: list[dict[str, Any]] = field(default_factory=list)

    def to_records(self) -> list[dict[str, Any]]:
        out = [{"a": a, "b": b, "similarity": float(self.matrix[i, j])}
               for i, a in enumerate(self.rows) for j, b in enumerate(self.cols)]
        return out + [{"a": w["name"], "b": w["name"], **w} for w in self.weights]

    def views(self, **options: Any) -> list[ViewSpec]:
        views = [heatmap(f"Representation similarity ({self.method})", self.matrix, rows=self.rows,
                         cols=self.cols, caption=f"{self.examples} examples")]
        if self.weights:
            views.append(table("Weight changes", self.weights))
        return finish(views, **options)


def svcca(x: torch.Tensor, y: torch.Tensor, keep: float = 0.99) -> float:
    """SVCCA: SVD-reduce each side to ``keep`` variance, then mean canonical correlation."""
    def reduce(a: torch.Tensor) -> torch.Tensor:
        a = a.double() - a.double().mean(0)
        u, s, _ = torch.linalg.svd(a, full_matrices=False)
        energy = (s ** 2).cumsum(0) / (s ** 2).sum().clamp_min(1e-30)
        k = int((energy < keep).sum()) + 1
        return u[:, :k] * s[:k]

    def basis(a: torch.Tensor) -> torch.Tensor:
        q, _ = torch.linalg.qr(a - a.mean(0))
        return q
    qx, qy = basis(reduce(x)), basis(reduce(y))
    correlations = torch.linalg.svdvals(qx.T @ qy).clamp(0, 1)
    return float(correlations.mean())


def cosine(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.shape != y.shape:
        return float("nan")
    return float(torch.nn.functional.cosine_similarity(x.double(), y.double(), dim=-1).mean())


def _matrix(method: str, a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> torch.Tensor:
    fn = {"cka_linear": linear_cka, "svcca": svcca, "cosine": cosine}[method]
    return torch.tensor([[fn(a[r], b[c]) for c in b] for r in a], dtype=torch.float32)


@builtin
class Similarity(Technique):
    name = "similarity"
    tier = "U"
    result_kind = "introspection.similarity"
    summary = "layer x layer CKA / SVCCA / cosine over a dataset"

    def run(self, subject: Any, inputs: Any = None, *, sites: Any = None,
            method: str = "cka_minibatch", pool: str = "mean", max_batches: int | None = None,
            **params: Any) -> Any:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        index = subject.index
        if sites is None:
            sites = "role:block" if index.resolve_modules("role:block", allow_empty=True) else \
                [p for p in index.paths() if p and "." not in p]
        paths = index.resolve_modules(sites)
        assumptions = [f"representations pooled with pool={pool!r} to [examples, features]"]
        if method == "cka_minibatch":
            matrix, examples, notes = _minibatch(subject, inputs, paths, pool, max_batches)
            assumptions.append("minibatch CKA: unbiased HSIC accumulated over batches "
                               "(Nguyen et al. 2021); approximate")
        else:
            feats, _, notes = collect_features(subject, inputs, paths, pool=pool,
                                               max_batches=max_batches)
            matrix = _matrix(method, feats, feats)
            examples = next(iter(feats.values())).shape[0]
        labels = [p or "<root>" for p in paths]
        payload = SimilarityPayload(method, labels, labels, matrix, examples)
        return make_result(self, payload, subject, assumptions=[*assumptions, *notes],
                           provenance={"inputs": input_fingerprint(inputs), "method": method})


def _minibatch(subject: Any, data: Any, paths: list[str], pool: str, max_batches: int | None,
               other: Any = None, other_paths: list[str] | None = None
               ) -> tuple[torch.Tensor, int, list[str]]:
    acc = CKAAccumulator(paths, other_paths)
    notes: list[str] = []
    examples = 0
    sel = ",".join(p or "<root>" for p in paths)
    for batch in batches(subject, data, max_batches):
        feats = _features(subject, batch, paths, sel, pool, notes)
        cols = None
        if other is not None:
            cols = _features(other, batch, other_paths, ",".join(p or "<root>" for p in other_paths),
                             pool, notes)
        acc.update(feats, cols)
        examples += next(iter(feats.values())).shape[0]
    return acc.finalize(), examples, notes


def _features(subject: Any, batch: Any, paths: list[str], selector: str, pool: str,
              notes: list[str]) -> dict[str, torch.Tensor]:
    trace = subject.inspector.trace(batch, sites=selector, keep="tensors", call_tree=False,
                                    max_bytes=None)
    out = {}
    for path in paths:
        record = max((r for r in trace.records.values() if r.key.path == path and r.key.op is None),
                     key=lambda r: r.occurrence)
        out[path], how = pooled(record.value, record.axes, pool, path)
        if f"{path}: {how}" not in notes:
            notes.append(f"{path}: {how}")
    return out


def compare(a: Any, b: Any, data: Any, *, sites_a: Any = None, sites_b: Any = None,
            method: str = "cka_minibatch", pool: str = "mean", max_batches: int | None = None) -> Any:
    """Compare two models / checkpoints on the same data (CKA etc.) and their weights."""
    from topologies.introspection.api import Inspector, introspect

    ia = a if isinstance(a, Inspector) else introspect(a)
    ib = b if isinstance(b, Inspector) else introspect(b)
    def default(insp: Any, sites: Any) -> list[str]:
        if sites is None:
            sites = "role:block" if insp.index.resolve_modules("role:block", allow_empty=True) else \
                [p for p in insp.index.paths() if p and "." not in p]
        return insp.index.resolve_modules(sites)
    pa, pb = default(ia, sites_a), default(ib, sites_b)
    if method == "cka_minibatch":
        matrix, examples, notes = _minibatch(ia.subject, data, pa, pool, max_batches, ib.subject, pb)
    else:
        fa, _, notes = collect_features(ia.subject, data, pa, pool=pool, max_batches=max_batches)
        fb, _, more = collect_features(ib.subject, data, pb, pool=pool, max_batches=max_batches)
        notes += more
        matrix = _matrix(method, fa, fb)
        examples = next(iter(fa.values())).shape[0]
    weights = weight_changes(ia.model, ib.model)
    payload = SimilarityPayload(method, [f"a:{p or '<root>'}" for p in pa],
                                [f"b:{p or '<root>'}" for p in pb], matrix, examples, weights)
    technique = Similarity()
    return make_result(technique, payload, ia.subject, kind="introspection.compare",
                       name="compare", assumptions=[f"pool={pool!r}", *notes,
                                                    *([("weights compared where state-dict keys "
                                                       "and shapes match")] if weights else [])],
                       provenance={"model_b": ib.subject.describe(), "method": method})


def weight_changes(a: torch.nn.Module, b: torch.nn.Module) -> list[dict[str, Any]]:
    sa, sb = a.state_dict(), b.state_dict()
    rows = []
    for name, wa in sa.items():
        wb = sb.get(name)
        if wb is None or wb.shape != wa.shape or not wa.is_floating_point():
            continue
        x, y = wa.double().reshape(-1), wb.double().reshape(-1)
        rows.append({"name": name, "relative_change": float((y - x).norm() / x.norm().clamp_min(1e-30)),
                     "cosine": float(torch.nn.functional.cosine_similarity(x, y, dim=0))})
    return rows
