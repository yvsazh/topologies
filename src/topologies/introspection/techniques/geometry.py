"""Representation geometry (U): PCA, effective rank, participation ratio, anisotropy, TwoNN."""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from topologies.introspection.payloads import Payload, input_fingerprint, make_result
from topologies.introspection.techniques import Technique, builtin
from topologies.introspection.techniques._features import collect_features
from topologies.introspection.views import finish, lines
from topologies.visualization.views import ViewSpec


@dataclass
class GeometryPayload(Payload):
    sites: dict[str, dict[str, Any]]
    projections: dict[str, torch.Tensor] = field(default_factory=dict)
    labels: list[Any] | None = None

    def to_records(self) -> list[dict[str, Any]]:
        return [{"site": s, **{k: v for k, v in d.items() if isinstance(v, (int, float))}}
                for s, d in self.sites.items()]

    def views(self, **options: Any) -> list[ViewSpec]:
        views = []
        curves = [{"y": d["explained_variance"][:50], "label": s} for s, d in self.sites.items()]
        if curves:
            views.append(lines("Cumulative explained variance", curves, x_label="components",
                               y_label="fraction"))
        for site, points in list(self.projections.items())[:4]:
            views.append(ViewSpec("scatter", f"PCA projection: {site}", {
                "points": points.tolist(), "labels": self.labels, "site": site,
                "x_label": "PC1", "y_label": "PC2"}))
        return finish(views, **options)


def spectrum(x: torch.Tensor) -> torch.Tensor:
    x = x.double() - x.double().mean(0)
    s = torch.linalg.svdvals(x)
    return s ** 2 / max(x.shape[0] - 1, 1)


def effective_rank(eig: torch.Tensor) -> float:
    p = eig.clamp_min(0) / eig.clamp_min(0).sum().clamp_min(1e-30)
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def participation_ratio(eig: torch.Tensor) -> float:
    eig = eig.clamp_min(0)
    return float(eig.sum() ** 2 / (eig ** 2).sum().clamp_min(1e-30))


def anisotropy(x: torch.Tensor, limit: int = 2000, seed: int = 0) -> float:
    if x.shape[0] > limit:
        g = torch.Generator().manual_seed(seed)
        x = x[torch.randperm(x.shape[0], generator=g)[:limit]]
    n = torch.nn.functional.normalize(x.double(), dim=-1)
    sims = n @ n.T
    m = sims.shape[0]
    return float((sims.sum() - sims.diagonal().sum()) / max(m * (m - 1), 1))


def twonn(x: torch.Tensor, limit: int = 10_000, seed: int = 0) -> float | None:
    """TwoNN intrinsic dimension (Facco et al. 2017) on at most ``limit`` points."""
    if x.shape[0] > limit:
        g = torch.Generator().manual_seed(seed)
        x = x[torch.randperm(x.shape[0], generator=g)[:limit]]
    if x.shape[0] < 3:
        return None
    d = torch.cdist(x.double(), x.double())
    d.fill_diagonal_(math.inf)
    r = d.topk(2, largest=False).values
    mu = r[:, 1] / r[:, 0].clamp_min(1e-30)
    mu = mu[torch.isfinite(mu) & (mu > 1)]
    if mu.numel() < 2:
        return None
    return float(mu.numel() / mu.log().sum())


@builtin
class Geometry(Technique):
    name = "geometry"
    tier = "U"
    result_kind = "introspection.geometry"
    summary = "PCA spectrum, effective rank, participation ratio, anisotropy, TwoNN dimension"

    def run(self, subject: Any, inputs: Any = None, *, sites: Any = None, pool: str = "mean",
            labels: Any = None, project: Callable[[torch.Tensor], torch.Tensor] | None = None,
            max_batches: int | None = None, max_examples: int = 10_000, **params: Any) -> Any:
        index = subject.index
        if sites is None:
            sites = "role:block" if index.resolve_modules("role:block", allow_empty=True) else \
                [p for p in index.paths() if p and "." not in p]
        paths = index.resolve_modules(sites)
        feats, ys, notes = collect_features(subject, inputs, paths, pool=pool, labels=labels,
                                            max_batches=max_batches, max_examples=max_examples)
        out: dict[str, dict[str, Any]] = {}
        projections = {}
        for path, x in feats.items():
            eig = spectrum(x)
            total = eig.sum().clamp_min(1e-30)
            label = path or "<root>"
            out[label] = {"examples": x.shape[0], "features": x.shape[1],
                          "effective_rank": effective_rank(eig),
                          "participation_ratio": participation_ratio(eig),
                          "anisotropy": anisotropy(x), "twonn_dimension": twonn(x),
                          "explained_variance": (eig.cumsum(0) / total).float().tolist()}
            if project is not None:
                projections[label] = project(x)[:, :2].float()
            else:
                centred = x.double() - x.double().mean(0)
                _, _, vh = torch.linalg.svd(centred, full_matrices=False)
                projections[label] = (centred @ vh[:2].T).float()
        payload = GeometryPayload(out, projections, ys.tolist() if ys is not None else None)
        return make_result(self, payload, subject, assumptions=[
            "effective rank = exp(entropy of the normalised PCA spectrum)",
            "TwoNN on at most 10000 sampled points; anisotropy on at most 2000",
            "2-D projection: PCA unless project= is given", *notes],
            provenance={"inputs": input_fingerprint(inputs)})
