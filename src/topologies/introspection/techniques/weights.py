"""Static weight statistics: norms, spectra, stable rank, heavy tails, layer similarity (U)."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from topologies.introspection.payloads import Payload, make_result
from topologies.introspection.sites import glob_regex
from topologies.introspection.techniques import Technique, builtin
from topologies.introspection.views import bars, finish, table
from topologies.visualization.views import ViewSpec

EXACT_SVD_LIMIT = 4096


@dataclass
class WeightsPayload(Payload):
    parameters: list[dict[str, Any]]
    spectra: dict[str, list[float]] = field(default_factory=dict)
    similarity: list[dict[str, Any]] = field(default_factory=list)

    def to_records(self) -> list[dict[str, Any]]:
        return [{k: v for k, v in row.items() if not isinstance(v, (list, dict))}
                for row in self.parameters]

    def views(self, **options: Any) -> list[ViewSpec]:
        matrices = [r for r in self.parameters if r.get("stable_rank") is not None]
        views = [table("Parameters", self.to_records()[:60])]
        if matrices:
            views.append(bars("Stable rank", [r["name"] for r in matrices],
                              [r["stable_rank"] for r in matrices]))
            alphas = [r for r in matrices if r.get("alpha") is not None]
            if alphas:
                views.append(bars("Heavy-tail alpha (heuristic)", [r["name"] for r in alphas],
                                  [r["alpha"] for r in alphas],
                                  caption="power-law fit to the tail of the eigenvalue density; "
                                          "a heuristic, not a quality guarantee"))
        for name, values in list(self.spectra.items())[:4]:
            views.append(ViewSpec("spectrum", f"Singular values: {name}", {"values": values,
                                                                           "site": name}))
        return finish(views, **options)


def matrix_view(name: str, tensor: torch.Tensor) -> tuple[torch.Tensor | None, str | None]:
    """2-D view used for spectra; conv kernels become ``(out, in * k...)``."""
    if tensor.ndim < 2:
        return None, None
    if tensor.ndim == 2:
        return tensor, None
    return tensor.reshape(tensor.shape[0], -1), (f"{name}: reshaped {tuple(tensor.shape)} -> " \
        f"(out, in*k) = {(tensor.shape[0], int(tensor[0].numel()))}")


def singular_values(matrix: torch.Tensor, *, rank: int = 256, seed: int = 0) -> tuple[torch.Tensor, bool]:
    m = matrix.detach().float().cpu()
    if max(m.shape) <= EXACT_SVD_LIMIT:
        return torch.linalg.svdvals(m), False
    q = min(rank, *m.shape)
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        _, s, _ = torch.svd_lowrank(m, q=q, niter=4)
    finally:
        torch.random.set_rng_state(state)
    return s, True


def heavy_tail_alpha(singular: torch.Tensor, *, tail: float = 0.5) -> float | None:
    """Power-law exponent of the empirical spectral density tail (WeightWatcher-style)."""
    from topologies.analysis import fit_power_law

    eig = (singular.double() ** 2).sort().values
    eig = eig[eig > 0]
    if eig.numel() < 8:
        return None
    tail_eig = eig[int(len(eig) * (1 - tail)):]
    if tail_eig.numel() < 4 or float(tail_eig.max()) <= float(tail_eig.min()):
        return None
    # Complementary CDF of the tail: P(lambda >= x) ~ x^(1 - alpha).
    n = tail_eig.numel()
    ccdf = [(n - i) / n for i in range(n)]
    try:
        fit = fit_power_law(tail_eig.tolist(), ccdf)
    except ValueError:
        return None
    slope = fit.parameters.get("exponent", fit.parameters.get("b"))
    if slope is None or not math.isfinite(slope):
        return None
    return 1.0 - float(slope)


def weight_stats(named: Mapping[str, torch.Tensor] | Any, *, params: str = "*",
                 near_zero: float = 1e-6, spectra: bool = True, rank: int = 256,
                 max_spectra: int = 64) -> tuple[WeightsPayload, list[str]]:
    """Statistics for an ``nn.Module`` (its parameters) or a plain ``state_dict``."""
    items = dict(named.named_parameters()) if hasattr(named, "named_parameters") else dict(named)
    pattern = glob_regex(params) if params not in ("*", "**") else None
    rows: list[dict[str, Any]] = []
    spectra_out: dict[str, list[float]] = {}
    assumptions: list[str] = []
    previous: tuple[str, torch.Tensor] | None = None
    similarity = []
    for name, tensor in items.items():
        if pattern is not None and not pattern.fullmatch(name):
            continue
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            continue
        w = tensor.detach().float()
        row: dict[str, Any] = {"name": name, "shape": "x".join(map(str, w.shape)),
                               "numel": w.numel(), "norm": float(w.norm()),
                               "mean": float(w.mean()), "std": float(w.std()) if w.numel() > 1 else 0.0,
                               "near_zero_fraction": float((w.abs() < near_zero).float().mean())}
        matrix, note = matrix_view(name, w)
        if matrix is not None and spectra and len(spectra_out) < max_spectra:
            s, approx = singular_values(matrix, rank=rank)
            top = float(s[0]) if s.numel() else 0.0
            row.update(sigma_max=top, sigma_min=float(s[-1]) if s.numel() else 0.0,
                       stable_rank=float((s ** 2).sum() / top ** 2) if top > 0 else None,
                       condition=float(s[0] / s[-1]) if s.numel() and float(s[-1]) > 0 else None,
                       alpha=heavy_tail_alpha(s), spectrum="randomized" if approx else "exact")
            spectra_out[name] = s.tolist()
            if note:
                assumptions.append(note)
            if approx:
                assumptions.append(f"{name}: randomized SVD (top {len(s)} values)")
        if previous is not None and previous[1].shape == w.shape:
            cos = torch.nn.functional.cosine_similarity(previous[1].reshape(1, -1), w.reshape(1, -1))
            similarity.append({"a": previous[0], "b": name, "cosine": float(cos)})
        if w.ndim >= 2:
            previous = (name, w)
        rows.append(row)
    return WeightsPayload(rows, spectra_out, similarity), assumptions


@builtin
class Weights(Technique):
    name = "weights"
    tier = "U"
    result_kind = "introspection.weights"
    summary = "weight norms, spectra, stable rank, heavy-tail alpha (static)"

    def run(self, subject: Any, inputs: Any = None, *, params: str = "*", state_dict: Any = None,
            **options: Any) -> Any:
        source = state_dict if state_dict is not None else subject.model
        payload, assumptions = weight_stats(source, params=params, **options)
        return make_result(self, payload, subject,
                           assumptions=[("heavy-tail alpha is a heuristic power-law fit (via "
                                        "analysis.fit_power_law) to the eigenvalue tail"),
                                        f"exact fp32 SVD up to {EXACT_SVD_LIMIT}, randomized above",
                                        *assumptions],
                           provenance={"params": params, "source": "state_dict" if state_dict
                                       is not None else "model"})
