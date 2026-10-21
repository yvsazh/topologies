"""``synth.random``: statistical data of any dimension.

Scalars (``synth.uniform``, ``synth.normal``, ... in the core) and here:

    synth.random.wishart(4)                         # a random SPD covariance [4, 4]
    synth.random.copula([synth.uniform(0, 1), synth.lognormal(0, 1)], corr=0.7)
    synth.random.simplex(5)                         # a point on the 5-simplex
    synth.random.low_rank((32, 16), rank=3)         # a low-rank matrix
    synth.random.sparse((64,), density=0.1)
    synth.random.gaussian_field((64, 64), length_scale=8.0)   # 1-D / 2-D / 3-D fields (FFT)
    c = synth.random.clusters(n_clusters=4, dim=8)  # {"x": point, "cluster": id}
    m = synth.random.manifold("swiss_roll", dim=3)  # {"x": point, "t": intrinsic coords}
    synth.random.heavy_tail((16,), alpha=2.5)       # Student-t entries

Tensor-valued nodes draw with ``ctx.rng.torch``, so every value depends only on
``(seed, record, path)``. Dataset-level structure (cluster centres, manifold
embeddings) comes from a ``structure_seed`` parameter, not from the record.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

from topologies import synth


def _shape(shape: Any) -> tuple[int, ...]:
    return (int(shape),) if isinstance(shape, (int, float)) else tuple(int(s) for s in shape)


@synth.node(type=synth.Tensor, tags=["random", "matrix"], example={"dim": 4})
def wishart(r, *, dim: int = 4, df: int | None = None, scale: float = 1.0):
    """A random symmetric positive definite matrix ``[dim, dim]`` (Wishart, ``df`` >= dim)."""
    return _wishart(r, int(dim), df, scale)


def _wishart(r: Any, dim: int, df: int | None, scale: float) -> torch.Tensor:
    df = int(df or dim + 2)
    g = torch.randn(df, dim, generator=r.torch, dtype=torch.float64) * math.sqrt(scale)
    return (g.t() @ g / df).float()


@synth.node(type=synth.Tensor, tags=["random", "matrix"], example={"dim": 4})
def correlation(r, *, dim: int = 4, df: int | None = None):
    """A random correlation matrix (a Wishart draw rescaled to unit diagonal)."""
    w = _wishart(r, int(dim), df, 1.0)
    d = w.diagonal().sqrt()
    return w / d[:, None] / d[None, :]


@synth.node(type=synth.Tensor, tags=["random", "simplex"], example={"dim": 5})
def simplex(r, *, dim: int = 3, concentration: float = 1.0):
    """A point on the ``dim``-simplex (Dirichlet with equal ``concentration``)."""
    return torch.tensor(r.dirichlet([concentration] * int(dim)), dtype=torch.float32)


@synth.node(type=synth.Tensor, tags=["random", "copula"],
            example={"marginals": ["uniform", "exponential"], "corr": 0.6})
def copula(r, *, marginals: Sequence[Any], corr: Any = 0.5):
    """Correlated values with given marginals through a Gaussian copula.

    ``marginals``: names (``"uniform"``, ``"normal"``, ``"exponential"``, ``"lognormal"``)
    or callables ``ppf(u)``; ``corr``: one correlation for every pair or a matrix.
    """
    k = len(marginals)
    c = torch.full((k, k), float(corr), dtype=torch.float64) if not isinstance(
        corr, (list, tuple, torch.Tensor)) else torch.as_tensor(corr, dtype=torch.float64)
    c.fill_diagonal_(1.0)
    z = torch.linalg.cholesky(c) @ torch.randn(k, generator=r.torch, dtype=torch.float64)
    u = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    out = []
    for m, ui in zip(marginals, u.tolist()):
        ui = min(max(ui, 1e-12), 1 - 1e-12)
        if callable(m):
            out.append(float(m(ui)))
        elif m == "uniform":
            out.append(ui)
        elif m == "normal":
            out.append(float(z[len(out)]))
        elif m == "exponential":
            out.append(-math.log(1 - ui))
        elif m == "lognormal":
            out.append(math.exp(float(z[len(out)])))
        else:
            raise synth.SynthError(f"Unknown marginal {m!r}.", stage="random",
                                   hint="Use uniform / normal / exponential / lognormal or a "
                                        "ppf(u) callable.")
    return torch.tensor(out, dtype=torch.float32)


@synth.node(type=synth.Tensor, tags=["random", "matrix"], example={"shape": [8, 6], "rank": 2})
def low_rank(r, *, shape: Sequence[int] = (16, 16), rank: int = 2, noise: float = 0.0):
    """A ``[m, n]`` matrix of rank ``rank`` (+ optional Gaussian ``noise``)."""
    m, n = _shape(shape)
    a = torch.randn(m, rank, generator=r.torch)
    b = torch.randn(rank, n, generator=r.torch)
    x = a @ b / math.sqrt(rank)
    return x + noise * torch.randn(m, n, generator=r.torch) if noise else x


@synth.node(type=synth.Tensor, tags=["random", "sparse"], example={"shape": [32], "density": 0.1})
def sparse(r, *, shape: Sequence[int] = (64,), density: float = 0.1, scale: float = 1.0):
    """A tensor whose entries are non-zero with probability ``density``."""
    s = _shape(shape)
    mask = torch.rand(s, generator=r.torch) < density
    return torch.randn(s, generator=r.torch) * scale * mask


@synth.node(type=synth.Tensor, tags=["random", "field"],
            example={"shape": [32, 32], "length_scale": 4.0})
def gaussian_field(r, *, shape: Sequence[int] = (64, 64), length_scale: float = 8.0,
                   power: float = 2.0):
    """A stationary Gaussian random field in 1-D, 2-D or 3-D (spectral synthesis by FFT):
    white noise filtered by a Gaussian kernel of ``length_scale`` samples; unit variance."""
    s = _shape(shape)
    noise = torch.randn(s, generator=r.torch, dtype=torch.float64)
    freqs = torch.meshgrid(*[torch.fft.fftfreq(n, dtype=torch.float64) for n in s],
                           indexing="ij")
    k2 = sum(f ** 2 for f in freqs)
    filt = torch.exp(-0.5 * (2 * math.pi * length_scale) ** 2 * k2 * (power / 2))
    f = torch.fft.ifftn(torch.fft.fftn(noise) * filt).real
    return ((f - f.mean()) / f.std().clamp_min(1e-12)).float()


@synth.node(type=synth.Tensor, tags=["random", "heavy-tail"], example={"shape": [16], "alpha": 2.5})
def heavy_tail(r, *, shape: Sequence[int] = (16,), alpha: float = 2.5, scale: float = 1.0):
    """Student-t entries with ``alpha`` degrees of freedom (heavy tails for small alpha)."""
    s = _shape(shape)
    z = torch.randn(s, generator=r.torch, dtype=torch.float64)
    g = torch.zeros(s, dtype=torch.float64)
    flat = g.view(-1)
    for i in range(flat.numel()):
        flat[i] = r.gammavariate(alpha / 2, 2.0)
    return (scale * z / torch.sqrt(g / alpha)).float()


def _structure(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


class clusters(synth.Node):
    """A point from one of ``n_clusters`` Gaussian clusters: ``{"x": [dim], "cluster": id}``.

    Centres are fixed by ``structure_seed`` (the same for every record); ``weights``
    sets cluster frequencies; ``spread`` the within-cluster standard deviation.
    """

    tags = ("random", "clusters")
    example = {"n_clusters": 3, "dim": 4}

    def __init__(self, n_clusters: int = 4, dim: int = 8, *, spread: Any = 0.5,
                 separation: float = 3.0, weights: Sequence[float] | None = None,
                 structure_seed: int = 0) -> None:
        self.n_clusters, self.dim = int(n_clusters), int(dim)
        self.spread, self.separation = spread, float(separation)
        self.weights = list(weights) if weights is not None else None
        self.structure_seed = int(structure_seed)
        self._centres: torch.Tensor | None = None

    def centres(self) -> torch.Tensor:
        if self._centres is None:
            g = _structure(self.structure_seed)
            self._centres = torch.randn(self.n_clusters, self.dim, generator=g) * self.separation
        return self._centres

    def sample(self, ctx: Any) -> dict[str, Any]:
        r = ctx.rng
        k = r.choices(range(self.n_clusters), weights=self.weights, k=1)[0] if self.weights \
            else r.randrange(self.n_clusters)
        s = float(ctx.resolve(self.spread))
        x = self.centres()[k] + s * torch.randn(self.dim, generator=r.torch)
        return {"x": x, "cluster": int(k)}


_MANIFOLDS = ("swiss_roll", "sphere", "torus", "circle", "s_curve")


class manifold(synth.Node):
    """A point on a manifold embedded in ``dim`` dimensions: ``{"x": [dim], "t": coords}``.

    ``kind``: swiss_roll / s_curve (2-D sheets), sphere / torus (2-D surfaces), circle.
    Points are embedded by a fixed random rotation (``structure_seed``) when ``dim`` > 3,
    plus isotropic ``noise``.
    """

    tags = ("random", "manifold")
    example = {"kind": "swiss_roll", "dim": 3}

    def __init__(self, kind: str = "swiss_roll", dim: int = 3, *, noise: Any = 0.0,
                 structure_seed: int = 0) -> None:
        if kind not in _MANIFOLDS:
            raise synth.SynthError(f"Unknown manifold {kind!r}.", stage="random",
                                   hint=f"One of {list(_MANIFOLDS)}.")
        self.kind, self.dim, self.noise = kind, int(dim), noise
        self.structure_seed = int(structure_seed)
        self._rot: torch.Tensor | None = None

    def sample(self, ctx: Any) -> dict[str, Any]:
        r = ctx.rng
        if self.kind == "swiss_roll":
            t = 1.5 * math.pi * (1 + 2 * r.random())
            h = 21 * r.random()
            p, coords = [t * math.cos(t), h, t * math.sin(t)], [t, h]
        elif self.kind == "s_curve":
            t = 3 * math.pi * (r.random() - 0.5)
            h = 2 * r.random()
            p, coords = [math.sin(t), h, math.copysign(1, t) * (math.cos(t) - 1)], [t, h]
        elif self.kind == "sphere":
            v = [r.gauss(0, 1) for _ in range(3)]
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            p = [x / n for x in v]
            coords = [math.acos(max(-1, min(1, p[2]))), math.atan2(p[1], p[0])]
        elif self.kind == "torus":
            a, b = 2 * math.pi * r.random(), 2 * math.pi * r.random()
            p = [(2 + math.cos(b)) * math.cos(a), (2 + math.cos(b)) * math.sin(a), math.sin(b)]
            coords = [a, b]
        else:
            a = 2 * math.pi * r.random()
            p, coords = [math.cos(a), math.sin(a)], [a]
        x = torch.tensor(p, dtype=torch.float32)
        if self.dim > len(p):
            if self._rot is None:
                q, _ = torch.linalg.qr(torch.randn(self.dim, self.dim,
                                                   generator=_structure(self.structure_seed)))
                self._rot = q[:, : len(p)]
            x = self._rot @ x
        elif self.dim < len(p):
            x = x[: self.dim]
        s = float(ctx.resolve(self.noise))
        if s:
            x = x + s * torch.randn(x.shape, generator=r.torch)
        return {"x": x, "t": coords}


__all__ = ["clusters", "copula", "correlation", "gaussian_field", "heavy_tail", "low_rank",
           "manifold", "simplex", "sparse", "wishart"]
