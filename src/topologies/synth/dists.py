"""Random nodes: scalar and vector distributions (parameters may themselves be nodes).

    synth.uniform(80, 800); synth.normal(0, 1); synth.lognormal(0, 0.5)
    synth.beta(2, 5); synth.gamma(2.0, 1.5); synth.poisson(3.0); synth.bernoulli(0.1)
    synth.intuniform(1, 6); synth.loguniform(1e-3, 1); synth.truncnorm(0, 1, -2, 2)
    synth.choice(["en", "uk", "de"], weights=[0.6, 0.3, 0.1]); synth.categorical([0.2, 0.8])
    synth.zipf(1000, s=1.2); synth.dirichlet([1.0] * 16); synth.mvnormal(mean, cov)

With constant parameters every distribution is also a valid *attribute*
(``Spec(attributes={"length": synth.intuniform(4, 60)})``): ``choice`` /
``bernoulli`` / ``categorical`` become categorical attributes (coverage samplers),
the others continuous ones (stratified / Latin hypercube through ``ppf``).
``experiments`` distributions (``topologies.experiments.uniform`` ...) are
accepted anywhere a node is.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from topologies.synth.core import Node, as_param, has_nodes
from topologies.synth.ctor import factory
from topologies.synth.errors import SynthError


class Dist(Node):
    """A random scalar/vector node: ``draw(rng, **params)`` with resolved parameters."""

    _names: tuple[str, ...] = ()

    def __init__(self, **params: Any) -> None:
        for k in self._names:
            setattr(self, k, as_param(params[k]))
        name = params.get("name")
        if name:
            self.__dict__["_name"] = name

    def params(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self._names}

    def sample(self, ctx: Any) -> Any:
        return self.draw(ctx.rng, **ctx.resolve_params(self))

    def draw(self, rng: Any, **p: Any) -> Any:
        raise NotImplementedError

    def ppf(self, u: float) -> Any:
        raise SynthError(f"{self.node_kind()} has no inverse CDF; LHS cannot stratify it.",
                         stage="attributes", hint="Use sampler='random' for this attribute.")

    def constant(self) -> bool:
        return not any(has_nodes(v) for v in self.params().values())

    def as_attr(self) -> Any:
        """The equivalent attribute (constant parameters only)."""
        from topologies.synth.attributes import Continuous

        if not self.constant():
            raise SynthError(f"{self.node_kind()} with node parameters cannot be an attribute.",
                             stage="attributes", hint="Use constant parameters, or sample it "
                                                      "inside the generator instead.")
        return Continuous(_AsDistribution(self))

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.params().items())
        return f"{self.node_kind()}({args})"


class _AsDistribution:
    """``experiments``-style distribution view of a constant ``Dist`` (attributes)."""

    def __init__(self, node: Dist) -> None:
        self.node = node

    def sample(self, rng: Any) -> Any:
        return self.node.draw(rng, **self.node.params())

    def ppf(self, u: float) -> Any:
        return self.node.ppf(u)

    def __repr__(self) -> str:
        return repr(self.node)


class Uniform(Dist):
    _names = ("low", "high")

    def draw(self, rng: Any, *, low: float, high: float) -> float:
        return rng.uniform(low, high)

    def ppf(self, u: float) -> float:
        return self.low + (self.high - self.low) * u


class LogUniform(Dist):
    _names = ("low", "high")

    def draw(self, rng: Any, *, low: float, high: float) -> float:
        return math.exp(rng.uniform(math.log(low), math.log(high)))

    def ppf(self, u: float) -> float:
        lo, hi = math.log(self.low), math.log(self.high)
        return math.exp(lo + (hi - lo) * u)

    def node_kind(self) -> str:
        return "loguniform"


class IntUniform(Dist):
    _names = ("low", "high")

    def draw(self, rng: Any, *, low: int, high: int) -> int:
        return rng.randint(int(low), int(high))

    def ppf(self, u: float) -> int:
        lo, hi = int(self.low), int(self.high)
        return lo + min(int(u * (hi - lo + 1)), hi - lo)

    def node_kind(self) -> str:
        return "intuniform"


class Normal(Dist):
    _names = ("mu", "sigma")

    def draw(self, rng: Any, *, mu: float, sigma: float) -> float:
        return rng.gauss(mu, sigma)

    def ppf(self, u: float) -> float:
        return statistics.NormalDist(self.mu, self.sigma).inv_cdf(min(max(u, 1e-12), 1 - 1e-12))


class LogNormal(Dist):
    _names = ("mu", "sigma")

    def draw(self, rng: Any, *, mu: float, sigma: float) -> float:
        return math.exp(rng.gauss(mu, sigma))

    def ppf(self, u: float) -> float:
        return math.exp(statistics.NormalDist(self.mu, self.sigma).inv_cdf(
            min(max(u, 1e-12), 1 - 1e-12)))

    def node_kind(self) -> str:
        return "lognormal"


class TruncNormal(Dist):
    _names = ("mu", "sigma", "low", "high")

    def draw(self, rng: Any, *, mu: float, sigma: float, low: float, high: float) -> float:
        return rng.truncnorm(mu, sigma, low, high)

    def node_kind(self) -> str:
        return "truncnorm"


class Beta(Dist):
    _names = ("a", "b")

    def draw(self, rng: Any, *, a: float, b: float) -> float:
        return rng.betavariate(a, b)


class Gamma(Dist):
    _names = ("shape", "scale")

    def draw(self, rng: Any, *, shape: float, scale: float) -> float:
        return rng.gammavariate(shape, scale)


class Exponential(Dist):
    _names = ("rate",)

    def draw(self, rng: Any, *, rate: float) -> float:
        return rng.expovariate(rate)

    def ppf(self, u: float) -> float:
        return -math.log(1 - min(u, 1 - 1e-12)) / self.rate


class Poisson(Dist):
    _names = ("lam",)

    def draw(self, rng: Any, *, lam: float) -> int:
        return rng.poisson(lam)


class Zipf(Dist):
    _names = ("n", "s")

    def draw(self, rng: Any, *, n: int, s: float) -> int:
        return rng.zipf(int(n), float(s))


class Bernoulli(Dist):
    _names = ("p",)

    def draw(self, rng: Any, *, p: float) -> bool:
        return rng.random() < p

    def as_attr(self) -> Any:
        from topologies.synth.attributes import Categorical

        return Categorical([True, False], [self.p, 1 - self.p])


class Choice(Dist):
    _names = ("values", "weights")

    def draw(self, rng: Any, *, values: Sequence[Any], weights: Sequence[float] | None) -> Any:
        vals = list(values)
        if weights is None:
            return vals[rng.randrange(len(vals))]
        return rng.choices(vals, weights=list(weights), k=1)[0]

    def as_attr(self) -> Any:
        from topologies.synth.attributes import Categorical

        return Categorical(list(self.values), None if self.weights is None else
                           list(self.weights))


class Categorical(Dist):
    """An index (list of probabilities) or a key (``{key: probability}``)."""

    _names = ("probs",)

    def draw(self, rng: Any, *, probs: Any) -> Any:
        if isinstance(probs, Mapping):
            return rng.choices(list(probs), weights=[float(w) for w in probs.values()], k=1)[0]
        return rng.choices(range(len(probs)), weights=[float(w) for w in probs], k=1)[0]

    def as_attr(self) -> Any:
        from topologies.synth.attributes import Categorical as CatAttr

        p = self.probs
        if isinstance(p, Mapping):
            return CatAttr(list(p), [float(w) for w in p.values()])
        return CatAttr(list(range(len(p))), [float(w) for w in p])


class Dirichlet(Dist):
    _names = ("alpha",)

    def draw(self, rng: Any, *, alpha: Any) -> list[float]:
        import torch

        a = alpha.tolist() if isinstance(alpha, torch.Tensor) else list(alpha)
        return rng.dirichlet([float(x) for x in a])


class MVNormal(Dist):
    """A ``torch`` vector ``[d]`` from ``N(mean, cov)`` (Cholesky; ``cov`` may be a node)."""

    _names = ("mean", "cov")

    def draw(self, rng: Any, *, mean: Any, cov: Any) -> Any:
        import torch

        m = torch.as_tensor(mean, dtype=torch.float64)
        c = torch.as_tensor(cov, dtype=torch.float64)
        if c.dim() == 1:
            c = torch.diag(c)
        jitter = 0.0
        for _ in range(6):
            try:
                L = torch.linalg.cholesky(c + jitter * torch.eye(c.shape[0], dtype=c.dtype))
                break
            except RuntimeError:
                jitter = jitter * 10 if jitter else 1e-10
        else:
            raise SynthError("mvnormal: the covariance is not positive definite.",
                             stage="node", hint="Use a symmetric PSD matrix (e.g. "
                                                "synth.random.wishart(d)).")
        z = torch.randn(m.shape[0], generator=rng.torch, dtype=torch.float64)
        return (m + L @ z).float()

    def node_kind(self) -> str:
        return "mvnormal"


# ---------------------------------------------------------------- factories


@factory
def uniform(low: float, high: float, *, name: str | None = None) -> Uniform:
    """Uniform float in ``[low, high)``."""
    return Uniform(low=low, high=high, name=name)


@factory
def loguniform(low: float, high: float, *, name: str | None = None) -> LogUniform:
    """Log-uniform float in ``[low, high)`` (scales, rates, learning rates)."""
    return LogUniform(low=low, high=high, name=name)


@factory
def intuniform(low: int, high: int, *, name: str | None = None) -> IntUniform:
    """Integer in ``[low, high]`` (both included)."""
    return IntUniform(low=low, high=high, name=name)


@factory
def normal(mu: float = 0.0, sigma: float = 1.0, *, name: str | None = None) -> Normal:
    return Normal(mu=mu, sigma=sigma, name=name)


@factory
def lognormal(mu: float = 0.0, sigma: float = 1.0, *, name: str | None = None) -> LogNormal:
    return LogNormal(mu=mu, sigma=sigma, name=name)


@factory
def truncnorm(mu: float, sigma: float, low: float = -math.inf, high: float = math.inf, *,
              name: str | None = None) -> TruncNormal:
    """Normal restricted to ``[low, high]``."""
    return TruncNormal(mu=mu, sigma=sigma, low=low, high=high, name=name)


@factory
def beta(a: float, b: float, *, name: str | None = None) -> Beta:
    return Beta(a=a, b=b, name=name)


@factory
def gamma(shape: float, scale: float = 1.0, *, name: str | None = None) -> Gamma:
    return Gamma(shape=shape, scale=scale, name=name)


@factory
def exponential(rate: float = 1.0, *, name: str | None = None) -> Exponential:
    return Exponential(rate=rate, name=name)


@factory
def poisson(lam: float, *, name: str | None = None) -> Poisson:
    return Poisson(lam=lam, name=name)


@factory
def zipf(n: int, s: float = 1.1, *, name: str | None = None) -> Zipf:
    """Rank in ``1..n`` with ``P(k) ~ 1/k^s``."""
    return Zipf(n=n, s=s, name=name)


@factory
def bernoulli(p: float = 0.5, *, name: str | None = None) -> Bernoulli:
    return Bernoulli(p=p, name=name)


@factory
def choice(values: Sequence[Any], weights: Sequence[float] | None = None, *,
           name: str | None = None) -> Choice:
    """One of ``values`` (uniform, or by ``weights``)."""
    vals = list(values)
    if not vals:
        raise SynthError("choice() needs at least one value.", stage="node")
    if weights is not None and not has_nodes(weights) and len(list(weights)) != len(vals):
        raise SynthError("choice(): one weight per value.", stage="node")
    return Choice(values=vals, weights=None if weights is None else list(weights), name=name)


@factory
def categorical(probs: Any, *, name: str | None = None) -> Categorical:
    """An index drawn with ``probs`` (a list) or a key (``{key: probability}``)."""
    return Categorical(probs=probs, name=name)


@factory
def dirichlet(alpha: Any, *, name: str | None = None) -> Dirichlet:
    """A point on the simplex (a list of floats summing to 1)."""
    return Dirichlet(alpha=alpha, name=name)


@factory
def mvnormal(mean: Any, cov: Any, *, name: str | None = None) -> MVNormal:
    """A vector from a multivariate normal (``cov``: matrix or diagonal)."""
    return MVNormal(mean=mean, cov=cov, name=name)


def from_distribution(d: Any) -> Node:
    """The node equivalent of an ``experiments`` distribution."""
    from topologies.experiments import distributions as D

    if isinstance(d, D.choice):
        return choice(list(d.values))
    if isinstance(d, D.uniform):
        return uniform(d.low, d.high)
    if isinstance(d, D.loguniform):
        return loguniform(d.low, d.high)
    if isinstance(d, D.intuniform):
        return intuniform(int(d.low), int(d.high))
    if isinstance(d, D.intloguniform):
        from topologies.synth.core import apply

        return apply(_int_floor, loguniform(d.low, d.high + 1), d.high)
    return _ExpDist(d)


def _int_floor(x: float, hi: int) -> int:
    return min(int(hi), int(x))


class _ExpDist(Node):
    """Any object with ``sample(rng)`` (an ``experiments`` distribution) as a node."""

    def __init__(self, dist: Any) -> None:
        self.dist = dist

    def params(self) -> dict[str, Any]:
        return {}

    def node_kind(self) -> str:
        return type(self.dist).__name__

    def sample(self, ctx: Any) -> Any:
        return self.dist.sample(ctx.rng)

    def fingerprint_config(self) -> Any:
        return repr(self.dist)


__all__ = [
    "Bernoulli",
    "Beta",
    "Categorical",
    "Choice",
    "Dirichlet",
    "Dist",
    "Exponential",
    "Gamma",
    "IntUniform",
    "LogNormal",
    "LogUniform",
    "MVNormal",
    "Normal",
    "Poisson",
    "TruncNormal",
    "Uniform",
    "Zipf",
    "bernoulli",
    "beta",
    "categorical",
    "choice",
    "dirichlet",
    "exponential",
    "from_distribution",
    "gamma",
    "intuniform",
    "lognormal",
    "loguniform",
    "mvnormal",
    "normal",
    "poisson",
    "truncnorm",
    "uniform",
    "zipf",
]
