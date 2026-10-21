"""Seeded randomness: one ``Rand`` per (spec seed, record index, node path).

``Rand`` is a ``random.Random`` (every stdlib method works: ``uniform``,
``randint``, ``choice``, ``choices``, ``shuffle``, ``sample``, ``gauss``,
``betavariate``, ...) driven by a counter-based splitmix64 stream, so creating
one per node and record costs ~0.2 µs and values are identical on every
platform. It adds distribution helpers and a matching ``torch.Generator``:

    r = Rand(derive_seed(0, 17, "audio.f0"))
    r.normal(0, 1); r.lognormal(0, 0.5); r.weighted({"en": 0.6, "uk": 0.4})
    r.zipf(1000); r.poisson(3.5); r.truncnorm(50, 20, 0, 120); r.dirichlet([1, 1, 1])
    torch.randn(16, generator=r.torch)       # same seed, torch side
    r.child("noise")                         # an independent, stable sub-stream

Seeds: ``derive_seed(*parts)`` hashes any parts; the executor derives one
base per node path once (``path_base(seed, path)``) and mixes in the record
index per record (``record_seed(base, index)``), which is cheap.
"""

from __future__ import annotations

import bisect
import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

_M64 = 0xFFFFFFFFFFFFFFFF
_GOLDEN = 0x9E3779B97F4A7C15
_INV53 = 1.0 / 9007199254740992.0


def derive_seed(*parts: Any) -> int:
    """Stable 63-bit seed from any parts (``repr`` based; ints/strings/tuples are stable)."""
    h = hashlib.blake2b(repr(parts).encode("utf-8", "surrogatepass"), digest_size=8)
    return int.from_bytes(h.digest(), "little") & ((1 << 63) - 1)


def mix64(x: int) -> int:
    """splitmix64 finalizer: a well-mixed 64-bit value from ``x``."""
    x = (x + _GOLDEN) & _M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _M64
    return x ^ (x >> 31)


def path_base(seed: int, path: str) -> int:
    """Per-node base seed (computed once per bound node)."""
    return derive_seed("node", int(seed), str(path))


def record_seed(base: int, index: int) -> int:
    """Seed of one node for one record: ``mix(base, index)`` (63 bits)."""
    return mix64(base ^ mix64(int(index) & _M64)) & ((1 << 63) - 1)


class Rand(random.Random):
    """``random.Random`` on a splitmix64 stream + helpers (see module docstring)."""

    def __init__(self, seed: int = 0) -> None:
        self._seed0 = int(seed) & ((1 << 63) - 1)
        self._s = self._seed0
        self._torch: Any = None
        self.gauss_next = None

    # -- the generator ---------------------------------------------------------
    def seed(self, a: Any = None, version: int = 2) -> None:
        self._seed0 = (a if isinstance(a, int) else derive_seed(a)) & ((1 << 63) - 1)
        self._s = self._seed0
        self.gauss_next = None

    def random(self) -> float:
        x = self._s = (self._s + _GOLDEN) & _M64
        x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _M64
        x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _M64
        return ((x ^ (x >> 31)) >> 11) * _INV53

    def getrandbits(self, k: int) -> int:
        out, n = 0, 0
        while n < k:
            x = self._s = (self._s + _GOLDEN) & _M64
            x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _M64
            x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _M64
            out |= (x ^ (x >> 31)) << n
            n += 64
        return out & ((1 << k) - 1) if k else 0

    def getstate(self) -> Any:
        return (self._seed0, self._s, self.gauss_next)

    def setstate(self, state: Any) -> None:
        self._seed0, self._s, self.gauss_next = state

    @property
    def seed_value(self) -> int:
        return self._seed0

    @property
    def torch(self) -> Any:
        """A ``torch.Generator`` seeded from the same seed (created on first use)."""
        if self._torch is None:
            import torch

            self._torch = torch.Generator().manual_seed(self._seed0)
        return self._torch

    def child(self, *parts: Any) -> Rand:
        """Independent RNG for a sub-task (``rng.child("noise")``), reproducible."""
        return Rand(derive_seed(self._seed0, *parts))

    # -- distributions -----------------------------------------------------
    def normal(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        return self.gauss(mu, sigma)

    def lognormal(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        return math.exp(self.gauss(mu, sigma))

    def loguniform(self, low: float, high: float) -> float:
        return math.exp(self.uniform(math.log(low), math.log(high)))

    def intuniform(self, low: int, high: int) -> int:
        """Integer in ``[low, high]`` (both ends included)."""
        return self.randint(int(low), int(high))

    def beta(self, a: float, b: float) -> float:
        return self.betavariate(a, b)

    def gamma(self, shape: float, scale: float = 1.0) -> float:
        return self.gammavariate(shape, scale)

    def exponential(self, rate: float = 1.0) -> float:
        return self.expovariate(rate)

    def weighted(self, choices: Mapping[Any, float] | Sequence[Any],
                 weights: Sequence[float] | None = None) -> Any:
        """One value from ``{value: weight}`` or ``(values, weights)``."""
        if isinstance(choices, Mapping):
            values, weights = list(choices), [float(w) for w in choices.values()]
        else:
            values = list(choices)
        if weights is None:
            return values[self.randrange(len(values))]
        return self.choices(values, weights=weights, k=1)[0]

    def bernoulli(self, p: float) -> bool:
        return self.random() < p

    def zipf(self, n: int, s: float = 1.1) -> int:
        """Rank in ``1..n`` with ``P(k) ~ 1/k^s`` (few frequent, long tail)."""
        cdf = _zipf_cdf(int(n), float(s))
        return min(n, bisect.bisect_left(cdf, self.random()) + 1)

    def power_law(self, alpha: float, xmin: float = 1.0, xmax: float | None = None) -> float:
        """Pareto-like value ``>= xmin`` with density ``~ x^-alpha`` (``alpha > 1``)."""
        if alpha <= 1:
            raise ValueError("power_law needs alpha > 1.")
        while True:
            x = xmin * (1.0 - self.random()) ** (-1.0 / (alpha - 1.0))
            if xmax is None or x <= xmax:
                return x

    def truncnorm(self, mu: float, sigma: float, lo: float = -math.inf,
                  hi: float = math.inf) -> float:
        """Normal(mu, sigma) restricted to ``[lo, hi]`` (rejection, then clamping)."""
        for _ in range(64):
            x = self.gauss(mu, sigma)
            if lo <= x <= hi:
                return x
        return min(hi, max(lo, mu))

    def dirichlet(self, alpha: Sequence[float]) -> list[float]:
        """A point on the simplex (``sum == 1``) with concentration ``alpha``."""
        g = [self.gammavariate(float(a), 1.0) for a in alpha]
        total = sum(g) or 1.0
        return [x / total for x in g]

    def lognormal_amount(self, median: float, spread: float = 0.8, *, digits: int = 2) -> float:
        """Positive money-like amount around ``median`` (log-normal)."""
        return round(math.exp(self.gauss(math.log(max(median, 1e-9)), spread)), digits)

    def correlated(self, rho: float) -> tuple[float, float]:
        """Two standard normals with correlation ``rho``."""
        a, b = self.gauss(0, 1), self.gauss(0, 1)
        return a, rho * a + math.sqrt(max(0.0, 1 - rho * rho)) * b

    def poisson(self, lam: float) -> int:
        if lam <= 0:
            return 0
        if lam > 30:
            return max(0, round(self.gauss(lam, math.sqrt(lam))))
        limit, k, p = math.exp(-lam), 0, 1.0
        while True:
            p *= self.random()
            if p <= limit:
                return k
            k += 1

    def maybe(self, value: Any, p: float, default: Any = None) -> Any:
        """``value`` with probability ``p`` else ``default``."""
        return value if self.random() < p else default

    def __repr__(self) -> str:
        return f"Rand({self._seed0})"


_ZIPF: dict[tuple[int, float], list[float]] = {}


def _zipf_cdf(n: int, s: float) -> list[float]:
    cdf = _ZIPF.get((n, s))
    if cdf is None:
        w = [1.0 / (k ** s) for k in range(1, n + 1)]
        total, acc, cdf = sum(w), 0.0, []
        for x in w:
            acc += x / total
            cdf.append(acc)
        _ZIPF[(n, s)] = cdf
    return cdf


#: Historical name (attributes, samplers and ops use it).
SynthRandom = Rand

__all__ = ["Rand", "SynthRandom", "derive_seed", "mix64", "path_base", "record_seed"]
