"""Search-space distributions for ``space="random"`` (``.grid(n)`` discretizes them)."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from topologies.experiments.errors import ExperimentError

# ---------------------------------------------------------------- distributions


class Distribution:
    """Base class: ``sample(rng)`` for ``space="random"``, ``grid(n)`` to discretize."""

    def sample(self, rng: random.Random) -> Any:
        raise NotImplementedError

    def grid(self, n: int) -> list[Any]:
        raise NotImplementedError

    def describe(self) -> str:
        return repr(self)


@dataclass(frozen=True)
class choice(Distribution):
    """Uniform choice over ``values``."""

    values: tuple

    def __init__(self, values: Sequence[Any]) -> None:
        if not len(values):
            raise ExperimentError("choice() needs at least one value.", stage="space")
        object.__setattr__(self, "values", tuple(values))

    def sample(self, rng: random.Random) -> Any:
        return self.values[rng.randrange(len(self.values))]

    def grid(self, n: int | None = None) -> list[Any]:
        return list(self.values)


@dataclass(frozen=True)
class uniform(Distribution):
    """Uniform float in ``[low, high]``."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if not self.high > self.low:
            raise ExperimentError(f"uniform({self.low}, {self.high}) needs high > low.",
                                  stage="space")

    def sample(self, rng: random.Random) -> float:
        return rng.uniform(self.low, self.high)

    def grid(self, n: int) -> list[float]:
        if n < 2:
            return [(self.low + self.high) / 2]
        return [self.low + (self.high - self.low) * i / (n - 1) for i in range(n)]


@dataclass(frozen=True)
class loguniform(Distribution):
    """Log-uniform float in ``[low, high]`` (both > 0), e.g. learning rates."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if not (self.low > 0 and self.high > self.low):
            raise ExperimentError(f"loguniform({self.low}, {self.high}) needs 0 < low < high.",
                                  stage="space")

    def sample(self, rng: random.Random) -> float:
        return math.exp(rng.uniform(math.log(self.low), math.log(self.high)))

    def grid(self, n: int) -> list[float]:
        lo, hi = math.log(self.low), math.log(self.high)
        if n < 2:
            return [math.exp((lo + hi) / 2)]
        return [float(f"{math.exp(lo + (hi - lo) * i / (n - 1)):.6g}") for i in range(n)]


@dataclass(frozen=True)
class intuniform(Distribution):
    """Uniform integer in ``[low, high]`` (inclusive)."""

    low: int
    high: int

    def __post_init__(self) -> None:
        if not self.high >= self.low:
            raise ExperimentError(f"intuniform({self.low}, {self.high}) needs high >= low.",
                                  stage="space")

    def sample(self, rng: random.Random) -> int:
        return rng.randint(int(self.low), int(self.high))

    def grid(self, n: int) -> list[int]:
        vals = uniform(self.low - 1e-9, self.high + 1e-9).grid(n) if n > 1 else [
            (self.low + self.high) / 2]
        return sorted({round(v) for v in vals})


@dataclass(frozen=True)
class intloguniform(Distribution):
    """Log-uniform integer in ``[low, high]`` (both >= 1), e.g. widths."""

    low: int
    high: int

    def __post_init__(self) -> None:
        if not (self.low >= 1 and self.high >= self.low):
            raise ExperimentError(
                f"intloguniform({self.low}, {self.high}) needs 1 <= low <= high.", stage="space")

    def sample(self, rng: random.Random) -> int:
        v = math.exp(rng.uniform(math.log(self.low), math.log(self.high + 1)))
        return min(int(self.high), max(int(self.low), int(v)))

    def grid(self, n: int) -> list[int]:
        if self.low == self.high:
            return [int(self.low)]
        return sorted({round(v) for v in loguniform(self.low, self.high).grid(n)})


__all__ = ["Distribution", "choice", "intloguniform", "intuniform", "loguniform", "uniform"]
