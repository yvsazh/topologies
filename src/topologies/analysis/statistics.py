"""Pure-Python linear and power-law fitting with explicit uncertainty semantics."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from topologies.analysis.core import field_value


@dataclass
class FitArtifact:
    """Fit diagnostics; ``residuals``/RMSE/R² use ``residual_domain``.

    For power laws that domain is ``log_y`` and original-space residuals are
    separately available as ``original_residuals``. Bootstrap intervals are
    confidence intervals for the fitted mean curve, not prediction intervals.
    """

    method: str
    parameters: dict[str, float]
    observations: list[tuple[float, float]]
    n: int
    x_range: tuple[float, float]
    residuals: list[float]
    residual_domain: str
    rmse: float
    r_squared: float | None
    original_residuals: list[float] | None = None
    assumptions: list[str] = field(default_factory=list)
    parameter_uncertainty: dict[str, float] | None = None
    interval: dict[str, Any] | None = None
    kind: str = "fitted"

    def predict(self, x: float, *, extrapolation_ratio: float = 2.0) -> dict[str, Any]:
        x = _finite(x, "prediction x")
        threshold = _finite(extrapolation_ratio, "extrapolation_ratio")
        if threshold <= 0:
            raise ValueError("extrapolation_ratio must be finite and positive")
        if self.method == "power_law" and x <= 0:
            raise ValueError("power-law prediction requires positive x")
        value = _finite(_predict(self.method, self.parameters, x), "prediction")
        lo, hi = self.x_range
        outside = x < lo or x > hi
        edge = lo if x < lo else hi
        if self.method == "power_law":
            ratio = max(x / edge, edge / x) if outside else 1.0
        else:
            span = max(hi - lo, abs(edge), 1.0)
            ratio = 1.0 + abs(x-edge) / span if outside else 1.0
        out: dict[str, Any] = {"x": x, "value": value,
                               "kind": "extrapolated" if outside else "interpolated",
                               "extrapolated": outside,
                               "large_extrapolation": outside and ratio >= threshold,
                               "range_ratio": ratio}
        if self.interval and self.interval.get("parameter_draws"):
            samples = [_finite(_predict(self.method, params, x), "bootstrap prediction")
                       for params in self.interval["parameter_draws"]]
            alpha = (1 - self.interval["confidence"]) / 2
            out["confidence_interval"] = [
                _quantile(sorted(samples), alpha), _quantile(sorted(samples), 1-alpha)]
            out["interval_kind"] = "confidence"
        return out


def fit_linear(x: list[float], y: list[float], *, bootstrap: int = 0,
               confidence: float = .95, seed: int = 0) -> FitArtifact:
    return _fit("linear", x, y, bootstrap=bootstrap, confidence=confidence, seed=seed)


def fit_power_law(x: list[float], y: list[float], *, bootstrap: int = 0,
                  confidence: float = .95, seed: int = 0) -> FitArtifact:
    return _fit("power_law", x, y, bootstrap=bootstrap, confidence=confidence, seed=seed)


def Fit(x: Any, y: Any, *, method: str = "linear", bootstrap: int = 0,
        confidence: float = .95, seed: int = 0):
    """Create a pipeline stage fitting selected record fields."""
    from topologies.analysis.core import Stage
    if method not in ("linear", "power_law"):
        raise ValueError("method must be 'linear' or 'power_law'")

    def run(records: Any) -> FitArtifact:
        return _fit(method, [field_value(r, x) for r in records],
                    [field_value(r, y) for r in records],
                    bootstrap=bootstrap, confidence=confidence, seed=seed)
    return Stage(f"fit:{method}", run, kind="fitted")


def _fit(method: str, x: list[float], y: list[float], *, bootstrap: int,
         confidence: float, seed: int) -> FitArtifact:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("fit requires at least two paired observations")
    if bootstrap < 0 or not 0 < confidence < 1:
        raise ValueError("bootstrap must be non-negative and confidence in (0, 1)")
    pairs = [(_finite(a, "x"), _finite(b, "y")) for a, b in zip(x, y)]
    if method == "power_law" and any(a <= 0 or b <= 0 for a, b in pairs):
        raise ValueError("power-law fit requires positive x and y")
    xx, yy = ([math.log(a) for a, _ in pairs], [math.log(b) for _, b in pairs]) \
        if method == "power_law" else ([a for a, _ in pairs], [b for _, b in pairs])
    if max(xx) == min(xx):
        raise ValueError("fit requires at least two distinct x values")
    slope, intercept = _least_squares(xx, yy)
    if not math.isfinite(slope) or not math.isfinite(intercept):
        raise ValueError("fit produced non-finite parameters")
    if method == "power_law":
        try:
            params = {"a": math.exp(intercept), "b": slope}
        except OverflowError as e:
            raise ValueError("fit produced non-finite parameters") from e
        if params["a"] <= 0:
            raise ValueError("fit produced parameters outside the finite numeric range")
        residual_domain = "log_y"
    else:
        params = {"slope": slope, "intercept": intercept}
        residual_domain = "y"
    predicted_fit = [intercept + slope * a for a in xx]
    residuals_fit = [b - p for b, p in zip(yy, predicted_fit)]
    original_residuals = [b - _finite(_predict(method, params, a), "fitted value")
                          for a, b in pairs]
    rmse = math.sqrt(sum(r*r for r in residuals_fit) / len(residuals_fit))
    mean = sum(yy) / len(yy)
    total = sum((v - mean)**2 for v in yy)
    r2 = None if total == 0 else 1 - sum(r*r for r in residuals_fit) / total
    if not math.isfinite(rmse) or (r2 is not None and not math.isfinite(r2)):
        raise ValueError("fit diagnostics are outside the finite numeric range")
    artifact = FitArtifact(method, params, pairs, len(pairs),
                           (min(a for a, _ in pairs), max(a for a, _ in pairs)),
                            residuals_fit, residual_domain, rmse, r2,
                            original_residuals=original_residuals,
                            assumptions=[("Least squares assumes independent observations with "
                                          "comparable error scale.")] + ([
                                              ("Power-law parameters and diagnostics are fit in "
                                               "log-log space; interpretation assumes "
                                               "multiplicative errors.")]
                                              if method == "power_law" else []))
    if bootstrap:
        artifact.interval = _bootstrap(artifact, bootstrap, confidence, seed)
        draws = artifact.interval["parameter_draws"]
        artifact.parameter_uncertainty = {
            key: math.sqrt(sum((draw[key] - sum(d[key] for d in draws) / len(draws))**2
                               for draw in draws) / (len(draws)-1))
            for key in params
        }
    return artifact


def _least_squares(x: list[float], y: list[float]) -> tuple[float, float]:
    mx, my = sum(x)/len(x), sum(y)/len(y)
    denom = sum((v-mx)**2 for v in x)
    if denom == 0:
        raise ValueError("fit requires at least two distinct x values")
    slope = sum((a-mx)*(b-my) for a, b in zip(x, y)) / denom
    return slope, my - slope*mx


def _predict(method: str, params: dict[str, float], x: float) -> float:
    try:
        value = (params["slope"] * x + params["intercept"] if method == "linear"
                 else params["a"] * x ** params["b"])
    except (OverflowError, ZeroDivisionError) as e:
        raise ValueError("prediction is outside the finite numeric range") from e
    if not math.isfinite(value):
        raise ValueError("prediction is outside the finite numeric range")
    return value


def _finite(v: float, label: str) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{label} values must be finite numbers") from e
    if not math.isfinite(f):
        raise ValueError(f"{label} values must be finite numbers")
    return f


def _bootstrap(fit: FitArtifact, n: int, confidence: float, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    preds: dict[float, list[float]] = {}
    for x, _ in fit.observations:
        preds[x] = []
    draws = []
    for _ in range(n):
        sample = [rng.choice(fit.observations) for _ in fit.observations]
        try:
            sub = _fit(fit.method, [p[0] for p in sample], [p[1] for p in sample],
                       bootstrap=0, confidence=confidence, seed=seed)
        except ValueError:
            continue
        draws.append(sub.parameters)
        for x, values in preds.items():
            values.append(_predict(fit.method, sub.parameters, x))
    if len(draws) < 2:
        raise ValueError("too few non-degenerate bootstrap refits")
    alpha = (1-confidence)/2
    pinterval = lambda values: [
        _quantile(sorted(values), alpha), _quantile(sorted(values), 1-alpha)]
    return {"type": "confidence", "confidence": confidence, "resamples": n,
            "successful_resamples": len(draws),
            "parameters": {k: pinterval([d[k] for d in draws]) for k in fit.parameters},
            "parameter_draws": draws,
            "predictions": {repr(x): pinterval(values) for x, values in preds.items()
                            if len(values) >= 2}}


def _quantile(values: list[float], q: float) -> float:
    pos = (len(values)-1)*q
    i = int(pos)
    return values[i] + (values[min(i+1, len(values)-1)]-values[i])*(pos-i)
