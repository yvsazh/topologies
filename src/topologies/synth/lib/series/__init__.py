"""``synth.series``: composable time series with exact anomaly / event masks.

    ts = synth.series.TimeSeries(length=256, channels=2, components=[
        synth.series.trend(slope=synth.uniform(-0.02, 0.02)),
        synth.series.seasonality(periods=[24, 168], amplitude=synth.uniform(0.5, 2)),
        synth.series.ar(coeffs=[0.6, -0.2], sigma=0.3),
        synth.series.regime_switch(n_regimes=2, switch_prob=0.01, scale=[1.0, 2.5]),
        synth.series.spikes(rate=0.01, scale=5.0),          # labelled anomalies
        synth.series.level_shift(rate=0.005, size=3.0),     # labelled events
        synth.series.noise(sigma=0.2),
        synth.series.missing(rate=0.02),
    ], cross_corr=0.6)

A record: ``{"series": [[c0, c1], ...] (or [x...] for one channel), "anomaly_mask":
[0/1...], "event_mask": [0/1...], "regime": [...]}``; masks are exact by
construction. Component parameters may be nodes (sampled per record; their paths
live under the series node) or attribute names (``amplitude="amp"`` reads the
record's attribute ``amp``). ``label=fn(info)`` adds a series-level label.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import torch

from topologies import synth


def _val(v: Any, ctx: Any) -> Any:
    if isinstance(v, synth.Node):
        return ctx.sample(v)
    if hasattr(v, "sample") and hasattr(v, "grid"):  # an experiments distribution
        return v.sample(ctx.rng)
    if isinstance(v, str) and v in ctx.attrs:
        return ctx.attrs[v]
    if isinstance(v, (list, tuple)):
        return [_val(x, ctx) for x in v]
    return v


class Component(synth.Node):
    """Base series component: ``apply(x [T, C] float64, info, ctx) -> x``."""

    tags = ("series", "component")

    def p(self, name: str, ctx: Any) -> Any:
        return _val(getattr(self, name), ctx)

    def apply(self, x: torch.Tensor, info: dict[str, Any], ctx: Any) -> torch.Tensor:
        raise NotImplementedError

    def sample(self, ctx: Any) -> Any:
        raise synth.SynthError(f"{type(self).__name__} is a series component.", stage="series",
                               hint="Put it in synth.series.TimeSeries(components=[...]).")


class trend(Component):
    """Linear + quadratic trend ``slope * t + curvature * t^2``."""

    example = {"slope": 0.01}

    def __init__(self, slope: Any = 0.01, curvature: Any = 0.0) -> None:
        self.slope, self.curvature = slope, curvature

    def apply(self, x, info, ctx):
        t = torch.arange(x.shape[0], dtype=torch.float64)
        slope, curv = float(self.p("slope", ctx)), float(self.p("curvature", ctx))
        return x + (slope * t + curv * t * t)[:, None]


class seasonality(Component):
    """Sum of sinusoids with ``periods`` (random phases)."""

    example = {"periods": [24], "amplitude": 1.0}

    def __init__(self, periods: Sequence[Any] = (24,), amplitude: Any = 1.0) -> None:
        self.periods, self.amplitude = list(periods), amplitude

    def apply(self, x, info, ctx):
        t = torch.arange(x.shape[0], dtype=torch.float64)
        amps, periods = self.p("amplitude", ctx), self.p("periods", ctx)
        amps = amps if isinstance(amps, list) else [amps] * len(periods)
        for per, amp in zip(periods, amps):
            phase = ctx.rng.uniform(0, 2 * math.pi)
            x = x + (float(amp) * torch.sin(2 * math.pi * t / float(per) + phase))[:, None]
        return x


class ar(Component):
    """Autoregressive noise ``y_t = sum_k c_k y_{t-k} + e_t``."""

    example = {"coeffs": [0.5], "sigma": 0.5}

    def __init__(self, coeffs: Sequence[Any] = (0.5,), sigma: Any = 0.5) -> None:
        self.coeffs, self.sigma = list(coeffs), sigma

    def apply(self, x, info, ctx):
        coeffs = [float(c) for c in self.p("coeffs", ctx)]
        sigma = float(self.p("sigma", ctx))
        t_len, c_len = x.shape
        e = torch.randn(t_len, c_len, generator=ctx.rng.torch, dtype=torch.float64) * sigma
        y = torch.zeros(t_len, c_len, dtype=torch.float64)
        for t in range(t_len):
            acc = e[t].clone()
            for k, c in enumerate(coeffs, 1):
                if t - k >= 0:
                    acc += c * y[t - k]
            y[t] = acc
        return x + y


class regime_switch(Component):
    """Markov regimes rescaling / offsetting the series (``info["regime"]`` per step)."""

    example = {"n_regimes": 2, "switch_prob": 0.02}

    def __init__(self, n_regimes: int = 2, switch_prob: Any = 0.01, scale: Any = None,
                 offset: Any = None) -> None:
        self.n_regimes, self.switch_prob, self.scale, self.offset = n_regimes, switch_prob, \
            scale, offset

    def apply(self, x, info, ctx):
        n = int(self.p("n_regimes", ctx))
        p = float(self.p("switch_prob", ctx))
        scales, offsets = self.p("scale", ctx), self.p("offset", ctx)
        scales = scales if isinstance(scales, list) else [1.0 + k for k in range(n)]
        offsets = offsets if isinstance(offsets, list) else [0.0] * n
        r = ctx.rng
        cur, regime = r.randrange(n), []
        for _ in range(x.shape[0]):
            if r.random() < p:
                cur = (cur + 1 + r.randrange(max(1, n - 1))) % n
            regime.append(cur)
        info["regime"] = regime
        sc = torch.tensor([float(scales[k]) for k in regime], dtype=torch.float64)[:, None]
        of = torch.tensor([float(offsets[k]) for k in regime], dtype=torch.float64)[:, None]
        return x * sc + of


class spikes(Component):
    """Point anomalies (``anomaly_mask`` = 1 where a spike is added)."""

    example = {"rate": 0.02, "scale": 5.0}

    def __init__(self, rate: Any = 0.01, scale: Any = 5.0) -> None:
        self.rate, self.scale = rate, scale

    def apply(self, x, info, ctx):
        rate, scale = float(self.p("rate", ctx)), float(self.p("scale", ctx))
        mask = info.setdefault("anomaly", [0] * x.shape[0])
        r = ctx.rng
        for t in range(x.shape[0]):
            if r.random() < rate:
                c = r.randrange(x.shape[1])
                x[t, c] += scale * (1 if r.random() < 0.5 else -1) * (1 + r.random())
                mask[t] = 1
        return x


class level_shift(Component):
    """Persistent level changes (``event_mask`` = 1 at each change)."""

    example = {"rate": 0.01, "size": 3.0}

    def __init__(self, rate: Any = 0.005, size: Any = 3.0) -> None:
        self.rate, self.size = rate, size

    def apply(self, x, info, ctx):
        rate, size = float(self.p("rate", ctx)), float(self.p("size", ctx))
        mask = info.setdefault("event", [0] * x.shape[0])
        shift = torch.zeros(x.shape[1], dtype=torch.float64)
        out = x.clone()
        r = ctx.rng
        for t in range(x.shape[0]):
            if r.random() < rate:
                shift = shift + size * (1 if r.random() < 0.5 else -1)
                mask[t] = 1
            out[t] = x[t] + shift
        return out


class noise(Component):
    """White Gaussian noise."""

    example = {"sigma": 0.1}

    def __init__(self, sigma: Any = 0.1) -> None:
        self.sigma = sigma

    def apply(self, x, info, ctx):
        s = float(self.p("sigma", ctx))
        return x + torch.randn(x.shape, generator=ctx.rng.torch, dtype=torch.float64) * s


class missing(Component):
    """Values missing completely at random (``None`` in the stored series)."""

    example = {"rate": 0.05}

    def __init__(self, rate: Any = 0.02) -> None:
        self.rate = rate

    def apply(self, x, info, ctx):
        m = torch.rand(x.shape, generator=ctx.rng.torch, dtype=torch.float64) < \
            float(self.p("rate", ctx))
        info["missing"] = int(m.sum())
        return x.masked_fill(m, float("nan"))


class TimeSeries(synth.Node):
    """A record of one composed series with its masks (see module docstring)."""

    tags = ("series", "anomaly", "forecast")
    example = {"length": 64, "components": [{"call": "synth.series.seasonality"},
                                            {"call": "synth.series.noise"}]}

    def __init__(self, *, length: Any = 128, channels: int = 1,
                 components: Sequence[Component] = (), cross_corr: float = 0.0,
                 label: Callable[[dict[str, Any]], Any] | None = None,
                 label_name: str = "label") -> None:
        if not components:
            raise synth.SynthError("TimeSeries needs components=[...].", stage="series",
                                   hint="components=[synth.series.seasonality(), "
                                        "synth.series.noise()]")
        self.length, self.channels, self.components = length, int(channels), list(components)
        self.cross_corr, self.label, self.label_name = float(cross_corr), label, label_name
        self.evidence = "construction"
        self.produces = synth.Schema(series=synth.Series(channels=self.channels
                                                         if self.channels > 1 else None),
                                     anomaly_mask=synth.Tokens(vocab=2),
                                     event_mask=synth.Tokens(vocab=2))

    def params(self) -> dict[str, Any]:
        return {"length": self.length, "components": list(self.components)}

    def sample(self, ctx: Any) -> dict[str, Any]:
        t_len = int(_val(self.length, ctx))
        x = torch.zeros(t_len, self.channels, dtype=torch.float64)
        info: dict[str, Any] = {}
        for k, comp in enumerate(self.components):
            x = comp.apply(x, info, ctx.child(f"c{k}"))
        if self.channels > 1 and self.cross_corr:
            mix = torch.full((self.channels, self.channels), self.cross_corr,
                             dtype=torch.float64)
            mix.fill_diagonal_(1.0)
            x = torch.where(torch.isnan(x), x,
                            torch.nan_to_num(x) @ torch.linalg.cholesky(mix).T)
        rows = x.tolist() if self.channels > 1 else [r[0] for r in x.tolist()]
        rows = [([None if math.isnan(v) else round(v, 6) for v in r] if isinstance(r, list)
                 else (None if math.isnan(r) else round(r, 6))) for r in rows]
        rec: dict[str, Any] = {"series": rows,
                               "anomaly_mask": info.get("anomaly", [0] * t_len),
                               "event_mask": info.get("event", [0] * t_len)}
        if "regime" in info:
            rec["regime"] = info["regime"]
        if self.label is not None:
            rec[self.label_name] = self.label({**info, "anomaly_mask": rec["anomaly_mask"],
                                               "event_mask": rec["event_mask"],
                                               "attrs": dict(ctx.attrs), "series": rows})
        ctx.log(length=t_len)
        return rec


__all__ = ["Component", "TimeSeries", "ar", "level_shift", "missing", "noise", "regime_switch",
           "seasonality", "spikes", "trend"]
