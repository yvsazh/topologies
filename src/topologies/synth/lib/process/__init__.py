"""Reproducible sequences, event times, and hidden-state processes.

    synth.process.markov(states=["dry", "wet"], transitions=[[.9, .1], [.3, .7]])
    synth.process.random_walk(steps=64, dim=2)

All factories return nodes. State labels are generated with their observations.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from topologies import synth


def _count(value: int, name: str = "steps") -> int:
    if isinstance(value, bool) or int(value) != value or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _positive(value: float, name: str, *, zero: bool = False) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0 or (value == 0 and not zero):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return value


def _weights(values: Sequence[float], size: int) -> list[float]:
    weights = [float(x) for x in values]
    if len(weights) != size or any(not math.isfinite(x) or x < 0 for x in weights):
        raise ValueError("Probability row must match its states and contain finite nonnegative weights")
    if sum(weights) <= 0:
        raise ValueError("Probability row must have positive total weight")
    return weights


def _indices(r, states, transitions, steps, start):
    n = _count(steps)
    if not states or len(transitions) != len(states):
        raise ValueError("States must be nonempty and transitions must be a square matrix")
    rows = [_weights(row, len(states)) for row in transitions]
    if start is None:
        current = r.randrange(len(states))
    else:
        try:
            current = list(states).index(start)
        except ValueError as exc:
            raise ValueError("Starting state must occur in states") from exc
    result = []
    for _ in range(n):
        result.append(current)
        current = r.weighted(list(range(len(states))), rows[current])
    return result


@synth.node(type=synth.Json, tags=["process", "markov"],
            example={"states": ["dry", "wet"], "transitions": [[.9, .1], [.3, .7]], "steps": 16})
def markov(r, *, states: Sequence[Any], transitions: Sequence[Sequence[float]],
           steps: int = 128, start: Any = None):
    """Draw state values from a finite Markov chain; transition rows are weights."""
    return [states[i] for i in _indices(r, states, transitions, steps, start)]


@synth.node(type=synth.Json, tags=["process", "hmm"], example={
    "states": ["cold", "hot"], "transitions": [[.9, .1], [.2, .8]],
    "emissions": [[.8, .2], [.1, .9]], "symbols": ["coat", "shirt"], "steps": 16})
def hmm(ctx, *, states: Sequence[Any], transitions: Sequence[Sequence[float]],
        emissions: Sequence[Sequence[float]], symbols: Sequence[Any],
        steps: int = 128, start: Any = None):
    """Categorical hidden Markov model, returning exact states and observations."""
    if not symbols or len(emissions) != len(states):
        raise ValueError("Emission rows must match states; symbols must be nonempty")
    rows = [_weights(row, len(symbols)) for row in emissions]
    indices = _indices(ctx.rng.child("states"), states, transitions, steps, start)
    hidden = [states[i] for i in indices]
    r = ctx.rng.child("observations")
    observations = [r.weighted(list(symbols), rows[i]) for i in indices]
    ctx.latent(states=hidden)
    return {"states": hidden, "observations": observations}


@synth.node(type=synth.Series, tags=["process", "walk"], example={"steps": 64, "dim": 2})
def random_walk(r, *, steps: int = 128, dim: int = 1, scale: float = 1.,
                drift: float = 0., start: float = 0.):
    """Gaussian increments; output includes one point per step, after its increment."""
    n, d = _count(steps), _count(dim, "dim")
    if d == 0 or not math.isfinite(float(drift)) or not math.isfinite(float(start)):
        raise ValueError("dim must be positive; start and drift must be finite")
    scale = _positive(scale, "scale", zero=True)
    increments = torch.randn((n, d), generator=r.torch) * scale + drift
    values = increments.cumsum(0) + start
    return values[:, 0] if d == 1 else values


def _arma(r, steps, ar_coeffs, ma_coeffs, scale, mean, initial):
    n = _count(steps)
    scale = _positive(scale, "scale", zero=True)
    ar_coeffs, ma_coeffs = list(ar_coeffs), list(ma_coeffs)
    if any(not math.isfinite(float(x)) for x in [*ar_coeffs, *ma_coeffs, mean, initial]):
        raise ValueError("Coefficients, mean, and initial value must be finite")
    values, errors = [], []
    for k in range(n):
        error = r.normal(0., scale)
        value = mean + error
        value += sum(a * ((values[k-j-1] if k > j else initial) - mean)
                     for j, a in enumerate(ar_coeffs))
        value += sum(a * (errors[k-j-1] if k > j else 0.)
                     for j, a in enumerate(ma_coeffs))
        values.append(value)
        errors.append(error)
    return torch.tensor(values, dtype=torch.float32)


@synth.node(type=synth.Series, tags=["process", "ar"], example={"coefficients": [.8], "steps": 64})
def ar(r, *, coefficients: Sequence[float] = (.8,), steps: int = 128,
       scale: float = 1., mean: float = 0., initial: float = 0.):
    """Autoregressive sequence centered on mean; no stationarity constraint is imposed."""
    return _arma(r, steps, coefficients, (), scale, mean, initial)


@synth.node(type=synth.Series, tags=["process", "arma"],
            example={"ar": [.7], "ma": [.2], "steps": 64})
def arma(r, *, ar: Sequence[float] = (.8,), ma: Sequence[float] = (.2,),
         steps: int = 128, scale: float = 1., mean: float = 0., initial: float = 0.):
    """Autoregressive moving-average sequence with zero prehistory innovations."""
    return _arma(r, steps, ar, ma, scale, mean, initial)


def _event_times(r, duration, mean, distribution, shape, max_events):
    duration = _positive(duration, "duration", zero=True)
    mean = _positive(mean, "mean")
    cap = _count(max_events, "max_events")
    if distribution not in {"exponential", "gamma", "fixed"}:
        raise ValueError("distribution must be exponential, gamma, or fixed")
    shape = _positive(shape, "shape")
    times, elapsed = [], 0.
    while elapsed < duration:
        if distribution == "fixed":
            wait = mean
        elif distribution == "gamma":
            wait = r.gamma(shape, mean / shape)
        else:
            wait = -math.log1p(-max(r.random(), 1e-15)) * mean
        if wait <= 0 or not math.isfinite(wait):
            raise ValueError("Interarrival times must be positive and finite")
        elapsed += wait
        if elapsed > duration:
            break
        if len(times) == cap:
            raise ValueError("Event count exceeds max_events; increase max_events")
        times.append(elapsed)
    return times


@synth.node(type=synth.Series, tags=["process", "events"],
            example={"rate": 4., "duration": 2.})
def poisson_events(r, *, rate: float = 1., duration: float = 10., max_events: int = 100_000):
    """Sorted arrivals of a homogeneous Poisson process, in seconds."""
    rate = _positive(rate, "rate", zero=True)
    if rate == 0:
        _positive(duration, "duration", zero=True)
        _count(max_events, "max_events")
        return torch.empty(0)
    return torch.tensor(_event_times(r, duration, 1. / rate, "exponential", 1., max_events))


@synth.node(type=synth.Series, tags=["process", "renewal"],
            example={"mean": .25, "duration": 2., "distribution": "gamma"})
def renewal(r, *, mean: float = 1., duration: float = 10.,
            distribution: str = "exponential", shape: float = 2., max_events: int = 100_000):
    """Renewal arrival times with exponential, gamma, or fixed interarrival times."""
    return torch.tensor(_event_times(r, duration, mean, distribution, shape, max_events))


@synth.node(type=synth.Json, tags=["process", "state-machine"], example={"spec": {
    "start": "request", "transitions": {"request": [{"to": "done", "value": "ok"}],
                                        "done": []}}, "steps": 8})
def state_machine(ctx, *, spec: Mapping[str, Any], steps: int = 128):
    """Weighted edges ``{to, weight=1, value=None}``; terminal states end the sequence."""
    n = _count(steps)
    transitions = spec.get("transitions", {})
    state = spec.get("start")
    if state not in transitions:
        raise ValueError("State-machine start must occur in transitions")
    states, values = [], []
    for _ in range(n):
        states.append(state)
        edges = transitions[state]
        if not edges:
            break
        weights = _weights([edge.get("weight", 1.) for edge in edges], len(edges))
        edge = ctx.rng.weighted(list(edges), weights)
        state = edge["to"]
        if state not in transitions:
            raise ValueError("Every destination state must occur in transitions")
        values.append(edge.get("value"))
    ctx.latent(states=states)
    return {"states": states, "values": values}


@synth.node(type=synth.Json, tags=["process", "regimes"], example={
    "regimes": [{"mean": -2., "scale": .1}, {"mean": 2., "scale": .1}],
    "transitions": [[.95, .05], [.05, .95]], "steps": 32})
def regime_switch(ctx, *, regimes: Sequence[Mapping[str, float]],
                  transitions: Sequence[Sequence[float]], steps: int = 128, start: int = 0):
    """Gaussian AR(1) regimes; output includes exact regime indices for every value."""
    for regime in regimes:
        _positive(regime.get("scale", 1.), "scale", zero=True)
        if any(not math.isfinite(float(regime.get(k, v)))
               for k, v in (("mean", 0.), ("phi", 0.))):
            raise ValueError("Regime means and coefficients must be finite")
    indices = _indices(ctx.rng.child("regimes"), list(range(len(regimes))),
                       transitions, steps, start)
    r, values = ctx.rng.child("values"), []
    previous = 0.
    for index in indices:
        regime = regimes[index]
        mean = float(regime.get("mean", 0.))
        previous = mean + regime.get("phi", 0.) * (previous - mean)
        previous += r.normal(0., regime.get("scale", 1.))
        values.append(previous)
    ctx.latent(regimes=indices)
    return {"values": values, "regimes": indices}


__all__ = ["ar", "arma", "hmm", "markov", "poisson_events", "random_walk", "regime_switch",
           "renewal", "state_machine"]
