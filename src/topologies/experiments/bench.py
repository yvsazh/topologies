"""Component micro-benchmarks and property checks, built on ``Experiment``.

    from topologies.experiments import bench
    from topologies.nn import MultiHeadAttention
    from topologies.layers import SlidingWindowAttention, LinearAttention

    r = bench.run(
        "attention-zoo",
        components={"mha": bench.make(MultiHeadAttention, heads=4, causal=True),
                    "sliding": SlidingWindowAttention, "linear": LinearAttention},
        inputs=bench.sequence(dim=128, batch=8, lengths=[128, 256, 512, 1024]),
        checks="auto", repeats=20, warmup=5,
    )
    r.show()                  # metrics per component x size + check matrix (✓ ✗ –)
    r.scaling("fwd_ms")       # fitted exponent: fwd_ms ∝ T^k

Trials are ``component x input_size [x dtype]`` of an ordinary experiment, so
``topologies exp show / pivot / report`` work on benchmarks too.

Components: a class (called with the dim under ``dim`` / ``d_model`` /
``embed_dim`` / ``hidden_size`` / ``channels`` / ``in_channels``), a factory
``fn(dim)``, ``bench.make(cls, **kw)`` or an ``nn.Module`` instance (used as is).

Checks (``pass`` / ``fail(message)`` / ``n/a(reason)``): ``shape``, ``finite``,
``grad`` (incl. dead parameters), ``causal`` (sequence + declared causal),
``mask`` (when ``forward`` takes ``mask``), ``batch_independence``,
``determinism``, ``length_generalization``, ``dtype`` and ``compile`` (opt-in).
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from topologies.experiments.bench_checks import (
    ALL_CHECKS,
    AUTO_CHECKS,
    _out,
    _params_of,
    run_checks,
)
from topologies.experiments.errors import ExperimentError

DIM_ALIASES = ("dim", "d_model", "embed_dim", "hidden_size", "channels", "in_channels")
DEFAULT_MEASURE = ("params", "fwd_ms", "fwd_bwd_ms", "fwd_flops", "peak_mem_mb",
                   "tokens_per_s")


# ---------------------------------------------------------------- components


@dataclass
class Make:
    """``bench.make(cls, **kw)``: a component with fixed kwargs (``causal=`` declares it)."""

    cls: Any
    kwargs: dict[str, Any] = field(default_factory=dict)
    causal: bool | None = None

    def to_config(self) -> dict[str, Any]:
        from topologies.experiments.spec import canonical

        return {"cls": canonical(self.cls, warn=False),
                "kwargs": canonical(self.kwargs, warn=False), "causal": self.causal}


def make(cls: Any, *, causal: bool | None = None, **kwargs: Any) -> Make:
    """Component spec; ``causal=True`` is passed to ``cls`` when it accepts it and
    always marks the component as causal for the ``causal`` check."""
    return Make(cls=cls, kwargs=dict(kwargs), causal=causal)


def build_component(spec: Any, dim: int | None) -> tuple[nn.Module, bool]:
    """``(module, declared_causal)`` for a component spec (see module docstring)."""
    if isinstance(spec, nn.Module):
        return spec, bool(getattr(spec, "causal", False))
    cls, kw, causal = spec, {}, None
    if isinstance(spec, Make):
        cls, kw, causal = spec.cls, dict(spec.kwargs), spec.causal
    names, var_kw = _params_of(cls)
    if causal is not None and ("causal" in names or var_kw):
        kw.setdefault("causal", causal)
    alias = next((a for a in DIM_ALIASES if a in names), None) if dim is not None else None
    if alias is not None and alias not in kw:
        module = cls(**{alias: dim, **kw})
    elif dim is not None and alias is None and not isinstance(cls, type):
        module = cls(dim, **kw)  # factory fn(dim)
    else:
        module = cls(**kw)
    if not isinstance(module, nn.Module):
        raise ExperimentError(f"Component {spec!r} built {type(module).__name__}, "
                              "not an nn.Module.", stage="bench")
    declared = causal if causal is not None else bool(getattr(module, "causal", False))
    return module, bool(declared)


# ---------------------------------------------------------------- inputs


@dataclass
class Inputs:
    """Input builder: ``kind`` is ``sequence`` | ``image`` | ``custom``."""

    kind: str
    sizes: list[int]
    dim: int | None = None
    batch: int = 8
    channels: int | None = None
    fn: Callable[[int], Any] | None = None

    def make(self, size: int, device: Any, dtype: torch.dtype = torch.float32,
             *, scale: float = 1.0, seed: int = 0) -> tuple[tuple, dict]:
        g = torch.Generator().manual_seed(seed)
        if self.kind == "sequence":
            x = torch.randn(self.batch, int(size), int(self.dim), generator=g) * scale
            return (x.to(device=device, dtype=dtype),), {}
        if self.kind == "image":
            x = torch.randn(self.batch, int(self.channels), int(size), int(size),
                            generator=g) * scale
            return (x.to(device=device, dtype=dtype),), {}
        out = self.fn(int(size))  # type: ignore[misc]
        if isinstance(out, dict):
            return (), out
        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
            return tuple(out[0]), dict(out[1])
        return (out if isinstance(out, tuple) else (out,)), {}

    def tokens(self, size: int) -> int | None:
        return self.batch * int(size) if self.kind == "sequence" else None

    def describe(self) -> dict[str, Any]:
        return {"kind": self.kind, "sizes": self.sizes, "dim": self.dim, "batch": self.batch,
                "channels": self.channels}

    def to_config(self) -> dict[str, Any]:
        from topologies.experiments.spec import canonical

        return {**self.describe(), "fn": canonical(self.fn, warn=False) if self.fn else None}


def sequence(dim: int, batch: int = 8, lengths: Sequence[int] = (128, 256, 512)) -> Inputs:
    """``[B, T, D]`` inputs for every ``T`` in ``lengths``."""
    return Inputs("sequence", [int(t) for t in lengths], dim=int(dim), batch=int(batch))


def image(channels: int, batch: int = 8, sizes: Sequence[int] = (32, 64)) -> Inputs:
    """``[B, C, H, W]`` inputs (square, ``H = W = size``)."""
    return Inputs("image", [int(s) for s in sizes], dim=int(channels), batch=int(batch),
                  channels=int(channels))


def custom(fn: Callable[[int], Any], sizes: Sequence[int], *, dim: int | None = None) -> Inputs:
    """``fn(size) -> tensor | (args...) | {kwargs} | ((args...), {kwargs})``."""
    return Inputs("custom", [int(s) for s in sizes], dim=dim, fn=fn)


# ---------------------------------------------------------------- helpers


def _sync(device: torch.device) -> None:
    try:
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
    except Exception:  # noqa: BLE001, S110 - device sync is advisory for timing
        pass


def _time(fn: Callable[[], Any], device: torch.device, repeats: int,
          warmup: int) -> tuple[float, float]:
    for _ in range(max(0, warmup)):
        fn()
    _sync(device)
    times = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    p90 = times[min(len(times) - 1, math.ceil(0.9 * len(times)) - 1)]
    return statistics.median(times), p90


# ---------------------------------------------------------------- trial function


def _device(name: str | None) -> torch.device:
    from topologies.experiments.runner import resolve_device

    return torch.device(resolve_device(name or "auto"))


_DTYPES = {"fp32": torch.float32, "float32": torch.float32, "bf16": torch.bfloat16,
           "bfloat16": torch.bfloat16, "fp16": torch.float16, "float16": torch.float16}


def bench_trial(ctx: Any, component: Any, input_size: int, dtype: str = "fp32",
                **_: Any) -> Any:
    """Built-in Level-3 trial: measure one component at one input size."""
    from topologies.build.params import count_params
    from topologies.experiments.trial import TrialOutput

    p = ctx.params
    inputs: Inputs = p["bench.inputs"]
    device = _device(p.get("bench.device"))
    dt = _DTYPES.get(str(dtype), torch.float32)
    torch.manual_seed(ctx.seed)
    module, causal = build_component(component, inputs.dim)
    module = module.to(device=device, dtype=dt)
    measure = tuple(p.get("bench.measure") or DEFAULT_MEASURE)
    repeats, warmup = int(p.get("bench.repeats", 20)), int(p.get("bench.warmup", 5))
    args, kw = inputs.make(input_size, device, dt)
    metrics: dict[str, float] = {}
    notes: dict[str, str] = {}
    if "params" in measure:
        metrics["params"] = float(count_params(module))
    module.eval()

    def fwd() -> Any:
        with torch.no_grad():
            return module(*args, **kw)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    mem0 = _mps_mem() if device.type == "mps" else None
    med, p90 = _time(fwd, device, repeats, warmup)
    if "fwd_ms" in measure:
        metrics["fwd_ms"], metrics["fwd_ms_p90"] = med, p90
    if "tokens_per_s" in measure and inputs.tokens(input_size):
        metrics["tokens_per_s"] = inputs.tokens(input_size) / (med / 1000.0)
    if "samples_per_s" in measure or inputs.kind != "sequence":
        metrics["samples_per_s"] = inputs.batch / (med / 1000.0)
    if "fwd_flops" in measure:
        try:
            from topologies.experiments.costs import flop_counter

            counter = flop_counter()
            with counter:
                fwd()
            flops = float(counter.get_total_flops())
            if flops > 0:
                metrics["fwd_flops"] = flops
        except Exception as e:  # noqa: BLE001
            notes["fwd_flops"] = f"{type(e).__name__}: {e}"[:200]
    if "fwd_bwd_ms" in measure:
        module.train()
        grad_args = tuple(a.detach().requires_grad_(a.is_floating_point())
                          if isinstance(a, torch.Tensor) else a for a in args)

        def fwd_bwd() -> None:
            module.zero_grad(set_to_none=True)
            _out(module(*grad_args, **kw)).float().pow(2).mean().backward()

        try:
            metrics["fwd_bwd_ms"], metrics["fwd_bwd_ms_p90"] = _time(
                fwd_bwd, device, max(1, repeats // 2), max(0, warmup // 2))
        except Exception as e:  # noqa: BLE001
            notes["fwd_bwd_ms"] = f"{type(e).__name__}: {e}"[:200]
        module.zero_grad(set_to_none=True)
        module.eval()
    if "peak_mem_mb" in measure:
        if device.type == "cuda":
            metrics["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
        elif device.type == "mps" and mem0 is not None:
            metrics["peak_mem_mb"] = max(0.0, (_mps_mem() or mem0) - mem0)
            notes["peak_mem_mb"] = "MPS driver memory delta (sampled)"
        else:
            notes["peak_mem_mb"] = "not measured on CPU"
    extras: dict[str, Any] = {"device": str(device), "causal": causal, "notes": notes}
    checks = p.get("bench.checks")
    if checks and int(input_size) == min(inputs.sizes):
        extras["checks"] = run_checks(module, inputs, int(input_size), device, names=checks,
                                      causal=causal, expect_shape=p.get("bench.expect_shape"))
    history = [{"epoch": 0, **metrics}]
    return TrialOutput(metrics=metrics, history=history, extras=extras)


def _mps_mem() -> float | None:
    try:
        return float(torch.mps.driver_allocated_memory()) / 2**20
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- experiment / results


def _check_names(checks: Any) -> list[str]:
    if checks in (None, False, "none"):
        return []
    if checks == "auto":
        return list(AUTO_CHECKS)
    names = [checks] if isinstance(checks, str) else list(checks)
    unknown = [c for c in names if c not in ALL_CHECKS]
    if unknown:
        import difflib

        sug = difflib.get_close_matches(unknown[0], ALL_CHECKS, n=1)
        raise ExperimentError(f"Unknown check {unknown[0]!r}.", stage="bench",
                              hint=f"Did you mean {sug[0]!r}?" if sug else None,
                              available=list(ALL_CHECKS))
    return names


def experiment(name: str, *, components: dict[str, Any], inputs: Inputs,
               checks: Any = "auto", measure: Sequence[str] = DEFAULT_MEASURE,
               repeats: int = 20, warmup: int = 5, device: str = "auto",
               dtype: str | Sequence[str] = "fp32", expect_shape: Any = None,
               **options: Any) -> Any:
    """The benchmark as an ``Experiment`` (axes ``component x input_size [x dtype]``)."""
    from topologies.experiments.core import Experiment

    if not isinstance(components, dict) or not components:
        raise ExperimentError("components must be a non-empty {label: component} dict.",
                              stage="bench")
    if not isinstance(inputs, Inputs):
        raise ExperimentError("inputs must come from bench.sequence/image/custom.",
                              stage="bench")
    axes: dict[str, Any] = {"component": dict(components),
                            "input_size": list(inputs.sizes)}
    dtypes = [dtype] if isinstance(dtype, str) else list(dtype)
    if len(dtypes) > 1:
        axes["dtype"] = dtypes
    fixed = {"bench.inputs": inputs, "bench.checks": _check_names(checks),
             "bench.measure": list(measure), "bench.repeats": int(repeats),
             "bench.warmup": int(warmup), "bench.device": device,
             "bench.expect_shape": expect_shape}
    if len(dtypes) == 1:
        fixed["dtype"] = dtypes[0]
    options.setdefault("goal", ("fwd_ms", "min"))
    options.setdefault("monitor", "lines")
    return Experiment(name, axes=axes, fixed=fixed, trial_fn=bench_trial, **options)


class BenchResult:
    """``ResultSet`` wrapper with benchmark views (``show``, ``scaling``, check matrix)."""

    def __init__(self, results: Any) -> None:
        self.results = results

    def __getattr__(self, name: str) -> Any:
        return getattr(self.results, name)

    def table(self, metric: str = "fwd_ms") -> Any:
        """Components x input sizes for ``metric``."""
        return self.results.pivot("component", "input_size", metric=metric)

    def check_matrix(self) -> Any:
        from topologies.experiments.results import Table

        comps: dict[str, dict[str, dict[str, str]]] = {}
        for r in self.results.records:
            checks = (r.extras or {}).get("checks")
            if checks:
                comps[str(r.config_labels.get("component"))] = checks
        names = [c for c in ALL_CHECKS if any(c in v for v in comps.values())]
        icon = {"pass": "✓", "fail": "✗", "n/a": "–"}
        rows, disp, notes = [], [], []
        for comp, checks in comps.items():
            rows.append([comp, *[checks.get(n, {}).get("status") for n in names]])
            disp.append([comp, *[icon.get(checks.get(n, {}).get("status"), "?")
                                 for n in names]])
            for n in names:
                c = checks.get(n, {})
                if c.get("status") == "fail":
                    notes.append(f"✗ {comp} · {n}: {c.get('message')}")
        return Table(["component", *names], rows, display=disp,
                     meta={"title": "checks  (✓ pass · ✗ fail · – n/a)", "notes": notes,
                           "json": comps})

    def scaling(self, metric: str = "fwd_ms") -> Any:
        """Fitted exponent per component: ``metric ∝ size^k`` (log-log least squares)."""
        from topologies.experiments.results import Table
        from topologies.experiments.stats import loglog_slope

        series: dict[str, list[tuple[float, float]]] = {}
        for r in self.results.records:
            v = r.metric(metric)
            if r.status == "completed" and v is not None:
                series.setdefault(str(r.config_labels.get("component")), []).append(
                    (float(r.config_labels.get("input_size")), v))
        rows = []
        for comp, pts in series.items():
            pts.sort()
            k = loglog_slope([x for x, _ in pts], [y for _, y in pts])
            rows.append([comp, k, len(pts)])
        disp = [[c, "—" if k is None else f"{k:.2f}", str(n)] for c, k, n in rows]
        return Table(["component", f"exponent ({metric} ∝ size^k)", "sizes"], rows,
                     display=disp, meta={"title": f"scaling  {metric}",
                                         "json": {c: k for c, k, _ in rows}})

    def exponents(self, metric: str = "fwd_ms") -> dict[str, float | None]:
        return {row[0]: row[1] for row in self.scaling(metric).rows}

    def show(self) -> None:
        from topologies.experiments.render.console import render_table

        out = []
        for m in ("fwd_ms", "fwd_bwd_ms"):
            if any(r.metric(m) is not None for r in self.results.records):
                out += render_table(self.table(m)) + [""]
        out += render_table(self.check_matrix()) + [""]
        out += render_table(self.scaling("fwd_ms"))
        failed = [r for r in self.results.records if r.status != "completed"]
        for r in failed:
            out.append(f" ✗ {r.describe()}: {(r.error or {}).get('message', r.status)}"[:160])
        print("\n".join(out))

    def __repr__(self) -> str:
        return f"BenchResult({self.results!r})"


def run(name: str, *, components: dict[str, Any], inputs: Inputs, checks: Any = "auto",
        measure: Sequence[str] = DEFAULT_MEASURE, repeats: int = 20, warmup: int = 5,
        device: str = "auto", dtype: str | Sequence[str] = "fp32", **options: Any
        ) -> BenchResult:
    """Run (or resume) a component benchmark; see the module docstring."""
    exp = experiment(name, components=components, inputs=inputs, checks=checks,
                     measure=measure, repeats=repeats, warmup=warmup, device=device,
                     dtype=dtype, **options)
    return BenchResult(exp.run())


__all__ = ["ALL_CHECKS", "AUTO_CHECKS", "BenchResult", "Inputs", "Make", "bench_trial",
           "build_component", "custom", "experiment", "image", "make", "run", "run_checks",
           "sequence"]
