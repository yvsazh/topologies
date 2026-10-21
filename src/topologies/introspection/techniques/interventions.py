"""Causal techniques (U): ablation scans, activation patching, causal tracing, steering.

Only these techniques report *effects*, and each effect is tied to the intervention
that produced it (mean ablation differs from zero ablation). Off-distribution
effects of ablation are a known caveat.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from topologies._execution import ModelCall
from topologies.introspection import _tree
from topologies.introspection.errors import IntrospectionError, SemanticsRequired
from topologies.introspection.inputs import Batch, normalize_inputs
from topologies.introspection.interventions import Ablate, Intervention, Patch, Replace, Steer
from topologies.introspection.objectives import Objective, primary
from topologies.introspection.payloads import Payload, input_fingerprint, make_result
from topologies.introspection.techniques import Technique, builtin
from topologies.introspection.views import bars, finish, heatmap
from topologies.visualization.views import ViewSpec

# --------------------------------------------------------------------------- metrics


def output_metric(metric: Any) -> tuple[Callable[[Any, Any], float], str]:
    """``(fn(clean_output, output) -> float, description)``.

    Default: KL(clean ‖ intervened) over the last axis when the output looks like logits
    (floating, >= 2 dims), otherwise the relative L2 change.
    """
    if metric is None:
        def default(clean: Any, out: Any) -> float:
            a, b = primary(clean).float(), primary(out).float()
            if a.ndim >= 2 and a.shape[-1] > 1:
                p = a.log_softmax(-1)
                q = b.log_softmax(-1)
                return float((p.exp() * (p - q)).sum(-1).mean())
            return float((b - a).norm() / a.norm().clamp_min(1e-12))
        return default, ("KL(clean || intervened) over the last output axis (logits assumed); " \
            "relative L2 for non-logit outputs")
    if isinstance(metric, Objective):
        return (lambda clean, out: float(metric.compute(None, None, out))), \
            f"objective value {metric.describe()}"
    if callable(metric):
        return (lambda clean, out: float(metric(out))), getattr(metric, "__name__", "metric")
    raise TypeError("metric must be None, an Objective or callable(output) -> float")


def _value(metric: Any, output: Any) -> float:
    if isinstance(metric, Objective):
        return float(metric.compute(None, None, output))
    return float(metric(output))


# --------------------------------------------------------------------------- batch expansion


def _batch_size(batch: Batch) -> int | None:
    return batch.size


def _expand(batch: Batch, copies: int) -> Batch | None:
    """Repeat every tensor with the batch dimension ``copies`` times (``None`` if impossible)."""
    size = batch.size
    if size is None or batch.call is None:
        return None
    def rep(t: torch.Tensor) -> torch.Tensor:
        return t.repeat(copies, *([1] * (t.ndim - 1))) if t.ndim and t.shape[0] == size else t
    args = _tree.tree_map(rep, batch.call.args)
    kwargs = _tree.tree_map(rep, dict(batch.call.kwargs))
    return Batch(ModelCall(tuple(args), kwargs), meta=dict(batch.meta))


def _rows(output: Any, start: int, size: int) -> Any:
    return _tree.tree_map(lambda t: t[start:start + size] if t.ndim and t.shape[0] >= start + size
                          else t, output)


def batch_independent(inspector: Any, batch: Batch, *, atol: float = 1e-5) -> bool:
    """Run ``[x; x]`` and ``[x; x + noise]``: the first rows must equal ``x`` alone."""
    size = batch.size
    if size is None or batch.call is None:
        return False
    alone = primary(inspector.trace(batch, sites=None, keep="none", call_tree=False).output)
    doubled = _expand(batch, 2)
    if doubled is None:
        return False
    out = primary(inspector.trace(doubled, sites=None, keep="none", call_tree=False).output)
    if out.shape[0] != 2 * size or not torch.allclose(out[:size], alone, atol=atol) or \
            not torch.allclose(out[size:], alone, atol=atol):
        return False
    generator = torch.Generator().manual_seed(0)

    def perturb(t: torch.Tensor) -> torch.Tensor:
        if t.is_floating_point() and t.ndim and t.shape[0] == size:
            noise = torch.randn(t.shape, generator=generator).to(t)
            return torch.cat([t, t + noise * (t.std() + 1)])
        if t.ndim and t.shape[0] == size:
            return torch.cat([t, t.flip(0)])
        return t
    mixed = Batch(ModelCall(_tree.tree_map(perturb, batch.call.args),
                            _tree.tree_map(perturb, dict(batch.call.kwargs))), meta=batch.meta)
    out = primary(inspector.trace(mixed, sites=None, keep="none", call_tree=False).output)
    return out.shape[0] == 2 * size and bool(torch.allclose(out[:size], alone, atol=atol))


class _RowAblate(Intervention):
    """Applies a different ablation to each batch copy (rows ``j*B:(j+1)*B``)."""

    def __init__(self, plans: list[tuple[str, Ablate]], size: int):
        self.plans = plans
        self.size = size
        self.site = plans[0][1].site

    def paths(self, index: Any) -> list[str]:
        return list(dict.fromkeys(path for path, _ in self.plans))

    def install(self, session: Any, hooks: Any) -> None:
        from topologies.introspection.interventions import _axes

        by_path: dict[str, list[tuple[int, Ablate]]] = {}
        for j, (path, spec) in enumerate(self.plans):
            by_path.setdefault(path, []).append((j, spec))

        def post(frame: Any, module: Any, args: tuple, kwargs: dict, output: Any) -> Any:
            plans = by_path.get(frame.path)
            if not plans:
                return None
            leaf = "" if isinstance(output, torch.Tensor) else _tree.primary_path(output)
            if leaf is None:
                return None
            value = _tree.get_path(output, leaf) if leaf else output
            pieces = list(value.split(self.size, dim=0))
            for j, spec in plans:
                piece = pieces[j]
                ctx = {"session": session, "frame": frame, "module": module, "path": frame.path,
                       "axes": _axes(session, frame.path, module, piece)}
                pieces[j] = spec.edit(piece, ctx)
            new = torch.cat(pieces, dim=0)
            return _tree.set_path(output, leaf, new) if leaf else new
        hooks.post_handlers.append(post)


# --------------------------------------------------------------------------- payloads


@dataclass
class ScanPayload(Payload):
    labels: list[str]
    effects: list[float]
    metric: str
    value: Any
    batch_expanded: bool
    clean: float | None = None

    def to_records(self) -> list[dict[str, Any]]:
        return [{"target": label, "effect": effect} for label, effect in zip(self.labels, self.effects)]

    def views(self, **options: Any) -> list[ViewSpec]:
        return finish([bars(f"Ablation effect ({self.value} ablation)", self.labels, self.effects,
                            y_label="effect", caption=self.metric)], **options)

    def findings(self, params: dict[str, Any]) -> list[Any]:
        from topologies.introspection.report import Finding

        if not self.effects:
            return []
        order = sorted(range(len(self.effects)), key=lambda i: -abs(self.effects[i]))[:3]
        return [Finding("ablation_effect", f"Largest {self.value}-ablation effect: "
                        f"{self.labels[order[0]]}", "info", "measured",
                        tuple(self.labels[i] for i in order),
                        {self.labels[i]: self.effects[i] for i in order} | {"metric": self.metric},
                        f"replacing these outputs with their {self.value} value changed the "
                        "output most; this measures this intervention only (off-distribution "
                        "caveat applies)", None)]


@dataclass
class PatchingPayload(Payload):
    sites: list[str]
    positions: list[int] | None
    effects: torch.Tensor           # [S, P] normalised (m_patched - m_base) / (m_target - m_base)
    raw: torch.Tensor               # [S, P] m_patched
    clean: float
    corrupt: float
    direction: str
    metric: str
    tokens: list[str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_records(self) -> list[dict[str, Any]]:
        rows = []
        for i, site in enumerate(self.sites):
            for j in range(self.effects.shape[1]):
                rows.append({"site": site, "position": self.positions[j] if self.positions else None,
                             "effect": float(self.effects[i, j]), "metric": float(self.raw[i, j])})
        return rows

    def views(self, **options: Any) -> list[ViewSpec]:
        cols = self.tokens if self.tokens and self.positions and len(self.tokens) == len(
            self.positions) else self.positions
        return finish([heatmap(f"Activation patching ({self.direction})", self.effects,
                               rows=self.sites, cols=cols, center=0.0,
                               caption=f"normalised effect; metric: {self.metric}; "
                                       f"clean={self.clean:.4g}, corrupt={self.corrupt:.4g}",
                               x_label="position", y_label="site")], **options)


@dataclass
class SteerPayload(Payload):
    site: str
    alpha: float
    change: dict[str, float]
    output: Any = None

    def to_records(self) -> list[dict[str, Any]]:
        return [{"site": self.site, "alpha": self.alpha, **self.change}]

    def views(self, **options: Any) -> list[ViewSpec]:
        return finish([bars("Output change under steering", list(self.change),
                            list(self.change.values()))], **options)


# --------------------------------------------------------------------------- techniques


def _one(subject: Any, inputs: Any) -> Batch:
    return next(iter(normalize_inputs(subject, inputs)))


def _run(subject: Any, batch: Batch, specs: tuple = (), sites: Any = None) -> Any:
    session = subject.inspector.session(sites=sites, interventions=specs, call_tree=False,
                                        keep="tensors" if sites else "none", max_bytes=None,
                                        store_device="same")
    return session.run_batch(batch)


@builtin
class AblationScan(Technique):
    name = "interventions.ablation_scan"
    tier = "U"
    result_kind = "introspection.ablation_scan"
    summary = "effect of ablating each site (or unit) on an output metric"

    def run(self, subject: Any, inputs: Any = None, *, sites: Any = None, metric: Any = None,
            value: Any = "mean", reference: Any = None, units: bool | list[int] = False,
            chunk: int = 16, **params: Any) -> Any:
        index = subject.index
        if sites is None:
            sites = "role:block" if index.resolve_modules("role:block", allow_empty=True) else \
                [p for p in index.paths() if p and "." not in p]
        paths = index.resolve_modules(sites)
        batch = _one(subject, inputs)
        fn, description = output_metric(metric)
        clean = _run(subject, batch).output
        plans: list[tuple[str, str, Ablate]] = []
        for path in paths:
            label = path or "<root>"
            if units is False:
                plans.append((label, path, Ablate(label, value=value, reference=reference)))
            else:
                axes = subject.declared_axes(path)
                width = _width(subject, batch, path, axes)
                chosen = range(width) if units is True else units
                plans.extend((f"{label}[{u}]", path, Ablate(label, units=[u], value=value,
                                                             reference=reference)) for u in chosen)
        expanded = batch_independent(subject.inspector, batch)
        effects: list[float] = []
        assumptions = [f"metric: {description}", f"ablation value: {value}"]
        if expanded:
            size = batch.size
            for start in range(0, len(plans), chunk):
                part = plans[start:start + chunk]
                big = _expand(batch, len(part))
                spec = _RowAblate([(p, a) for _, p, a in part], size)
                out = _run(subject, big, (spec,)).output
                for j in range(len(part)):
                    effects.append(fn(clean, _rows(out, j * size, size)))
            assumptions.append("batch expansion: several ablations per forward (batch "
                               "independence verified at runtime)")
        else:
            for _, _, spec in plans:
                effects.append(fn(clean, _run(subject, batch, (spec,)).output))
            assumptions.append("sequential ablations (batch independence check failed)")
        if value == "mean" and reference is None:
            assumptions.append("mean ablation without reference= uses the batch mean of the "
                               "inputs themselves")
        payload = ScanPayload([p[0] for p in plans], effects, description, value, expanded)
        return make_result(self, payload, subject, assumptions=assumptions,
                           provenance={"inputs": input_fingerprint(inputs), "sites": str(sites)})


def _width(subject: Any, batch: Batch, path: str, axes: Any) -> int:
    trace = subject.inspector.trace(batch, sites=path or "<root>", keep="summary", call_tree=False)
    record = next(iter(trace.records.values()))
    axes = record.axes
    if axes is None or axes.feature_axis is None:
        raise SemanticsRequired(f"a unit scan at {path!r} needs a feature axis; declare it")
    return record.shape[axes.feature_axis]


@builtin
class Patching(Technique):
    name = "interventions.patching"
    tier = "U"
    result_kind = "introspection.patching"
    summary = "activation patching between clean and corrupt runs (normalised effects)"

    def run(self, subject: Any, inputs: Any = None, *, corrupt: Any = None, sites: Any = None,
            metric: Any = None, positions: Any = None, direction: str = "denoising",
            corrupt_interventions: tuple = (), **params: Any) -> Any:
        if metric is None:
            raise ValueError("patching needs metric=... (e.g. LogitDiff(a, b) or callable(output))")
        if direction not in ("denoising", "noising"):
            raise ValueError("direction must be 'denoising' or 'noising'")
        index = subject.index
        paths = index.resolve_modules(sites if sites is not None else "role:residual")
        clean_batch = _one(subject, inputs)
        corrupt_batch = _one(subject, corrupt) if corrupt is not None else clean_batch
        selector = ",".join(p or "<root>" for p in paths)
        clean = _run(subject, clean_batch, (), selector)
        bad = _run(subject, corrupt_batch, tuple(corrupt_interventions), selector)
        m_clean, m_corrupt = _value(metric, clean.output), _value(metric, bad.output)
        if direction == "denoising":
            source, base_batch, base_specs = clean, corrupt_batch, tuple(corrupt_interventions)
            base_value, target_value = m_corrupt, m_clean
        else:
            source, base_batch, base_specs = bad, clean_batch, ()
            base_value, target_value = m_clean, m_corrupt
        position_list = None
        if positions == "each":
            first = next(r for r in clean.records.values() if r.key.path == paths[0])
            axes = first.axes
            axis = axes.index("position") if axes is not None else None
            if axis is None:
                raise SemanticsRequired(f"positions='each' needs a position axis at {paths[0]!r}; "
                                        f"declare it, e.g. insp.declare({paths[0]!r}, axes='b t d')")
            for path in paths:
                a = next(r for r in clean.records.values() if r.key.path == path)
                b = next(r for r in bad.records.values() if r.key.path == path)
                if a.shape != b.shape:
                    raise IntrospectionError(
                        f"positional patching needs matching shapes at {path!r}: {a.shape} vs "
                        f"{b.shape}; use positions=None for pooled patching")
            position_list = list(range(first.shape[axis]))
        cols = position_list if position_list is not None else [None]
        raw = torch.zeros(len(paths), len(cols))
        for i, path in enumerate(paths):
            label = path or "<root>"
            shapes_differ = next(r for r in clean.records.values() if r.key.path == path).shape != \
                next(r for r in bad.records.values() if r.key.path == path).shape
            for j, pos in enumerate(cols):
                spec = Patch(label, source=source, positions=None if pos is None else [pos],
                             pool=shapes_differ and pos is None)
                out = _run(subject, base_batch, (*base_specs, spec)).output
                raw[i, j] = _value(metric, out)
        denominator = target_value - base_value
        effects = (raw - base_value) / denominator if abs(denominator) > 1e-12 else \
            torch.full_like(raw, float("nan"))
        tokens = clean_batch.meta.get("tokens", [None])[0] if positions == "each" else None
        payload = PatchingPayload([p or "<root>" for p in paths], position_list, effects, raw,
                                  m_clean, m_corrupt, direction, _describe(metric), tokens)
        warnings = [] if abs(denominator) > 1e-12 else [
            "clean and corrupt metrics are equal; normalised effects are undefined (raw values kept)"]
        return make_result(self, payload, subject, warnings=warnings, assumptions=[
            (f"{direction}: patch {'clean' if direction == 'denoising' else 'corrupt'} activations "
            f"into the {'corrupt' if direction == 'denoising' else 'clean'} run"),
            "normalised effect = (m_patched - m_base) / (m_target - m_base)"],
            provenance={"inputs": input_fingerprint(inputs), "corrupt": input_fingerprint(corrupt)})


def _describe(metric: Any) -> str:
    return metric.describe() if isinstance(metric, Objective) else getattr(metric, "__name__",
                                                                          "metric")


@builtin
class CausalTracing(Technique):
    name = "interventions.causal_tracing"
    tier = "A"
    requires = frozenset({"role:embed"})
    result_kind = "introspection.patching"
    summary = "ROME-style: corrupt the embedding with noise, restore (site, position) scans"

    def run(self, subject: Any, inputs: Any = None, *, metric: Any = None, sites: Any = None,
            noise: float | None = None, positions: Any = None, seed: int = 0, **params: Any) -> Any:
        embed = subject.index.resolve_modules("role:embed")[0]
        batch = _one(subject, inputs)
        clean = _run(subject, batch, (), embed or "<root>")
        emb = next(iter(clean.records.values())).value
        scale = noise if noise is not None else 3.0 * float(emb.float().std())
        generator = torch.Generator().manual_seed(seed)
        perturbation = torch.randn(emb.shape, generator=generator).to(emb) * scale
        corrupt_positions = positions

        def corrupt_fn(value: torch.Tensor, ctx: dict[str, Any]) -> torch.Tensor:
            delta = perturbation.to(value.device, value.dtype)
            if corrupt_positions is not None:
                mask = torch.zeros(value.shape[1], dtype=torch.bool, device=value.device)
                mask[list(corrupt_positions)] = True
                delta = delta * mask[None, :, None]
            return value + delta
        spec = Replace(embed or "<root>", corrupt_fn)
        if metric is None:
            from topologies.introspection.objectives import Predicted

            metric = Predicted()
            metric.per_example(None, None, clean.output)  # fix classes from the clean run
        result = Patching().run(subject, inputs, corrupt=inputs, sites=sites or "role:residual",
                                metric=metric, positions="each", corrupt_interventions=(spec,))
        result.assumptions.append(f"corruption: Gaussian noise (std={scale:.4g}, 3x the embedding "
                                  "std by default, following ROME) added at the embed output")
        result.provenance["technique"] = self.name
        result.payload.extra["noise_std"] = scale
        return result


@builtin
class SteerTechnique(Technique):
    name = "interventions.steer"
    tier = "U"
    result_kind = "introspection.steer"
    summary = "add a direction at a site and measure the output change"

    def run(self, subject: Any, inputs: Any = None, *, site: str = "", vector: Any = None,
            alpha: float = 1.0, positions: Any = None, **params: Any) -> Any:
        batch = _one(subject, inputs)
        clean = _run(subject, batch).output
        steered = _run(subject, batch, (Steer(site, vector, alpha, positions),)).output
        a, b = primary(clean).float(), primary(steered).float()
        change = {"relative_l2": float((b - a).norm() / a.norm().clamp_min(1e-12))}
        if a.ndim >= 2 and a.shape[-1] > 1:
            p, q = a.log_softmax(-1), b.log_softmax(-1)
            change["kl"] = float((p.exp() * (p - q)).sum(-1).mean())
            change["argmax_changed"] = float((a.argmax(-1) != b.argmax(-1)).float().mean())
        payload = SteerPayload(site, alpha, change, steered)
        return make_result(self, payload, subject, assumptions=[
            (f"steering adds {alpha} x vector at {site}; the effect is specific to this direction "
            "and scale")], provenance={"inputs": input_fingerprint(inputs)})


__all__ = ["AblationScan", "CausalTracing", "Patching", "SteerTechnique", "batch_independent",
           "output_metric"]
