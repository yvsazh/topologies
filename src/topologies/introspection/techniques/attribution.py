"""Attribution (D): saliency, input×gradient, Integrated Gradients, SmoothGrad, occlusion,
Grad-CAM and layer attribution.

Attribution is computed with respect to a **location**: the model input (floating
inputs only) or the output of one module (``at="blocks.2"``). Token models cannot
be attributed at the discrete input. They default to the ``embed`` role output,
then to the first ``nn.Embedding`` output (recorded as inferred), and otherwise
raise ``SemanticsRequired``. Attribution is correlational and describes this
objective's local sensitivity. It does not say which parts are responsible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from topologies._execution import ModelCall
from topologies.introspection import _tree
from topologies.introspection.errors import IntrospectionError, SemanticsRequired
from topologies.introspection.inputs import Batch, normalize_inputs
from topologies.introspection.objectives import Objective, target_objective
from topologies.introspection.payloads import Payload, input_fingerprint, make_result
from topologies.introspection.techniques import AVAILABLE, Availability, Technique, builtin
from topologies.introspection.views import finish, heatmap, plain
from topologies.visualization.views import ViewSpec

METHODS = ("saliency", "input_x_gradient", "integrated_gradients", "smoothgrad", "occlusion",
           "gradcam")


@dataclass
class AttributionPayload(Payload):
    method: str
    location: str
    target: str
    attributions: torch.Tensor
    per_position: torch.Tensor | None = None
    tokens: list[list[str]] | None = None
    completeness: dict[str, Any] | None = None
    inputs: torch.Tensor | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_records(self) -> list[dict[str, Any]]:
        rows = []
        scores = self.per_position if self.per_position is not None else \
            self.attributions.reshape(self.attributions.shape[0], -1)
        for b in range(scores.shape[0]):
            for i, value in enumerate(scores[b].tolist()[:512]):
                row = {"example": b, "index": i, "score": value}
                if self.tokens and b < len(self.tokens) and i < len(self.tokens[b]):
                    row["token"] = self.tokens[b][i]
                rows.append(row)
        return rows

    def views(self, **options: Any) -> list[ViewSpec]:
        caption = f"{self.method} at {self.location}; target: {self.target}"
        if self.completeness:
            caption += f"; completeness error {self.completeness['relative_error']:.2%}"
        views: list[ViewSpec] = []
        if self.per_position is not None and self.tokens:
            for b, toks in enumerate(self.tokens[:4]):
                views.append(ViewSpec("token_heat", f"Attribution (example {b})",
                                      {"tokens": toks, "scores": plain(self.per_position[b])}, {},
                                      caption))
        elif self.meta.get("kind") == "image":
            heat = self.attributions.abs().sum(1) if self.attributions.ndim == 4 else \
                self.attributions.abs()
            images = self.inputs if self.inputs is not None else heat
            views.append(ViewSpec("image_grid", "Attribution", {
                "images": [plain(img) for img in images[:8]],
                "overlay": [plain(h) for h in heat[:8]],
                "labels": [f"example {i}" for i in range(min(8, images.shape[0]))]}, {}, caption))
        else:
            matrix = self.per_position if self.per_position is not None else \
                self.attributions.reshape(self.attributions.shape[0], -1)
            views.append(heatmap("Attribution", matrix, center=0.0, caption=caption,
                                 x_label="feature / position", y_label="example"))
        return finish(views, **options)


# --------------------------------------------------------------------------- location plumbing


@dataclass
class Location:
    """Where attribution is taken: the input, or a module output (``path``)."""

    path: str | None                 # None = model input (first floating tensor)
    label: str
    reduce_axis: int | None = None   # feature axis summed for per-position scores
    source: str = "declared"


class _SetSite:
    """Intervention that replaces a module's output with a given tensor (gradient point)."""

    needs_grad = True

    def __init__(self, path: str, value: torch.Tensor):
        self.path = path
        self.value = value

    def paths(self, index: Any) -> list[str]:
        return [self.path]

    def kwarg_paths(self, index: Any) -> list[str]:
        return []

    def install(self, session: Any, hooks: Any) -> None:
        def post(frame: Any, module: nn.Module, args: tuple, kwargs: dict, output: Any) -> Any:
            if frame.path != self.path or frame.occurrence not in (0, -1):
                return None
            if isinstance(output, torch.Tensor):
                return self.value
            leaf = _tree.primary_path(output)
            return _tree.set_path(output, leaf or "", self.value)
        hooks.post_handlers.append(post)

    def cleanup(self, session: Any) -> None:
        return


def _primary_index(call: ModelCall) -> int:
    for i, arg in enumerate(call.args):
        if isinstance(arg, torch.Tensor):
            return i
    raise IntrospectionError("attribution at the input needs a positional tensor input")


def resolve_location(subject: Any, batch: Batch, at: Any) -> Location:
    index = subject.index
    if at not in (None, "input"):
        path = index.resolve_modules(at)[0] if not isinstance(at, str) or at != "<root>" else ""
        axes = subject.declared_axes(path)
        return Location(path, path or "<root>", axes.feature_axis if axes is not None else None,
                        "declared" if axes is not None else "unknown")
    if batch.call is not None:
        found = [a for a in batch.call.args if isinstance(a, torch.Tensor)]
        if found and found[0].is_floating_point():
            return Location(None, "input")
    embed = index.resolve_modules("role:embed", allow_empty=True)
    if embed:
        return Location(embed[0], embed[0], -1, "declared")
    for path in index.paths():
        if isinstance(index.module(path), nn.Embedding):
            return Location(path, path, -1, "inferred_from_module_type")
    raise SemanticsRequired(
        "inputs are not floating point, so attribution needs a differentiable site: pass "
        "at='<module path>' or declare one with insp.declare('<embedding path>', role='embed')")


class _Evaluator:
    """Evaluates objective value and gradient at a point of the location."""

    def __init__(self, subject: Any, batch: Batch, location: Location, objective: Objective,
                 training: bool = False):
        self.subject = subject
        self.batch = batch
        self.location = location
        self.objective = objective
        self.training = training

    def original(self) -> torch.Tensor:
        if self.location.path is None:
            call = self.batch.call
            return call.args[_primary_index(call)].detach()
        trace = self.subject.inspector.trace(self.batch, sites=self.location.path or "<root>",
                                             keep="tensors", call_tree=False, store_device="same",
                                             max_bytes=None)
        records = sorted(trace.records.values(), key=lambda r: r.occurrence)
        if not records:
            raise IntrospectionError(f"site {self.location.label!r} did not run")
        return records[0].value.detach()

    def __call__(self, point: torch.Tensor, *, need_grad: bool = True
                 ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from topologies.introspection.capture import CapturePolicy
        from topologies.introspection.session import Session

        point = point.detach().clone().requires_grad_(need_grad)
        batch = self.batch
        interventions: tuple = ()
        if self.location.path is None:
            call = batch.call
            i = _primary_index(call)
            args = list(call.args)
            args[i] = point
            batch = Batch(ModelCall(tuple(args), dict(call.kwargs)), None, None, dict(batch.meta))
        else:
            interventions = (_SetSite(self.location.path, point),)
        holder: dict[str, Any] = {}

        class _Probe(Objective):
            def compute(inner, session: Any, b: Any, output: Any) -> torch.Tensor:
                per = self.objective.per_example(session, b, output)
                value = self.objective.compute(session, b, output)
                holder["per"] = per.detach() if per is not None else value.detach().reshape(1)
                if need_grad and value.requires_grad:
                    session.hooks.phase = "backward"  # checkpoint recompute is not a new call
                    try:
                        holder["grad"] = torch.autograd.grad(value, point, allow_unused=True)[0]
                    finally:
                        session.hooks.phase = "forward"
                return value.detach()
        session = Session(self.subject, sites=None, policy=CapturePolicy(keep="none",
                                                                         training=self.training),
                          objective=_Probe(), interventions=interventions, call_tree=False)
        session.run_batch(batch)
        grad = holder.get("grad")
        if need_grad and grad is None:
            grad = torch.zeros_like(point)
        return holder["per"], grad


# --------------------------------------------------------------------------- methods


def integrated_gradients(evaluate: _Evaluator, x: torch.Tensor, baseline: torch.Tensor, steps: int
                         ) -> tuple[torch.Tensor, dict[str, Any]]:
    """Trapezoid Riemann sum along the straight path baseline -> x."""
    alphas = torch.linspace(0, 1, steps + 1, device=x.device)
    total = torch.zeros_like(x, dtype=torch.float32)
    for i, alpha in enumerate(alphas):
        _, grad = evaluate(baseline + alpha * (x - baseline))
        weight = 0.5 if i in (0, steps) else 1.0
        total += weight * grad.float()
    attributions = (x - baseline).float() * total / steps
    fx, _ = evaluate(x, need_grad=False)
    fb, _ = evaluate(baseline, need_grad=False)
    delta = float((fx - fb).sum())
    error = float(attributions.sum()) - delta
    relative = abs(error) / max(abs(delta), 1e-12)
    return attributions, {"sum_attributions": float(attributions.sum()), "f_x_minus_f_baseline": delta,
                          "error": error, "relative_error": relative, "steps": steps,
                          "method": "riemann_trapezoid"}


def make_baseline(x: torch.Tensor, baseline: Any) -> tuple[torch.Tensor, str]:
    if baseline is None or baseline == "zero":
        return torch.zeros_like(x), "zero baseline"
    if isinstance(baseline, torch.Tensor):
        return baseline.to(x.device, x.dtype).expand_as(x).clone(), "user tensor baseline"
    if baseline == "uniform":
        g = torch.Generator(device="cpu").manual_seed(0)
        lo, hi = float(x.min()), float(x.max())
        return (torch.rand(x.shape, generator=g) * (hi - lo) + lo).to(x), "uniform-noise baseline"
    if baseline == "mean":
        return x.mean(0, keepdim=True).expand_as(x).clone(), "batch-mean baseline"
    if baseline == "blur":
        if x.ndim != 4:
            raise ValueError("blur baseline needs [B, C, H, W] inputs")
        kernel = torch.ones(x.shape[1], 1, 5, 5, device=x.device, dtype=x.dtype) / 25
        return F.conv2d(x, kernel, padding=2, groups=x.shape[1]), "5x5 box-blur baseline"
    if callable(baseline):
        return baseline(x), "callable baseline"
    raise ValueError("baseline must be 'zero', 'uniform', 'mean', 'blur', a tensor or a callable")


def occlusion(evaluate: _Evaluator, x: torch.Tensor, window: tuple[int, ...] | None,
              stride: tuple[int, ...] | None, value: float) -> torch.Tensor:
    """Drop in the objective when a sliding window (over non-batch axes) is set to ``value``."""
    base, _ = evaluate(x, need_grad=False)
    dims = x.shape[1:]
    if window is None:
        window = (dims[0], *(max(1, d // 4) for d in dims[1:])) if len(dims) >= 2 else (1,)
    window = tuple(window)
    stride = tuple(stride) if stride is not None else window
    if len(window) != len(dims) or len(stride) != len(dims):
        raise ValueError(f"window/stride need {len(dims)} entries (non-batch axes)")
    scores = torch.zeros_like(x, dtype=torch.float32)
    counts = torch.zeros_like(x, dtype=torch.float32)
    starts = [range(0, max(d - w, 0) + 1, s) for d, w, s in zip(dims, window, stride)]
    import itertools

    for corner in itertools.product(*starts):
        region = (slice(None), *(slice(c, c + w) for c, w in zip(corner, window)))
        occluded = x.clone()
        occluded[region] = value
        out, _ = evaluate(occluded, need_grad=False)
        drop = (base - out).float()
        shape = (x.shape[0],) + (1,) * len(dims)
        scores[region] += drop.reshape(shape)
        counts[region] += 1
    return scores / counts.clamp_min(1)


def gradcam(subject: Any, batch: Batch, site: str, objective: Objective) -> tuple[torch.Tensor, Any]:
    path = subject.index.resolve_modules(site)[0]
    trace = subject.inspector.trace(batch, sites=f"{path},{path}:grad_output", keep="tensors",
                                    objective=objective, call_tree=False, store_device="same",
                                    max_bytes=None)
    records = {r.port: r for r in trace.records.values() if r.occurrence == 0}
    activation, grad = records.get("output"), records.get("grad_output")
    if activation is None or grad is None:
        raise IntrospectionError(f"Grad-CAM: no activation/gradient captured at {site!r}")
    axes = activation.axes
    if (axes is None or axes.names[:2] != ("batch", "channel")) and activation.value.ndim == 4:
        axes = _conv_descendant_axes(subject, path, activation.value)
    if axes is None or axes.names[:2] != ("batch", "channel") or len(axes.names) < 3:
        raise SemanticsRequired(f"Grad-CAM needs spatial axes at {site!r} (b c h w); declare them: "
                                f"insp.declare({site!r}, axes='b c h w')")
    a, g = activation.value.float(), grad.value.float()
    return _cam(a, g), axes


def _cam(a: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    spatial = tuple(range(2, a.ndim))
    weights = g.mean(dim=spatial, keepdim=True)
    return torch.relu((weights * a).sum(1))


def _conv_descendant_axes(subject: Any, path: str, value: torch.Tensor) -> Any:
    """``b c h w`` when the site's last conv/batch-norm descendant has the same channel count."""
    from topologies.introspection.semantics import Axes, SemanticSource

    module = subject.index.module(path)
    last = None
    for child in module.modules():
        if isinstance(child, (nn.Conv2d, nn.BatchNorm2d)):
            last = child
    channels = getattr(last, "out_channels", None) or getattr(last, "num_features", None)
    if channels is None or value.shape[1] != channels:
        return None
    return Axes(("batch", "channel", "height", "width"), SemanticSource.INFERRED_FROM_MODULE_TYPE,
                1, 0)


# --------------------------------------------------------------------------- technique


@builtin
class Attribution(Technique):
    name = "attribution"
    tier = "D"
    requires = frozenset({"differentiable_input"})
    result_kind = "introspection.attribution"
    summary = "saliency, input×grad, IG, SmoothGrad, occlusion, Grad-CAM, layer attribution"

    def check(self, subject: Any) -> Availability:
        index = subject.index
        if any(isinstance(index.module(p), (nn.Embedding,)) for p in index.paths()) or \
                index.resolve_modules("role:embed", allow_empty=True):
            return AVAILABLE
        return Availability(True, "needs floating inputs or at=<site>")

    def run(self, subject: Any, inputs: Any = None, *, method: str = "integrated_gradients",
            target: Any = "predicted", objective: Any = None, at: Any = None, site: Any = None,
            baseline: Any = None, steps: int = 32, n: int = 16, sigma: float = 0.15,
            window: tuple[int, ...] | None = None, stride: tuple[int, ...] | None = None,
            occlusion_value: float = 0.0, seed: int = 0, training: bool = False,
            **params: Any) -> Any:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        batch = next(iter(normalize_inputs(subject, inputs)))
        if batch.batch is not None:  # recipe dict -> positional call for gradient plumbing
            batch = _as_call(subject, batch)
        obj = objective if isinstance(objective, Objective) else target_objective(
            objective if objective is not None else target)
        assumptions = [f"objective: {obj.describe()}"]
        warnings: list[str] = []
        meta: dict[str, Any] = {}
        if method == "gradcam":
            if site is None and at is None:
                raise ValueError("gradcam needs site='<conv layer>'")
            cam, cam_axes = gradcam(subject, batch, site or at, obj)
            if cam_axes.source.value == "inferred_from_module_type" and cam_axes.names[1] == \
                    "channel" and not isinstance(subject.index.module(subject.index.resolve_modules(
                        site or at)[0]), nn.Conv2d | nn.BatchNorm2d):
                assumptions.append(f"Grad-CAM axes at {site or at} inferred from its last "
                                   "conv/batch-norm descendant (same channel count)")
            x = batch.call.args[_primary_index(batch.call)] if batch.call is not None else None
            if x is not None and x.ndim == 4 and cam.ndim == 3:
                cam = F.interpolate(cam[:, None], size=x.shape[-2:], mode="bilinear",
                                    align_corners=False)[:, 0]
                assumptions.append("Grad-CAM map upsampled bilinearly to the input size")
                meta["kind"] = "image"
            payload = AttributionPayload("gradcam", site or at, obj.describe(), cam.detach().cpu(),
                                         inputs=x.detach().cpu() if x is not None else None, meta=meta)
            return make_result(self, payload, subject, assumptions=assumptions,
                               provenance={"inputs": input_fingerprint(inputs), "method": method})
        location = resolve_location(subject, batch, at if at is not None else site)
        if location.source == "inferred_from_module_type":
            assumptions.append(f"token inputs: attribution taken at the first nn.Embedding output "
                               f"({location.label}); source=INFERRED_FROM_MODULE_TYPE")
        evaluate = _Evaluator(subject, batch, location, obj, training)
        x = evaluate.original().float() if location.path is not None else evaluate.original()
        if not x.is_floating_point():
            raise SemanticsRequired("attribution needs a floating location; pass at='<site>'")
        completeness = None
        if method == "saliency":
            _, grad = evaluate(x)
            attributions = grad.float().abs()
            assumptions.append("saliency = |gradient|")
        elif method == "input_x_gradient":
            _, grad = evaluate(x)
            attributions = (x * grad).float()
        elif method == "integrated_gradients":
            base, note = make_baseline(x, baseline)
            assumptions.append(f"integrated gradients: {note}, {steps} steps, trapezoid rule")
            attributions, completeness = integrated_gradients(evaluate, x, base, steps)
            if completeness["relative_error"] > 0.05:
                warnings.append(f"IG completeness error {completeness['relative_error']:.1%} > 5%; "
                                "increase steps=")
        elif method == "smoothgrad":
            g = torch.Generator(device="cpu").manual_seed(seed)
            scale = sigma * float(x.max() - x.min())
            total = torch.zeros_like(x, dtype=torch.float32)
            for _ in range(n):
                noise = torch.randn(x.shape, generator=g).to(x) * scale
                _, grad = evaluate(x + noise)
                total += grad.float()
            attributions = total / n
            assumptions.append(f"smoothgrad: {n} samples, sigma={sigma} x input range")
        else:  # occlusion
            attributions = occlusion(evaluate, x, window, stride, occlusion_value)
            assumptions.append(f"occlusion: window={window or 'auto'}, value={occlusion_value}; "
                               "score = drop of the objective")
        per_position = None
        tokens = batch.meta.get("tokens")
        if location.path is not None and location.reduce_axis is not None and attributions.ndim >= 3:
            per_position = attributions.sum(dim=location.reduce_axis)
            assumptions.append("per-position scores sum attributions over the feature axis")
        elif location.path is not None and attributions.ndim == 3:
            per_position = attributions.sum(-1)
            assumptions.append("per-position scores sum attributions over the last axis "
                               "(axes unknown)")
        if location.path is None and x.ndim == 4:
            meta["kind"] = "image"
        payload = AttributionPayload(method, location.label, obj.describe(),
                                     attributions.detach().cpu(),
                                     per_position.detach().cpu() if per_position is not None else None,
                                     tokens, completeness,
                                     x.detach().cpu() if location.path is None else None, meta)
        return make_result(self, payload, subject, warnings=warnings, assumptions=[
            *assumptions, ("attribution is correlational: it measures local sensitivity of the "
            "objective, not responsibility")], provenance={"inputs": input_fingerprint(inputs),
                                                          "method": method, "at": location.label})


def _as_call(subject: Any, batch: Batch) -> Batch:
    recipe = subject.recipe
    keys = [k for k in batch.batch if k not in recipe.target_keys()]
    primary = recipe.input_keys()[0] if recipe.input_keys() else keys[0]
    from topologies.training.adapters import bind_inputs

    args, kwargs = bind_inputs(subject.model, {k: batch.batch[k] for k in keys}, primary=primary)
    return Batch(ModelCall(tuple(args), dict(kwargs)), None, None, dict(batch.meta))
