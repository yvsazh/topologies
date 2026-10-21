"""Activation maximisation (D, experimental).

Optimises a continuous input, or a perturbation added at a site's output (``at=``), to
maximise an objective under optional regularisers. The only claim the result makes is
that it is an input that maximises this objective under these regularisers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from topologies._execution import ModelCall
from topologies.introspection.errors import SemanticsRequired
from topologies.introspection.inputs import Batch, normalize_inputs
from topologies.introspection.objectives import Objective, as_objective
from topologies.introspection.payloads import Payload, make_result
from topologies.introspection.techniques import Technique, builtin
from topologies.introspection.views import finish, lines, plain
from topologies.visualization.views import ViewSpec


class UnitActivation(Objective):
    """Mean activation of one unit (feature-axis index) at a site."""

    name = "unit_activation"

    def __init__(self, site: str, unit: int):
        self.site = site
        self.unit = unit
        self._value: torch.Tensor | None = None

    def paths(self, index: Any) -> list[str]:
        return [index.canonical(self.site)]

    def install(self, session: Any, hooks: Any) -> None:
        path = session.subject.index.canonical(self.site)
        self._axes = session.subject.declared_axes(path)

        def post(frame: Any, module: Any, args: tuple, kwargs: dict, output: Any) -> None:
            if frame.path == path and not frame.recompute:
                from topologies.introspection.objectives import primary
                from topologies.introspection.semantics import infer_axes

                self._value = primary(output)
                if self._axes is None:
                    self._axes = infer_axes(module, "output", self._value, None)
        hooks.post_handlers.append(post)

    def compute(self, session: Any, batch: Any, output: Any) -> torch.Tensor:
        value = self._value
        axes = self._axes
        if value is None:
            raise SemanticsRequired(f"site {self.site!r} did not run")
        if axes is None or axes.feature_axis is None:
            raise SemanticsRequired(f"unit maximisation at {self.site!r} needs a feature axis; "
                                    f"insp.declare({self.site!r}, axes='b t d')")
        return value.select(axes.feature_axis % value.ndim, self.unit).float().mean()

    def describe(self) -> str:
        return f"UnitActivation({self.site!r}, {self.unit})"


@dataclass
class MaximizationPayload(Payload):
    objective: str
    result: torch.Tensor
    history: list[float]
    regularizers: dict[str, float] = field(default_factory=dict)
    at: str = "input"

    def to_records(self) -> list[dict[str, Any]]:
        return [{"step": i, "objective": v} for i, v in enumerate(self.history)]

    def views(self, **options: Any) -> list[ViewSpec]:
        views = [lines("Objective during optimisation", [{"y": self.history, "label": "objective"}],
                       x_label="step", caption=f"{self.objective}; regularisers "
                                               f"{self.regularizers} (experimental)")]
        if self.result.ndim == 4:
            views.append(ViewSpec("image_grid", "Optimised input",
                                  {"images": [plain(x) for x in self.result[:8]],
                                   "labels": [f"#{i}" for i in range(min(8, self.result.shape[0]))]},
                                  {}, "an input that maximises this objective under these "
                                      "regularisers; nothing more"))
        return finish(views, **options)


def total_variation(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 3:
        return x.new_zeros(())
    dh = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    dw = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    return dh + dw


@builtin
class Maximize(Technique):
    name = "maximization"
    tier = "D"
    result_kind = "introspection.maximization"
    summary = "optimise an input (or a site perturbation) to maximise an objective (experimental)"

    def run(self, subject: Any, inputs: Any = None, *, objective: Any = None, init: Any = None,
            at: str | None = None, steps: int = 100, lr: float = 0.05,
            regularizers: dict[str, float] | None = None, seed: int = 0, **params: Any) -> Any:
        if objective is None:
            raise ValueError("maximize needs objective= (e.g. UnitActivation(site, unit), Logit(i))")
        objective = objective if isinstance(objective, Objective) else as_objective(objective)
        regularizers = dict(regularizers or {})
        generator = torch.Generator().manual_seed(seed)
        if at is None:
            if init is None:
                raise ValueError("maximize at the input needs init= (a tensor or a shape)")
            x0 = init.clone().float() if isinstance(init, torch.Tensor) else \
                torch.randn(tuple(init), generator=generator) * 0.1
            base = None
        else:
            if inputs is None:
                raise ValueError("maximize at a site needs inputs= to run the model on")
            base = next(iter(normalize_inputs(subject, inputs)))
            trace = subject.inspector.trace(base, sites=at, keep="tensors", call_tree=False,
                                            max_bytes=None)
            x0 = torch.zeros_like(next(iter(trace.records.values())).value.float())
        x = x0.to(subject.device).requires_grad_(True)
        optimizer = torch.optim.Adam([x], lr=lr)
        history: list[float] = []
        jitter = int(regularizers.get("jitter", 0))
        for _ in range(steps):
            point = x
            if jitter and x.ndim >= 3:
                shifts = [int(s) for s in torch.randint(-jitter, jitter + 1, (2,), generator=generator)]
                point = torch.roll(x, shifts=shifts, dims=(-2, -1))
            value, grad = _objective_grad(subject, objective, point.detach(), at, base)
            with torch.enable_grad():
                reg = x.new_zeros(())
                if regularizers.get("l2"):
                    reg = reg + regularizers["l2"] * x.pow(2).mean()
                if regularizers.get("tv"):
                    reg = reg + regularizers["tv"] * total_variation(x)
                reg_grad = torch.autograd.grad(reg, x)[0] if reg.requires_grad else torch.zeros_like(x)
            if jitter and x.ndim >= 3:
                grad = torch.roll(grad, shifts=[-s for s in shifts], dims=(-2, -1))
            optimizer.zero_grad()
            x.grad = -(grad - reg_grad)       # ascent on objective, descent on regularisers
            optimizer.step()
            history.append(value)
        payload = MaximizationPayload(objective.describe(), x.detach().cpu(), history, regularizers,
                                      at or "input")
        return make_result(self, payload, subject, assumptions=[
            ("experimental: an input that maximises this objective under these regularisers; "
            "it does not show what the model 'looks for' in natural data"),
            f"Adam, lr={lr}, {steps} steps, regularisers={regularizers}"])


def _objective_grad(subject: Any, objective: Objective, point: torch.Tensor, at: str | None,
                    base: Batch | None) -> tuple[float, torch.Tensor]:
    from topologies.introspection.capture import CapturePolicy
    from topologies.introspection.interventions import Replace
    from topologies.introspection.session import Session

    point = point.clone().requires_grad_(True)
    holder: dict[str, Any] = {}

    class _Probe(Objective):
        def install(inner, session: Any, hooks: Any) -> None:
            objective.install(session, hooks)

        def paths(inner, index: Any) -> list[str]:
            return objective.paths(index) if hasattr(objective, "paths") else []

        def compute(inner, session: Any, batch: Any, output: Any) -> torch.Tensor:
            value = objective.compute(session, batch, output)
            session.hooks.phase = "backward"
            try:
                holder["grad"] = torch.autograd.grad(value, point, allow_unused=True)[0]
            finally:
                session.hooks.phase = "forward"
            holder["value"] = float(value.detach())
            return value.detach()
    if at is None:
        batch = Batch(ModelCall((point,)))
        specs: tuple = ()
    else:
        batch = base
        specs = (Replace(at, lambda v, ctx: v + point.to(v.dtype)),)
    session = Session(subject, sites=None, policy=CapturePolicy(keep="none"), objective=_Probe(),
                      interventions=specs, call_tree=False)
    session.run_batch(batch)
    grad = holder.get("grad")
    return holder["value"], grad if grad is not None else torch.zeros_like(point)
