"""Gradient flow: per-parameter and per-activation gradient norms for a scalar objective (U)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from topologies.introspection.objectives import as_objective, default_objective
from topologies.introspection.payloads import Payload, input_fingerprint, make_result
from topologies.introspection.techniques import Technique, builtin
from topologies.introspection.views import bars, finish, lines, table
from topologies.visualization.views import ViewSpec


@dataclass
class GradientsPayload(Payload):
    objective: str
    value: float | None
    parameters: list[dict[str, Any]]
    modules: list[dict[str, Any]]
    activations: list[dict[str, Any]] = field(default_factory=list)
    unused: list[str] = field(default_factory=list)
    nonfinite_origin: dict[str, Any] | None = None

    def to_records(self) -> list[dict[str, Any]]:
        return self.parameters

    def views(self, **options: Any) -> list[ViewSpec]:
        views = []
        if self.activations:
            views.append(lines("Activation gradient norm (call order)",
                               [{"y": [a["grad_norm"] for a in self.activations], "label": "|dL/dh|"}],
                               x_ticks=[a["site"] for a in self.activations], y_label="L2 norm",
                               log_y=True, caption=f"objective: {self.objective}"))
        if self.modules:
            views.append(lines("Parameter gradient norm per module (gradient flow)",
                               [{"y": [m["grad_norm"] for m in self.modules], "label": "|dL/dW|"}],
                               x_ticks=[m["module"] for m in self.modules], y_label="L2 norm",
                               log_y=True))
            views.append(bars("Gradient-to-weight ratio", [m["module"] for m in self.modules],
                              [m["ratio"] for m in self.modules]))
        if self.unused:
            views.append(table("Parameters without gradient", [{"parameter": p} for p in self.unused]))
        return finish(views, **options)

    def findings(self, params: dict[str, Any]) -> list[Any]:
        from topologies.introspection.report import Finding

        out = []
        if self.nonfinite_origin:
            out.append(Finding("nonfinite_gradient", "Non-finite gradients", "critical", "measured",
                               (self.nonfinite_origin["site"],), dict(self.nonfinite_origin),
                               "the first site in backward order whose gradient has NaN/Inf",
                               "check the loss scale, learning rate and the forward non-finite origin"))
        norms = [m["grad_norm"] for m in self.modules if m["grad_norm"] > 0]
        ratio_limit = float(params.get("gradient_ratio_threshold", 1e3))
        if len(norms) >= 2 and min(norms) > 0:
            spread = max(norms) / min(norms)
            if spread > ratio_limit:
                out.append(Finding("gradient_spread", "Gradient norms differ strongly across modules",
                                   "warning", "measured", (self.modules[0]["module"],
                                                           self.modules[-1]["module"]),
                                   {"max_over_min": spread, "threshold": ratio_limit},
                                   "some modules receive orders of magnitude smaller gradients "
                                   "(vanishing) or larger ones (exploding) for this objective", None))
        if self.unused:
            out.append(Finding("unused_parameters", f"{len(self.unused)} parameter(s) got no gradient",
                               "notice", "measured", tuple(self.unused[:20]), {"count": len(self.unused)},
                               "these parameters did not influence the objective for these inputs",
                               None))
        return out


@builtin
class Gradients(Technique):
    name = "gradients"
    tier = "U"
    requires = frozenset({"objective"})
    result_kind = "introspection.gradients"
    summary = "per-parameter/per-activation gradient norms, gradient flow (needs an objective)"

    def run(self, subject: Any, inputs: Any = None, *, objective: Any = None, sites: Any = "leaf",
            training: bool = False, **params: Any) -> Any:
        objective = as_objective(objective) if objective is not None else \
            default_objective(subject, inputs)
        session = subject.inspector.session(sites=sites, objective=objective, call_tree=True,
                                            keep="summary", grads=True, training=training,
                                            parameter_grads=True, **params)
        trace = session.run(inputs)
        payload = summarize(subject, trace, session, objective)
        assumptions = [f"objective: {objective.describe()}",
                       "eval-mode gradients (dropout off, batch-norm running stats)" if not training
                       else "training-mode gradients (dropout/batch-norm in train mode)",
                       "gradients computed with torch.autograd.grad: parameter .grad is untouched"]
        return make_result(self, payload, subject, warnings=trace.warnings,
                           assumptions=[*assumptions, *trace.assumptions],
                           provenance={"inputs": input_fingerprint(inputs)})


def summarize(subject: Any, trace: Any, session: Any, objective: Any) -> GradientsPayload:
    grads = getattr(session, "param_grads", {}) or {}
    params = dict(subject.model.named_parameters())
    rows, unused = [], []
    module_sq: dict[str, list[float]] = {}
    for name, grad in grads.items():
        weight = params[name].detach().float()
        module = name.rsplit(".", 1)[0] if "." in name else "<root>"
        if grad is None:
            unused.append(name)
            continue
        g = grad.detach().float()
        gn, wn = float(g.norm()), float(weight.norm())
        rows.append({"name": name, "module": module, "grad_norm": gn, "weight_norm": wn,
                     "ratio": gn / wn if wn > 0 else math.inf,
                     "nonfinite": not bool(torch.isfinite(g).all())})
        acc = module_sq.setdefault(module, [0.0, 0.0])
        acc[0] += gn ** 2
        acc[1] += wn ** 2
    order = {c.path or "<root>": c.index for c in trace.calls}
    modules = sorted(({"module": m, "grad_norm": math.sqrt(v[0]), "weight_norm": math.sqrt(v[1]),
                       "ratio": math.sqrt(v[0]) / math.sqrt(v[1]) if v[1] > 0 else math.inf}
                      for m, v in module_sq.items()), key=lambda r: order.get(r["module"], 1 << 30))
    activations = []
    for record in sorted(trace.records.values(), key=lambda r: (r.meta.get("order", 0))):
        if record.port == "grad_output" and record.summary:
            norm = record.summary.get("l2")
            activations.append({"site": record.site.removesuffix(":grad_output"),
                                "occurrence": record.occurrence,
                                "grad_norm": float(norm) if isinstance(norm, (int, float)) else math.nan,
                                "nonfinite_frac": record.summary.get("nonfinite_frac")})
    origin = None
    for entry in reversed(activations):
        frac = entry.get("nonfinite_frac")
        if isinstance(frac, (int, float)) and frac > 0 or (isinstance(frac, str)):
            origin = {"site": entry["site"], "occurrence": entry["occurrence"], "order": "backward"}
            break
    if origin is None:
        bad = [r["name"] for r in rows if r["nonfinite"]]
        if bad:
            origin = {"site": bad[-1], "occurrence": 0, "order": "parameters"}
    value = trace.metadata.get("objective")
    return GradientsPayload(objective.describe(), value, rows, modules, activations, unused, origin)
