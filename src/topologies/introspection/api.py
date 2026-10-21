"""``introspect()`` and the ``Inspector`` facade."""
from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import torch.nn as nn

from topologies.introspection.capture import CapturePolicy
from topologies.introspection.semantics import Axes, Unit, module_feature_axes, require_feature_axis
from topologies.introspection.session import Session
from topologies.introspection.sites import ROOT
from topologies.introspection.subject import Subject
from topologies.introspection.trace import Trace

_POLICY_FIELDS = set(CapturePolicy.__dataclass_fields__)


def introspect(source: Any, *, tokenizer: Any = None, recipe: Any = None, adapter: Any = "auto",
               device: Any = None, policy: CapturePolicy | None = None) -> Inspector:
    """Open an inspector on a path, ``Pretrained``, ``Trainer`` or any ``nn.Module``."""
    subject = Subject.resolve(source, tokenizer=tokenizer, recipe=recipe, adapter=adapter,
                              device=device)
    return Inspector(subject, policy=policy)


class Inspector:
    """Discovery, capture and every technique for one subject (model)."""

    def __init__(self, subject: Subject, *, policy: CapturePolicy | None = None):
        self.subject = subject
        subject.inspector = self
        self.policy = policy or CapturePolicy()

    # -- metadata ------------------------------------------------------------------------

    @property
    def model(self) -> nn.Module:
        return self.subject.model

    @property
    def tokenizer(self) -> Any:
        return self.subject.tokenizer

    @property
    def recipe(self) -> Any:
        return self.subject.recipe

    @property
    def adapter(self) -> Any:
        return self.subject.adapter

    @property
    def index(self) -> Any:
        return self.subject.index

    def sites(self, selector: Any = "*") -> list[str]:
        """Module paths matching ``selector`` (``<root>`` for the model itself)."""
        return [p or ROOT for p in self.index.resolve_modules(selector)]

    def describe(self, site: str) -> dict[str, Any]:
        path = self.index.canonical(site)
        info = self.index.modules.get(path)
        if info is None:
            raise KeyError(self.index.unknown_message(site))
        module = self.index.module(path)
        out = info.to_dict()
        declared = self.subject.declared_axes(path)
        out["axes"] = declared.to_dict() if declared is not None else None
        out["children"] = [f"{path}.{n}" if path else n for n, _ in module.named_children()]
        out["parameters"] = {n: list(p.shape) for n, p in module.named_parameters(recurse=False)}
        return out

    def declare(self, selector: Any, *, axes: Any = None, role: str | None = None) -> Inspector:
        """Declare axes (``"b t d"``) and/or a semantic role for selected modules."""
        self.subject.declare(selector, axes=axes, role=role)
        return self

    def units(self, site: str, *, at: Any = None) -> list[Unit]:
        """Units along the feature axis of a site (runs nothing; needs known axes + width)."""
        path = self.index.canonical(site.split(":")[0])
        module = self.index.module(path)
        axes = self.subject.declared_axes(path)
        if axes is None:
            axes = module_feature_axes(module)
        require_feature_axis(site, axes)
        width = getattr(module, "out_features", None) or getattr(module, "out_channels", None) \
                or getattr(module, "embedding_dim", None) or getattr(module, "num_features", None) \
            or getattr(module, "hidden_size", None)
        if width is None:
            normalized = getattr(module, "normalized_shape", None)
            width = normalized[-1] if normalized else None
        if width is None:
            from topologies.introspection.errors import SemanticsRequired
            raise SemanticsRequired(f"width of {site!r} is unknown without running the model; "
                                    "use insp.trace(x, sites=site) and the record shape")
        return [Unit(site, i, at) for i in range(int(width))]

    def _policy(self, **overrides: Any) -> CapturePolicy:
        known = {k: v for k, v in overrides.items() if k in _POLICY_FIELDS}
        return CapturePolicy.build(self.policy, **known)

    # -- capture ---------------------------------------------------------------------------

    def session(self, *, sites: Any = "*", objective: Any = None, interventions: tuple = (),
                ops: Any = None, call_tree: bool = True, grad_inputs: bool = False,
                parameter_grads: bool = False, **policy: Any) -> Session:
        unknown = set(policy) - _POLICY_FIELDS
        if unknown:
            raise TypeError(f"unknown option(s): {', '.join(sorted(unknown))}")
        return Session(self.subject, sites=sites, policy=self._policy(**policy),
                       objective=_objective(objective), interventions=tuple(interventions),
                       ops=ops, call_tree=call_tree, grad_inputs=grad_inputs,
                       parameter_grads=parameter_grads)

    def trace(self, inputs: Any, sites: Any = "*", *, keep: str = "summary",
              ports: tuple[str, ...] | str = ("output",), ops: Any = None, objective: Any = None,
              call_tree: bool = True, **policy: Any) -> Trace:
        """Run once and record summaries (default) or tensors at the selected sites."""
        ports = (ports,) if isinstance(ports, str) else tuple(ports)
        session = self.session(sites=sites, objective=objective, ops=ops, call_tree=call_tree,
                               keep=keep, ports=ports, **policy)
        return session.run(inputs)

    def collect(self, data: Any, sites: Any, reducers: Any = ("moments",), *,
                max_batches: int | None = None, batch_size: int | None = None,
                progress: bool = True, visualizer: Any = None, resume: Any = None,
                checkpoint_every: int = 10, **policy: Any) -> Any:
        """Stream a dataset through streaming reducers (no activations are stored)."""
        from topologies.introspection.collection import collect

        return collect(self, data, sites, reducers, max_batches=max_batches,
                       batch_size=batch_size, progress=progress, visualizer=visualizer,
                       resume=resume, checkpoint_every=checkpoint_every,
                       policy=self._policy(keep="none", **policy))

    def run_with(self, inputs: Any, *interventions: Any, sites: Any = None,
                 keep: str = "summary", **policy: Any) -> Trace:
        """Run once with interventions applied (and optional capture at ``sites``)."""
        session = self.session(sites=sites, interventions=interventions, call_tree=False,
                               keep=keep, **policy)
        return session.run(inputs)

    @contextlib.contextmanager
    def intervene(self, *interventions: Any) -> Iterator[nn.Module]:
        """Apply interventions to plain ``model(...)`` calls inside the block.

        Occurrence filters (``site@k``) count calls across the whole block. Mode and grad
        settings are the caller's own; nothing is captured.
        """
        session = self.session(sites=None, interventions=interventions, call_tree=False,
                               keep="none")
        with session.open():
            yield self.model

    # -- techniques ---------------------------------------------------------------------------

    def capabilities(self) -> Any:
        """Which techniques can run on this model, and why the others cannot."""
        from topologies.introspection.techniques import capabilities

        return capabilities(self.subject)

    def run(self, technique: str, inputs: Any = None, **params: Any) -> Any:
        """Run any registered technique by name; returns an ``AnalysisResult``."""
        from topologies.introspection.errors import SemanticsRequired
        from topologies.introspection.techniques import DEFAULT_TECHNIQUES

        cls = DEFAULT_TECHNIQUES.get(technique)
        instance = cls()
        status = instance.check_call(self.subject, params)
        if not status:
            message = f"{technique} is unavailable: {status.reason}"
            if status.fix:
                message += f". Fix: {status.fix}"
            raise SemanticsRequired(message) if cls.tier == "A" else _unavailable(message)
        return instance.run(self.subject, inputs, **params)

    def structure(self) -> Any:
        return self.run("structure")

    def call_tree(self, inputs: Any) -> Any:
        return self.run("structure.call_tree", inputs)

    def graph(self, inputs: Any, level: str = "module", **params: Any) -> Any:
        return self.run("structure.graph", inputs, level=level, **params)

    def activations(self, inputs_or_data: Any, sites: Any = "leaf", **params: Any) -> Any:
        return self.run("activations", inputs_or_data, sites=sites, **params)

    def anomalies(self, inputs: Any, **params: Any) -> Any:
        return self.run("anomalies", inputs, **params)

    def gradients(self, inputs: Any, objective: Any = None, **params: Any) -> Any:
        return self.run("gradients", inputs, objective=objective, **params)

    def attribute(self, inputs: Any, target: Any = "predicted",
                  method: str = "integrated_gradients", at: Any = None, **params: Any) -> Any:
        return self.run("attribution", inputs, target=target, method=method, at=at, **params)

    def attention(self, inputs: Any, sites: Any = None, heads: Any = None, queries: Any = None,
                  **params: Any) -> Any:
        return self.run("attention", inputs, sites=sites, heads=heads, queries=queries, **params)

    def ablation_scan(self, inputs: Any, sites: Any = None, metric: Any = None,
                      value: Any = "mean", reference: Any = None, **params: Any) -> Any:
        return self.run("interventions.ablation_scan", inputs, sites=sites, metric=metric,
                        value=value, reference=reference, **params)

    def patching(self, clean: Any, corrupt: Any, sites: Any = None, metric: Any = None,
                 positions: Any = None, **params: Any) -> Any:
        return self.run("interventions.patching", clean, corrupt=corrupt, sites=sites,
                        metric=metric, positions=positions, **params)

    def causal_tracing(self, inputs: Any, **params: Any) -> Any:
        return self.run("interventions.causal_tracing", inputs, **params)

    def steer(self, inputs: Any, site: str, vector: Any, alpha: float = 1.0, **params: Any) -> Any:
        return self.run("interventions.steer", inputs, site=site, vector=vector, alpha=alpha,
                        **params)

    def similarity(self, data: Any, sites: Any = None, method: str = "cka_minibatch",
                   pool: str = "mean", **params: Any) -> Any:
        return self.run("similarity", data, sites=sites, method=method, pool=pool, **params)

    def geometry(self, data: Any, sites: Any = None, **params: Any) -> Any:
        return self.run("geometry", data, sites=sites, **params)

    def top_examples(self, data: Any, site: str, units: Any = None, k: int = 20,
                     **params: Any) -> Any:
        return self.run("neurons.top_examples", data, site=site, units=units, k=k, **params)

    def selectivity(self, data: Any, site: str, labels: Any = None, **params: Any) -> Any:
        return self.run("neurons.selectivity", data, site=site, labels=labels, **params)

    def probe(self, data: Any, labels: Any = None, site: str = "", **params: Any) -> Any:
        return self.run("probing.probe", data, labels=labels, site=site, **params)

    def tcav(self, inputs: Any, *, concept: Any, random: Any, site: str, **params: Any) -> Any:
        return self.run("probing.tcav", inputs, concept=concept, random=random, site=site,
                        **params)

    def generation(self, prompt: Any, max_new_tokens: int = 20, sites: Any = None,
                   **params: Any) -> Any:
        return self.run("process.generation", prompt, max_new_tokens=max_new_tokens, sites=sites,
                        **params)

    def diffusion(self, **params: Any) -> Any:
        return self.run("process.diffusion", None, **params)

    def process(self, fn: Any, step_module: str | None = None, **params: Any) -> Any:
        return self.run("process.loop", fn, step_module=step_module, **params)

    def lens(self, inputs: Any, kind: str = "logit", **params: Any) -> Any:
        return self.run("lens", inputs, kind=kind, **params)

    def maximize(self, objective: Any, init: Any = None, steps: int = 100, **params: Any) -> Any:
        """Activation maximisation (experimental)."""
        return self.run("maximization", params.pop("inputs", None), objective=objective,
                        init=init, steps=steps, **params)

    def dictionary(self, data: Any, site: str, **params: Any) -> Any:
        """Sparse autoencoder over a site's activations (experimental)."""
        return self.run("dictionary", data, site=site, **params)

    def weights(self, params: str = "*", **options: Any) -> Any:
        return self.run("weights", None, params=params, **options)

    def report(self, inputs: Any = None, data: Any = None, preset: str = "quick", *,
               include: Any = None, exclude: Any = None, objective: Any = None,
               strict: bool = False, **params: Any) -> Any:
        """Evidence-backed diagnostic report; techniques that cannot run are listed, not faked."""
        from topologies.introspection.report import build_report

        return build_report(self, inputs, data=data, preset=preset, include=include,
                            exclude=exclude, objective=objective, strict=strict, **params)

    def __repr__(self) -> str:
        return (f"Inspector({type(self.model).__name__}, adapter="
                f"{getattr(self.adapter, 'name', None)!r}, device={self.subject.device})")


def _objective(objective: Any) -> Any:
    if objective is None:
        return None
    from topologies.introspection.objectives import as_objective

    return as_objective(objective)


def trace(model: nn.Module, inputs: Any, sites: Any = "*", **policy: Any) -> Trace:
    """Functional shortcut: ``introspect(model).trace(inputs, sites, **policy)``."""
    return introspect(model).trace(inputs, sites, **policy)


__all__ = ["Axes", "Inspector", "introspect", "trace"]


def _unavailable(message: str) -> Exception:
    from topologies.introspection.errors import IntrospectionError

    return IntrospectionError(message)
