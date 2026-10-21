"""Bridges into the analysis layer, the Trainer and experiments."""
from __future__ import annotations

from typing import Any

from topologies.training.callbacks import Callback


class Introspect:
    """``Analysis`` object: ``analyze(model, analysis=Introspect("activations", inputs=x))``.

    ``data`` may be an ``nn.Module``, an ``Inspector`` or anything ``introspect`` accepts.
    Inputs come from the constructor or from ``context.parameters["inputs"]``.
    """

    def __init__(self, technique: str, inputs: Any = None, **params: Any):
        self.technique = technique
        self.inputs = inputs
        self.params = params
        self.name = f"introspect:{technique}"

    def run(self, data: Any, context: Any) -> Any:
        from topologies.introspection.api import Inspector, introspect

        inspector = data if isinstance(data, Inspector) else introspect(data)
        parameters = dict(getattr(context, "parameters", {}) or {})
        inputs = self.inputs if self.inputs is not None else parameters.pop("inputs", None)
        parameters.pop("inputs", None)
        return inspector.run(self.technique, inputs, **{**parameters, **self.params})


class IntrospectionCallback(Callback):
    """Trainer callback: cheap introspection every ``every_n_steps`` steps.

    It traces summaries (RMS, zero fraction, non-finite fraction) at ``sites`` on a fixed
    ``probe_batch`` and, with ``gradients=True``, reads per-module gradient norms from
    ``p.grad`` *after* the backward pass, without an extra backward. Scalars are added to
    the step metrics as ``introspection/<site>/<stat>``, so they reach every live visualizer
    (the Matplotlib live sink plots them) and the ``trainer.history`` epoch averages
    (``train_introspection/...``). Between probes, the last measured values are carried
    forward, so an epoch average is the mean over steps of the most recent probe. The
    exact probe rows are kept in ``callback.history``. Failures never break training: the trainer's
    callback list swallows them. Overhead is bounded by ``every_n_steps`` and ``max_sites``.
    """

    def __init__(self, probe_batch: Any = None, *, sites: Any = "role:block", every_n_steps: int = 100,
                 gradients: bool = False, stats: tuple[str, ...] = ("rms", "zero_frac",
                                                                     "nonfinite_frac"),
                 max_sites: int = 32, prefix: str = "introspection"):
        if every_n_steps < 1:
            raise ValueError("every_n_steps must be >= 1")
        self.probe_batch = probe_batch
        self.sites = sites
        self.every_n_steps = every_n_steps
        self.gradients = gradients
        self.stats = tuple(stats)
        self.max_sites = max_sites
        self.prefix = prefix
        self.history: list[dict[str, Any]] = []
        self._inspector: Any = None
        self._paths: list[str] | None = None
        self._last: dict[str, float] = {}

    def _setup(self, trainer: Any) -> None:
        from topologies.introspection.api import Inspector
        from topologies.introspection.subject import Subject

        if self._inspector is None or self._inspector.subject.source_model is not trainer.model:
            subject = Subject.resolve(trainer)
            self._inspector = Inspector(subject)
            sites = self.sites
            index = subject.index
            if sites == "role:block" and not index.resolve_modules("role:block", allow_empty=True):
                sites = [p for p in index.paths() if p and "." not in p]
            self._paths = index.resolve_modules(sites)[: self.max_sites]

    def on_step_end(self, trainer: Any, step: int, metrics: dict) -> None:
        if step % self.every_n_steps:
            metrics.update(self._last)
            return
        self._setup(trainer)
        row: dict[str, Any] = {"step": step}
        if self.probe_batch is not None:
            trace = self._inspector.trace(self.probe_batch, sites=self._paths, keep="summary",
                                          call_tree=False)
            for record in trace.records.values():
                summary = record.summary or {}
                if record.occurrence:
                    continue
                values = dict(summary)
                mean, std = summary.get("mean"), summary.get("std")
                if isinstance(mean, float) and isinstance(std, float):
                    values["rms"] = (mean ** 2 + std ** 2) ** 0.5
                for stat in self.stats:
                    if isinstance(values.get(stat), (int, float)):
                        row[f"{self.prefix}/{record.site}/{stat}"] = float(values[stat])
        if self.gradients:
            model = self._inspector.model
            for path in self._paths or []:
                module = model.get_submodule(path) if path else model
                grads = [p.grad for p in module.parameters() if p.grad is not None]
                if grads:
                    norm = sum(float(g.detach().float().norm()) ** 2 for g in grads) ** 0.5
                    row[f"{self.prefix}/{path or '<root>'}/grad_norm"] = norm
        self.history.append(row)
        self._last = {k: v for k, v in row.items() if k != "step"}
        metrics.update(self._last)


__all__ = ["Introspect", "IntrospectionCallback"]
