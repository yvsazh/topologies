"""Session: the only place that executes the model under instrumentation.

Sequence (§4.3): resolve sites and policy → pre-flight budget → enter
``model_execution`` → register hooks (+ function mode when requested) → forward
→ optional objective + backward → ``finally`` remove every handle and restore
flags. Sessions never nest on the same model.
"""
from __future__ import annotations

import contextlib
import os
import weakref
from collections.abc import Callable, Iterator
from typing import Any

import torch
import torch.nn as nn

from topologies._execution import model_execution
from topologies.introspection import _tree
from topologies.introspection.capture import CapturePolicy, Recorder, default_axes_for, summary_dict
from topologies.introspection.errors import IntrospectionError
from topologies.introspection.hooks import HookManager, assert_no_leaked_hooks, hook_counts
from topologies.introspection.inputs import Batch, normalize_inputs
from topologies.introspection.sites import SiteKey, SiteRequest, resolve_sites
from topologies.introspection.trace import SiteRecord, Trace

_ACTIVE: dict[int, Session] = {}


def _debug() -> bool:
    return os.environ.get("TOPOLOGIES_INTROSPECTION_DEBUG", "") not in ("", "0", "false")


class Session:
    """Compose hooks, recorder, interventions, functional capture and an objective."""

    def __init__(self, subject: Any, *, sites: Any = "*", policy: CapturePolicy | None = None,
                 objective: Any = None, interventions: tuple = (), ops: Any = None,
                 functional: Any = None, call_tree: bool = True,
                 recorder_factory: Callable[..., Any] | None = None,
                 grad_inputs: bool = False, parameter_grads: bool = False):
        self.subject = subject
        self.model: nn.Module = subject.model
        self.policy = policy or CapturePolicy()
        self.objective = objective
        self.interventions = tuple(interventions)
        self.ops = ops
        self.functional = functional
        self.call_tree = call_tree
        self.recorder_factory = recorder_factory
        self.grad_inputs = grad_inputs
        self.parameter_grads = parameter_grads
        self.param_grads: dict[str, torch.Tensor | None] = {}
        self.input_grads: list[torch.Tensor | None] = []
        self.grad_input_tensors: list[torch.Tensor] = []
        index = subject.index
        self.requests, self.params = resolve_sites(sites, index, default_ports=self.policy.ports,
                                                   allow_empty=sites in (None, (), []))
        self.warnings: list[str] = list(getattr(subject, "warnings", []))
        self.assumptions: list[str] = []
        module_paths = list(dict.fromkeys(r.path for r in self.requests if r.op is None))
        if len(module_paths) > self.policy.max_sites:
            self.warnings.append(f"{len(module_paths)} sites selected (max_sites="
                                 f"{self.policy.max_sites}); narrow sites=... to reduce overhead")
        intervention_paths = [p for spec in self.interventions for p in spec.paths(index)]
        if objective is not None and hasattr(objective, "paths"):
            intervention_paths += list(objective.paths(index))
        if functional is None and (ops or any(r.op is not None for r in self.requests)):
            from topologies.introspection.functional import FunctionalInterceptor

            functional = FunctionalInterceptor(self, ops=ops)
        self.functional = functional
        needs_stack = functional is not None and functional.needs_mode
        mha_paths = list(getattr(functional, "mha_paths", []))
        tracked = list(dict.fromkeys([*module_paths, *intervention_paths, *mha_paths]))
        everything = index.paths()
        if call_tree or (needs_stack and not functional.scoped):
            global_mode = len(tracked) < len(everything)
            tracked = everything
        elif needs_stack:
            # Scoped function mode: frames are only needed inside the requested subtrees.
            scope = functional.scope_paths
            tracked = list(dict.fromkeys([*tracked, *(p for p in everything if any(
                p == s or not s or p.startswith(s + ".") for s in scope))]))
            global_mode = False
        else:
            global_mode = False
        kwarg_paths = {r.path for r in self.requests if r.op is None and r.port.startswith(
            ("input.", "grad_input.")) and not r.port.split(".")[1].isdigit()}
        kwarg_paths |= {p for spec in self.interventions for p in spec.kwarg_paths(index)}
        kwarg_paths |= set(mha_paths)
        protect = []
        if self.policy.keep == "tensors" and self.policy.clone == "auto":
            protect = [p for p in everything if getattr(index.module(p), "inplace", False) is True]
        self.hooks = HookManager(index, tracked=tracked, kwarg_paths=kwarg_paths,
                                 protect_paths=protect, call_tree=call_tree,
                                 timing=self.policy.timing, global_mode=global_mode)
        self.recorder: Any = None
        self._persistent = False
        self._grad_flags: list[tuple[torch.Tensor, bool]] = []

    # -- properties -----------------------------------------------------------------------

    @property
    def needs_grad(self) -> bool:
        return (self.objective is not None or self.policy.grads or self.grad_inputs
                or any(r.port.startswith("grad_") for r in self.requests)
                or any(getattr(s, "needs_grad", False) for s in self.interventions))

    def _exec_mode(self) -> tuple[bool, bool]:
        """(inference_mode, no_grad) for forward-only runs.

        ``inference_mode`` is used unless tensors are kept: in-place modification checks
        need version counters, which inference tensors do not have.
        """
        if self.needs_grad:
            return False, False
        if self.policy.keep == "tensors":
            return False, True
        return True, False

    # -- lifecycle ---------------------------------------------------------------------------

    def _new_recorder(self, batch: Batch | None) -> Any:
        subject = self.subject
        size = batch.size if batch is not None else None
        axes_for = default_axes_for(subject.declared_axes, size)
        if self.recorder_factory is not None:
            return self.recorder_factory(self, batch)
        recorder = Recorder(self.policy, self.requests, axes_for=axes_for, batch_size=size,
                            store=getattr(self, "store_factory", lambda: None)())
        return recorder

    def _install_handlers(self) -> None:
        hooks = self.hooks
        hooks.pre_handlers.clear()
        hooks.post_handlers.clear()
        if self.functional is not None:
            self.functional.install(self, hooks)
        for spec in self.interventions:
            spec.install(self, hooks)
        if self.objective is not None:
            self.objective.install(self, hooks)
        recorder = self.recorder

        def pre(frame: Any, module: nn.Module, args: tuple, kwargs: dict) -> None:
            if not frame.recompute and recorder.wants(frame.path):
                recorder.on_pre(frame.path, frame.occurrence, module, args, kwargs)

        def post(frame: Any, module: nn.Module, args: tuple, kwargs: dict, output: Any) -> Any:
            if not frame.recompute and recorder.wants(frame.path):
                return recorder.on_post(frame.path, frame.occurrence, module, output)
            return None
        hooks.pre_handlers.append(pre)
        hooks.post_handlers.append(post)
        hooks.protect = getattr(recorder, "protect", None)

    def _enter_guard(self) -> None:
        key = id(self.model)
        if key in _ACTIVE and _ACTIVE[key] is not self:
            raise IntrospectionError(
                f"an introspection session is already running on this {type(self.model).__name__}; "
                "sessions may not nest on the same model (nest on different models instead)")
        _ACTIVE[key] = self

    def _exit_guard(self) -> None:
        if _ACTIVE.get(id(self.model)) is self:
            del _ACTIVE[id(self.model)]

    @contextlib.contextmanager
    def persistent(self) -> Iterator[Session]:
        """Register hooks once for many ``run`` calls (dataset-scale collection)."""
        self._enter_guard()
        baseline = hook_counts(self.model) if _debug() else None
        self._persistent = True
        try:
            self.hooks.attach()
            yield self
        finally:
            self._persistent = False
            self.hooks.detach()
            self._exit_guard()
            if baseline is not None:
                assert_no_leaked_hooks(self.model, baseline=baseline)

    @contextlib.contextmanager
    def open(self) -> Iterator[Session]:
        """Install hooks and handlers for raw model calls (``Inspector.intervene``)."""
        self._enter_guard()
        self.hooks.reset()
        self.recorder = _NullRecorder()
        self._install_handlers()
        try:
            self.hooks.attach()
            yield self
        finally:
            self.hooks.detach()
            for spec in self.interventions:
                spec.cleanup(self)
            self._exit_guard()

    def run(self, inputs: Any) -> Trace:
        """Execute one invocation (tensor/str/dict/ModelCall/callable/Batch) and return a Trace."""
        batches = list(normalize_inputs(self.subject, inputs))
        if len(batches) != 1:
            if all(b.call is not None for b in batches) and batches:
                raise IntrospectionError("Session.run takes one invocation; use insp.collect(...) "
                                         "for iterables of batches")
            raise IntrospectionError("Session.run takes exactly one invocation")
        return self.run_batch(batches[0])

    def run_batch(self, batch: Batch) -> Trace:
        if not self._persistent:
            self._enter_guard()
        baseline = hook_counts(self.model) if _debug() and not self._persistent else None
        self.hooks.reset()
        self.recorder = self._new_recorder(batch)
        self._install_handlers()
        inference, no_grad = self._exec_mode()
        output = None
        scalar = None
        try:
            with model_execution(self.model, training=self.policy.training, inference=inference,
                                 preserve_rng=True, seed=self.policy.seed), \
                    torch.no_grad() if no_grad else contextlib.nullcontext():
                if not self._persistent:
                    self.hooks.attach()
                functional = self.functional.enter(self) if self.functional is not None else None
                try:
                    if self.needs_grad:
                        self.recorder.grad_enabled = True
                        batch = self._prepare_grad_inputs(batch)
                        self._ensure_graph()
                    runner = getattr(self.objective, "run", None)
                    output = runner(self, batch) if callable(runner) else \
                        batch.run(self.subject, self.model)
                    if self.objective is not None:
                        scalar = self._backward(batch, output)
                finally:
                    if functional is not None:
                        self.functional.exit(self)
        finally:
            if not self._persistent:
                self.hooks.detach()
                self._exit_guard()
            self._restore_grad_flags()
            for spec in self.interventions:
                spec.cleanup(self)
            if hasattr(self.recorder, "remove_grad_hooks"):
                self.recorder.remove_grad_hooks()
            if baseline is not None:
                assert_no_leaked_hooks(self.model, baseline=baseline)
        return self._finish(batch, output, scalar)

    # -- gradients ----------------------------------------------------------------------------

    def _prepare_grad_inputs(self, batch: Batch) -> Batch:
        self.grad_input_tensors = []
        if not self.grad_inputs or batch.call is None:
            return batch

        def mark(t: torch.Tensor) -> torch.Tensor:
            return t.detach().clone().requires_grad_(True) if t.is_floating_point() else t
        args = _tree.tree_map(mark, batch.call.args)
        kwargs = _tree.tree_map(mark, dict(batch.call.kwargs))
        self.grad_input_tensors = [t for _, t in _tree.leaves((args, kwargs)) if t.requires_grad]
        return batch.with_tensors(args, kwargs)

    def _ensure_graph(self) -> None:
        """A frozen model has no autograd graph: enable ``requires_grad`` temporarily."""
        params = [p for p in self.model.parameters() if p.is_floating_point()]
        if self.grad_input_tensors or any(p.requires_grad for p in params) or not params:
            return
        for p in params:
            self._grad_flags.append((p, p.requires_grad))
            p.requires_grad_(True)
        note = ("parameters were frozen; requires_grad was enabled for this run only (restored "
                "afterwards) so gradients could flow")
        if note not in self.assumptions:
            self.assumptions.append(note)

    def _backward(self, batch: Batch, output: Any) -> torch.Tensor:
        """Gradients via ``torch.autograd.grad`` (no ``.grad`` accumulation on parameters)."""
        scalar = self.objective.compute(self, batch, output)
        if not isinstance(scalar, torch.Tensor) or scalar.numel() != 1:
            raise IntrospectionError("objective must return a scalar tensor")
        scalar = scalar.reshape(())
        if not scalar.requires_grad:
            return scalar.detach()
        named = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        if self.params:
            named = [(n, p) for n, p in named if n in self.params]
        want_params = self.parameter_grads or bool(self.params) or \
            getattr(self.objective, "parameter_grads", False)
        params = [p for _, p in named] if want_params else []
        inputs = list(self.grad_input_tensors)
        watched = list(getattr(self.recorder, "watched", []))
        targets = inputs + params + watched
        self.hooks.phase = "backward"
        try:
            if targets:
                grads = torch.autograd.grad(scalar, targets, allow_unused=True)
                self.input_grads = list(grads[:len(inputs)])
                if want_params:
                    self.param_grads = {n: g for (n, _), g in zip(
                        named, grads[len(inputs):len(inputs) + len(params)])}
        finally:
            self.hooks.phase = "forward"
        return scalar.detach()

    def _restore_grad_flags(self) -> None:
        for tensor, flag in self._grad_flags:
            tensor.requires_grad_(flag)
        self._grad_flags.clear()

    # -- results --------------------------------------------------------------------------------

    def _finish(self, batch: Batch, output: Any, scalar: Any) -> Trace:
        recorder = self.recorder
        if not hasattr(recorder, "finalize"):
            return Trace(output, self.hooks.call_tree(), {}, self.policy)
        pending, warnings = recorder.finalize()
        store = recorder.store
        order = {}
        for call in self.hooks.calls:
            order.setdefault((call.path, call.occurrence), call.index)
        vectors = [p.summary for p in pending.values() if p.summary is not None]
        stacked = torch.stack([v.to("cpu") if v.device.type != "cpu" else v for v in vectors]) \
            if vectors else None
        records: dict[SiteKey, SiteRecord] = {}
        cursor = 0
        for key, p in pending.items():
            summary = None
            if p.summary is not None and stacked is not None:
                summary = summary_dict(stacked[cursor])
                cursor += 1
            value = store.tensors.get(key) if p.has_value else None
            loader = None
            if value is None and p.has_value and hasattr(store, "loader"):
                loader = store.loader(key)
            records[key] = SiteRecord(key.site, key.port, key.occurrence, p.shape, p.dtype,
                                      p.device, p.axes, value, summary, None, key,
                                      {**p.meta, "order": order.get((key.path, key.occurrence), 0)},
                                      loader)
        store_device = self.policy.store_device
        detached = _tree.tree_map(
            lambda t: t.detach() if store_device == "same" else t.detach().to(store_device), output)
        meta = dict(batch.meta)
        if scalar is not None:
            meta["objective"] = float(scalar)
        assumptions = [*self.assumptions, *getattr(recorder, "assumptions", []),
                       *meta.pop("assumptions", [])]
        if self.policy.keep == "tensors" and not self.needs_grad:
            assumptions.append("forward ran under no_grad (not inference_mode) so in-place "
                               "modification of kept tensors can be detected")
        provenance = {"subject": self.subject.describe(), "policy": self.policy.to_dict(),
                      "requests": len(self.requests),
                      "inputs": _tree.describe(batch.tensors()) if batch.tensors() is not None
                      else "callable"}
        trace = Trace(detached, self.hooks.call_tree(), records, self.policy,
                      warnings=[*self.warnings, *warnings],
                      assumptions=list(dict.fromkeys(assumptions)), metadata=meta,
                      provenance=provenance)
        trace._model_ref = weakref.ref(self.model)
        return trace


class _NullRecorder:
    warnings: list[str] = []
    assumptions: list[str] = []
    grad_enabled = False
    store = None
    watched: list[torch.Tensor] = []

    def wants(self, path: str) -> bool:
        return False

    def protect(self, args: tuple) -> None:
        return

    def remove_grad_hooks(self) -> None:
        return


__all__ = ["Session", "SiteRequest"]
