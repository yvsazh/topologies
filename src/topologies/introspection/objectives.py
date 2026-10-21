"""Scalar objectives for gradient-based techniques.

An objective turns the model output (and the batch) into one scalar. Batch-wise
objectives sum per-example values, so gradients stay per example for models
that are independent across the batch.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from topologies.introspection import _tree
from topologies.introspection.errors import ObjectiveRequired


class Objective:
    """Base class. ``compute(session, batch, output) -> scalar tensor``."""

    name = "objective"
    parameter_grads = False

    def compute(self, session: Any, batch: Any, output: Any) -> torch.Tensor:
        raise NotImplementedError

    def per_example(self, session: Any, batch: Any, output: Any) -> torch.Tensor | None:
        """Per-example values ``[B]`` when the objective is a sum over the batch."""
        return None

    def install(self, session: Any, hooks: Any) -> None:
        """Objectives that read internal sites register handlers here."""

    def describe(self) -> str:
        return repr(self)


def primary(output: Any, path: str | None = None) -> torch.Tensor:
    value = _tree.get_path(output, path) if path else output
    if isinstance(value, torch.Tensor):
        return value
    found = _tree.first_tensor(value)
    if found is None:
        raise ObjectiveRequired("the model output has no tensor; pass a custom objective")
    return found[1]


def _select(out: torch.Tensor, index: int | None, position: int | None) -> torch.Tensor:
    """``[B, C]`` -> ``out[:, index]``; ``[B, T, V]`` -> ``out[:, position, index]``."""
    if out.ndim >= 3 and position is not None:
        out = out.select(1, position)
    if index is not None:
        out = out[..., index]
    return out.reshape(out.shape[0], -1).sum(-1) if out.ndim > 1 else out


@dataclass
class Output(Objective):
    """Sum (or mean) of the primary output, optionally of one output index."""

    index: int | None = None
    position: int | None = None
    reduce: str = "sum"
    path: str | None = None
    name = "output"

    def per_example(self, session: Any, batch: Any, output: Any) -> torch.Tensor:
        out = primary(output, self.path)
        if self.index is not None or self.position is not None:
            return _select(out, self.index, self.position)
        return out.reshape(out.shape[0], -1).sum(-1) if out.ndim else out.reshape(1)

    def compute(self, session: Any, batch: Any, output: Any) -> torch.Tensor:
        values = self.per_example(session, batch, output)
        return values.mean() if self.reduce == "mean" else values.sum()


@dataclass
class Logit(Objective):
    """One class/token logit of the primary output (``position`` for ``[B, T, V]`` outputs)."""

    index: int
    position: int | None = -1
    path: str | None = None
    name = "logit"

    def per_example(self, session: Any, batch: Any, output: Any) -> torch.Tensor:
        return _select(primary(output, self.path), self.index, self.position)

    def compute(self, session: Any, batch: Any, output: Any) -> torch.Tensor:
        return self.per_example(session, batch, output).sum()


@dataclass
class Predicted(Objective):
    """The logit of each example's own arg-max prediction (fixed on the first evaluation)."""

    position: int | None = -1
    path: str | None = None
    name = "predicted"

    def __post_init__(self) -> None:
        self.classes: torch.Tensor | None = None

    def per_example(self, session: Any, batch: Any, output: Any) -> torch.Tensor:
        out = primary(output, self.path)
        if out.ndim >= 3 and self.position is not None:
            out = out.select(1, self.position)
        out = out.reshape(out.shape[0], -1)
        if self.classes is None or self.classes.shape[0] != out.shape[0]:
            self.classes = out.detach().argmax(-1)
        return out.gather(1, self.classes.to(out.device)[:, None]).squeeze(1)

    def compute(self, session: Any, batch: Any, output: Any) -> torch.Tensor:
        return self.per_example(session, batch, output).sum()


@dataclass
class LogitDiff(Objective):
    """``logit[a] - logit[b]`` (mean over the batch): the standard patching metric."""

    a: int
    b: int
    position: int | None = -1
    path: str | None = None
    name = "logit_diff"

    def per_example(self, session: Any, batch: Any, output: Any) -> torch.Tensor:
        out = primary(output, self.path)
        return _select(out, self.a, self.position) - _select(out, self.b, self.position)

    def compute(self, session: Any, batch: Any, output: Any) -> torch.Tensor:
        return self.per_example(session, batch, output).mean()

    def value(self, output: Any) -> torch.Tensor:
        return self.per_example(None, None, output).mean()


@dataclass
class Loss(Objective):
    """The recipe loss (``recipe.batch_to_loss``), or ``fn(output, target)`` with dataset targets."""

    fn: Callable[..., torch.Tensor] | None = None
    name = "loss"
    parameter_grads = True

    def run(self, session: Any, batch: Any) -> Any:
        """Loss objectives own the forward: the recipe computes output and loss together."""
        subject = session.subject
        if self.fn is None and subject.recipe is not None and batch.batch is not None:
            loss, metrics = subject.recipe.batch_to_loss(subject.model, batch.batch,
                                                         subject.recipe_cfg(), None, None)
            self._loss = loss
            self.metrics = {k: float(v) for k, v in metrics.items()}
            return loss
        self._loss = None
        return batch.run(subject, subject.model)

    def compute(self, session: Any, batch: Any, output: Any) -> torch.Tensor:
        if getattr(self, "_loss", None) is not None:
            return self._loss
        target = batch.meta.get("target")
        if target is None:
            targets = batch.meta.get("targets") or {}
            target = next(iter(targets.values()), None)
        if self.fn is None or target is None:
            raise ObjectiveRequired(
                "Loss() needs a recipe batch with targets, or Loss(fn) plus (input, target) data; "
                "otherwise pass objective=Output() / Logit(i) / a callable(output) -> scalar")
        return self.fn(primary(output), target)


@dataclass
class SiteNorm(Objective):
    """Norm of the activation at one module output (activation maximisation, probes)."""

    site: str
    ord: float = 2.0
    unit: int | None = None
    name = "site_norm"

    def install(self, session: Any, hooks: Any) -> None:
        path = session.subject.index.canonical(self.site)
        self._value = None

        def post(frame: Any, module: Any, args: tuple, kwargs: dict, output: Any) -> None:
            if frame.path == path and not frame.recompute:
                self._value = primary(output)
        hooks.post_handlers.append(post)

    def paths(self, index: Any) -> list[str]:
        return [index.canonical(self.site)]

    def compute(self, session: Any, batch: Any, output: Any) -> torch.Tensor:
        value = self._value
        if value is None:
            raise ObjectiveRequired(f"site {self.site!r} did not run")
        if self.unit is not None:
            axes = session.subject.declared_axes(session.subject.index.canonical(self.site))
            axis = axes.feature_axis if axes is not None and axes.feature_axis is not None else -1
            value = value.select(axis, self.unit)
        return torch.linalg.vector_norm(value.float(), ord=self.ord)


@dataclass
class Custom(Objective):
    fn: Callable[..., torch.Tensor]
    name = "custom"

    def compute(self, session: Any, batch: Any, output: Any) -> torch.Tensor:
        try:
            return self.fn(output, batch)
        except TypeError:
            return self.fn(output)


def as_objective(value: Any) -> Objective:
    """Accept an Objective, a callable, an int (logit), ``"loss"``, ``"output"``, ``"predicted"``
    or ``("token", position, token_id)``."""
    if isinstance(value, Objective):
        return value
    if isinstance(value, bool):
        raise TypeError("objective cannot be a bool")
    if isinstance(value, int):
        return Logit(value)
    if isinstance(value, str):
        named = {"loss": Loss, "output": Output, "predicted": Predicted}
        if value not in named:
            raise ValueError(f"unknown objective {value!r}; use {sorted(named)}")
        return named[value]()
    if isinstance(value, tuple) and len(value) == 3 and value[0] == "token":
        return Logit(int(value[2]), position=int(value[1]))
    if callable(value):
        return Custom(value)
    raise TypeError(f"cannot build an objective from {type(value).__name__}")


def target_objective(target: Any) -> Objective:
    """Attribution ``target=``: int class, ``"predicted"``, ``("token", pos, id)`` or an Objective."""
    if target is None or target == "predicted":
        return Predicted()
    return as_objective(target)


def default_objective(subject: Any, inputs: Any) -> Objective:
    """``Loss()`` when the subject has a recipe and the batch has targets; else raise."""
    from topologies.introspection.inputs import normalize_inputs

    if subject.recipe is not None:
        batch = next(iter(normalize_inputs(subject, inputs)))
        if batch.batch is not None and any(k in batch.batch for k in subject.recipe.target_keys()):
            return Loss()
    raise ObjectiveRequired(
        "no scalar objective could be resolved: pass objective=Loss() with a recipe batch that "
        "has targets, objective=Output(), objective=Logit(class_id), LogitDiff(a, b), or any "
        "callable(output) -> scalar")


__all__ = ["Custom", "Logit", "LogitDiff", "Loss", "Objective", "Output", "Predicted", "SiteNorm",
           "as_objective", "default_objective", "primary", "target_objective"]
