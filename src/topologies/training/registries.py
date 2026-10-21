"""Optimizer and scheduler registries (+ linear warmup that composes with any schedule).

Built-ins are registered through the same mechanism users get:

    from topologies.training import register_optimizer, register_scheduler

    register_optimizer("adagrad", lambda params, lr, weight_decay=0.0, **kw:
                       torch.optim.Adagrad(params, lr=lr, weight_decay=weight_decay))
    register_scheduler("halve", lambda opt, total_steps, **kw:
                       torch.optim.lr_scheduler.StepLR(opt, max(1, total_steps // 2), 0.5))

    Trainer(model, train, optimizer="adagrad", scheduler="halve")

Optimizer factories are called as ``factory(params, **kwargs)``: ``lr`` and
``weight_decay`` always, plus every ``OptimizerConfig`` field the user set
explicitly and ``OptimizerConfig.kwargs``; keyword arguments the factory's
signature does not accept are dropped (unless it takes ``**kwargs``).
Scheduler factories are called as ``factory(optimizer, total_steps, **kwargs)``.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable
from typing import Any

import torch

from topologies.training.errors import TrainingError

OptimizerFactory = Callable[..., torch.optim.Optimizer]
SchedulerFactory = Callable[..., Any]

_OPTIMIZERS: dict[str, OptimizerFactory] = {}
_SCHEDULERS: dict[str, SchedulerFactory] = {}
#: Names whose factories receive every config field (exact legacy calls).
_BUILTIN_OPTIMIZERS: set[str] = set()


def _norm(name: str) -> str:
    return str(name).strip().lower()


# ---------------------------------------------------------------- optimizers


def register_optimizer(name: str, factory: OptimizerFactory, *, overwrite: bool = True) -> None:
    """Make ``optimizer=name`` (and ``OptimizerConfig(name=...)``) build ``factory``."""
    key = _norm(name)
    if not key:
        raise ValueError("optimizer name must be non-empty.")
    if not callable(factory):
        raise TypeError(f"optimizer factory for {name!r} must be callable.")
    if key in _OPTIMIZERS and not overwrite:
        raise ValueError(f"optimizer {name!r} is already registered.")
    _OPTIMIZERS[key] = factory


def list_optimizers() -> list[str]:
    return sorted(_OPTIMIZERS)


def get_optimizer_factory(name: str) -> OptimizerFactory:
    key = _norm(name)
    if key not in _OPTIMIZERS:
        raise TrainingError(f"Unknown optimizer {name!r}.", stage="optimizer",
                            available=list_optimizers())
    return _OPTIMIZERS[key]


def is_builtin_optimizer(name: str) -> bool:
    return _norm(name) in _BUILTIN_OPTIMIZERS


def accepted_kwargs(fn: Any, kwargs: dict[str, Any], *, skip: int = 0) -> dict[str, Any]:
    """``kwargs`` filtered to what ``fn`` accepts (all of them if it takes ``**kw``).

    ``skip`` leading positional parameters (``params``/``optimizer``...) are
    never filled from ``kwargs``.
    """
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(p.kind is p.VAR_KEYWORD for p in params):
        return dict(kwargs)
    names = {p.name for p in params[skip:]
             if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    return {k: v for k, v in kwargs.items() if k in names}


def _adamw(params, *, lr, weight_decay=0.0, eps=1e-8, betas=None,
           momentum=None, **kwargs):
    kw: dict[str, Any] = {"lr": lr, "weight_decay": weight_decay, "eps": eps}
    if betas is not None:
        kw["betas"] = betas
    return torch.optim.AdamW(params, **kw, **kwargs)


def _adam(params, *, lr, weight_decay=0.0, eps=1e-8, betas=None,
          momentum=None, **kwargs):
    kw: dict[str, Any] = {"lr": lr, "weight_decay": weight_decay, "eps": eps}
    if betas is not None:
        kw["betas"] = betas
    return torch.optim.Adam(params, **kw, **kwargs)


def _sgd(params, *, lr, weight_decay=0.0, momentum=0.9, **kwargs):
    return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, **kwargs)


def _rmsprop(params, *, lr, weight_decay=0.0, momentum=0.9, eps=1e-8, **_):
    return torch.optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay,
                               eps=eps)


def _lion(params, *, lr, weight_decay=0.0, **_):
    try:
        from torch.optim import Lion as _Lion  # type: ignore[attr-defined]
    except Exception as e:
        raise TrainingError(
            "Optimizer 'lion' is not available in this PyTorch build.",
            stage="optimizer",
            optimizer_name="lion",
            hint="Use optimizer='adamw' instead, upgrade PyTorch, or register your own "
                 "implementation with register_optimizer('lion', factory).",
            available=[n for n in list_optimizers() if n != "lion"],
        ) from e
    return _Lion(params, lr=lr, weight_decay=weight_decay)


for _name, _fn in (("adamw", _adamw), ("adam", _adam), ("sgd", _sgd),
                   ("rmsprop", _rmsprop)):
    register_optimizer(_name, _fn)

from topologies.implementations.optimizers.adabelief import adabelief as _adabelief
from topologies.implementations.optimizers.adafactor import adafactor as _adafactor
from topologies.implementations.optimizers.adahessian import adahessian as _adahessian
from topologies.implementations.optimizers.adam8bit import adam8bit as _adam8bit
from topologies.implementations.optimizers.adam_mini import adam_mini as _adam_mini
from topologies.implementations.optimizers.ademamix import ademamix as _ademamix
from topologies.implementations.optimizers.asam import asam as _asam
from topologies.implementations.optimizers.ema import ema as _ema
from topologies.implementations.optimizers.fadam import fadam as _fadam
from topologies.implementations.optimizers.fisher_sam import fisher_sam as _fisher_sam
from topologies.implementations.optimizers.flora import flora as _flora
from topologies.implementations.optimizers.galore import galore as _galore
from topologies.implementations.optimizers.grokfast import grokfast as _grokfast
from topologies.implementations.optimizers.gsam import gsam as _gsam
from topologies.implementations.optimizers.kfac import kfac as _kfac
from topologies.implementations.optimizers.lamb import lamb as _lamb
from topologies.implementations.optimizers.lars import lars as _lars
from topologies.implementations.optimizers.lion import lion as _lion
from topologies.implementations.optimizers.lookahead import lookahead as _lookahead
from topologies.implementations.optimizers.model_soup import model_soup as _model_soup
from topologies.implementations.optimizers.muon import muon as _muon
from topologies.implementations.optimizers.novograd import novograd as _novograd
from topologies.implementations.optimizers.polynomial_decay import (
    polynomial_decay as _polynomial_decay,
)
from topologies.implementations.optimizers.qlora import qlora as _qlora
from topologies.implementations.optimizers.radam import radam as _radam
from topologies.implementations.optimizers.relora import relora as _relora
from topologies.implementations.optimizers.sam import sam as _sam
from topologies.implementations.optimizers.schedule_free import schedule_free as _schedule_free
from topologies.implementations.optimizers.shampoo import shampoo as _shampoo
from topologies.implementations.optimizers.sm3 import sm3 as _sm3
from topologies.implementations.optimizers.soap import soap as _soap
from topologies.implementations.optimizers.sophia import sophia as _sophia
from topologies.implementations.optimizers.swa import swa as _swa
from topologies.implementations.optimizers.swad import swad as _swad
from topologies.implementations.optimizers.swalp import swalp as _swalp
from topologies.implementations.optimizers.twa import twa as _twa

register_optimizer("adafactor", _adafactor)
register_optimizer("lamb", _lamb)
register_optimizer("lars", _lars)
register_optimizer("radam", _radam)
register_optimizer("adabelief", _adabelief)
register_optimizer("novograd", _novograd)
register_optimizer("lion", _lion)
register_optimizer("ademamix", _ademamix)
register_optimizer("sophia", _sophia)
register_optimizer("shampoo", _shampoo)
register_optimizer("fadam", _fadam)
register_optimizer("swad", _swad)
register_optimizer("twa", _twa)
register_optimizer("model-soup", _model_soup)
register_optimizer("swalp", _swalp)
register_optimizer("polynomial-decay", _polynomial_decay)
register_optimizer("ema", _ema)
register_optimizer("fisher-sam", _fisher_sam)
register_optimizer("lookahead", _lookahead)
register_optimizer("swa", _swa)
register_optimizer("gsam", _gsam)
register_optimizer("asam", _asam)
register_optimizer("sam", _sam)
register_optimizer("qlora", _qlora)
register_optimizer("relora", _relora)
register_optimizer("flora", _flora)
register_optimizer("adahessian", _adahessian)
register_optimizer("galore", _galore)
register_optimizer("sm3", _sm3)
register_optimizer("adam-8bit", _adam8bit)
register_optimizer("grokfast", _grokfast)
register_optimizer("schedule-free", _schedule_free)
register_optimizer("soap", _soap)
register_optimizer("muon", _muon)
register_optimizer("kfac", _kfac)
register_optimizer("adam-mini", _adam_mini)


def split_decay_params(params: Any, *, weight_decay: float, model: Any = None) -> list[dict]:
    """Group parameters for ``decay_policy="matrices"``.

    Decay applies to tensors with ``ndim >= 2`` that are not the weight of an
    ``nn.Embedding`` (or ``EmbeddingBag``). Biases, norms, gains, and other
    scalars are in a second group with zero decay. ``model`` is used only to
    recognize embedding parameters; without it, every matrix decays.
    """
    import torch.nn as nn

    embed_ids: set[int] = set()
    if model is not None:
        for mod in model.modules():
            if isinstance(mod, (nn.Embedding, nn.EmbeddingBag)):
                embed_ids.add(id(mod.weight))
    decay: list[Any] = []
    frozen: list[Any] = []
    seen: set[int] = set()
    flat = []
    for item in params:
        if isinstance(item, dict):
            flat.extend(item.get("params", ()))
        else:
            flat.append(item)
    for param in flat:
        if id(param) in seen or not getattr(param, "requires_grad", False):
            continue
        seen.add(id(param))
        if getattr(param, "ndim", 0) >= 2 and id(param) not in embed_ids:
            decay.append(param)
        else:
            frozen.append(param)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(weight_decay)})
    if frozen:
        groups.append({"params": frozen, "weight_decay": 0.0})
    if not groups:
        raise TrainingError("Model has no trainable parameters for the optimizer.", stage="optimizer")
    return groups


def build_optimizer(name_or_cls: Any, params: Any, cfg: Any, *,
                    model: Any = None) -> torch.optim.Optimizer:
    """Build an optimizer from a registered name or ``Optimizer`` subclass + ``OptimizerConfig``.

    ``decay_policy="matrices"`` (see :func:`split_decay_params`) replaces a flat
    parameter list with decay / no-decay groups. ``"all"`` leaves ``params``
    unchanged.
    """
    if getattr(cfg, "decay_policy", "all") == "matrices":
        params = split_decay_params(params, weight_decay=cfg.weight_decay, model=model)
    full = {"lr": cfg.lr, "weight_decay": cfg.weight_decay, "momentum": cfg.momentum,
            "eps": cfg.eps, "betas": cfg.betas}
    factory = (name_or_cls if not isinstance(name_or_cls, str)
               else get_optimizer_factory(name_or_cls))
    explicit = set(getattr(cfg, "model_fields_set", set()))
    kw: dict[str, Any] = {"lr": cfg.lr, "weight_decay": cfg.weight_decay}
    optimizer_name = _norm(name_or_cls) if isinstance(name_or_cls, str) else ""
    common_options = {"adam": {"eps", "betas"}, "adamw": {"eps", "betas"},
                      "sgd": {"momentum"}, "rmsprop": {"momentum", "eps"}}
    for k in ("momentum", "eps", "betas"):
        if k in explicit and full[k] is not None and k in common_options.get(optimizer_name, {"momentum", "eps", "betas"}):
            kw[k] = full[k]
    kw.update(dict(cfg.kwargs or {}))
    return factory(params, **accepted_kwargs(factory, kw, skip=1))


# ---------------------------------------------------------------- schedulers


def register_scheduler(name: str, factory: SchedulerFactory, *, overwrite: bool = True) -> None:
    """Make ``scheduler=name`` build ``factory(optimizer, total_steps, **kwargs)``."""
    key = _norm(name)
    if not key or key == "none":
        raise ValueError("scheduler name must be non-empty and not 'none'.")
    if not callable(factory):
        raise TypeError(f"scheduler factory for {name!r} must be callable.")
    if key in _SCHEDULERS and not overwrite:
        raise ValueError(f"scheduler {name!r} is already registered.")
    _SCHEDULERS[key] = factory


def list_schedulers() -> list[str]:
    return ["none", *sorted(_SCHEDULERS)]


def get_scheduler_factory(name: str) -> SchedulerFactory:
    key = _norm(name)
    if key not in _SCHEDULERS:
        raise TrainingError(f"Unknown scheduler {name!r}.", stage="scheduler",
                            available=list_schedulers())
    return _SCHEDULERS[key]


def _cosine(opt, total_steps, *, T_max=None, min_lr_ratio=0.0, **_):
    """Cosine from the base LR down to ``min_lr_ratio`` times that LR."""
    t_max = max(1, int(T_max if T_max is not None else (total_steps or 100)))
    floor = float(min_lr_ratio)

    def _lr(step: int) -> float:
        progress = min(1.0, step / t_max)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor + (1.0 - floor) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _lr)
    scheduler.T_max = t_max  # the horizon, as on CosineAnnealingLR
    return scheduler


def _linear(opt, total_steps, *, end_factor=0.0, min_lr_ratio=None, total_steps_=None, **_):
    """Linear decay from the base LR to ``min_lr_ratio`` (or ``end_factor``) times that LR."""
    total_f = float(total_steps_ if total_steps_ is not None else (total_steps or 100))
    floor = float(end_factor if min_lr_ratio is None else min_lr_ratio)

    def _lr(step: int) -> float:
        frac = min(1.0, step / max(1.0, total_f))
        return 1.0 - frac * (1.0 - floor)

    return torch.optim.lr_scheduler.LambdaLR(opt, _lr)


def _wsd(opt, total_steps, *, decay_ratio=0.2, decay_shape="linear", min_lr_ratio=0.0, **_):
    """Warmup-stable-decay over ``total_steps`` after any outer warmup.

    The stable phase holds the base LR. The last ``decay_ratio`` of the steps
    decay to ``min_lr_ratio`` times that LR with ``decay_shape`` of ``linear``,
    ``cosine``, or ``sqrt``. Pair with ``warmup_steps`` / ``warmup_ratio`` for
    the warmup phase: that wrapper runs first and this schedule sees only the
    remaining steps.
    """
    total = max(1, int(total_steps or 1))
    ratio = float(decay_ratio)
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"decay_ratio must be in [0, 1); got {decay_ratio!r}.")
    shape = str(decay_shape)
    if shape not in ("linear", "cosine", "sqrt"):
        raise ValueError(
            f"decay_shape must be 'linear', 'cosine', or 'sqrt'; got {decay_shape!r}."
        )
    floor = float(min_lr_ratio)
    decay_start = total * (1.0 - ratio)

    def _lr(step: int) -> float:
        if float(step) <= decay_start:
            return 1.0
        span = max(1.0, float(total) - decay_start)
        progress = min(1.0, (float(step) - decay_start) / span)
        if shape == "linear":
            return 1.0 - progress * (1.0 - floor)
        if shape == "cosine":
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return floor + (1.0 - floor) * cosine
        return floor + (1.0 - floor) * (1.0 - math.sqrt(progress))

    return torch.optim.lr_scheduler.LambdaLR(opt, _lr)


def _step(opt, total_steps, *, step_size=10, gamma=0.1, **_):
    return torch.optim.lr_scheduler.StepLR(opt, step_size=int(step_size), gamma=float(gamma))


def _onecycle(opt, total_steps, *, max_lr=None, **_):
    if max_lr is None:
        try:
            max_lr = float(opt.param_groups[0].get("lr", 3e-4))
        except Exception:  # noqa: BLE001
            max_lr = 3e-4
    return torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(max_lr),
                                               total_steps=max(2, total_steps or 100))


def _constant(opt, total_steps, **_):
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0)


def _inverse_sqrt(opt, total_steps, *, timescale=None, **_):
    """``lr * sqrt(timescale / max(step, timescale))``: flat, then 1/sqrt(step) decay."""
    ts = float(timescale if timescale is not None else max(1, (total_steps or 100) // 10))
    ts = max(1.0, ts)
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda step: math.sqrt(ts / max(step, ts)))


for _name, _fn in (("cosine", _cosine), ("linear", _linear), ("step", _step),
                   ("onecycle", _onecycle), ("constant", _constant),
                   ("inverse_sqrt", _inverse_sqrt), ("wsd", _wsd)):
    register_scheduler(_name, _fn)

from topologies.implementations.schedulers.grafted import grafted as _grafted_sched

register_scheduler("grafted", _grafted_sched)

from topologies.implementations.schedulers.cooldown import cooldown as _cooldown_sched

register_scheduler("cooldown", _cooldown_sched)

from topologies.implementations.schedulers.clr import clr as _clr_sched

register_scheduler("clr", _clr_sched)

from topologies.implementations.schedulers.trapezoidal import trapezoidal as _trapezoidal_sched

register_scheduler("trapezoidal", _trapezoidal_sched)

from topologies.implementations.schedulers.cosine_min import cosine_min as _cosine_min_sched

register_scheduler("cosine-min", _cosine_min_sched)

from topologies.implementations.schedulers.exponential import exponential as _exponential_sched

register_scheduler("exponential", _exponential_sched)

from topologies.implementations.schedulers.multistep import multistep as _multistep_sched

register_scheduler("multistep", _multistep_sched)

from topologies.implementations.schedulers.reduce_on_plateau import (
    reduce_on_plateau as _reduce_on_plateau_sched,
)

register_scheduler("reduce-on-plateau", _reduce_on_plateau_sched)

from topologies.implementations.schedulers.cosine_restarts import (
    cosine_restarts as _cosine_restarts_sched,
)

register_scheduler("cosine-restarts", _cosine_restarts_sched)

from topologies.implementations.schedulers.warmup_linear import (
    warmup_linear as _warmup_linear_sched,
)

register_scheduler("warmup-linear", _warmup_linear_sched)

from topologies.implementations.schedulers.wsd import wsd as _wsd_phases


def _wsd_sched(opt, total_steps, **kwargs):
    """One ``wsd`` name for both parameterizations.

    ``decay_ratio`` / ``decay_shape`` (``linear``, ``cosine``, ``sqrt``) give the
    last fraction of the run to the decay; ``warmup_steps`` / ``stable_steps`` /
    ``decay`` (``cosine``, ``linear``) set explicit phase lengths. ``min_lr_ratio``
    works with both.
    """
    if "decay_ratio" in kwargs or "decay_shape" in kwargs:
        return _wsd(opt, total_steps, **kwargs)
    return _wsd_phases(opt, total_steps, **kwargs)


register_scheduler("wsd", _wsd_sched)

from topologies.implementations.schedulers.sgdr import sgdr as _sgdr_sched

register_scheduler("sgdr", _sgdr_sched)

from topologies.implementations.schedulers.noam import noam as _noam_sched

register_scheduler("noam", _noam_sched)

_SCHED_ALIASES = {"one-cycle": "onecycle", "one_cycle": "onecycle",
                  "inverse-sqrt": "inverse_sqrt", "rsqrt": "inverse_sqrt"}


def canonical_scheduler(name: str) -> str:
    key = _norm(name)
    return _SCHED_ALIASES.get(key, key)


def build_scheduler(name: str, opt: Any, total_steps: int, kwargs: dict | None) -> Any:
    kw = dict(kwargs or {})
    if canonical_scheduler(name) == "linear" and "total_steps" in kw:
        kw["total_steps_"] = kw.pop("total_steps")  # legacy kwarg name
    factory = get_scheduler_factory(canonical_scheduler(name))
    return factory(opt, total_steps, **kw)


def call_scheduler_factory(factory: Callable, opt: Any, total_steps: int) -> Any:
    """User factory: ``fn(optimizer)`` (legacy) or ``fn(optimizer, total_steps)``."""
    try:
        params = [p for p in inspect.signature(factory).parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                  and p.default is p.empty]
        two = len(params) >= 2 or any(p.name == "total_steps" for p in
                                      inspect.signature(factory).parameters.values())
    except (TypeError, ValueError):
        two = False
    return factory(opt, total_steps) if two else factory(opt)


# ---------------------------------------------------------------- warmup


class WarmupScheduler:
    """Linear warmup in front of any scheduler (or none).

    After ``s`` optimizer steps the learning rate is ``lr * s / warmup`` for
    ``s < warmup`` (so step 0 uses 0, ``warmup / 2`` uses half), then the
    wrapped schedule runs from its own start. Checkpointable like a torch
    scheduler.
    """

    def __init__(self, optimizer: Any, warmup_steps: int, inner: Any = None) -> None:
        self.optimizer = optimizer
        self.warmup_steps = max(0, int(warmup_steps))
        self.inner = inner
        self._count = 0
        self._target = [float(g["lr"]) for g in optimizer.param_groups]
        self._apply(self._mult())

    def _mult(self) -> float:
        if self.warmup_steps <= 0:
            return 1.0
        return min(1.0, self._count / self.warmup_steps)

    def _apply(self, mult: float) -> None:
        for g, lr in zip(self.optimizer.param_groups, self._target):
            g["lr"] = lr * mult

    def step(self) -> None:
        self._count += 1
        if self._count <= self.warmup_steps:
            self._apply(self._mult())
        elif self.inner is not None:
            self.inner.step()

    def get_last_lr(self) -> list[float]:
        return [float(g["lr"]) for g in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        return {"count": self._count, "target": list(self._target),
                "warmup_steps": self.warmup_steps,
                "inner": self.inner.state_dict() if self.inner is not None else None}

    def load_state_dict(self, state: dict) -> None:
        self._count = int(state.get("count", 0))
        self._target = [float(x) for x in state.get("target", self._target)]
        if self.inner is not None and state.get("inner") is not None:
            self.inner.load_state_dict(state["inner"])
        if self._count <= self.warmup_steps:
            self._apply(self._mult())

    def __repr__(self) -> str:
        inner = type(self.inner).__name__ if self.inner is not None else None
        return f"WarmupScheduler(warmup_steps={self.warmup_steps}, inner={inner})"


__all__ = [
    "WarmupScheduler",
    "accepted_kwargs",
    "build_optimizer",
    "build_scheduler",
    "get_optimizer_factory",
    "get_scheduler_factory",
    "list_optimizers",
    "list_schedulers",
    "register_optimizer",
    "register_scheduler",
    "split_decay_params",
]
