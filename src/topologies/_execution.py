"""Shared explicit model-call, recursive transfer, and execution lifecycle helpers."""
from __future__ import annotations

import random
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

_TRANSFER_HANDLERS: dict[type, Any] = {}


def register_transfer_handler(container_type: type, handler: Any) -> None:
    """Register reconstruction handler ``handler(value, transfer_leaf)``."""
    if not isinstance(container_type, type) or not callable(handler):
        raise TypeError("container_type must be a type and handler callable")
    if container_type in _TRANSFER_HANDLERS:
        raise ValueError(f"Transfer handler already registered for {container_type!r}")
    _TRANSFER_HANDLERS[container_type] = handler


@dataclass
class ModelCall:
    """An unambiguous invocation; a raw tensor is represented as one positional arg."""

    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)


def transfer(value: Any, device: str | torch.device, *, non_blocking: bool = False,
             _path: str = "input") -> Any:
    """Recursively transfer tensor leaves; never silently leave failed tensors behind."""
    if isinstance(value, torch.Tensor):
        try:
            return value.to(device, non_blocking=non_blocking)
        except Exception as exc:
            raise RuntimeError(f"Tensor transfer failed at {_path}: {exc}") from exc
    for container_type, handler in _TRANSFER_HANDLERS.items():
        if isinstance(value, container_type):
            try:
                return handler(value, lambda child, suffix="item": transfer(
                    child, device, non_blocking=non_blocking, _path=f"{_path}.{suffix}"))
            except Exception as exc:
                raise RuntimeError(f"Container transfer failed at {_path}: {exc}") from exc
    if isinstance(value, Mapping):
        pairs = ((k, transfer(v, device, non_blocking=non_blocking, _path=f"{_path}.{k}"))
                 for k, v in value.items())
        try:
            return type(value)(pairs)
        except Exception as exc:
            raise RuntimeError(f"Container reconstruction failed at {_path}: {exc}") from exc
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(transfer(v, device, non_blocking=non_blocking,
                                      _path=f"{_path}.{name}")
                             for name, v in zip(value._fields, value)))
    if isinstance(value, tuple):
        return tuple(transfer(v, device, non_blocking=non_blocking, _path=f"{_path}[{i}]")
                     for i, v in enumerate(value))
    if isinstance(value, list):
        return [transfer(v, device, non_blocking=non_blocking, _path=f"{_path}[{i}]")
                for i, v in enumerate(value)]
    return value


@contextmanager
def execution_rng(*, preserve_rng: bool = True, seed: int | None = None) -> Iterator[None]:
    """Preserve framework/Python/loaded-NumPy RNG state and optionally seed a scoped execution.

    NumPy is inspected only in ``sys.modules`` so this helper does not make an
    optional dependency eager. Explicit seeds use NumPy-compatible uint32 range.
    A supplied seed always restores the caller's streams, even when preservation
    is otherwise disabled.
    """
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or
                             not 0 <= seed <= 0xFFFFFFFF):
        raise ValueError("seed must be an integer in [0, 2**32 - 1]")
    restore = preserve_rng or seed is not None
    devices = list(range(torch.cuda.device_count())) if restore and torch.cuda.is_available() else []
    rng_context = torch.random.fork_rng(devices=devices, enabled=restore)
    python_state = random.getstate() if restore else None
    numpy = sys.modules.get("numpy")
    numpy_state = numpy.random.get_state() if restore and numpy is not None else None
    try:
        with rng_context:
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                random.seed(seed)
                if numpy is not None:
                    numpy.random.seed(seed)
            yield
    finally:
        if python_state is not None:
            random.setstate(python_state)
        if numpy_state is not None:
            numpy.random.set_state(numpy_state)


GradMode = Literal["inference", "no_grad", "enable", "unchanged"]
_GRAD_MODES: dict[str, Any] = {
    "inference": torch.inference_mode,
    "no_grad": torch.no_grad,
    "enable": torch.enable_grad,
    "unchanged": nullcontext,
}


@contextmanager
def model_execution(model: Any, *, training: bool = False, inference: bool = True,
                    preserve_rng: bool = True, seed: int | None = None,
                    grad_mode: GradMode | None = None) -> Iterator[None]:
    """Set mode and grad policy, restoring every submodule's mode even on exceptions.

    ``grad_mode``: ``"inference"`` (``torch.inference_mode``; outputs can never
    join autograd), ``"no_grad"`` (outputs stay usable as autograd inputs),
    ``"enable"`` or ``"unchanged"`` (leave the caller's grad mode alone, e.g.
    for generators that yield while suspended). Default: ``"inference"`` when
    ``inference`` else ``"enable"``.
    """
    mode = grad_mode or ("inference" if inference else "enable")
    if mode not in _GRAD_MODES:
        raise ValueError(f"grad_mode must be one of {sorted(_GRAD_MODES)}, got {mode!r}")
    modules = list(model.modules()) if callable(getattr(model, "modules", None)) else []
    flags = [module.training for module in modules]
    try:
        with execution_rng(preserve_rng=preserve_rng, seed=seed):
            if modules:
                model.train(training)
            with _GRAD_MODES[mode]():
                yield
    finally:
        for module, flag in zip(modules, flags):
            module.training = flag


def _tensors_depth_first(value: Any) -> Iterator[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _tensors_depth_first(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors_depth_first(item)


def primary_output(out: Any, *, prefer: str | None = "logits", floating: bool = False,
                   strict: bool = True) -> Any:
    """The primary tensor of a model output (one rule for every call site).

    A tensor is returned as is. A mapping yields its ``prefer`` key when present
    (``"logits"`` by default), otherwise its values in order; tuples/lists are
    searched depth-first for the first tensor (the first floating-point tensor
    when ``floating``). With ``strict=False`` an output without a matching
    tensor is returned unchanged (sequences: their first element) instead of
    raising ``TypeError``.
    """
    if isinstance(out, torch.Tensor):
        return out
    candidate = out
    if isinstance(out, Mapping) and prefer is not None and prefer in out:
        candidate = out[prefer]
        if isinstance(candidate, torch.Tensor):
            return candidate
    for tensor in _tensors_depth_first(candidate):
        if not floating or tensor.is_floating_point():
            return tensor
    if not strict:
        if isinstance(out, (tuple, list)) and out:
            return out[0]
        return out
    raise TypeError(f"Model returned {type(out).__name__} without a "
                    f"{'floating-point ' if floating else ''}tensor output.")

