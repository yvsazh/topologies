"""Scoped hook registration, call stack, occurrence counters and the call tree.

Per-module forward(-pre) hooks with kwargs are registered only on tracked
modules. When every module is tracked for the call tree while only a few are
captured (``call_tree=True``) a single global hook pair is used instead.
All handles are removed in :meth:`HookManager.detach`, which sessions call in
``finally``.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch.nn.modules import module as module_hooks

from topologies.introspection import _tree
from topologies.introspection.sites import SiteIndex
from topologies.introspection.trace import CallRecord, CallTree

_CONTAINERS = (nn.ModuleList, nn.ModuleDict, nn.ParameterList, nn.ParameterDict)
PreHandler = Callable[["Frame", nn.Module, tuple, dict], Any]
PostHandler = Callable[["Frame", nn.Module, tuple, dict, Any], Any]


@dataclass
class Frame:
    path: str
    occurrence: int
    call_index: int | None
    module: nn.Module
    recompute: bool = False
    op_counts: dict[str, int] = field(default_factory=dict)
    start: float | None = None


class HookManager:
    """Owns every hook handle of one session and the bookkeeping they share."""

    def __init__(self, index: SiteIndex, *, tracked: Iterable[str], kwarg_paths: Iterable[str] = (),
                 protect_paths: Iterable[str] = (), call_tree: bool = True, timing: bool = False,
                 global_mode: bool = False, record_shapes: bool = True):
        self.index = index
        # Containers (ModuleList/ModuleDict) have no forward; tracking them is noise.
        self.tracked = tuple(p for p in dict.fromkeys(tracked)
                             if not isinstance(index.module(p), _CONTAINERS))
        self.kwarg_paths = set(kwarg_paths)
        self.protect_paths = set(protect_paths)
        self.call_tree_enabled = call_tree
        self.timing = timing
        self.global_mode = global_mode
        self.record_shapes = record_shapes
        self.pre_handlers: list[PreHandler] = []
        self.post_handlers: list[PostHandler] = []
        self.protect: Callable[[tuple], None] | None = None
        self.phase = "forward"
        self.stack: list[Frame] = []
        self.counters: dict[str, int] = {}
        self.calls: list[CallRecord] = []
        self.recompute_calls = 0
        self._handles: list[Any] = []
        self._ids = {id(index.module(p)): p for p in self.tracked}
        self._kwargs: dict[int, dict] = {}
        self._attached = False

    # -- lifecycle --------------------------------------------------------------------

    def attach(self) -> None:
        if self._attached:
            return
        self._attached = True
        try:
            if self.global_mode:
                self._handles.append(module_hooks.register_module_forward_pre_hook(self._global_pre))
                self._handles.append(module_hooks.register_module_forward_hook(
                    self._global_post, with_kwargs=True, always_call=True))
                for path in self.kwarg_paths:
                    module = self.index.module(path)
                    self._handles.append(module.register_forward_pre_hook(
                        self._kwargs_pre, with_kwargs=True))
            else:
                for path in self.tracked:
                    module = self.index.module(path)
                    self._handles.append(module.register_forward_pre_hook(
                        self._module_pre, with_kwargs=True))
                    self._handles.append(module.register_forward_hook(
                        self._module_post, with_kwargs=True, always_call=True))
            for path in self.protect_paths:
                module = self.index.module(path)
                self._handles.append(module.register_forward_pre_hook(self._protect_pre))
        except BaseException:
            self.detach()
            raise

    def detach(self) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        self._attached = False
        self.stack.clear()
        self._kwargs.clear()

    def reset(self) -> None:
        """Start a new run: occurrences and the call tree restart at zero."""
        self.counters.clear()
        self.calls = []
        self.stack.clear()
        self.recompute_calls = 0
        self.phase = "forward"

    def call_tree(self) -> CallTree:
        return CallTree(list(self.calls), tuple(self.index.paths()), self.tracked)

    @property
    def current(self) -> Frame | None:
        return self.stack[-1] if self.stack else None

    # -- hook bodies ----------------------------------------------------------------------

    def _module_pre(self, module: nn.Module, args: tuple, kwargs: dict) -> Any:
        path = self._ids.get(id(module))
        if path is None:
            return None
        return self._enter(path, module, args, kwargs)

    def _module_post(self, module: nn.Module, args: tuple, *rest: Any) -> Any:
        # Normal calls pass (kwargs, output); the always_call exception path may pass (output,).
        kwargs, output = rest if len(rest) == 2 else ({}, rest[0] if rest else None)
        path = self._ids.get(id(module))
        if path is None:
            return None
        return self._exit(path, module, args, kwargs, output)

    def _global_pre(self, module: nn.Module, args: tuple) -> Any:
        path = self._ids.get(id(module))
        if path is None:
            return None
        if path in self.kwarg_paths:
            return None  # entered by the per-module kwargs hook, which sees kwargs
        result = self._enter(path, module, args, {})
        return None if result is None else result[0]

    def _kwargs_pre(self, module: nn.Module, args: tuple, kwargs: dict) -> Any:
        path = self._ids.get(id(module))
        return None if path is None else self._enter(path, module, args, kwargs)

    def _global_post(self, module: nn.Module, args: tuple, *rest: Any) -> Any:
        kwargs, output = rest if len(rest) == 2 else ({}, rest[0] if rest else None)
        path = self._ids.get(id(module))
        if path is None:
            return None
        return self._exit(path, module, args, kwargs, output)

    def _protect_pre(self, module: nn.Module, args: tuple) -> None:
        if self.protect is not None and self.phase == "forward":
            self.protect(args)

    def _enter(self, path: str, module: nn.Module, args: tuple, kwargs: dict) -> Any:
        recompute = self.phase == "backward"
        if recompute:
            self.recompute_calls += 1
            occurrence = -1
            call_index = None
        else:
            occurrence = self.counters.get(path, 0)
            self.counters[path] = occurrence + 1
            call_index = None
            if self.call_tree_enabled:
                parent = next((f.call_index for f in reversed(self.stack)
                               if f.call_index is not None), None)
                depth = (self.calls[parent].depth + 1) if parent is not None else 0
                call_index = len(self.calls)
                self.calls.append(CallRecord(
                    call_index, path, occurrence, parent, depth, type(module).__name__,
                    inputs=_tree.describe(args) if self.record_shapes else None))
        frame = Frame(path, occurrence, call_index, module, recompute)
        if self.timing:
            _sync(args)
            frame.start = time.perf_counter()
        self.stack.append(frame)
        replaced = None
        for handler in self.pre_handlers:
            result = handler(frame, module, args, kwargs)
            if result is not None:
                args, kwargs = result
                replaced = (args, kwargs)
        return replaced

    def _exit(self, path: str, module: nn.Module, args: tuple, kwargs: dict, output: Any) -> Any:
        frame = None
        while self.stack:
            candidate = self.stack.pop()
            if candidate.module is module:
                frame = candidate
                break
        if frame is None:
            return None
        replaced = None
        for handler in self.post_handlers:
            result = handler(frame, module, args, kwargs, output if replaced is None else replaced)
            if result is not None:
                replaced = result
        if frame.call_index is not None:
            record = self.calls[frame.call_index]
            final = output if replaced is None else replaced
            if output is None and not isinstance(final, torch.Tensor):
                record.error = True
            if self.record_shapes:
                record.outputs = _tree.describe(final)
                tensors = [t for _, t in _tree.leaves(final)]
                record.dtypes = tuple(dict.fromkeys(str(t.dtype).replace("torch.", "")
                                                    for t in tensors))
                record.devices = tuple(dict.fromkeys(str(t.device) for t in tensors))
            if self.timing and frame.start is not None:
                _sync(final)
                record.duration = time.perf_counter() - frame.start
        return replaced


def _sync(value: Any) -> None:
    found = _tree.first_tensor(value)
    if found is not None and found[1].device.type == "cuda":
        torch.cuda.synchronize(found[1].device)


def leaked_hooks(model: nn.Module) -> list[str]:
    """Paths of modules that still carry forward/backward hooks (tests and debug mode)."""
    leaked = []
    for name, module in model.named_modules():
        if module._forward_hooks or module._forward_pre_hooks or module._backward_hooks or \
                getattr(module, "_backward_pre_hooks", None):
            leaked.append(name or "<root>")
    if module_hooks._global_forward_hooks or module_hooks._global_forward_pre_hooks:
        leaked.append("<global>")
    return leaked


def assert_no_leaked_hooks(model: nn.Module, *, baseline: dict[str, int] | None = None) -> None:
    """Raise if introspection left hooks behind (``baseline`` = pre-existing hook counts)."""
    counts = hook_counts(model)
    if baseline is not None:
        extra = {k: v for k, v in counts.items() if v != baseline.get(k, 0)}
    else:
        extra = {k: v for k, v in counts.items() if v}
    if extra:
        raise AssertionError(f"leaked hooks: {extra}")


def hook_counts(model: nn.Module) -> dict[str, int]:
    counts = {}
    for name, module in model.named_modules():
        n = (len(module._forward_hooks) + len(module._forward_pre_hooks) +
             len(module._backward_hooks) + len(getattr(module, "_backward_pre_hooks", {})))
        if n:
            counts[name or "<root>"] = n
    n = len(module_hooks._global_forward_hooks) + len(module_hooks._global_forward_pre_hooks)
    if n:
        counts["<global>"] = n
    return counts
