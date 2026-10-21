"""Functional interception (tier F): see inside ``forward`` without changing the model.

* ``scaled_dot_product_attention``: the original op computes the output, so outputs
  are unchanged. When weights are requested they are **recomputed** as
  ``softmax(q @ kᵀ · scale + mask)`` in float32. The recomputation honours bool
  and float masks, ``is_causal`` (top-left aligned, as PyTorch documents it) and
  GQA head expansion.
* ``nn.MultiheadAttention``: within the session, ``need_weights=True`` and
  ``average_attn_weights=False`` are forced, and the caller's requested return value
  is restored. The math path can differ from the fast path in the last ulp.
* **op mode** (``ops="add,softmax,matmul"``, or ``"*"`` with a cap) records the outputs
  of intercepted torch functions and tensor methods as ``<path>/fn:<op>#i`` sites. It is
  opt-in and costly. Site names come from the innermost running module, which is a heuristic.
"""
from __future__ import annotations

import inspect
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

from topologies.introspection import _tree
from topologies.introspection.semantics import Axes, SemanticSource, infer_axes
from topologies.introspection.sites import SiteKey

ATTENTION_AXES = Axes(("batch", "heads", "query", "key"), SemanticSource.DECLARED, None, 0)
_SDPA = {F.scaled_dot_product_attention, getattr(torch._C._nn, "scaled_dot_product_attention", None)}
_SKIP = {"__get__", "__set__", "size", "dim", "numel", "shape", "device", "dtype", "is_floating_point",
         "requires_grad", "_version", "data_ptr", "untyped_storage", "storage", "stride",
         "is_contiguous", "__len__", "__iter__", "__bool__", "__index__", "__repr__", "__format__",
         "item", "tolist", "element_size", "nelement", "ndimension", "is_complex", "grad_fn",
         "is_leaf", "names", "layout", "__hash__", "__eq__", "__ne__", "is_inference",
         "register_hook", "detach", "__getitem__", "__setitem__"}
_MHA_PARAMS = ("query", "key", "value", "key_padding_mask", "need_weights", "attn_mask",
               "average_attn_weights", "is_causal")


def op_name(func: Any) -> str:
    if func in _SDPA:
        return "sdpa"
    name = getattr(func, "__name__", None) or str(func)
    stripped = name.strip("_")
    if stripped.startswith("i") and name.startswith("__i") and stripped[1:] in ("add", "mul", "sub"):
        return stripped[1:] + "_"
    return stripped.removeprefix("r") if name.startswith("__r") and stripped[1:] in (
        "add", "mul", "sub", "matmul", "truediv") else stripped


def sdpa_weights(q: torch.Tensor, k: torch.Tensor, attn_mask: Any = None, is_causal: bool = False,
                 scale: float | None = None, enable_gqa: bool = False) -> torch.Tensor:
    """``softmax(q kᵀ · scale + bias)`` in float32 with PyTorch's mask semantics."""
    qf, kf = q.float(), k.float()
    if qf.shape[-3] != kf.shape[-3]:
        if not enable_gqa and qf.ndim >= 3 and qf.shape[-3] % kf.shape[-3]:
            raise ValueError("query/key head counts differ and are not a multiple (GQA)")
        kf = kf.repeat_interleave(qf.shape[-3] // kf.shape[-3], dim=-3)
    scale = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
    scores = qf @ kf.transpose(-2, -1) * scale
    length, source = scores.shape[-2], scores.shape[-1]
    if is_causal:
        allowed = torch.ones(length, source, dtype=torch.bool, device=scores.device).tril(0)
        scores = scores.masked_fill(~allowed, float("-inf"))
    if attn_mask is not None:
        mask = attn_mask.to(scores.device)
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, float("-inf"))
        else:
            scores = scores + mask.float()
    return torch.softmax(scores, dim=-1)


_NAMES: dict[Any, str] = {}


class _Mode(TorchFunctionMode):
    def __init__(self, owner: FunctionalInterceptor):
        super().__init__()
        self.owner = owner

    def __torch_function__(self, func: Any, types: Any, args: tuple = (), kwargs: Any = None) -> Any:
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        try:
            self.owner.observe(func, args, kwargs, out)
        except Exception as exc:  # noqa: BLE001 - observation must never change behaviour
            self.owner.errors.append(f"{op_name(func)}: {exc}")
        return out


class FunctionalInterceptor:
    """Owned by one session: torch function mode + ``nn.MultiheadAttention`` weight capture."""

    def __init__(self, session: Any, *, ops: Any = None, graph: bool = False, max_ops: int = 10_000):
        self.session = session
        self.requests = [r for r in session.requests if r.op is not None]
        if ops in (None, False):
            self.ops: set[str] | None = set()
        elif ops in ("*", True):
            self.ops = None
        else:
            self.ops = {o.strip() for o in (ops.split(",") if isinstance(ops, str) else ops)}
        self.ops_all = ops in ("*", True)
        self.graph = graph
        self.max_ops = max_ops
        self.count = 0
        self.errors: list[str] = []
        self.mha_paths = [p for p in dict.fromkeys(r.path for r in self.requests if r.op == "mha")]
        self.names = {r.op for r in self.requests if r.op != "mha"} | (self.ops or set())
        # Requests alone (no op mode, no graph): the mode is only active inside the requested
        # modules, entered/exited by their hooks, so the rest of the model runs untouched.
        self.scoped = not graph and not self.ops_all and not self.ops
        self.scope_paths = {r.path for r in self.requests if r.op != "mha"}
        self._depth = 0
        self._mode: _Mode | None = None
        self._mha_state: dict[int, list[tuple[bool, bool]]] = {}
        self.warned_cap = False

    @property
    def active(self) -> bool:
        return bool(self.requests) or self.ops is None or bool(self.ops) or self.graph

    @property
    def needs_mode(self) -> bool:
        return any(r.op != "mha" for r in self.requests) or self.ops is None or bool(self.ops) \
            or self.graph

    # -- session lifecycle ---------------------------------------------------------------

    def install(self, session: Any, hooks: Any) -> None:
        if self.scoped and self.needs_mode:
            hooks.pre_handlers.append(self._scope_enter)
            hooks.post_handlers.append(self._scope_exit)
        if self.mha_paths:
            hooks.pre_handlers.append(self._mha_pre)
            hooks.post_handlers.insert(0, self._mha_post)
            session.assumptions.append(
                "nn.MultiheadAttention ran with need_weights=True (math path); outputs may differ "
                "from the fast path in the last ulp")

    def enter(self, session: Any) -> Any:
        self.count = 0
        self._depth = 0
        if not self.needs_mode or self.scoped:
            return self if self.scoped else None
        self._mode = _Mode(self)
        self._mode.__enter__()
        return self._mode

    def _scope_enter(self, frame: Any, module: nn.Module, args: tuple, kwargs: dict) -> None:
        if frame.path in self.scope_paths and not frame.recompute:
            if self._depth == 0:
                self._mode = _Mode(self)
                self._mode.__enter__()
            self._depth += 1

    def _scope_exit(self, frame: Any, module: nn.Module, args: tuple, kwargs: dict,
                    output: Any) -> None:
        if frame.path in self.scope_paths and not frame.recompute and self._depth:
            self._depth -= 1
            if self._depth == 0 and self._mode is not None:
                self._mode.__exit__(None, None, None)
                self._mode = None

    def exit(self, session: Any) -> None:
        self._depth = 0
        if self._mode is not None:
            self._mode.__exit__(None, None, None)
            self._mode = None
        for error in self.errors:
            message = f"functional capture skipped an op ({error})"
            if message not in session.warnings:
                session.warnings.append(message)
        self.errors.clear()

    # -- observation -------------------------------------------------------------------------

    def _wanted(self, path: str, op: str, index: int, occurrence: int) -> list[Any]:
        out = []
        for req in self.requests:
            if req.op != op:
                continue
            if not (path == req.path or (req.path and path.startswith(req.path + ".")) or
                    req.path == ""):
                continue
            if req.op_index is not None and req.op_index != index:
                continue
            if req.occurrence != "*" and req.occurrence != occurrence:
                continue
            out.append(req)
        return out

    def observe(self, func: Any, args: tuple, kwargs: dict, out: Any) -> None:
        name = _NAMES.get(func)
        if name is None:
            name = _NAMES[func] = op_name(func)
        if not (self.ops_all or self.graph) and name not in self.names:
            return
        if name in _SKIP or name.startswith("__get"):
            return
        session = self.session
        frame = session.hooks.current
        if frame is None or frame.recompute:
            return
        index = frame.op_counts.get(name, 0)
        frame.op_counts[name] = index + 1
        recorder = session.recorder
        if self.graph and hasattr(recorder, "on_op"):
            if self.count < self.max_ops:
                node = f"{frame.path or '<root>'}/fn:{name}#{index}@{frame.occurrence}"
                recorder.on_op(node, f"{name} in {frame.path or '<root>'}",
                               [t for _, t in _tree.leaves((args, kwargs))],
                               [t for _, t in _tree.leaves(out)], frame.path.count(".") + 2)
            self.count += 1
            return
        wanted = self._wanted(frame.path, name, index, frame.occurrence)
        record_output = self.ops_all or (self.ops is not None and name in self.ops)
        if not wanted and not record_output:
            return
        if self.count >= self.max_ops:
            if not self.warned_cap:
                self.warned_cap = True
                session.warnings.append(f"op capture stopped after max_ops={self.max_ops} ops")
            return
        self.count += 1
        ports = {r.port for r in wanted} or {"output"}
        if name == "sdpa" and "weights" in ports:
            self._sdpa(frame, index, args, kwargs)
            ports.discard("weights")
        if ports and (record_output or wanted):
            found = out if isinstance(out, torch.Tensor) else (_tree.first_tensor(out) or (None, None))[1]
            if isinstance(found, torch.Tensor):
                key = SiteKey(frame.path, "output", frame.occurrence, name, index)
                # Op outputs never inherit the module's declared axes.
                self._capture(key, frame, found, infer_axes(None, "output", found, None))

    def _capture(self, key: SiteKey, frame: Any, value: torch.Tensor, axes: Axes | None,
                 presliced: bool = False) -> None:
        recorder = self.session.recorder
        if hasattr(recorder, "capture"):
            recorder.capture(key, frame.path, frame.module, value, axes=axes,
                             reduce=not presliced, meta={"op": key.op})
        elif hasattr(recorder, "feed_value"):
            recorder.feed_value(key, frame.module, value, axes)

    def _sdpa(self, frame: Any, index: int, args: tuple, kwargs: dict) -> None:
        names = ("query", "key", "value", "attn_mask", "dropout_p", "is_causal", "scale",
                 "enable_gqa")
        bound = dict(zip(names, args))
        bound.update(kwargs)
        q, k = bound["query"], bound["key"]
        mask = bound.get("attn_mask")
        policy = self.session.policy
        slicing = policy.slice if isinstance(policy.slice, dict) else {}
        if "query" in slicing or "heads" in slicing:
            q_index = slicing.get("query")
            if isinstance(q_index, int):
                q_index = slice(q_index, q_index + 1 if q_index != -1 else None)
            if q_index is not None:
                q = q[..., q_index, :]
                if isinstance(mask, torch.Tensor) and mask.ndim >= 2 and mask.shape[-2] > 1:
                    mask = mask[..., q_index, :]
                if bound.get("is_causal"):
                    # Row-slicing changes causal alignment: build the explicit mask instead.
                    length, source = bound["query"].shape[-2], k.shape[-2]
                    full = torch.ones(length, source, dtype=torch.bool, device=q.device).tril(0)
                    rows = full[q_index]
                    mask = rows if mask is None else (mask & rows if mask.dtype == torch.bool
                                                       else mask.masked_fill(~rows, float("-inf")))
                    bound["is_causal"] = False
            heads = slicing.get("heads")
            if heads is not None:
                kv_ratio = q.shape[-3] // k.shape[-3]
                hsel = slice(heads, heads + 1) if isinstance(heads, int) else heads
                q = q[..., hsel, :, :]
                k = k.repeat_interleave(kv_ratio, dim=-3)[..., hsel, :, :]
                if isinstance(mask, torch.Tensor) and mask.ndim == 4 and mask.shape[1] > 1:
                    mask = mask[:, hsel]
        heads_n = q.shape[-3] if q.ndim >= 3 else 1
        size = q.shape[0] * heads_n * q.shape[-2] * k.shape[-2] * 4
        recorder = self.session.recorder
        if hasattr(recorder, "_check_budget") and policy.keep == "tensors":
            recorder._check_budget(size, SiteKey(frame.path, "weights", frame.occurrence, "sdpa", index))
        weights = sdpa_weights(q, k, mask, bool(bound.get("is_causal", False)), bound.get("scale"),
                               bool(bound.get("enable_gqa", False)))
        if self.session.policy.training and float(bound.get("dropout_p", 0.0) or 0.0) > 0:
            note = "attention weights are pre-dropout (training mode with dropout_p > 0)"
            if note not in self.session.warnings:
                self.session.warnings.append(note)
        key = SiteKey(frame.path, "weights", frame.occurrence, "sdpa", index)
        self._capture(key, frame, weights, ATTENTION_AXES, presliced=True)

    # -- nn.MultiheadAttention ----------------------------------------------------------------

    def _mha_pre(self, frame: Any, module: nn.Module, args: tuple, kwargs: dict) -> Any:
        if frame.recompute or frame.path not in self.mha_paths or not isinstance(
                module, nn.MultiheadAttention):
            return None
        merged = dict(zip(_MHA_PARAMS, args))
        merged.update(kwargs)
        wanted = bool(merged.get("need_weights", True))
        average = bool(merged.get("average_attn_weights", True))
        self._mha_state.setdefault(id(module), []).append((wanted, average))
        merged["need_weights"] = True
        merged["average_attn_weights"] = False
        positional = tuple(merged.pop(n) for n in _MHA_PARAMS[:3])
        return positional, merged

    def _mha_post(self, frame: Any, module: nn.Module, args: tuple, kwargs: dict, output: Any) -> Any:
        if frame.recompute or frame.path not in self.mha_paths or not isinstance(
                module, nn.MultiheadAttention):
            return None
        stack = self._mha_state.get(id(module))
        wanted, average = stack.pop() if stack else (False, True)
        if not isinstance(output, tuple) or len(output) != 2 or output[1] is None:
            return None
        attn, weights = output
        if weights.ndim == 3:  # unbatched input: [H, L, S]
            weights = weights[None]
        index = frame.op_counts.get("mha", 0)
        frame.op_counts["mha"] = index + 1
        if any(r.op == "mha" and r.port == "weights" and (r.occurrence == "*" or
                                                           r.occurrence == frame.occurrence)
               for r in self.requests if r.path == frame.path):
            key = SiteKey(frame.path, "weights", frame.occurrence, "mha", index)
            self._capture(key, frame, weights.detach(), ATTENTION_AXES)
        restored = None if not wanted else (output[1].mean(dim=-3) if average else output[1])
        return attn, restored


def mha_signature_ok() -> bool:
    params = list(inspect.signature(nn.MultiheadAttention.forward).parameters)[1:]
    return tuple(params[:len(_MHA_PARAMS)]) == _MHA_PARAMS


__all__ = ["ATTENTION_AXES", "FunctionalInterceptor", "op_name", "sdpa_weights"]
