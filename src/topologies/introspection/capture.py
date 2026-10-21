"""Capture policy, per-call summaries, the recorder and tensor storage backends."""
from __future__ import annotations

import inspect
import math
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn

from topologies.introspection import _tree
from topologies.introspection.errors import CaptureBudgetExceeded, SemanticsRequired
from topologies.introspection.semantics import Axes, SemanticSource, infer_axes
from topologies.introspection.sites import SiteKey, SiteRequest

SUMMARY_FIELDS = ("min", "max", "mean", "std", "abs_mean", "l2", "zero_frac", "nonfinite_frac")
INPLACE_TYPES = (nn.ReLU, nn.Dropout, nn.Hardtanh, nn.LeakyReLU, nn.ReLU6, nn.SiLU, nn.ELU,
                 nn.SELU, nn.CELU, nn.Threshold, nn.Hardswish, nn.Hardsigmoid, nn.Mish,
                 nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)


@dataclass(frozen=True)
class CapturePolicy:
    """Every capture knob in one immutable place (see §5.1 of the design)."""

    keep: Literal["none", "summary", "tensors"] = "summary"
    ports: tuple[str, ...] = ("output",)
    occurrences: Literal["all", "first", "last"] | tuple[int, ...] = "all"
    store_device: str = "cpu"
    store_dtype: torch.dtype | None = None
    slice: Callable | Mapping[str, Any] | None = None
    sample_positions: int | None = None
    max_bytes: int | None = 2**30
    spill_dir: str | Path | None = None
    max_sites: int = 512
    grads: bool = False
    clone: Literal["auto", "always", "never"] = "auto"
    timing: bool = False
    training: bool = False
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.keep not in ("none", "summary", "tensors"):
            raise ValueError("keep must be 'none', 'summary' or 'tensors'")
        if self.clone not in ("auto", "always", "never"):
            raise ValueError("clone must be 'auto', 'always' or 'never'")
        if isinstance(self.ports, str):
            object.__setattr__(self, "ports", (self.ports,))
        if isinstance(self.occurrences, list):
            object.__setattr__(self, "occurrences", tuple(self.occurrences))
        if self.max_bytes is not None and self.max_bytes < 0:
            raise ValueError("max_bytes must be >= 0 or None")

    @classmethod
    def build(cls, base: CapturePolicy | None = None, **overrides: Any) -> CapturePolicy:
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(overrides) - known)
        if unknown:
            raise TypeError(f"unknown CapturePolicy option(s): {', '.join(unknown)}")
        values = {k: v for k, v in overrides.items() if v is not None or k in (
            "max_bytes", "store_dtype", "slice", "sample_positions", "spill_dir", "seed")}
        return replace(base or cls(), **values)

    def to_dict(self) -> dict[str, Any]:
        out = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, torch.dtype):
                value = str(value)
            elif callable(value):
                value = getattr(value, "__name__", "callable")
            elif isinstance(value, Path):
                value = str(value)
            elif isinstance(value, Mapping):
                value = {str(k): repr(v) for k, v in value.items()}
            out[f.name] = list(value) if isinstance(value, tuple) else value
        return out


_ONES: dict[tuple[int, torch.device], torch.Tensor] = {}


def _ones(n: int, device: torch.device) -> torch.Tensor:
    key = (n, device)
    ones = _ONES.get(key)
    if ones is None:
        if len(_ONES) > 64:
            _ONES.clear()
        ones = _ONES[key] = torch.ones(n, device=device)
    return ones


def summary_vector(value: torch.Tensor) -> torch.Tensor:
    """Eight float32 statistics computed on the device with few fused reductions.

    Sum and sum of squares use BLAS dot products; variance is ``E[x²] - E[x]²`` in float64
    on the two scalars (cheap, but subject to cancellation when ``|mean| >> std``). On CPU
    the non-finite count only runs when min/max are not finite (no device sync there);
    on accelerators it always runs to avoid a host sync.
    """
    x = value.detach()
    if x.is_complex():
        x = x.abs()
    x = x.float()
    n = x.numel()
    if n == 0:
        return torch.full((len(SUMMARY_FIELDS),), float("nan"), device=x.device)
    flat = x.reshape(-1)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    lo, hi = torch.aminmax(flat)
    total = torch.dot(flat, _ones(n, flat.device)).double()
    squares = torch.dot(flat, flat).double()
    mean = total / n
    var = (squares / n - mean * mean).clamp_min(0)
    abs_mean = torch.linalg.vector_norm(flat, ord=1) / n
    zeros = (n - torch.count_nonzero(flat)).float() / n
    if flat.device.type == "cpu" and bool(torch.isfinite(lo) & torch.isfinite(hi)):
        nonfinite = torch.zeros((), device=flat.device)
    else:
        nonfinite = (~torch.isfinite(flat)).float().mean()
    return torch.stack([lo, hi, mean.float(), var.sqrt().float(), abs_mean, squares.sqrt().float(),
                        zeros, nonfinite])


def summary_dict(vector: torch.Tensor | list[float]) -> dict[str, float]:
    values = vector.tolist() if isinstance(vector, torch.Tensor) else list(vector)
    return {name: (float(v) if math.isfinite(v) else (str(v))) for name, v in zip(SUMMARY_FIELDS, values)}


def tensor_bytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    return int(math.prod(shape)) * torch.empty((), dtype=dtype).element_size()


# --------------------------------------------------------------------------- storage


class MemoryStore:
    """Kept tensors on ``store_device``; GPU→CPU copies are non-blocking into pinned memory."""

    def __init__(self, policy: CapturePolicy):
        self.policy = policy
        self.tensors: dict[Any, torch.Tensor] = {}
        self.bytes = 0
        self._pending_sync = False

    def convert(self, value: torch.Tensor) -> torch.Tensor:
        policy = self.policy
        out = value.detach()
        if policy.store_dtype is not None and out.is_floating_point():
            out = out.to(policy.store_dtype)
        if policy.store_device != "same" and out.device != torch.device(policy.store_device):
            target = torch.device(policy.store_device)
            if out.device.type == "cuda" and target.type == "cpu":
                pinned = torch.empty(out.shape, dtype=out.dtype, pin_memory=True)
                pinned.copy_(out, non_blocking=True)
                self._pending_sync = True
                out = pinned
            else:
                out = out.to(target)
        return out

    def put(self, key: Any, value: torch.Tensor) -> torch.Tensor:
        self.tensors[key] = value
        self.bytes += value.numel() * value.element_size()
        return value

    def drop(self, key: Any) -> None:
        old = self.tensors.pop(key, None)
        if old is not None:
            self.bytes -= old.numel() * old.element_size()

    def get(self, key: Any) -> torch.Tensor:
        return self.tensors[key]

    def finalize(self) -> None:
        if self._pending_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._pending_sync = False


class SpillStore(MemoryStore):
    """Like ``MemoryStore`` until the budget is reached; then tensors spill to disk shards.

    Shards are ``spill_dir/<site-hash>/<occurrence>-<n>.pt`` written with ``torch.save`` and
    loadable with ``weights_only=True``. ``Trace.tensor`` loads them on demand.
    """

    def __init__(self, policy: CapturePolicy, run_tag: int = 0):
        super().__init__(policy)
        self.spill = Path(policy.spill_dir)  # type: ignore[arg-type]
        self.paths: dict[Any, Path] = {}
        self.run_tag = run_tag
        self.spilled_bytes = 0

    def put(self, key: Any, value: torch.Tensor) -> torch.Tensor:
        size = value.numel() * value.element_size()
        budget = self.policy.max_bytes
        if budget is None or self.bytes + size <= budget:
            return super().put(key, value)
        import hashlib

        digest = hashlib.sha1(str(getattr(key, "site", key)).encode()).hexdigest()[:12]
        folder = self.spill / digest
        folder.mkdir(parents=True, exist_ok=True)
        occurrence = getattr(key, "occurrence", 0)
        path = folder / f"{occurrence}-{self.run_tag}-{len(self.paths)}.pt"
        torch.save({"value": value.contiguous().cpu()}, path)
        self.paths[key] = path
        self.spilled_bytes += size
        return value

    def drop(self, key: Any) -> None:
        super().drop(key)
        path = self.paths.pop(key, None)
        if path is not None and path.exists():
            path.unlink()

    def loader(self, key: Any) -> Callable[[], torch.Tensor] | None:
        path = self.paths.get(key)
        if path is None:
            return None
        return lambda: torch.load(path, weights_only=True)["value"]


def make_store(policy: CapturePolicy) -> MemoryStore:
    return SpillStore(policy) if policy.spill_dir is not None else MemoryStore(policy)


def budget_message(needed: int, budget: int, where: str) -> str:
    return (f"capture would keep {needed / 2**20:.1f} MiB of tensors ({where}), above "
            f"max_bytes={budget / 2**20:.1f} MiB. Remedies: narrow sites=..., keep only part of "
            "each tensor with slice={'position': -1} or sample_positions=N, halve memory with "
            "store_dtype=torch.float16, raise max_bytes=..., or set spill_dir=... to spill "
            "shards to disk.")


# --------------------------------------------------------------------------- recorder


@dataclass
class PendingRecord:
    key: SiteKey
    module_path: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    axes: Axes | None
    summary: torch.Tensor | None = None
    has_value: bool = False
    grad_summary: torch.Tensor | None = None
    has_grad: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


class Recorder:
    """Turns hook callbacks into records according to the policy and resolved requests."""

    def __init__(self, policy: CapturePolicy, requests: list[SiteRequest], *,
                 axes_for: Callable[[str, str, nn.Module | None, torch.Tensor], Axes | None],
                 batch_size: int | None = None, store: Any = None):
        self.policy = policy
        self.axes_for = axes_for
        self.batch_size = batch_size
        self.store = store or make_store(policy)
        self.by_path: dict[str, list[SiteRequest]] = {}
        for req in requests:
            if req.op is None:
                self.by_path.setdefault(req.path, []).append(req)
        self.op_requests = [r for r in requests if r.op is not None]
        self.records: dict[SiteKey, PendingRecord] = {}
        self.warnings: list[str] = []
        self.assumptions: list[str] = []
        self._aliased: dict[SiteKey, tuple[torch.Tensor, int | None]] = {}
        self._grad_handles: list[Any] = []
        self.watched: list[torch.Tensor] = []
        self.grad_enabled = False
        self._warned: set[str] = set()
        self._signatures: dict[int, inspect.Signature | None] = {}

    # -- requests ---------------------------------------------------------------

    def wants(self, path: str) -> bool:
        return path in self.by_path

    def _occurrence_ok(self, req: SiteRequest, occurrence: int) -> bool:
        if req.occurrence != "*":
            return req.occurrence == occurrence
        occ = self.policy.occurrences
        if occ == "first":
            return occurrence == 0
        if isinstance(occ, tuple):
            return occurrence in occ
        return True

    # -- hook entry points --------------------------------------------------------

    def on_pre(self, path: str, occurrence: int, module: nn.Module, args: tuple,
               kwargs: Mapping[str, Any] | None) -> None:
        for req in self.by_path.get(path, ()):
            if not req.port.startswith(("input", "grad_input")) or not self._occurrence_ok(
                    req, occurrence):
                continue
            value = self._select_input(module, req.port.removeprefix("grad_"), args, kwargs or {})
            if value is None:
                self._warn_once(f"{path}:{req.port}", f"site {path or '<root>'}:{req.port} "
                                "has no tensor value; nothing recorded")
                continue
            key = SiteKey(path, req.port, occurrence)
            if req.port.startswith("grad_"):
                self._watch_grad(key, path, module, value)
            else:
                self.capture(key, path, module, value)

    def on_post(self, path: str, occurrence: int, module: nn.Module, output: Any) -> Any:
        """Capture output ports; returns a replacement output only when a gradient is watched.

        A watched output is handed downstream as an identical clone, so a following in-place
        op (``ReLU(inplace=True)``) cannot rewrite the tensor whose gradient is recorded.
        """
        watched_before = len(self.watched)
        for req in self.by_path.get(path, ()):
            if not req.port.startswith(("output", "grad_output")) or not self._occurrence_ok(
                    req, occurrence):
                continue
            port = req.port.removeprefix("grad_")
            sub = port[len("output."):] if port.startswith("output.") else None
            try:
                value = _tree.get_path(output, sub) if sub else output
            except (KeyError, IndexError, AttributeError, ValueError):
                self._warn_once(f"{path}:{req.port}", f"{path}: output has no path {sub!r}")
                continue
            if not isinstance(value, torch.Tensor):
                found = _tree.first_tensor(value)
                if found is None:
                    continue
                self._note_primary(path, found[0])
                value = found[1]
            key = SiteKey(path, req.port, occurrence)
            if req.port.startswith("grad_"):
                self._watch_grad(key, path, module, value)
            else:
                self.capture(key, path, module, value)
        if self.policy.grads and self.grad_enabled:
            found = output if isinstance(output, torch.Tensor) else (
                (_tree.first_tensor(output) or (None, None))[1])
            if isinstance(found, torch.Tensor) and path in self.by_path:
                key = SiteKey(path, "grad_output", occurrence)
                if key not in self.records:
                    self._watch_grad(key, path, module, found)
        if isinstance(output, torch.Tensor) and any(t is output for t in
                                                    self.watched[watched_before:]):
            return output.clone()
        return None

    def _note_primary(self, path: str, leaf: str) -> None:
        self._warn_once(f"primary:{path}", None)
        note = (f"port 'output' of {path or '<root>'} is a structured value; the first tensor "
                f"leaf (output.{leaf}) is recorded")
        if note not in self.assumptions:
            self.assumptions.append(note)

    def _select_input(self, module: nn.Module, port: str, args: tuple,
                      kwargs: Mapping[str, Any]) -> torch.Tensor | None:
        if port == "input":
            found = _tree.first_tensor((args, dict(kwargs)))
            return None if found is None else found[1]
        spec = port[len("input."):]
        head, _, rest = spec.partition(".")
        if head.isdigit():
            index = int(head)
            value = args[index] if index < len(args) else None
        elif head in kwargs:
            value = kwargs[head]
        else:
            value = self._bind(module, args, kwargs).get(head)
        if value is not None and rest:
            value = _tree.get_path(value, rest)
        if value is not None and not isinstance(value, torch.Tensor):
            found = _tree.first_tensor(value)
            value = None if found is None else found[1]
        return value

    def _bind(self, module: nn.Module, args: tuple, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        key = id(type(module))
        if key not in self._signatures:
            try:
                self._signatures[key] = inspect.signature(module.forward)
            except (TypeError, ValueError):
                self._signatures[key] = None
        sig = self._signatures[key]
        if sig is None:
            return {}
        try:
            return dict(sig.bind_partial(*args, **kwargs).arguments)
        except TypeError:
            return {}

    # -- capture ------------------------------------------------------------------

    def capture(self, key: SiteKey, module_path: str, module: nn.Module | None,
                value: torch.Tensor, *, axes: Axes | None = None,
                meta: Mapping[str, Any] | None = None, reduce: bool = True) -> None:
        policy = self.policy
        if axes is None:
            axes = self.axes_for(module_path, key.port, module, value)
        if policy.occurrences == "last":
            for old in [k for k in self.records if k.path == key.path and k.port == key.port
                        and k.op == key.op and k.op_index == key.op_index]:
                self.records.pop(old)
                self.store.drop(old)
                self._aliased.pop(old, None)
        record = PendingRecord(key, module_path, tuple(value.shape), str(value.dtype).replace(
            "torch.", ""), str(value.device), axes, meta=dict(meta or {}))
        if policy.keep != "none":
            record.summary = summary_vector(value)
        if policy.keep == "tensors":
            reduced, stored_axes, note = self._reduce(value, axes, key) if reduce else \
                (value, axes, None)
            if note:
                record.meta.update(note, original_shape=tuple(value.shape))
                record.axes = stored_axes
            dtype = policy.store_dtype if policy.store_dtype is not None and \
                reduced.is_floating_point() else reduced.dtype
            self._check_budget(reduced.numel() * torch.empty((), dtype=dtype).element_size(), key)
            stored = self.store.convert(reduced)
            aliased = stored.untyped_storage().data_ptr() == value.untyped_storage().data_ptr() \
                if stored.device == value.device else False
            if aliased and policy.clone == "always":
                stored = stored.clone()
                aliased = False
            self.store.put(key, stored)
            record.has_value = True
            if key in getattr(self.store, "paths", {}):
                aliased = False
            if aliased:
                version = None if torch.is_inference(stored) else stored._version
                self._aliased[key] = (stored, version)
        self.records[key] = record

    def _reduce(self, value: torch.Tensor, axes: Axes | None, key: SiteKey
                ) -> tuple[torch.Tensor, Axes | None, dict[str, Any] | None]:
        """Apply ``slice`` / ``sample_positions`` before copying; returns stored axes too."""
        policy = self.policy
        out, note = value, None
        resolved: Axes | None = None
        if axes is not None and axes.source != SemanticSource.UNKNOWN:
            try:
                resolved = axes.resolve(value.ndim)
            except SemanticsRequired:
                resolved = None
        if policy.slice is not None:
            if callable(policy.slice):
                out = policy.slice(out)
                note = {"sliced": getattr(policy.slice, "__name__", "callable")}
                resolved = None
            else:
                index: list[Any] = [slice(None)] * out.ndim
                applied, dropped = {}, []
                for name, sel in policy.slice.items():
                    axis = resolved.index(name) if resolved is not None else None
                    if axis is None:
                        self._warn_once(f"slice:{key.site}:{name}",
                                        f"slice axis {name!r} is unknown at {key.site}; "
                                        "kept unsliced (declare axes to enable slicing)")
                        continue
                    index[axis] = sel
                    applied[name] = sel if isinstance(sel, int) else repr(sel)
                    if isinstance(sel, int):
                        dropped.append(axis)
                if applied:
                    out = out[tuple(index)]
                    note = {"sliced": applied}
                    if resolved is not None:
                        names = tuple(n for i, n in enumerate(resolved.names) if i not in dropped)
                        resolved = Axes.from_names(names, resolved.source)
        if policy.sample_positions is not None and resolved is not None and out.ndim >= 3 \
                and resolved.batch_axis == 0 and resolved.feature_axis in (-1, out.ndim - 1):
            flat = out.reshape(out.shape[0], -1, out.shape[-1])
            generator = torch.Generator().manual_seed(policy.seed or 0)
            count = min(policy.sample_positions, flat.shape[1])
            picks = torch.randperm(flat.shape[1], generator=generator)[:count].to(out.device)
            out = flat.index_select(1, picks)
            note = {**(note or {}), "sampled_positions": picks.tolist()}
            resolved = Axes.from_names(("batch", "position", "feature"), resolved.source)
        return out, (resolved if note else axes), note

    def _check_budget(self, size: int, key: SiteKey) -> None:
        """Raise before allocating when the running total would exceed ``max_bytes``."""
        budget = self.policy.max_bytes
        if budget is None or self.store.bytes + size <= budget:
            return
        spill = getattr(self.store, "spill", None)
        if spill is not None:
            return
        raise CaptureBudgetExceeded(budget_message(self.store.bytes + size, budget,
                                                   f"running total at {key}"))

    def protect(self, args: tuple) -> None:
        """Called before a known in-place module runs: clone kept tensors it would overwrite."""
        if not self._aliased:
            return
        ptrs = {a.untyped_storage().data_ptr() for a in args if isinstance(a, torch.Tensor)}
        for key, (stored, _) in list(self._aliased.items()):
            if stored.untyped_storage().data_ptr() in ptrs:
                copy = stored.clone()
                self.store.drop(key)
                self.store.put(key, copy)
                del self._aliased[key]

    # -- gradients ----------------------------------------------------------------

    def _watch_grad(self, key: SiteKey, path: str, module: nn.Module | None,
                    value: torch.Tensor) -> None:
        if not (self.grad_enabled and value.requires_grad):
            self._warn_once(f"grad:{key.site}", f"{key.site}: no gradient flows here "
                            "(no objective, or the value does not require grad)")
            return
        axes = self.axes_for(path, "output", module, value)
        record = PendingRecord(key, path, tuple(value.shape), str(value.dtype).replace("torch.", ""),
                               str(value.device), axes)
        self.records[key] = record

        def hook(grad: torch.Tensor, key: SiteKey = key, record: PendingRecord = record) -> None:
            if self.policy.keep != "none":
                record.summary = summary_vector(grad)
            if self.policy.keep == "tensors":
                self._check_budget(grad.numel() * grad.element_size(), key)
                stored = self.store.convert(grad).clone()
                self.store.drop(key)
                self.store.put(key, stored)
                record.has_value = True
        self._grad_handles.append(value.register_hook(hook))
        self.watched.append(value)

    def remove_grad_hooks(self) -> None:
        for handle in self._grad_handles:
            handle.remove()
        self._grad_handles.clear()
        self.watched.clear()

    # -- finishing ------------------------------------------------------------------

    def _warn_once(self, tag: str, message: str | None) -> None:
        if tag in self._warned:
            return
        self._warned.add(tag)
        if message:
            self.warnings.append(message)

    def check_versions(self) -> None:
        for key, (stored, version) in self._aliased.items():
            if version is not None and not torch.is_inference(stored) and stored._version != version:
                self.warnings.append(
                    f"{key}: the kept tensor was modified in place after capture; use "
                    "clone='always' to keep the value as it was produced")

    def finalize(self) -> tuple[dict[SiteKey, PendingRecord], list[str]]:
        self.remove_grad_hooks()
        self.check_versions()
        self.store.finalize()
        return self.records, self.warnings


def default_axes_for(overrides: Callable[[str, str], Axes | None], batch_size: int | None
                     ) -> Callable[[str, str, nn.Module | None, torch.Tensor], Axes | None]:
    def axes_for(path: str, port: str, module: nn.Module | None, value: torch.Tensor) -> Axes:
        declared = overrides(path, port)
        if declared is not None:
            return declared
        return infer_axes(module, port, value, batch_size)
    return axes_for


def warn_dropped(message: str) -> None:
    warnings.warn(message, RuntimeWarning, stacklevel=3)
