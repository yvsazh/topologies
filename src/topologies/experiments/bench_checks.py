"""Property checks for ``topologies.experiments.bench`` (each returns
``{"status": "pass" | "fail" | "n/a", "message": ...}`` and never raises)."""

from __future__ import annotations

import copy
import inspect
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn as nn

LENGTH_ATTRS = ("max_seq", "max_len", "context_length", "max_seq_len", "block_size",
                "max_position_embeddings", "window_max")
AUTO_CHECKS = ("shape", "finite", "grad", "causal", "mask", "batch_independence",
               "determinism", "length_generalization", "dtype")
ALL_CHECKS = (*AUTO_CHECKS, "compile")


def _params_of(fn: Any) -> tuple[set[str], bool]:
    try:
        ps = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return set(), True
    return ({p.name for p in ps if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)},
            any(p.kind is p.VAR_KEYWORD for p in ps))


def _out(o: Any) -> torch.Tensor:
    if isinstance(o, torch.Tensor):
        return o
    if isinstance(o, dict):
        o = list(o.values())
    for v in o:
        if isinstance(v, torch.Tensor):
            return v
    raise TypeError(f"forward returned {type(o).__name__} without a tensor")


def _accepts_mask(module: nn.Module) -> bool:
    names, _ = _params_of(module.forward)
    return "mask" in names


def _close(a: torch.Tensor, b: torch.Tensor, tol: float = 1e-5) -> bool:
    if a.shape != b.shape:
        return False
    return bool(torch.allclose(a.float(), b.float(), atol=tol, rtol=tol))


def _construction_length(module: nn.Module) -> int | None:
    for attr in LENGTH_ATTRS:
        v = getattr(module, attr, None)
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            return v
    return None


# ---------------------------------------------------------------- checks


def run_checks(module: nn.Module, inputs: Any, size: int, device: torch.device, *,
               names: Sequence[str], causal: bool, expect_shape: Any = None
               ) -> dict[str, dict[str, str]]:
    """``{check: {"status": "pass"|"fail"|"n/a", "message": ...}}``; never raises."""
    out: dict[str, dict[str, str]] = {}

    def res(status: str, message: str = "") -> dict[str, str]:
        return {"status": status, "message": message}

    seq = inputs.kind == "sequence"
    for name in names:
        try:
            out[name] = _CHECKS[name](module, inputs, size, device, causal=causal, seq=seq,
                                      expect_shape=expect_shape, res=res)
        except Exception as e:  # noqa: BLE001
            out[name] = res("fail", f"{type(e).__name__}: {str(e).splitlines()[0][:200]}")
    return out


def _c_shape(m, inputs, size, device, *, expect_shape, res, **_):
    args, kw = inputs.make(size, device)
    m.eval()
    with torch.no_grad():
        y = _out(m(*args, **kw))
    want = tuple(expect_shape) if expect_shape is not None else (
        tuple(args[0].shape) if args else None)
    if want is None:
        return res("n/a", "no reference input")
    return res("pass") if tuple(y.shape) == want else res(
        "fail", f"output {tuple(y.shape)} != expected {want}")


def _c_finite(m, inputs, size, device, *, res, **_):
    m.eval()
    with torch.no_grad():
        for scale in (1.0, 100.0):
            args, kw = inputs.make(size, device, scale=scale)
            y = _out(m(*args, **kw))
            if not bool(torch.isfinite(y).all()):
                return res("fail", f"non-finite output for {scale:g} * randn")
    return res("pass")


def _c_grad(m, inputs, size, device, *, res, **_):
    m.train()
    m.zero_grad(set_to_none=True)
    args, kw = inputs.make(size, device)
    _out(m(*args, **kw)).float().pow(2).mean().backward()
    bad, dead = [], []
    for n, p in m.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None or not bool(p.grad.abs().sum() > 0):
            dead.append(n)
        elif not bool(torch.isfinite(p.grad).all()):
            bad.append(n)
    m.zero_grad(set_to_none=True)
    if bad:
        return res("fail", "non-finite grads: " + ", ".join(bad[:6]))
    if dead:
        return res("fail", "dead parameters (no or zero grad): " + ", ".join(dead[:6]))
    return res("pass")


def _c_causal(m, inputs, size, device, *, causal, seq, res, **_):
    if not seq:
        return res("n/a", "not a sequence component")
    if not causal:
        return res("n/a", "not declared causal (bench.make(..., causal=True))")
    m.eval()
    args, kw = inputs.make(size, device)
    x = args[0]
    k = x.shape[1] // 2
    x2 = x.clone()
    x2[:, k + 1:] = torch.randn_like(x2[:, k + 1:]) * 3.0
    with torch.no_grad():
        a, b = _out(m(x, *args[1:], **kw)), _out(m(x2, *args[1:], **kw))
    if _close(a[:, :k + 1], b[:, :k + 1], 1e-4):
        return res("pass")
    diff = float((a[:, :k + 1] - b[:, :k + 1]).abs().max())
    return res("fail", f"outputs at positions <= {k} change when later inputs change "
                       f"(max diff {diff:.3g})")


def _c_mask(m, inputs, size, device, *, seq, res, **_):
    if not seq:
        return res("n/a", "not a sequence component")
    if not _accepts_mask(m):
        return res("n/a", "forward takes no mask")
    m.eval()
    args, _kw = inputs.make(size, device)
    x = args[0]
    T = x.shape[1]
    real = max(1, T - T // 4)
    mask = torch.ones(x.shape[0], T, dtype=torch.long, device=x.device)
    mask[:, real:] = 0
    x2 = x.clone()
    x2[:, real:] = torch.randn_like(x2[:, real:]) * 3.0
    with torch.no_grad():
        a, b = _out(m(x, mask=mask)), _out(m(x2, mask=mask))
    if _close(a[:, :real], b[:, :real], 1e-4):
        return res("pass")
    return res("fail", "real positions depend on the content of padded positions")


def _c_batch(m, inputs, size, device, *, res, **_):
    args, kw = inputs.make(size, device)
    x = args[0] if args else None
    if x is None or x.shape[0] < 2:
        return res("n/a", "needs a batch of >= 2 tensor inputs")
    m.eval()
    x2 = x.clone()
    x2[1:] = torch.randn_like(x2[1:]) * 3.0
    with torch.no_grad():
        a, b = _out(m(x, *args[1:], **kw)), _out(m(x2, *args[1:], **kw))
    return res("pass") if _close(a[:1], b[:1], 1e-4) else res(
        "fail", "sample 0 changes when other samples change (eval mode)")


def _c_determinism(m, inputs, size, device, *, res, **_):
    from topologies._execution import execution_rng

    m.train()
    outs = []
    for _ in range(2):
        with execution_rng(seed=0), torch.no_grad():  # scoped: caller's RNG untouched
            args, kw = inputs.make(size, device)
            outs.append(_out(m(*args, **kw)).clone())
    m.eval()
    return res("pass") if torch.equal(outs[0], outs[1]) else res(
        "fail", "same seed gives different outputs")


def _c_length(m, inputs, size, device, *, seq, res, **_):
    if not seq:
        return res("n/a", "not a sequence component")
    limit = _construction_length(m)
    if limit is None:
        return res("n/a", "no construction-time length")
    m.eval()
    args, kw = inputs.make(2 * limit, device)
    try:
        with torch.no_grad():
            y = _out(m(*args, **kw))
    except Exception as e:  # noqa: BLE001
        return res("fail", f"fails at 2x construction length {2 * limit}: "
                           f"{type(e).__name__}")
    return res("pass", f"ok at {2 * limit} tokens") if bool(torch.isfinite(y).all()) else \
        res("fail", f"non-finite at {2 * limit} tokens")


def _c_dtype(m, inputs, size, device, *, res, **_):
    dtypes = [torch.bfloat16] + ([torch.float16] if device.type == "cuda" else [])
    m.eval()
    args, kw = inputs.make(size, device)
    with torch.no_grad():
        ref = _out(m(*args, **kw)).float()
    msgs = []
    for dt in dtypes:
        mm = copy.deepcopy(m).to(dtype=dt)
        a2 = tuple(a.to(dt) if isinstance(a, torch.Tensor) and a.is_floating_point() else a
                   for a in args)
        with torch.no_grad():
            y = _out(mm(*a2, **kw)).float()
        if not bool(torch.isfinite(y).all()):
            return res("fail", f"{dt} output is not finite")
        rel = float((y - ref).norm() / ref.norm().clamp_min(1e-12))
        if rel >= 5e-2:
            return res("fail", f"{dt} relative error {rel:.3g} >= 5e-2")
        msgs.append(f"{str(dt).replace('torch.', '')} rel err {rel:.2g}")
    return res("pass", "; ".join(msgs))


def _c_compile(m, inputs, size, device, *, res, **_):
    compiled = torch.compile(m)
    args, kw = inputs.make(size, device)
    m.eval()
    with torch.no_grad():
        y = _out(compiled(*args, **kw))
    return res("pass") if bool(torch.isfinite(y).all()) else res("fail", "non-finite output")


_CHECKS: dict[str, Callable[..., dict[str, str]]] = {
    "shape": _c_shape, "finite": _c_finite, "grad": _c_grad, "causal": _c_causal,
    "mask": _c_mask, "batch_independence": _c_batch, "determinism": _c_determinism,
    "length_generalization": _c_length, "dtype": _c_dtype, "compile": _c_compile}


__all__ = ["ALL_CHECKS", "AUTO_CHECKS", "run_checks"]
