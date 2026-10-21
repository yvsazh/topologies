"""Measured prefill and decode latency for models with a KV cache.

Constructing :class:`DecodeProfile` runs the model. Peak memory is the CUDA
allocator peak of the measured iterations; it is ``0`` on CPU.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from topologies.build.params import count_params


@dataclass
class DecodeProfile:
    """Timing of ``init_cache`` + ``step`` for one model.

    ``cache_bytes`` is the live cache after a prefill. ``cache_bytes_per_token``
    divides that by the prompt length. ``flops_per_token`` is one prefill
    forward divided by ``batch_size * prompt_len``, or ``2 * params`` when the
    flop counter returns nothing.
    """

    prefill_tok_s: float
    decode_tok_s: float
    ms_per_token_p50: float
    ms_per_token_p90: float
    peak_mem_mb: float
    cache_bytes: int
    cache_bytes_per_token: float
    params: int
    flops_per_token: float

    def __init__(self, model: nn.Module, *, batch_size: int = 1, prompt_len: int = 512,
                 new_tokens: int = 256, device: torch.device | str | None = None,
                 dtype: torch.dtype | None = None, warmup: int = 3, iterations: int = 10,
                 threads: int | None = None) -> None:
        if not callable(getattr(model, "init_cache", None)) or not callable(getattr(model, "step", None)):
            raise TypeError("DecodeProfile needs a model with init_cache and step.")
        _check_int("batch_size", batch_size, 1)
        _check_int("prompt_len", prompt_len, 1)
        _check_int("new_tokens", new_tokens, 1)
        _check_int("warmup", warmup, 0)
        _check_int("iterations", iterations, 1)
        limit = getattr(model, "max_seq", None)
        if isinstance(limit, int) and prompt_len + new_tokens > limit:
            raise ValueError(
                f"prompt_len + new_tokens ({prompt_len + new_tokens}) exceeds max_seq ({limit})."
            )
        vocab = int(getattr(model, "vocab_size", 0) or 0)
        if vocab < 1:
            raise ValueError("model.vocab_size must be positive.")
        device_t = _device_of(model) if device is None else torch.device(device)
        if device is not None or dtype is not None:
            model.to(device=device_t, dtype=dtype)
        was = model.training
        previous_threads = torch.get_num_threads()
        if threads is not None:
            _check_int("threads", threads, 1)
            torch.set_num_threads(threads)
        model.eval()
        try:
            prefill_s, decode_s, cache_bytes, peak = _measure(
                model, batch_size, prompt_len, new_tokens, vocab, device_t, warmup, iterations
            )
        finally:
            model.train(was)
            if threads is not None:
                torch.set_num_threads(previous_threads)
        mean_prefill = sum(prefill_s) / len(prefill_s)
        mean_decode = sum(decode_s) / len(decode_s)
        ms = sorted(t * 1e3 for t in decode_s)
        object.__setattr__(self, "prefill_tok_s", (batch_size * prompt_len) / mean_prefill)
        object.__setattr__(self, "decode_tok_s", batch_size / mean_decode)
        object.__setattr__(self, "ms_per_token_p50", _quantile(ms, 0.50))
        object.__setattr__(self, "ms_per_token_p90", _quantile(ms, 0.90))
        object.__setattr__(self, "peak_mem_mb", peak)
        object.__setattr__(self, "cache_bytes", cache_bytes)
        object.__setattr__(self, "cache_bytes_per_token", cache_bytes / prompt_len)
        params = count_params(model)
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "flops_per_token", _flops_per_token(
            model, batch_size, prompt_len, vocab, device_t, params
        ))

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _measure(model: nn.Module, batch: int, prompt_len: int, new_tokens: int, vocab: int,
             device: torch.device, warmup: int, iterations: int
             ) -> tuple[list[float], list[float], int, float]:
    prefill_s: list[float] = []
    decode_s: list[float] = []
    cache_bytes = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for i in range(warmup + iterations):
            tokens = torch.randint(0, vocab, (batch, prompt_len), device=device)
            cache = model.init_cache(batch, device=device, dtype=_dtype_of(model))
            elapsed = _elapsed(device, lambda tokens=tokens, cache=cache: model.step(tokens, cache))
            keep = i >= warmup
            if keep:
                prefill_s.append(elapsed)
                cache_bytes = _tensor_bytes(cache)
            for _ in range(new_tokens):
                nxt = torch.randint(0, vocab, (batch, 1), device=device)
                elapsed = _elapsed(device, lambda nxt=nxt, cache=cache: model.step(nxt, cache))
                if keep:
                    decode_s.append(elapsed)
    peak = 0.0
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return prefill_s, decode_s, cache_bytes, peak


def _flops_per_token(model: nn.Module, batch: int, prompt_len: int, vocab: int,
                     device: torch.device, params: int) -> float:
    tokens = torch.randint(0, vocab, (batch, prompt_len), device=device)
    total = 0.0
    try:
        from topologies.experiments.costs import flop_counter
        counter = flop_counter()
        with torch.no_grad(), counter:
            model(tokens)
        total = float(counter.get_total_flops())
    except (RuntimeError, AttributeError, TypeError):
        total = 0.0
    if total <= 0:
        total = 2.0 * params * batch * prompt_len
    return total / (batch * prompt_len)


def _elapsed(device: torch.device, fn) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - start


def _tensor_bytes(obj: object, seen: set[int] | None = None) -> int:
    seen = seen or set()
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_bytes(item, seen) for item in obj)
    if hasattr(obj, "__dict__"):
        return sum(_tensor_bytes(value, seen) for value in vars(obj).values())
    return 0


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    index = round((len(sorted_values) - 1) * q)
    return sorted_values[max(0, min(len(sorted_values) - 1, index))]


def _device_of(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _dtype_of(model: nn.Module) -> torch.dtype:
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float32


def _check_int(name: str, value: int, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
