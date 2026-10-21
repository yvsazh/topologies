"""Explicit runtime profiling; arbitrary callable side effects cannot be rolled back."""
from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Mapping
from typing import Any

from topologies.analysis.core import AnalysisContext, AnalysisResult


class RuntimeProfile:
    """Measure supplied inputs as ``target(input)``, ``target(*tuple)``, or ``target(**dict)``.

    Model profiling always uses eval/no-grad and restores every submodule's prior
    training flag. RNG is forked/restored. Arbitrary buffer or external side effects
    cannot be undone. CUDA peak is allocator peak including warmup, reset before run.
    """
    name = "runtime_profile"

    def __init__(self, target: Callable | Any = None, *, inputs: Any = None, warmup: int = 1,
                 iterations: int = 10, tokens_per_call: int | None = None,
                 samples_per_call: int | None = None, model_input_mode: bool = True,
                 device: str | Any | None = None):
        for value, label, minimum in ((warmup, "warmup", 0), (iterations, "iterations", 1)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{label} must be an integer >= {minimum}")
        for value, label in ((tokens_per_call, "tokens_per_call"), (samples_per_call, "samples_per_call")):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"{label} must be a positive integer")
        if target is not None and not callable(target):
            raise TypeError("target must be callable")
        self.target = target
        self.inputs = inputs
        self.warmup = warmup
        self.iterations = iterations
        self.tokens_per_call = tokens_per_call
        self.samples_per_call = samples_per_call
        self.model_input_mode = model_input_mode
        self.device = device

    def run(self, data: Any, context: AnalysisContext) -> AnalysisResult:
        import torch

        target = self.target if self.target is not None else data
        inputs = self.inputs if self.inputs is not None else context.parameters.get("inputs")
        if inputs is None:
            raise ValueError("explicit inputs are required")
        if not callable(target):
            raise TypeError("profile target must be callable")
        is_model = isinstance(target, torch.nn.Module)
        modules = list(target.modules()) if is_model else []
        flags = [module.training for module in modules]
        selected_device = self._select_device(target, inputs, torch)
        cuda_devices = [(selected_device.index if selected_device.index is not None
                         else torch.cuda.current_device())] if selected_device.type == "cuda" else []
        samples: list[float] = []
        peak_bytes = None
        try:
            with torch.random.fork_rng(devices=cuda_devices, enabled=True):
                if is_model:
                    target.eval()
                if selected_device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(selected_device)
                with torch.no_grad() if is_model else contextlib.nullcontext():
                    for _ in range(self.warmup):
                        self._call(target, inputs)
                    for _ in range(self.iterations):
                        self._synchronize(selected_device, torch)
                        start = time.perf_counter()
                        self._call(target, inputs)
                        self._synchronize(selected_device, torch)
                        samples.append(time.perf_counter() - start)
                if selected_device.type == "cuda":
                    peak_bytes = torch.cuda.max_memory_allocated(selected_device)
        finally:
            for module, flag in zip(modules, flags):
                module.training = flag

        mean = sum(samples) / len(samples)
        payload = {"latency_seconds": samples, "mean_latency_seconds": mean,
                   "throughput_calls_per_second": 1 / mean if mean else None,
                   "tokens_per_second": self.tokens_per_call / mean if self.tokens_per_call and mean else None,
                   "samples_per_second": self.samples_per_call / mean if self.samples_per_call and mean else None,
                   "peak_memory_bytes": peak_bytes,
                   "peak_memory_kind": "measured_cuda_allocator_peak_including_warmup" if peak_bytes is not None else "unknown",
                   "device": str(selected_device)}
        return AnalysisResult(payload, kind="measured", name=self.name,
                              assumptions=["explicit inputs, warmup, and iteration counts",
                                           "CUDA peak allocator statistics reset before warmup; reset is a global allocator side effect",
                                           "CPU peak allocator memory is unknown",
                                           "RNG state is restored; arbitrary callable buffer/external side effects cannot be rolled back"])

    def _call(self, target: Callable, inputs: Any) -> Any:
        if isinstance(inputs, Mapping):
            return target(**inputs)
        if isinstance(inputs, tuple):
            return target(*inputs)
        return target(inputs)

    def _select_device(self, target: Any, inputs: Any, torch):
        if self.device is not None:
            return torch.device(self.device)
        if isinstance(target, torch.nn.Module):
            for tensor in (*target.parameters(), *target.buffers()):
                return tensor.device
        def tensor_device(obj):
            if isinstance(obj, torch.Tensor):
                return obj.device
            if isinstance(obj, Mapping):
                return next((d for value in obj.values() if (d := tensor_device(value)) is not None), None)
            if isinstance(obj, (list, tuple)):
                return next((d for value in obj if (d := tensor_device(value)) is not None), None)
            return None
        return tensor_device(inputs) or torch.device("cpu")

    @staticmethod
    def _synchronize(device, torch) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()
