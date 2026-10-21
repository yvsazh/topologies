"""Bounded, explicit data and inference-pipeline profiling.

Loader wait is consumer-visible and is not mislabeled as worker transform time.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import os
import queue
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from topologies._execution import ModelCall, model_execution, transfer


def _percentile(samples: list[float], q: float) -> float | None:
    if not samples:
        return None
    values = sorted(samples)
    return values[min(math.ceil(q * len(values)) - 1, len(values) - 1)]


@dataclass
class DataProfile:
    batches: int
    startup_seconds: float
    wait_seconds: list[float]
    elapsed_seconds: float = 0.0
    first_batch_seconds: float | None = None
    warmup_batches: int = 0
    units: dict[str, int] = field(default_factory=dict)
    memory_bytes: int | None = None
    mode: str = "consumer-visible asynchronous wall time"
    loader_settings: dict[str, Any] = field(default_factory=dict)
    worker_events: list[dict[str, Any]] = field(default_factory=list)
    worker_events_dropped: int = 0
    worker_events_pending: int = 0
    unit: str = "batch"
    batch_cardinalities: list[int] = field(default_factory=list)

    @property
    def batches_per_second(self) -> float:
        return self.batches / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0

    @property
    def p50_wait_seconds(self) -> float | None: return _percentile(self.wait_seconds, .50)
    @property
    def p95_wait_seconds(self) -> float | None: return _percentile(self.wait_seconds, .95)

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.update(p50_wait_seconds=self.p50_wait_seconds, p95_wait_seconds=self.p95_wait_seconds,
                     batches_per_second=self.batches_per_second)
        return value


@dataclass
class PipelineProfile:
    data: DataProfile
    preparation_seconds: list[float]
    transfer_seconds: list[float]
    model_seconds: list[float]
    end_to_end_seconds: list[float]
    synchronized_diagnostics: bool
    shapes: list[Any] = field(default_factory=list)
    memory_bytes: int | None = None
    recommendations: list[str] = field(default_factory=list)
    operator_events: list[dict[str, Any]] = field(default_factory=list)
    operator_attribution: str = "inclusive operator averages; not additive with model wall time"
    padding_fractions: list[float] = field(default_factory=list)
    postprocessing_seconds: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        def stats(xs): return {"raw_seconds": xs, "p50_seconds": _percentile(xs, .5), "p95_seconds": _percentile(xs, .95)}
        return {"data": self.data.as_dict(), "preparation": stats(self.preparation_seconds),
                "transfer": stats(self.transfer_seconds), "model": stats(self.model_seconds),
                "postprocessing": stats(self.postprocessing_seconds),
                "end_to_end": stats(self.end_to_end_seconds),
                "synchronized_diagnostics": self.synchronized_diagnostics,
                "shapes": self.shapes, "memory_bytes": self.memory_bytes,
                "recommendations": self.recommendations, "operator_events": self.operator_events,
                "operator_attribution": self.operator_attribution,
                "padding_fraction": {"raw": self.padding_fractions,
                                     "p50": _percentile(self.padding_fractions, .5),
                                     "p95": _percentile(self.padding_fractions, .95)}}


def _settings(loader: Any) -> dict[str, Any]:
    return {key: getattr(loader, key, None) for key in
            ("batch_size", "num_workers", "pin_memory", "persistent_workers", "prefetch_factor", "drop_last")}


class BoundedEventCollector:
    """Spawn-pickleable bounded event channel with explicit feeder-safe accounting.

    A shared semaphore bounds accepted-but-not-drained events. Accepted and drained
    counters—not Queue.empty/qsize—govern reads, so feeder-thread delay cannot make
    an accepted event look absent. A timed-out drain leaves it pending for a later
    drain; worker failure can therefore delay at most ``drain_timeout`` per call.
    """
    def __init__(self, max_events: int = 256, *, context: str = "spawn"):
        if max_events < 1: raise ValueError("max_events must be positive")
        ctx = mp.get_context(context)
        self.queue = ctx.Queue(maxsize=max_events)
        self._slots = ctx.BoundedSemaphore(max_events)
        self._lock = ctx.RLock()
        self._accepted = ctx.Value("L", 0)
        self._consumed = ctx.Value("L", 0)
        self._dropped = ctx.Value("L", 0)

    @property
    def dropped(self) -> int:
        with self._dropped.get_lock(): return int(self._dropped.value)

    @property
    def pending(self) -> int:
        with self._lock: return int(self._accepted.value - self._consumed.value)

    @property
    def accepted(self) -> int:
        with self._lock: return int(self._accepted.value)

    def emit(self, name: str, seconds: float) -> None:
        from torch.utils.data import get_worker_info
        worker = get_worker_info()
        event = {"name": name, "seconds": float(seconds), "pid": os.getpid(),
                 "worker_id": worker.id if worker else None, "kind": "transform" if name.startswith("transform:") else "collate"}
        if not self._slots.acquire(False):
            with self._dropped.get_lock(): self._dropped.value += 1
            return
        # Keep accepted-count publication ordered before a consumer can account
        # the just-enqueued message. put_nowait cannot exceed the shared slot cap.
        with self._lock:
            try: self.queue.put_nowait(event)
            except queue.Full:
                self._slots.release()
                with self._dropped.get_lock(): self._dropped.value += 1
            else:
                self._accepted.value += 1

    def drain(self, limit: int = 4096, *, timeout: float = 0.1) -> list[dict[str, Any]]:
        if limit < 0: raise ValueError("drain limit must be nonnegative")
        if timeout < 0: raise ValueError("timeout must be nonnegative")
        events=[]
        deadline=time.monotonic()+timeout
        for _ in range(limit):
            with self._lock: pending=self._accepted.value-self._consumed.value
            if pending <= 0: break
            remaining=deadline-time.monotonic()
            if remaining <= 0: break
            try: event=self.queue.get(timeout=remaining)
            except queue.Empty: break
            with self._lock: self._consumed.value += 1
            self._slots.release()
            events.append(event)
        return events


class InstrumentedCallable:
    """Top-level, spawn-pickleable wrapper around a named transform/collator."""
    def __init__(self, function: Callable[..., Any], name: str, collector: BoundedEventCollector):
        if not name: raise ValueError("instrumented callable requires a name")
        self.function, self.name, self.collector = function, name, collector

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        start=time.perf_counter()
        try: return self.function(*args, **kwargs)
        finally: self.collector.emit(self.name, time.perf_counter()-start)


def instrument_transform(function: Callable[..., Any], name: str, collector: BoundedEventCollector) -> InstrumentedCallable:
    return InstrumentedCallable(function, f"transform:{name}", collector)


def instrument_collator(function: Callable[..., Any], name: str, collector: BoundedEventCollector) -> InstrumentedCallable:
    return InstrumentedCallable(function, f"collate:{name}", collector)


def _cardinality(batch: Any) -> int | None:
    if isinstance(batch, torch.Tensor) and batch.ndim: return int(batch.shape[0])
    if isinstance(batch, (list, tuple)): return len(batch)
    if isinstance(batch, dict):
        vals=[_cardinality(x) for x in batch.values()]
        return next((n for n in vals if n is not None), None)
    return None


def profile_data(loader: Any, *, max_batches: int = 100, warmup_batches: int = 0,
                 repeats: int = 1, loader_factory: Callable[[], Any] | None = None,
                 event_collector: BoundedEventCollector | None = None,
                 max_event_drain: int = 4096, unit: str = "batch",
                 event_drain_timeout: float = 0.1) -> DataProfile:
    if max_batches < 1 or warmup_batches < 0 or repeats < 1:
        raise ValueError("max_batches/repeats must be positive and warmup_batches nonnegative")
    if repeats > 1 and loader_factory is None:
        raise ValueError("Repeated profiling requires loader_factory; a one-shot iterator cannot replay")
    if max_event_drain < 0: raise ValueError("max_event_drain must be nonnegative")
    if event_drain_timeout < 0: raise ValueError("event_drain_timeout must be nonnegative")
    if not unit: raise ValueError("unit must be a nonempty label")
    waits: list[float] = []
    cardinalities=[]
    startup_samples: list[float] = []
    first_samples: list[float] = []
    batches = 0
    start_all = time.perf_counter()
    for repeat in range(repeats):
        source = loader_factory() if loader_factory else loader
        start = time.perf_counter()
        iterator = iter(source)
        startup_samples.append(time.perf_counter() - start)
        first = None
        for index in range(max_batches + warmup_batches):
            tick = time.perf_counter()
            try: batch=next(iterator)
            except StopIteration: break
            duration = time.perf_counter() - tick
            if first is None: first = time.perf_counter() - start
            if index >= warmup_batches:
                waits.append(duration)
                batches += 1
                cardinalities.append(_cardinality(batch))
        if first is not None: first_samples.append(first)
        if loader_factory is None and repeat == 0: break
    elapsed = time.perf_counter() - start_all
    events=event_collector.drain(max_event_drain, timeout=event_drain_timeout) if event_collector else []
    return DataProfile(batches, statistics.median(startup_samples), waits, elapsed,
                       statistics.median(first_samples) if first_samples else None,
                       warmup_batches, {}, None, loader_settings=_settings(loader),
                       worker_events=events, worker_events_dropped=event_collector.dropped if event_collector else 0,
                       worker_events_pending=event_collector.pending if event_collector else 0,
                       unit=unit, batch_cardinalities=[n for n in cardinalities if n is not None])


def _sync(device: torch.device) -> None:
    if device.type == "cuda": torch.cuda.synchronize(device)


def _shape(value: Any) -> Any:
    if isinstance(value, torch.Tensor): return tuple(value.shape)
    if isinstance(value, dict): return {str(k): _shape(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_shape(v) for v in value]
    return None


def profile_pipeline(model: Any, loader: Any, *, adapter: Callable[[Any], Any] | None = None,
                     device: str | torch.device = "cpu", max_batches: int = 100,
                     warmup_batches: int = 1, non_blocking: bool = False,
                     synchronized_diagnostics: bool = True, loader_factory: Callable[[], Any] | None = None,
                     postprocess: Callable[[Any], Any] | None = None,
                     operator_profile: bool = False, max_operator_events: int = 100,
                     event_collector: BoundedEventCollector | None = None,
                     max_event_drain: int = 4096, event_drain_timeout: float = 0.1,
                     padding_measure: Callable[[Any], tuple[int, int]] | None = None) -> PipelineProfile:
    """Profile inference. Adapter returns ModelCall (or raw call args); no target assumptions."""
    target = torch.device(device)
    if max_batches < 1 or warmup_batches < 0: raise ValueError("invalid profiling bounds")
    if max_operator_events < 1: raise ValueError("max_operator_events must be positive")
    if max_event_drain < 0 or event_drain_timeout < 0: raise ValueError("invalid worker event drain bounds")
    if loader_factory is None and iter(loader) is loader and warmup_batches:
        # Warmup is separately named; it still consumes a one-shot stream explicitly.
        pass
    iterator = iter(loader)
    prep, transfers, models, posts, end_to_end, shapes, padding = [], [], [], [], [], [], []
    wait_samples=[]
    start_all = time.perf_counter()
    startup = time.perf_counter()-start_all
    batch_count=0
    profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU] +
        ([torch.profiler.ProfilerActivity.CUDA] if target.type == "cuda" else []),
        acc_events=True) if operator_profile else None
    if profiler: profiler.__enter__()
    try:
      with model_execution(model, inference=True, preserve_rng=True):
        for index in range(max_batches + warmup_batches):
            whole = time.perf_counter()
            wait_tick = time.perf_counter()
            try: batch=next(iterator)
            except StopIteration: break
            wait_samples.append(time.perf_counter()-wait_tick)
            if padding_measure:
                effective,total=padding_measure(batch)
                if total <= 0 or effective < 0 or effective > total:
                    raise ValueError("padding_measure must return 0 <= effective <= positive total")
                padding.append(1.0-effective/total)
            t=time.perf_counter(); prepared=adapter(batch) if adapter else batch
            prep_time=time.perf_counter()-t
            call = prepared if isinstance(prepared, ModelCall) else ModelCall((prepared,))
            shapes.append(_shape((call.args, call.kwargs)))
            if synchronized_diagnostics: _sync(target)
            t=time.perf_counter(); moved=transfer((call.args, call.kwargs), target, non_blocking=non_blocking)
            if synchronized_diagnostics: _sync(target)
            transfer_time=time.perf_counter()-t
            moved_call=ModelCall(moved[0], moved[1])
            if synchronized_diagnostics: _sync(target)
            t=time.perf_counter(); output=model(*moved_call.args, **moved_call.kwargs)
            if synchronized_diagnostics: _sync(target)
            model_time=time.perf_counter()-t
            t=time.perf_counter()
            if postprocess: output=postprocess(output)
            if synchronized_diagnostics: _sync(target)
            post_time=time.perf_counter()-t
            if profiler: profiler.step()
            if index >= warmup_batches:
                prep.append(prep_time); transfers.append(transfer_time); models.append(model_time)
                if postprocess: posts.append(post_time)
                end_to_end.append(time.perf_counter()-whole); batch_count+=1
    finally:
      if profiler: profiler.__exit__(None, None, None)
    elapsed=time.perf_counter()-start_all
    events=event_collector.drain(max_event_drain, timeout=event_drain_timeout) if event_collector else []
    data=DataProfile(batch_count, startup, wait_samples, elapsed, None, warmup_batches,
                     loader_settings=_settings(loader), mode=("synchronized diagnostic spans; end-to-end wall time" if synchronized_diagnostics else "asynchronous realistic wall time"),
                     worker_events=events, worker_events_dropped=event_collector.dropped if event_collector else 0)
    data.worker_events_pending=event_collector.pending if event_collector else 0
    rec=[]
    worker_seconds=[event["seconds"] for event in data.worker_events]
    if worker_seconds and statistics.mean(worker_seconds) > (statistics.mean(models) if models else 0):
        rec.append("Measured worker transform/collate cost is high; trial vectorization, batching, or deterministic caching.")
    elif data.p95_wait_seconds and sum(wait_samples)/max(1,len(wait_samples)) > (statistics.mean(models) if models else 0):
        rec.append("Consumer-visible loader wait is high; instrument worker transforms before attributing the cause.")
    if transfers and statistics.mean(transfers) > (statistics.mean(models) if models else float("inf")):
        rec.append("Measured transfer exceeds model span; test pinned/nonblocking transfer on this target.")
    if padding and statistics.mean(padding) >= .2:
        rec.append("Measured padding waste is at least 20%; trial length bucketing or application-compatible packing.")
    op_events=[]
    if profiler:
        try:
            for event in profiler.key_averages()[:max_operator_events]:
                op_events.append({"name": event.key, "cpu_time_total_us": event.cpu_time_total,
                                  "cuda_time_total_us": getattr(event, "cuda_time_total", None),
                                  "calls": event.count, "inclusive": True})
        except Exception as exc:  # noqa: BLE001 - profiler support varies by backend; report, don't fail
            op_events=[{"unavailable": str(exc)}]
    return PipelineProfile(data, prep, transfers, models, end_to_end, synchronized_diagnostics, shapes, None, rec, op_events,
                           padding_fractions=padding, postprocessing_seconds=posts)
