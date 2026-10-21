"""Process-pool runner: ``workers > 1`` or ``devices=[...]``.

- ``multiprocessing`` spawn context; each worker runs one trial at a time.
- Worker ``i`` gets ``devices[i]``; CPU-only pools split threads:
  ``Experiment(threads_per_worker=None)`` means ``max(1, cpu_count // workers)``.
  Different intra-op thread counts can change CPU reduction order, so results may
  drift slightly from a sequential run; ``result.json`` records ``torch_threads``.
- A worker that dies (OOM killer, segfault) fails only its running trial
  (``WorkerCrash`` with the exit code) and is replaced (at most ``2 * workers``
  restarts); the run never hangs waiting for it.
- MPS has one device: pools fall back to ``workers=1`` (with a warning) unless
  every trial runs on CPU.
- Workers write only inside their trial directory and stream progress to the
  coordinator through a queue; only the coordinator appends to ``trials.jsonl``.
- Pruning in workers compares against the trials finished when the trial
  was dispatched (a snapshot; documented).
"""

from __future__ import annotations

import os
import pickle
import queue as _queue
import time
import warnings
from typing import Any

from topologies.experiments.errors import ExperimentError
from topologies.experiments.trial import Trial


def _find_unpicklable(obj: Any, path: str, depth: int = 0) -> tuple[str, Any] | None:
    try:
        pickle.dumps(obj)
        return None
    except Exception:  # noqa: BLE001, S110 - pickling user objects may raise anything
        pass
    if depth < 4:
        if isinstance(obj, dict):
            for k, v in obj.items():
                hit = _find_unpicklable(v, f"{path}[{k!r}]", depth + 1)
                if hit:
                    return hit
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                hit = _find_unpicklable(v, f"{path}[{i}]", depth + 1)
                if hit:
                    return hit
        elif hasattr(obj, "__dict__") and not isinstance(obj, type) and not callable(obj):
            for k, v in vars(obj).items():
                hit = _find_unpicklable(v, f"{path}.{k}", depth + 1)
                if hit:
                    return hit
    return path, obj


def check_picklable(exp: Any, trials: list[Trial]) -> None:
    """Raise ``ExperimentError`` naming the first value a worker could not receive."""
    runtime = exp.runtime()
    candidates: list[tuple[str, Any]] = [
        (f"Experiment({field}=...)", getattr(runtime, field))
        for field in ("task", "trial_fn", "build_model", "build_data", "make_trainer",
                      "evaluate", "pruner")]
    for t in trials:
        for k, v in t.params.items():
            candidates.append((f"{k!r} (trial {' × '.join(map(str, t.labels.values()))})", v))
    for where, value in candidates:
        hit = _find_unpicklable(value, where)
        if hit is not None:
            path, bad = hit
            name = getattr(bad, "__qualname__", None) or type(bad).__name__
            raise ExperimentError(
                f"{path} = {name} cannot be sent to worker processes (not picklable).",
                stage="plan", key=path,
                hint="Define it as a module-level function/class (not a lambda or closure), "
                     "or run with workers=1.")


def _cpu_only(exp: Any, trials: list[Trial]) -> bool:
    if exp.devices:
        return all(str(d).startswith("cpu") for d in exp.devices)
    return all(str(t.params.get("trainer.device", "auto")).startswith("cpu") for t in trials)


def _mps_available() -> bool:
    try:
        import torch

        return bool(torch.backends.mps.is_available()) and not torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def _worker(index: int, payload: bytes, spec_path: str | None, device: str | None,
            threads: int | None, tasks: Any, results: Any, current: Any = None) -> None:
    import torch

    if threads:
        try:
            torch.set_num_threads(int(threads))
        except Exception:  # noqa: BLE001, S110 - thread tuning is advisory
            pass
    if spec_path:
        from topologies.experiments.spec import load_spec

        load_spec(spec_path)  # makes spec-file classes/functions importable here
    from topologies.experiments.cache import Cache
    from topologies.experiments.runner import execute_trial, resolve_device
    from topologies.experiments.store import ExperimentStore

    runtime, root = pickle.loads(payload)
    store = ExperimentStore(root, runtime.name)
    cache = Cache(store.root)
    while True:
        try:
            item = tasks.get()
        except (EOFError, OSError):
            return
        if item is None:
            return
        trial, peers = item
        if current is not None:
            # Shared memory, written synchronously: survives a hard crash, unlike a
            # queue message (whose feeder thread dies with the process).
            current.value = trial.trial_id.encode()[:255]

        def emit(ev: dict, _q: Any = results) -> None:
            try:
                _q.put(("event", ev))
            except Exception:  # noqa: BLE001, S110 - progress events are best-effort
                pass

        dev = resolve_device(device or trial.params.get("trainer.device"))
        try:
            result = execute_trial(runtime, trial, device=dev, cache=cache, store=store,
                                   emit=emit, peers=peers)
            results.put(("done", trial.trial_id, result))
        except KeyboardInterrupt:
            results.put(("interrupted", trial.trial_id, None))
            return
        except BaseException as e:  # noqa: BLE001 - report, keep the pool alive
            results.put(("crash", trial.trial_id, f"{type(e).__name__}: {e}"))


def run_parallel(exp: Any, pending: list[Trial], state: Any,
                 peers: dict[str, dict[int, float]]) -> None:
    import multiprocessing as mp

    from topologies.experiments.coordinator import _after_trial, run_sequential

    if not pending:
        return
    n = len(exp.devices) if exp.devices and exp.workers <= 1 else exp.workers
    n = max(1, min(n, len(pending)))
    if _mps_available() and not _cpu_only(exp, pending):
        warnings.warn("MPS exposes a single device: running with workers=1 "
                      "(set trainer.device='cpu' to parallelize on CPU).", stacklevel=3)
        n = 1
    if n == 1:
        run_sequential(exp, pending, state, peers)
        return
    check_picklable(exp, pending)
    payload = pickle.dumps((exp.runtime(), str(exp.store.root)))
    devices = list(exp.devices or [])
    threads = None
    if not devices or all(str(d).startswith("cpu") for d in devices):
        # Split the CPU between workers instead of oversubscribing it n-fold.
        threads = exp.threads_per_worker or max(1, (os.cpu_count() or 1) // n)
    ctx = mp.get_context("spawn")
    tasks, results = ctx.Queue(), ctx.Queue()
    spec_path = getattr(exp, "_spec_path", None)

    slots: dict[int, Any] = {}  # worker pid -> shared "current trial id" slot

    def start_worker(i: int) -> Any:
        current = ctx.Array("c", 256)
        p = ctx.Process(target=_worker, args=(i, payload, spec_path,
                                              devices[i % len(devices)] if devices else None,
                                              threads, tasks, results, current), daemon=True)
        p.start()
        slots[p.pid] = current
        return p

    procs = [start_worker(i) for i in range(n)]
    restarts = 0
    by_id = {t.trial_id: t for t in pending}
    queue_order = list(pending)
    running: set[str] = set()

    def dispatch() -> None:
        while queue_order and len(running) < n:
            t = queue_order.pop(0)
            running.add(t.trial_id)
            from topologies.experiments.coordinator import stamp_plan

            stamp_plan(exp, t)
            exp.store.log_event(t.trial_id, "running")
            state.trial_start(t)
            tasks.put((t, dict(peers)))

    def record_crash(t: Trial, message: str) -> None:
        from topologies.experiments.runner import assemble_result
        from topologies.experiments.store import now_iso

        res = assemble_result(exp.runtime(), t, None, status="failed",
                              status_reason=message[:300], stop_reason=None, out=None,
                              error={"type": "WorkerCrash", "message": message,
                                     "traceback": ""},
                              started_at=now_iso(), wall_s=0.0, adjustments=[])
        exp.store.write_result(t.trial_id, res)
        _after_trial(exp, state, t, res, peers)

    def reap_dead_workers() -> None:
        nonlocal restarts
        for p in list(procs):
            if p.is_alive():
                continue
            procs.remove(p)
            slot = slots.pop(p.pid, None)
            tid = slot.value.decode() if slot is not None and slot.value else None
            if tid is not None and tid in running:
                running.discard(tid)
                record_crash(by_id[tid], f"worker process died (exit code {p.exitcode})")
            if (running or queue_order) and restarts < 2 * n:
                restarts += 1
                procs.append(start_worker(n + restarts))
        if (running or queue_order) and not procs:
            raise ExperimentError("All worker processes died.", stage="run")

    try:
        dispatch()
        while running:
            try:
                msg = results.get(timeout=0.2)
            except _queue.Empty:
                reap_dead_workers()
                dispatch()
                continue
            kind = msg[0]
            if kind == "event":
                state.handle(msg[1])
                continue
            tid = msg[1]
            if tid not in running:
                continue  # already recorded (e.g. as a crash)
            running.discard(tid)
            t = by_id[tid]
            if kind == "done":
                _after_trial(exp, state, t, msg[2], peers)
            elif kind == "crash":
                record_crash(t, msg[2])
            else:
                raise KeyboardInterrupt
            dispatch()
    except KeyboardInterrupt:
        deadline = time.monotonic() + 3.0
        while running and time.monotonic() < deadline:
            try:
                msg = results.get(timeout=0.2)
                if msg[0] in ("done", "interrupted", "crash"):
                    running.discard(msg[1])
            except _queue.Empty:
                continue
        _mark_interrupted(exp, [by_id[i] for i in running])
        for i in running:
            exp.store.log_event(i, "interrupted")
        raise
    finally:
        for _ in procs:
            try:
                tasks.put(None)
            except Exception:  # noqa: BLE001, S110 - shutdown signal to possibly dead workers
                pass
        for p in procs:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()


def _mark_interrupted(exp: Any, trials: list[Trial]) -> None:
    from topologies.experiments.runner import assemble_result
    from topologies.experiments.store import now_iso

    for t in trials:
        res = exp.store.read_result(t.trial_id, warn=False)
        if res and res.get("status") == "interrupted":
            continue
        exp.store.write_result(t.trial_id, assemble_result(
            exp.runtime(), t, None, status="interrupted", status_reason="KeyboardInterrupt",
            stop_reason=None, out=None, error=None, started_at=now_iso(), wall_s=0.0,
            adjustments=[]))


__all__ = ["check_picklable", "run_parallel"]
