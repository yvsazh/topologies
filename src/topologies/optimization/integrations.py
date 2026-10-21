"""Experiment bridge: execution, transformation, and constraint verdicts stay distinct."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DEVICE_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()
_LOCK_ROOT = Path(
    os.environ.get(
        "TOPOLOGIES_DEVICE_LOCK_DIR", Path(tempfile.gettempdir()) / "topologies-device-locks"
    )
)


@dataclass(frozen=True)
class Constraint:
    metric: str
    limit: float
    direction: str = "max"
    unit: str = ""
    evidence_kind: str = "measured"  # measured or estimated
    relative: bool = False
    near_zero: float = 1e-12


@dataclass(frozen=True)
class ConstraintVerdict:
    metric: str
    status: str
    value: float | None
    limit: float
    unit: str
    evidence_kind: str
    reason: str | None = None


def check_constraint(
    constraint: Constraint,
    value: Any,
    *,
    evidence_kind: str = "measured",
    baseline: float | None = None,
) -> ConstraintVerdict:
    if constraint.direction not in {"max", "min"}:
        raise ValueError("constraint direction must be max or min")
    try:
        v = float(value)
        lim = float(constraint.limit)
        if not math.isfinite(v) or not math.isfinite(lim):
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        return ConstraintVerdict(
            constraint.metric,
            "indeterminate",
            None,
            constraint.limit,
            constraint.unit,
            evidence_kind,
            "missing or non-finite evidence",
        )
    if constraint.evidence_kind == "measured" and evidence_kind != "measured":
        return ConstraintVerdict(
            constraint.metric,
            "indeterminate",
            v,
            lim,
            constraint.unit,
            evidence_kind,
            "measured evidence required",
        )
    if constraint.relative:
        if (
            baseline is None
            or not math.isfinite(float(baseline))
            or abs(float(baseline)) <= constraint.near_zero
        ):
            return ConstraintVerdict(
                constraint.metric,
                "indeterminate",
                v,
                lim,
                constraint.unit,
                evidence_kind,
                "relative comparison baseline is near zero or unavailable",
            )
        v = (v - float(baseline)) / abs(float(baseline))
    passed = v <= lim if constraint.direction == "max" else v >= lim
    return ConstraintVerdict(
        constraint.metric, "pass" if passed else "fail", v, lim, constraint.unit, evidence_kind
    )


def feasible_pareto(
    records: Sequence[Mapping[str, Any]],
    objectives: Mapping[str, str],
    *,
    constraints: Sequence[Constraint] = (),
) -> Any:
    """Filter to completed, fully evidenced feasible records, then generic Pareto selection."""
    from topologies.analysis.records import pareto_front

    feasible = []
    for row in records:
        if (
            row.get("status", "completed") != "completed"
            or row.get("transformation_status", "completed") != "completed"
        ):
            continue
        verdicts = row.get("constraint_verdicts")
        if verdicts is None:
            evidence = row.get("evidence_kind", {})
            verdicts = [
                check_constraint(
                    c,
                    row.get(c.metric),
                    evidence_kind=evidence.get(c.metric, "measured"),
                    baseline=row.get(f"baseline_{c.metric}"),
                )
                for c in constraints
            ]
        if any(
            (v.status if isinstance(v, ConstraintVerdict) else v.get("status")) != "pass"
            for v in verdicts
        ):
            continue
        feasible.append(row)
    return pareto_front(feasible, objectives)


@contextmanager
def exclusive_device(
    device: str, *, lock_dir: str | os.PathLike[str] | None = None, timeout: float | None = 600
):
    """Cross-process exclusive benchmark lock, keyed by precise device identity.

    Uses advisory ``flock`` on Unix and a one-byte ``msvcrt`` lock on Windows.
    Set ``TOPOLOGIES_DEVICE_LOCK_DIR`` or ``lock_dir`` to choose a shared local
    runtime directory. A stable fingerprint (e.g. ``cuda:0|GPU-uuid``) avoids
    serializing unrelated devices.
    """
    identity = str(device).strip()
    if not identity:
        raise ValueError("device lock identity must not be empty")
    with _LOCKS_GUARD:
        lock = _DEVICE_LOCKS.setdefault(identity, threading.RLock())
    with lock:
        root = Path(lock_dir) if lock_dir is not None else _LOCK_ROOT
        root.mkdir(parents=True, exist_ok=True)
        lock_path = root / (hashlib.sha256(identity.encode()).hexdigest() + ".lock")
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            deadline = None if timeout is None else time.monotonic() + timeout
            while True:
                try:
                    if os.name == "nt":
                        import msvcrt

                        os.lseek(descriptor, 0, os.SEEK_SET)
                        if os.fstat(descriptor).st_size == 0:
                            os.write(descriptor, b"0")
                        os.lseek(descriptor, 0, os.SEEK_SET)
                        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except (BlockingIOError, OSError):
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"timed out waiting for benchmark device lock {identity!r}"
                        )
                    time.sleep(0.025)
            try:
                yield
            finally:
                if os.name == "nt":
                    import msvcrt

                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def trial_output(
    result: Any,
    *,
    metrics: dict[str, float] | None = None,
    constraints: Sequence[Constraint] = (),
    evidence_kind: Mapping[str, str] | None = None,
    artifacts: Mapping[str, Any] | None = None,
) -> Any:
    from topologies.experiments.trial import TrialOutput

    statuses = [
        check_constraint(
            c,
            (metrics or {}).get(c.metric),
            evidence_kind=(evidence_kind or {}).get(c.metric, "measured"),
        )
        for c in constraints
    ]
    return TrialOutput(
        metrics=dict(metrics or {}),
        extras={
            "optimization": result,
            "transformation_status": "completed",
            "constraint_verdicts": [v.__dict__ for v in statuses],
        },
        artifacts=dict(artifacts or {}),
    )


def optimization_trial(
    ctx: Any,
    optimize: Any,
    *,
    evaluate: Any,
    constraints: Sequence[Constraint] = (),
    benchmark: Any = None,
) -> Any:
    baseline = ctx.extras.get("model")
    if baseline is None:
        raise ValueError(
            "Put a freshly reconstructed source model in ctx.extras['model'] before calling the helper"
        )
    result = optimize(baseline, ctx)
    values = evaluate(result.model, ctx)
    if not isinstance(values, dict):
        raise TypeError("evaluate(result.model, ctx) must return a metric dictionary")
    evidence: dict[str, str] = {}
    if benchmark is not None:
        device = str(ctx.extras.get("benchmark_device", ctx.device))
        with exclusive_device(device):
            measured = benchmark(result.model, ctx)
        values.update(measured)
        evidence.update({key: "measured" for key in measured})
    output = trial_output(result, metrics=values, constraints=constraints, evidence_kind=evidence)
    output.extras["optimization_provenance"] = getattr(result, "provenance", None)
    return output
