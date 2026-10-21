"""Trials: planned trial, runtime context, outputs, statuses and ``result.json``.

``TrialContext`` is what stage overrides (``build_model(ctx)``, ...) and
Level-3 trial functions receive:

    @experiment(axes={"lr": [1e-3, 3e-4]}, goal=("val_acc", "max"))
    def sweep(ctx, lr):
        t = Trainer(model, train, val, lr=lr, callbacks=ctx.callbacks(),
                    **ctx.trainer_defaults())
        t.fit()
        return t.evaluate(val, prefix="val_")
"""

from __future__ import annotations

import math
import platform
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from topologies.experiments.errors import ExperimentError, Pruned, TrialTimeout

STATUSES = ("queued", "running", "completed", "failed", "diverged", "oom", "timeout",
            "pruned", "interrupted")
#: Non-completed statuses that a plain ``run()`` does not retry (use ``rerun="failed"``).
FINAL_FAILURES = frozenset({"failed", "diverged", "oom", "timeout", "pruned"})


@dataclass
class Trial:
    """One planned trial: a point of the space plus a seed."""

    trial_id: str
    labels: dict[str, Any]
    params: dict[str, Any]
    config: dict[str, Any]
    seed: int
    index: int = 0
    plan_error: str | None = None
    plan_total: int | None = None
    plan_done: int | None = None

    @property
    def config_labels(self) -> dict[str, Any]:
        return {k: v for k, v in self.labels.items() if k != "seed"}

    def describe(self) -> str:
        parts = [str(v) for k, v in self.labels.items() if k != "seed"]
        return " × ".join(parts) + f" · seed {self.seed}" if parts else f"seed {self.seed}"


@dataclass
class TrialOutput:
    """Rich return value of a Level-3 trial function (a plain metrics dict also works)."""

    metrics: dict[str, float]
    history: list[dict[str, Any]] | None = None
    extras: dict[str, Any] | None = None
    artifacts: dict[str, Any] | None = None


@dataclass
class Runtime:
    """Picklable per-experiment state that trials need (sent to workers)."""

    name: str
    task: Any
    goal: tuple[str, str]
    select: str
    budget: dict[str, float]
    root: str
    exp_dir: str
    trial_timeout: float | None = None
    pruner: Any = None
    save: str = "none"
    on_oom: str = "skip"
    trial_fn: Callable | None = None
    build_model: Callable | None = None
    build_data: Callable | None = None
    make_trainer: Callable | None = None
    evaluate: Callable | None = None
    default_device: str | None = None
    free_form: bool = False
    rung: dict[str, Any] | None = None  # successive halving: {"epochs": n, "resume": bool}
    budget_override: dict[str, float] | None = None  # e.g. the 2-step smoke run


class TrialContext:
    """Everything a trial (or a stage override) needs; see the module docstring."""

    def __init__(self, runtime: Runtime, trial: Trial, *, device: str, cache: Any,
                 emit: Callable[[dict], None] | None = None,
                 peers: dict[str, dict[int, float]] | None = None) -> None:
        self.runtime = runtime
        self.trial = trial
        self.experiment = runtime.name
        self.trial_id = trial.trial_id
        self.labels = dict(trial.labels)
        self.params = dict(trial.params)
        self.seed = int(trial.seed)
        self.task = runtime.task
        base = dict(getattr(self.task, "params", {}) or {})
        base.update({k[5:]: v for k, v in self.params.items() if k.startswith("task.")})
        self.task_params = base
        self.dir = Path(runtime.exp_dir) / "trials" / trial.trial_id
        self.device = device
        self.cache = cache
        self.extras: dict[str, Any] = {}
        self.info: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []
        self.peers = peers if peers is not None else {}
        self.adjustments: list[dict[str, Any]] = []
        self._emit = emit
        self._tokenizer: Any = None
        self._tokenizer_key: str | None = None
        self._tokenizer_ready = False
        self._datasets: Any = None
        self._stop_reason: str | None = None
        self._status_hint: str | None = None
        self._callbacks: list[Any] | None = None
        self._t0_report: float | None = None

    @property
    def budget(self) -> dict[str, float]:
        """Training budget of this trial (``train_for`` may itself be an axis)."""
        if self.runtime.budget_override:
            return dict(self.runtime.budget_override)
        value = self.params.get("train_for")
        if value is None:
            return dict(self.runtime.budget or {"epochs": 1})
        from topologies.experiments.routing import parse_train_for

        return parse_train_for(value)

    # -- lazy stages -------------------------------------------------------
    @property
    def tokenizer(self) -> Any:
        """The trial's tokenizer (trained/cached on first access), or None."""
        if not self._tokenizer_ready:
            from topologies.experiments.pipeline import get_tokenizer

            self._tokenizer, self._tokenizer_key = get_tokenizer(self)
            self._tokenizer_ready = True
        return self._tokenizer

    @property
    def tokenizer_key(self) -> str | None:
        _ = self.tokenizer
        return self._tokenizer_key

    @property
    def datasets(self) -> Any:
        """``Datasets(train, val, test)`` from the task (built on first access)."""
        if self._datasets is None:
            from topologies.experiments.pipeline import build_data

            self._datasets = build_data(self)
        return self._datasets

    @datasets.setter
    def datasets(self, value: Any) -> None:
        self._datasets = value

    # -- trainer integration -------------------------------------------------
    def callbacks(self) -> list[Any]:
        """Progress, divergence guard, time budget / timeout and pruning callbacks."""
        if self._callbacks is None:
            from topologies.experiments.callbacks import trial_callbacks

            self._callbacks = trial_callbacks(self)
        return list(self._callbacks)

    def trainer_defaults(self) -> dict[str, Any]:
        """``seed``, ``device``, quiet monitor and a jsonl log in the trial directory."""
        return {"seed": self.seed, "device": self.device, "monitor": "none",
                "logger": "jsonl", "log_dir": str(self.dir)}

    # -- manual loops ----------------------------------------------------------
    def report(self, epoch: int, metrics: dict[str, float]) -> None:
        """Record one epoch of a custom loop: curves, live progress, pruning, timeout.

        Raises ``Pruned`` / ``TrialTimeout`` when the trial should stop.
        """
        import time

        if self._t0_report is None:
            self._t0_report = time.monotonic()
        row = {"epoch": int(epoch), **{k: _num(v) for k, v in metrics.items()}}
        self.record_epoch(row)
        if self.runtime.trial_timeout is not None and \
                time.monotonic() - self._t0_report > self.runtime.trial_timeout:
            self.request_stop("timeout", status="timeout")
            raise TrialTimeout(f"trial exceeded trial_timeout={self.runtime.trial_timeout:g}s")
        if self.check_prune(int(epoch), row):
            raise Pruned("pruned")

    def record_epoch(self, row: dict[str, Any]) -> None:
        """Add derived metrics, keep the row, notify the coordinator."""
        try:
            if self.task is not None and hasattr(self.task, "derive"):
                row.update(self.task.derive(self, row))
        except Exception:  # noqa: BLE001, S110 - derived metrics are best-effort
            pass
        self.history.append(row)
        self.emit({"type": "epoch", "epoch": row.get("epoch"), "metrics": row})

    def check_prune(self, epoch: int, row: dict[str, Any]) -> bool:
        pruner = self.runtime.pruner
        metric, _mode = self.runtime.goal
        value = row.get(metric)
        if pruner is None or value is None or not isinstance(value, (int, float)):
            return False
        try:
            peers = {k: v for k, v in self.peers.items() if k != self.trial_id}
            if pruner.should_prune(self.trial_id, epoch, float(value), peers):
                self.request_stop("pruned", status="pruned")
                return True
        except Exception:  # noqa: BLE001, S110 - a broken pruner never kills the trial
            pass
        return False

    def log(self, **values: Any) -> None:
        """Extra scalars stored in the result's ``extras``."""
        self.extras.update({k: _num(v) for k, v in values.items()})

    def request_stop(self, reason: str, *, status: str | None = None) -> None:
        if self._stop_reason is None:
            self._stop_reason = reason
        if status is not None and self._status_hint is None:
            self._status_hint = status

    @property
    def should_stop(self) -> bool:
        return self._stop_reason is not None

    def emit(self, event: dict) -> None:
        if self._emit is None:
            return
        try:
            self._emit({"trial_id": self.trial_id, **event})
        except Exception:  # noqa: BLE001, S110 - observability never breaks a trial
            pass


def _num(v: Any) -> Any:
    try:
        f = float(v)
        return f if not isinstance(v, bool) else v
    except (TypeError, ValueError):
        return v


# ---------------------------------------------------------------- errors / status


def is_oom(exc: BaseException) -> bool:
    try:
        import torch

        oom_t = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_t is not None and isinstance(exc, oom_t):
            return True
    except Exception:  # noqa: BLE001, S110 - OOM detection probe; unknown means not OOM
        pass
    seen: set[int] = set()
    e: BaseException | None = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if isinstance(e, MemoryError) or "out of memory" in str(e).lower():
            return True
        e = e.__cause__ or e.__context__
    return False


def error_info(exc: BaseException) -> dict[str, Any]:
    import traceback

    return {"type": type(exc).__name__, "message": str(exc).strip()[:4000],
            "traceback": "".join(traceback.format_exception(type(exc), exc,
                                                            exc.__traceback__))[-20000:]}


def short_error(err: dict | None) -> str:
    """First line of the message (the decisive part) for tables and insights."""
    if not err:
        return ""
    msg = str(err.get("message", "")).strip().splitlines()
    first = msg[0] if msg else ""
    return f"{err.get('type', 'Error')}: {first}"[:200]


# ---------------------------------------------------------------- environment

_ENV_CACHE: dict[str, Any] = {}


def _git_hash() -> str | None:
    if "git" in _ENV_CACHE:
        return _ENV_CACHE["git"]
    out = None
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                           timeout=2, check=False)
        if r.returncode == 0:
            out = r.stdout.strip() or None
    except Exception:  # noqa: BLE001
        out = None
    _ENV_CACHE["git"] = out
    return out


def environment(device: str | None = None, precision: str | None = None) -> dict[str, Any]:
    try:
        import torch

        torch_v = torch.__version__
        threads = torch.get_num_threads()
    except Exception:  # noqa: BLE001
        torch_v, threads = None, None
    try:
        import topologies

        topo_v = topologies.__version__
    except Exception:  # noqa: BLE001
        topo_v = None
    return {"topologies": topo_v, "torch": torch_v, "python": platform.python_version(),
            "platform": f"{sys.platform}-{platform.machine()}", "device": device,
            "precision": precision, "torch_threads": threads, "git": _git_hash()}


# ---------------------------------------------------------------- result.json


def best_of(history: list[dict], metric: str, mode: str) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for row in history:
        v = row.get(metric)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            continue
        if best is None or (v < best["value"] if mode == "min" else v > best["value"]):
            best = {"value": float(v), "epoch": row.get("epoch")}
    return best


def goal_value(final: dict, history: list[dict], metric: str, mode: str,
               select: str) -> float | None:
    if select == "best":
        b = best_of(history, metric, mode)
        if b is not None:
            return b["value"]
    v = final.get(metric)
    if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
        return float(v)
    if select != "best":
        for row in reversed(history):
            v = row.get(metric)
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                return float(v)
    return None


class _ResultModel(BaseModel):
    """Schema of ``result.json`` (§7.4); ``validate_result`` checks files against it."""

    model_config = ConfigDict(extra="forbid")

    format: str
    version: int
    trial_id: str
    experiment: str
    labels: dict[str, Any]
    config: dict[str, Any]
    status: str
    status_reason: str | None
    stop_reason: str | None
    goal: dict[str, Any]
    final: dict[str, Any]
    best: dict[str, Any]
    history: list[dict[str, Any]]
    cost: dict[str, Any]
    build: dict[str, Any] | None
    extras: dict[str, Any]
    adjustments: list[dict[str, Any]]
    env: dict[str, Any]
    started_at: str | None
    finished_at: str | None
    error: dict[str, Any] | None


def validate_result(data: dict) -> dict:
    """Raise ``ExperimentError`` unless ``data`` matches the ``result.json`` schema."""
    try:
        m = _ResultModel(**data)
    except Exception as e:
        raise ExperimentError(f"Invalid result.json: {e}", stage="store") from e
    if m.format != "topologies-trial" or m.status not in STATUSES:
        raise ExperimentError(f"Invalid result.json format/status ({m.format}, {m.status}).",
                              stage="store")
    for k in ("metric", "mode", "value", "select"):
        if k not in m.goal:
            raise ExperimentError(f"result.json goal lacks {k!r}.", stage="store")
    return data


__all__ = [
    "FINAL_FAILURES",
    "STATUSES",
    "Runtime",
    "Trial",
    "TrialContext",
    "TrialOutput",
    "best_of",
    "environment",
    "error_info",
    "goal_value",
    "is_oom",
    "short_error",
    "validate_result",
]
