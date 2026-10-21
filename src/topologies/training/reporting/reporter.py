"""Fan-out of training facts to logger (stores) and monitor (presents)."""

from __future__ import annotations

import warnings
from typing import Any


class TrainingReporter:
    """Owns no training state; additive observability only."""

    def __init__(
        self,
        logger: Any | None = None,
        monitor: Any | None = None,
        rank: int = 0,
        log_every_steps: int = 10,
        visualizer: Any | None = None,
    ) -> None:
        self.logger = logger
        self.monitor = monitor
        self.visualizer = visualizer
        self.rank = int(rank or 0)
        try:
            self.log_every_steps = max(1, int(log_every_steps or 10))
        except Exception:  # noqa: BLE001
            self.log_every_steps = 10
        self._logger_ok = True
        self._monitor_ok = True
        self._visualizer_ok = True
        self._warned: set[str] = set()
        self._session_step = 0
        self._step_baseline = 0
        self._session_elapsed = 0.0

    # -- internals ---------------------------------------------------------
    def _warn_once(self, consumer: str, exc: BaseException) -> None:
        """One ``RuntimeWarning`` per failing sink; the sink is disabled by the caller."""
        if consumer in self._warned:
            return
        self._warned.add(consumer)
        warnings.warn(f"[topologies] {consumer} failed ({type(exc).__name__}: {exc}); "
                      "disabling it for the rest of this run.", RuntimeWarning, stacklevel=3)

    def _emit_logger(self, method: str, *args: Any) -> None:
        if self.rank != 0 or not self._logger_ok or self.logger is None:
            return
        try:
            fn = getattr(self.logger, method, None)
            if fn is None:
                return
            fn(*args)
        except Exception as e:  # noqa: BLE001
            self._warn_once("logger", e)
            self._logger_ok = False

    def _emit_monitor(self, method: str, *args: Any) -> None:
        if self.rank != 0 or not self._monitor_ok or self.monitor is None:
            return
        try:
            fn = getattr(self.monitor, method, None)
            if fn is None:
                return
            fn(*args)
        except Exception as e:  # noqa: BLE001
            self._warn_once("monitor", e)
            self._monitor_ok = False

    def _emit_visualizer(self, event: Any) -> None:
        if self.rank != 0 or self.visualizer is None or not self._visualizer_ok:
            return
        try:
            from topologies.training.reporting.events import (
                StepEvent,
                TrainBeginEvent,
                ValidationEvent,
            )
            from topologies.visualization import normalize_event

            if isinstance(event, TrainBeginEvent):
                self._session_step = int(getattr(event.info, "global_step", 0) or 0)
                self._step_baseline = self._session_step
                self._session_elapsed = float(getattr(event.info, "elapsed", 0.0) or 0.0)
            if isinstance(event, StepEvent):
                if self._step_baseline == 0 and self._session_step == 0:
                    self._step_baseline = max(0, int(event.global_step) - 1)
                self._session_step = int(event.global_step)
            elif isinstance(event, ValidationEvent):
                event = {**vars(event), "global_step": self._session_step, "kind": "training.validation_completed"}
            self.visualizer.handle_safe(normalize_event(
                event, step_baseline=self._step_baseline, elapsed_baseline=self._session_elapsed
            ))
        except Exception as exc:  # noqa: BLE001
            self._warn_once("visualizer", exc)
            self._visualizer_ok = False

    def on_error(self, error: BaseException) -> None:
        """Notify presentation sinks without changing the original exception."""
        if self.rank != 0 or self.visualizer is None or not self._visualizer_ok:
            return
        try:
            from topologies.visualization import normalize_event

            event = normalize_event(
                {
                    "kind": "error",
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
                source="training",
            )
            self.visualizer.handle_safe(event)
        except Exception as exc:  # noqa: BLE001
            self._warn_once("visualizer", exc)
            self._visualizer_ok = False

    # -- public event API --------------------------------------------------
    def on_train_begin(self, event: Any) -> None:
        self._emit_logger("on_train_begin", event)
        self._emit_monitor("on_train_begin", event)
        self._emit_visualizer(event)

    def on_epoch_begin(self, event: Any) -> None:
        self._emit_logger("on_epoch_begin", event)
        self._emit_monitor("on_epoch_begin", event)
        self._emit_visualizer(event)

    def wants_step(self, step: int) -> bool:
        """Whether a step event at ``step`` reaches any sink (rank 0, sink enabled).

        The Trainer skips building step events (and their device->host metric copy)
        when nothing would consume them.
        """
        if self.rank != 0:
            return False
        if self.monitor is not None and self._monitor_ok:
            return True
        if self.visualizer is not None and self._visualizer_ok:
            return True
        return (self.logger is not None and self._logger_ok
                and (step % self.log_every_steps == 0 or step <= 1))

    def on_step(self, event: Any) -> None:
        # Logger is throttled; monitor controls its own frequency.
        try:
            step = int(getattr(event, "global_step", 0))
        except Exception:  # noqa: BLE001
            step = 0
        if step % self.log_every_steps == 0 or step <= 1:
            self._emit_logger("on_step", event)
        self._emit_monitor("on_step", event)
        self._emit_visualizer(event)

    def on_validation_begin(self, event: Any) -> None:
        self._emit_logger("on_validation_begin", event)
        self._emit_monitor("on_validation_begin", event)
        self._emit_visualizer(event)

    def on_validation_end(self, event: Any) -> None:
        # Call only on_validation_end; BaseLogger routes it to on_validation,
        # and JsonlLogger/TextLogger implement single-write via that path.
        self._emit_logger("on_validation_end", event)
        self._emit_monitor("on_validation_end", event)
        self._emit_visualizer(event)

    def on_checkpoint(self, event: Any) -> None:
        self._emit_logger("on_checkpoint", event)
        self._emit_monitor("on_checkpoint", event)
        self._emit_visualizer(event)

    def on_epoch_end(self, event: Any) -> None:
        self._emit_logger("on_epoch_end", event)
        self._emit_monitor("on_epoch_end", event)
        self._emit_visualizer(event)

    def on_train_end(self, event: Any) -> None:
        self._emit_logger("on_train_end", event)
        self._emit_monitor("on_train_end", event)
        self._emit_visualizer(event)

    def close(self) -> None:
        if self.rank != 0:
            return
        if self.visualizer is not None:
            try:
                self.visualizer.close()
            except Exception as e:  # noqa: BLE001
                self._warn_once("visualizer", e)
        if self.logger is not None and self._logger_ok:
            try:
                close = getattr(self.logger, "close", None)
                if close is not None:
                    close()
            except Exception as e:  # noqa: BLE001
                self._warn_once("logger", e)
                self._logger_ok = False
        if self.monitor is not None and self._monitor_ok:
            try:
                close = getattr(self.monitor, "close", None)
                if close is not None:
                    close()
            except Exception as e:  # noqa: BLE001
                self._warn_once("monitor", e)
                self._monitor_ok = False
