"""Logger base: stores facts, never renders terminal UI."""

from __future__ import annotations

from typing import Any


class BaseLogger:
    rank: int = 0

    # -- legacy API (kept for backward compatibility) ----------------------
    def log(self, step: int, values: dict[str, Any]) -> None:
        """Legacy per-step hook. Default is a no-op."""
        return

    def close(self) -> None:
        pass

    # -- event API (new) ----------------------------------------------------
    def on_train_begin(self, event: Any) -> None:
        pass

    def on_epoch_begin(self, event: Any) -> None:
        pass

    def on_step(self, event: Any) -> None:
        """Default bridge: old loggers implementing only log() still get data."""
        try:
            step = int(getattr(event, "global_step", 0))
        except Exception:  # noqa: BLE001
            step = 0
        try:
            metrics = dict(getattr(event, "metrics", {}) or {})
        except Exception:  # noqa: BLE001
            metrics = {}
        try:
            self.log(step, metrics)
        except NotImplementedError:
            pass

    def on_validation_begin(self, event: Any) -> None:
        pass

    def on_validation(self, event: Any) -> None:
        pass

    def on_validation_end(self, event: Any) -> None:
        # Default: route to on_validation so subclasses overriding either work.
        try:
            self.on_validation(event)
        except Exception:  # noqa: BLE001, S110
            pass

    def on_checkpoint(self, event: Any) -> None:
        pass

    def on_epoch_end(self, event: Any) -> None:
        pass

    def on_train_end(self, event: Any) -> None:
        pass


class NullLogger(BaseLogger):
    def log(self, step: int, values: dict[str, Any]) -> None:
        pass


class ConsoleLogger(BaseLogger):
    """Deprecated: kept for backward imports. Use StandardMonitor instead."""

    def __init__(self, rank: int = 0) -> None:
        self.rank = rank

    def log(self, step: int, values: dict[str, Any]) -> None:
        if self.rank != 0:
            return
        items = " ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in (values or {}).items()
        )
        print(f"[step {step}] {items}", flush=True)
