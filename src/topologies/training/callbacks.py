"""Minimal callback system. Callbacks observe; they never own the loop."""

from __future__ import annotations

from typing import Any


class Callback:
    def on_train_begin(self, trainer: Any) -> None:
        pass

    def on_train_end(self, trainer: Any) -> None:
        pass

    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict) -> None:
        pass

    def on_step_begin(self, trainer: Any, step: int) -> None:
        pass

    def on_step_end(self, trainer: Any, step: int, metrics: dict) -> None:
        pass

    def on_validation_end(self, trainer: Any, metrics: dict) -> None:
        pass

    def on_checkpoint(self, trainer: Any, path: str) -> None:
        pass

    def on_exception(self, trainer: Any, exc: BaseException) -> None:
        pass


class CallbackList(Callback):
    def __init__(self, callbacks: list[Callback] | None = None) -> None:
        self.callbacks = list(callbacks or [])

    def _each(self, name: str, *args: Any) -> None:
        for cb in self.callbacks:
            try:
                getattr(cb, name)(*args)
            except Exception:  # noqa: BLE001, S112 - callbacks must not break training
                continue

    def on_train_begin(self, trainer: Any) -> None:
        self._each("on_train_begin", trainer)

    def on_train_end(self, trainer: Any) -> None:
        self._each("on_train_end", trainer)

    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        self._each("on_epoch_begin", trainer, epoch)

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict) -> None:
        self._each("on_epoch_end", trainer, epoch, metrics)

    def on_step_begin(self, trainer: Any, step: int) -> None:
        self._each("on_step_begin", trainer, step)

    def on_step_end(self, trainer: Any, step: int, metrics: dict) -> None:
        self._each("on_step_end", trainer, step, metrics)

    def on_validation_end(self, trainer: Any, metrics: dict) -> None:
        self._each("on_validation_end", trainer, metrics)

    def on_checkpoint(self, trainer: Any, path: str) -> None:
        self._each("on_checkpoint", trainer, path)

    def on_exception(self, trainer: Any, exc: BaseException) -> None:
        self._each("on_exception", trainer, exc)
