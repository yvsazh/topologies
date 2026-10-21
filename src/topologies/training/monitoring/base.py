"""Monitor base: presents facts, never writes training history."""

from __future__ import annotations

from typing import Any


class BaseMonitor:
    def on_train_begin(self, event: Any) -> None:
        pass

    def on_epoch_begin(self, event: Any) -> None:
        pass

    def on_step(self, event: Any) -> None:
        pass

    def on_validation_begin(self, event: Any) -> None:
        pass

    def on_validation_end(self, event: Any) -> None:
        pass

    def on_validation(self, event: Any) -> None:
        pass

    def on_checkpoint(self, event: Any) -> None:
        pass

    def on_epoch_end(self, event: Any) -> None:
        pass

    def on_train_end(self, event: Any) -> None:
        pass

    def set_plan(self, index: int, total: int, *, done: int | None = None) -> None:
        """Where this trial sits in the experiment.

        ``index`` is 1-based. ``done`` is how many other trials in the plan
        already finished. The default monitor ignores this.
        """

    def note(self, label: str, *, done: int | None = None, total: int | None = None) -> None:
        """A long stretch of work that is not a training step.

        ``done`` / ``total`` are counts of the same unit (blocks, batches).
        The default monitor ignores this. Dashboard and standard redraw it.
        """

    def close(self) -> None:
        pass


class NullMonitor(BaseMonitor):
    pass
