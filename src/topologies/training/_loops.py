"""Epoch loops for the Trainer: one batch loop (loss/step levels) and one epoch close."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from topologies.training._step_metrics import (
    _accumulate,
    _floats,
    _loss_value,
    _train_averages,
)
from topologies.training.distributed import (
    all_reduce_mean_dict,
    barrier,
    maybe_wrap_sampler,
)
from topologies.training.errors import TrainingError
from topologies.training.logging import BaseLogger
from topologies.training.recipe_base import EpochContext


class _LoopsMixin:
    """Trainer mixin (split out of engine.py; state lives on the Trainer)."""

    # -- level entry points -------------------------------------------------------
    def _fit_loss_level(self) -> dict:
        return self._run_batch_epochs(self._loss_step, flush_accumulation=True)

    def _fit_step_level(self) -> dict:
        # Step-level recipes manage accumulation themselves (StepContext).
        return self._run_batch_epochs(self._custom_step, flush_accumulation=False)

    def _fit_epoch_level(self) -> dict:
        max_epochs = self.config.max_epochs
        for epoch in range(self.current_epoch, max_epochs):
            if self._stop_requested:
                break
            self._begin_epoch(epoch, max_epochs)
            ctx = EpochContext(
                model=self.model, train_loader=self.train_loader, val_loader=self.val_loader,
                epoch=epoch, global_step=self.global_step, device=self.device,
                precision=self.precision, rank=self.dist.rank, world_size=self.dist.world_size,
                config=self.config, recipe_config=self.recipe_config,
                components=self.components, adapter=self.adapter, logger=self.logger,
                trainer=self,
            )
            try:
                result = self.recipe.epoch_loop(ctx)
            except TrainingError:
                raise
            except Exception as e:
                raise TrainingError(
                    f"Recipe epoch_loop failed: {e}", recipe=self.recipe.name, stage="recipe-loop"
                ) from e
            self.global_step = result.global_step
            if self._close_epoch(epoch, {"train_loss": result.loss or 0.0, **result.metrics}, {}):
                break
        return self._end_training()

    # -- shared machinery ---------------------------------------------------------
    def _max_steps_reached(self) -> bool:
        return self.config.max_steps is not None and self.global_step >= self.config.max_steps

    def _begin_epoch(self, epoch: int, max_epochs: int, loader: Any = None) -> None:
        from topologies.training.reporting.events import EpochBeginEvent

        self.current_epoch = epoch
        # The loop's loader (possibly wrapped with a distributed sampler) gets the epoch.
        self._propagate_set_epoch(loader if loader is not None else self.train_loader, epoch)
        self._propagate_set_epoch(self.val_loader, epoch)
        self.callbacks.on_epoch_begin(self, epoch)
        self.reporter.on_epoch_begin(EpochBeginEvent(epoch=epoch, total_epochs=max_epochs))
        self._epoch_t0 = time.monotonic()
        self._last_checkpoint_path = None

    def _run_batch_epochs(self, step_fn: Callable[[Any, int, int], Any], *,
                          flush_accumulation: bool) -> dict:
        """Epochs of per-batch steps; ``step_fn(batch, batch_idx, epoch) -> metrics``."""
        loader = maybe_wrap_sampler(self._require_loader(), strategy=self.dist.strategy)
        max_epochs = self.config.max_epochs
        steps_per_epoch = self._safe_steps_per_epoch()
        for epoch in range(self.current_epoch, max_epochs):
            if self._stop_requested or self._max_steps_reached():
                break
            self._begin_epoch(epoch, max_epochs, loader)
            self.model.train()
            epoch_loss: Any = 0.0
            n_batches = updates = 0
            sums: dict[str, Any] = {}
            for batch_idx, batch in enumerate(loader):
                if self._max_steps_reached():
                    break
                self.callbacks.on_step_begin(self, self.global_step)
                self._run_optimization_step_hooks(self.global_step)
                before = self.global_step
                metrics = step_fn(batch, batch_idx, epoch)
                self.callbacks.on_step_end(self, self.global_step, metrics)
                epoch_loss = epoch_loss + _loss_value(metrics)
                n_batches += 1
                _accumulate(sums, metrics)
                if self.global_step != before:  # step events follow optimizer updates
                    updates += 1
                    self._maybe_step_checkpoint(epoch, n_batches)
                    self._emit_step(epoch, updates, steps_per_epoch, metrics)
                if self._stop_requested:
                    break  # finish this step, then close the epoch normally
            self._check_not_exhausted(loader, n_batches, epoch)
            if flush_accumulation and self._accum_count > 0:
                self._flush_accumulation()
                self._maybe_step_checkpoint(epoch, n_batches)
            val_metrics = self._maybe_validate(epoch)
            train = {"train_loss": _floats({"loss": epoch_loss})["loss"] / max(1, n_batches),
                     **_train_averages(sums, n_batches)}
            if self._close_epoch(epoch, train, val_metrics):
                break
        return self._end_training()

    def _emit_step(self, epoch: int, epoch_step: int, steps_per_epoch: int | None,
                   metrics: Any) -> None:
        if not self.reporter.wants_step(self.global_step):
            return  # no active sink: skip the event and its device->host copy
        from topologies.training.reporting.events import StepEvent

        event_metrics = dict(metrics)
        lr = self._current_lr()
        if lr is not None:
            event_metrics.setdefault("lr", float(lr))
        elapsed = time.monotonic() - self._train_t0 if self._train_t0 is not None else 0.0
        self.reporter.on_step(StepEvent(
            global_step=int(self.global_step), epoch=int(epoch), epoch_step=int(epoch_step),
            steps_per_epoch=steps_per_epoch, metrics=event_metrics, elapsed=float(elapsed)))

    def _close_epoch(self, epoch: int, train_metrics: dict, val_metrics: dict) -> bool:
        """History, logs, callbacks, early stopping, checkpoint, events. True = stop now."""
        from topologies.training.reporting.events import EpochEndEvent

        metrics = self._reduce_epoch_metrics({**train_metrics, **val_metrics}, val_metrics)
        self.history.append({"epoch": epoch, **metrics})
        self._log(epoch, metrics)
        self.callbacks.on_epoch_end(self, epoch, metrics)
        # Early-stopping state first, so the epoch's checkpoint records it.
        should_stop = self._check_early_stopping(metrics)
        self._maybe_checkpoint(epoch, metrics)
        barrier(self.dist)  # rank 0 finished writing before anyone resumes from it
        train = {k: v for k, v in metrics.items() if k not in val_metrics}
        duration = time.monotonic() - self._epoch_t0 if self._epoch_t0 is not None else 0.0
        self.reporter.on_epoch_end(EpochEndEvent(
            epoch=int(epoch), train_metrics=train, val_metrics=dict(val_metrics),
            best_metric=self.best_metric, duration=float(duration),
            checkpoint_path=self._last_checkpoint_path))
        if should_stop:
            self._early_stopped = True
            return True
        self.current_epoch = epoch + 1
        return self._stop_requested

    def _end_training(self) -> dict:
        from topologies.training.reporting.events import TrainEndEvent

        self.stop_reason = self._final_stop_reason()
        duration = time.monotonic() - self._train_t0 if self._train_t0 is not None else 0.0
        self.reporter.on_train_end(TrainEndEvent(
            total_epochs=int(self.current_epoch), global_step=int(self.global_step),
            duration=float(duration), best_metric=self.best_metric,
            best_epoch=self._best_epoch, history=list(self.history)))
        self.callbacks.on_train_end(self)
        return self._fit_result(self.current_epoch)

    def _reduce_epoch_metrics(self, metrics: dict, val_metrics: dict) -> dict[str, float]:
        """Average train metrics across ranks (one collective); validation metrics are
        already globally exact (``evaluate`` reduces weighted sums)."""
        out = {k: float(v) for k, v in metrics.items()}
        if self.dist.world_size <= 1:
            return out
        train = {k: v for k, v in out.items() if k not in val_metrics}
        out.update(all_reduce_mean_dict(train, self.dist, self.device))
        return out

    def _check_not_exhausted(self, loader: Any, n_batches: int, epoch: int) -> None:
        from topologies.training.engine import _is_one_shot

        if n_batches == 0 and _is_one_shot(loader):
            raise TrainingError(
                f"Training data is exhausted at epoch {epoch}: a one-shot iterator/generator "
                "can only be iterated once.",
                recipe=self.recipe.name, stage="data",
                hint="Pass a DataLoader, list or dataset to train for several epochs, or set "
                "max_epochs=1.")

    def _maybe_validate(self, epoch: int) -> dict[str, float]:
        from topologies.training.reporting.events import ValidationBeginEvent, ValidationEvent

        every = self.config.eval_every_epochs
        if every is None or self.val_loader is None or (epoch + 1) % max(1, every) != 0:
            return {}
        self.reporter.on_validation_begin(ValidationBeginEvent(epoch=int(epoch)))
        t0 = time.monotonic()
        metrics = self.validate()  # errors propagate as TrainingError
        self.reporter.on_validation_end(ValidationEvent(
            epoch=int(epoch), metrics=dict(metrics), duration=float(time.monotonic() - t0)))
        self.callbacks.on_validation_end(self, metrics)
        return metrics

    def _log(self, epoch: int, metrics: dict) -> None:
        # Event-aware loggers record epochs via on_epoch_end; only legacy
        # log()-only loggers get the epoch summary through log().
        if type(self.logger).on_epoch_end is not BaseLogger.on_epoch_end:
            return
        try:
            self.logger.log(self.global_step, {"epoch": epoch, **metrics})
        except Exception:  # noqa: BLE001, S110 - policy (b): a logger must not stop training
            pass
