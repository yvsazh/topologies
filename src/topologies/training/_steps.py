"""Per-batch training steps and optimizer services (also used by Level 2/3 recipe contexts)."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from topologies.training._step_metrics import (
    _batch_size_of,
    _StepMetrics,
)
from topologies.training.device import autocast_kwargs
from topologies.training.errors import TrainingError
from topologies.training.recipe_base import (
    StepContext,
)


class _StepsMixin:
    """Trainer mixin (split out of engine.py; state lives on the Trainer)."""

    def _count_batch(self, batch: dict) -> None:
        self.samples_seen += _batch_size_of(batch)
        tokens = batch.get("tokens")
        if not isinstance(tokens, torch.Tensor):
            return
        mask = batch.get("mask")
        if isinstance(mask, torch.Tensor) and mask.shape == tokens.shape:
            n = mask.detach().sum()  # stays on device: no sync per step
            acc = self._tokens_seen_acc
            self._tokens_seen_acc = n if acc is None else acc + n.to(acc.device)
        else:
            self._tokens_seen += int(tokens.numel())

    def _require_loader(self) -> DataLoader:
        if self.train_loader is None:
            raise TrainingError(
                f"Recipe {self.recipe.name!r} (level={self.recipe.level}) requires "
                "train_loader, but none was provided.",
                recipe=self.recipe.name,
                stage="config",
                hint="Pass train_loader=... or use a Level-3 recipe with env/components.",
            )
        return self.train_loader

    def _batch_to_dict(self, batch: Any) -> dict:
        """Map tuple/tensor batches onto the recipe's required keys (spec order)."""
        if isinstance(batch, dict):
            return batch
        keys = self.recipe.required_batch_keys()
        all_keys = list(self.recipe.expected_batch_spec())
        if isinstance(batch, (tuple, list)):
            # Required keys first; recipes whose input key is optional (e.g.
            # distillation: tokens|pixels) get "<input>, <required...>".
            targets = self.recipe.target_keys()
            with_targets = [k for k in all_keys if k in keys or k in targets]
            candidates = [keys, with_targets, all_keys]
            if not self.recipe.input_keys():
                optional_inputs = [k for k in all_keys if k not in keys and k not in targets]
                candidates += [[o, *keys] for o in optional_inputs]
            for candidate in candidates:
                if candidate and len(candidate) == len(batch):
                    return dict(zip(candidate, batch))
        elif isinstance(batch, torch.Tensor) and len(keys) >= 1:
            inputs = self.recipe.input_keys() or keys
            if len(keys) == 1 or (
                len(inputs) == 1 and self.recipe.optional_batch_keys() >= set(keys) - set(inputs)
            ):
                return {inputs[0]: batch}
        n = len(batch) if isinstance(batch, (tuple, list)) else 1
        raise TrainingError(
            "Training failed during batch validation.",
            recipe=self.recipe.name,
            stage="batch",
            expected="\n".join(
                f"    {k}: {v}" for k, v in self.recipe.expected_batch_spec().items()
            ),
            received=f"type={type(batch).__name__} with {n} field(s) (expected dict batch)",
            hint=(
                f"Tuple batches are mapped positionally to {keys}; otherwise wrap the "
                "dataset: topologies.data.DictDataset(ds, keys=(...)) or "
                f"topologies.data.for_recipe(ds, {self.recipe.name!r})."
            ),
        )

    def _prepare_batch(self, batch: Any) -> dict:
        batch = self._batch_to_dict(batch)
        try:
            adapted = self.adapter.adapted_batch(batch)
        except Exception as e:
            raise TrainingError(
                f"Adapter failed: {e}", stage="adapter", recipe=self.recipe.name
            ) from e
        moved = self._move_batch(adapted)
        # Strict batch checking against the recipe schema.
        self.recipe.validate_batch(moved, strict=self.config.strict_batch)
        return moved

    def _call_batch_to_loss(self, prepared: dict):
        fn = self.recipe.batch_to_loss
        if self._batch_to_loss_takes_adapter is None:  # resolved once, not per step
            import inspect as _insp

            try:
                params = _insp.signature(fn).parameters
                # Bound method: (model, batch, cfg, components[, adapter]).
                self._batch_to_loss_takes_adapter = len(params) >= 5
            except (ValueError, TypeError):
                self._batch_to_loss_takes_adapter = False
        if self._batch_to_loss_takes_adapter and self._user_adapter:
            return fn(self.model, prepared, self.recipe_config, self.components, self.adapter)
        return fn(self.model, prepared, self.recipe_config, self.components)

    def _maybe_preflight(self, prepared: dict) -> None:
        if self._preflight_done:
            return
        self._preflight_done = True
        try:
            self.recipe.preflight(
                self.model, prepared, self.recipe_config, self.components, self.adapter
            )
        except (TrainingError, NotImplementedError) as e:
            if isinstance(e, NotImplementedError):
                return  # the recipe has no checks
            raise
        except Exception as e:  # a failing check is reported, not hidden
            raise TrainingError(f"Recipe preflight check failed: {e}", recipe=self.recipe.name,
                                stage="preflight") from e

    def _loss_step(self, batch: Any, batch_idx: int, epoch: int) -> dict[str, float]:
        prepared = self._prepare_batch(batch)
        self._maybe_preflight(prepared)
        self._count_batch(prepared)
        name = "default"
        if self._accum_count == 0:
            self._opt_zero_grad(name)
        self.model.train()
        autocast = autocast_kwargs(self.device, self.precision)
        try:
            if autocast:
                with torch.autocast(**autocast):
                    loss, metrics = self._call_batch_to_loss(prepared)
            else:
                loss, metrics = self._call_batch_to_loss(prepared)
        except TrainingError:
            raise
        except Exception as e:
            raise TrainingError(
                f"Recipe batch_to_loss failed: {e}",
                recipe=self.recipe.name,
                stage="loss",
                device=str(self.device),
                precision=self.precision,
            ) from e
        if not isinstance(loss, torch.Tensor):
            raise TrainingError(
                "batch_to_loss must return (Tensor, dict).", recipe=self.recipe.name, stage="loss"
            )
        from topologies.training.aux import fold_leftover_aux

        loss, metrics = fold_leftover_aux(loss, metrics, self.model)
        scaled = loss / max(1, self.config.accumulation_steps)
        self._opt_backward(scaled, name)
        self._accum_count += 1
        self.micro_step += 1
        grad_norm = None
        if self._accum_count >= self.config.accumulation_steps:
            grad_norm = self._finish_accumulation(name, as_tensor=True)
        out = {k: (v.detach() if isinstance(v, torch.Tensor) else v)
               for k, v in metrics.items()}
        out.setdefault("loss", loss.detach())
        if grad_norm is not None:
            out["grad_norm"] = grad_norm
        # Values stay on the device; they are copied to the host only when read
        # (step events for active sinks, callbacks, epoch close).
        return _StepMetrics(out)

    def _finish_accumulation(self, name: str = "default", *, rescale: float = 1.0,
                             as_tensor: bool = False) -> Any:
        """Clip, step optimizer + scheduler for the current window; one optimizer update."""
        if rescale != 1.0:
            for p in self._params_for(name):
                if p.grad is not None:
                    p.grad.mul_(rescale)
        grad_norm = self._opt_clip(name, as_tensor=as_tensor)
        self._opt_step(name)
        self._sched_step(name)
        self._accum_count = 0
        self.global_step += 1
        return grad_norm

    def _flush_accumulation(self) -> None:
        """Step a partial accumulation window at epoch end (loss is rescaled to its mean).

        Each micro-loss was divided by ``accumulation_steps``; a window of ``k`` micro-batches
        is scaled by ``accumulation_steps / k`` so it averages like a full window.
        """
        if self._accum_count <= 0:
            return
        k = self._accum_count
        self._finish_accumulation("default", rescale=self.config.accumulation_steps / k)

    def _custom_step(self, batch: Any, batch_idx: int, epoch: int) -> dict[str, float]:
        prepared = self._prepare_batch(batch)
        self._maybe_preflight(prepared)
        self._count_batch(prepared)
        self._last_grad_norm = None

        ctx = StepContext(
            model=self.model,
            batch=prepared,
            batch_idx=batch_idx,
            epoch=epoch,
            global_step=self.global_step,
            device=self.device,
            precision=self.precision,
            scaler=self.scaler,
            accumulation_steps=self.config.accumulation_steps,
            is_accumulating=self._accum_count > 0,
            rank=self.dist.rank,
            world_size=self.dist.world_size,
            config=self.recipe_config,
            components=self.components,
            adapter=self.adapter,
            logger=self.logger,
            metrics={},
            _backward_fn=self._opt_backward,
            _zero_grad_fn=self._opt_zero_grad,
            _step_optimizer_fn=self._opt_step,
            _step_scheduler_fn=self._sched_step,
            _clip_fn=self._opt_clip,
            _log_fn=None,
        )
        # Wrap autocast around the whole custom step for simplicity.
        autocast = autocast_kwargs(self.device, self.precision)
        self._optimizer_stepped = False
        try:
            if autocast:
                with torch.autocast(**autocast):
                    result = self.recipe.training_step(ctx)
            else:
                result = self.recipe.training_step(ctx)
        except TrainingError:
            raise
        except Exception as e:
            raise TrainingError(
                f"Recipe training_step failed: {e}", recipe=self.recipe.name, stage="recipe-loop"
            ) from e
        self.micro_step += 1
        if self._optimizer_stepped:  # any optimizer update in this step = one global step
            self.global_step += 1
        # Merge StepContext.log values with StepResult.metrics (result wins).
        merged: dict[str, float] = {}
        for source in (ctx.metrics or {}, result.metrics or {}):
            for k, v in source.items():
                try:
                    merged[str(k)] = float(v)  # type: ignore[arg-type]
                except (TypeError, ValueError):  # non-numeric extras are not step metrics
                    continue
        out = dict(merged)
        out.setdefault("loss", float(result.loss))
        if self._last_grad_norm is not None and "grad_norm" not in out:
            out["grad_norm"] = self._last_grad_norm
        return out

    def _opt_zero_grad(self, name: str | None = None) -> None:
        targets = [name] if name else list(self.optimizers.keys())
        for n in targets:
            opt = self.optimizers.get(n)
            if opt is None:
                raise TrainingError(
                    f"Unknown optimizer {n!r}.",
                    stage="optimizer",
                    optimizer_name=n,
                    available=sorted(self.optimizers.keys()),
                )
            try:
                opt.zero_grad(set_to_none=True)
            except TypeError:  # optimizers without the set_to_none argument
                opt.zero_grad()

    def _opt_backward(self, loss: torch.Tensor, name: str = "default") -> None:
        try:
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
        except Exception as e:
            raise TrainingError(
                f"Backward failed (optimizer={name}): {e}",
                stage="backward",
                optimizer_name=name,
                device=str(self.device),
                precision=self.precision,
            ) from e

    def _opt_clip(self, name: str = "default", *, as_tensor: bool = False) -> Any:
        """Unscale (AMP), clip, and return the total gradient L2 norm (pre-clip)."""
        clip_norm = self.config.clip_grad_norm
        cfg = self._optimizer_configs.get(name)
        if cfg is not None and cfg.clip_grad_norm is not None:
            clip_norm = cfg.clip_grad_norm
        params = [p for p in self._params_for(name) if p.grad is not None]
        norm: Any = None
        try:
            if self.scaler is not None:
                opt = self.optimizers.get(name)
                if opt is not None:
                    self.scaler.unscale_(opt)
            if params:
                max_norm = float(clip_norm) if clip_norm is not None else float("inf")
                total = torch.nn.utils.clip_grad_norm_(params, max_norm)
                norm = total.detach() if as_tensor else float(total)
            if self.config.clip_grad_value is not None and params:
                torch.nn.utils.clip_grad_value_(params, float(self.config.clip_grad_value))
        except Exception as e:
            raise TrainingError(
                f"Gradient clipping failed: {e}", stage="backward", optimizer_name=name
            ) from e
        if norm is not None:
            self._last_grad_norm = norm
        return norm

    def _opt_step(self, name: str = "default") -> None:
        opt = self.optimizers.get(name)
        if opt is None:
            raise TrainingError(
                f"Unknown optimizer {name!r}.",
                stage="optimizer",
                optimizer_name=name,
                available=sorted(self.optimizers.keys()),
            )
        try:
            if self.scaler is not None:
                self.scaler.step(opt)
                self.scaler.update()
            else:
                opt.step()
            self._optimizer_stepped = True
            for hook in tuple(self._hooks["after_optimizer_step"]):
                hook(self, name)
        except Exception as e:
            raise TrainingError(
                f"Optimizer step failed ({name}): {e}", stage="optimizer", optimizer_name=name
            ) from e

    def _sched_step(self, name: str = "default") -> None:
        sched = self.schedulers.get(name)
        if sched is None:
            return
        try:
            sched.step()
        except Exception as e:
            raise TrainingError(
                f"Scheduler step failed ({name}): {e}", stage="scheduler", optimizer_name=name
            ) from e
