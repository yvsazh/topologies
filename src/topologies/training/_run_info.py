"""Run metadata for observability: RunInfo, step totals, current learning rate."""

from __future__ import annotations

from typing import Any


class _RunInfoMixin:
    """Trainer mixin (split out of engine.py; state lives on the Trainer)."""

    def _current_lr(self) -> float | None:
        opt = self.optimizers.get("default") or next(iter(self.optimizers.values()), None)
        if opt is None or not opt.param_groups:
            return None
        return float(opt.param_groups[0].get("lr", 0.0))

    def _safe_steps_per_epoch(self) -> int | None:
        """Optimizer steps per epoch (None when the loader has no length)."""
        try:
            if self.train_loader is None:
                return None
            n = len(self.train_loader)  # type: ignore[arg-type]
        except TypeError:
            return None
        return self._optimizer_steps_per_epoch(n)

    def _safe_batch_size(self) -> int | None:
        bs = getattr(self.train_loader, "batch_size", None)
        return int(bs) if bs is not None else None

    def _compute_total_steps(self, steps_per_epoch: int | None) -> int | None:
        max_steps = self.config.max_steps
        if steps_per_epoch is not None and steps_per_epoch > 0:
            total = int(steps_per_epoch) * int(self.config.max_epochs)
            if max_steps is not None:
                total = min(total, int(max_steps))
            return max(1, total)
        return max(1, int(max_steps)) if max_steps is not None else None

    def _build_run_info(self, steps_per_epoch: int | None, batch_size: int | None) -> Any:
        from topologies.training.reporting.events import RunInfo, count_params

        total, trainable, frozen = count_params(self.model)
        arch_id = getattr(self.model, "architecture_id", None) or None
        # Optimizer / scheduler / lr / wd (the "default" optimizer, else the first one).
        cfg = self._optimizer_configs.get("default") or next(
            iter(self._optimizer_configs.values()), None)
        opt_name = lr_v = wd_v = sched_s = None
        if cfg is not None:
            name = str(cfg.name or "")
            opt_name = {"adamw": "AdamW", "adam": "Adam", "sgd": "SGD"}.get(
                name.lower(), name.upper() or None)
            lr_v = float(cfg.lr)
            wd_v = float(cfg.weight_decay or 0.0)
            sched_s = str(cfg.scheduler) if cfg.scheduler else None
        if opt_name is None and self.optimizers:
            opt_name = type(next(iter(self.optimizers.values()))).__name__
        current = self._current_lr()  # may differ from the config after scheduling
        if current is not None:
            lr_v = current
        if sched_s is None:
            sched_s = next((type(x).__name__ for x in self.schedulers.values() if x is not None),
                           None)
        cfg_run = self.config
        return RunInfo(
            model_name=type(self.model).__name__,
            architecture_id=str(arch_id) if arch_id is not None else None,
            param_total=total,
            param_trainable=trainable,
            param_frozen=frozen,
            recipe_name=self.recipe.name,
            recipe_level=self.recipe.level,
            device=str(self.device),
            precision=str(self.precision),
            optimizer_name=opt_name,
            lr=lr_v,
            weight_decay=wd_v,
            scheduler=sched_s,
            max_epochs=int(cfg_run.max_epochs),
            max_steps=int(cfg_run.max_steps) if cfg_run.max_steps is not None else None,
            steps_per_epoch=steps_per_epoch,
            batch_size=batch_size,
            accumulation_steps=int(cfg_run.accumulation_steps),
            seed=int(cfg_run.seed) if cfg_run.seed is not None else None,
            checkpoint_dir=str(cfg_run.checkpoint_dir) if cfg_run.checkpoint_dir else None,
            eval_every=(int(cfg_run.eval_every_epochs)
                        if cfg_run.eval_every_epochs is not None else None),
            run_dir=str(cfg_run.log_dir) if cfg_run.log_dir else None,
            world_size=int(self.dist.world_size),
            rank=int(self.dist.rank),
            global_step=int(self.global_step),
            elapsed=0.0,
        )
