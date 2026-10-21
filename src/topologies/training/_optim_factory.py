"""Optimizer and scheduler construction for the Trainer (names, configs, classes, instances, factories)."""

from __future__ import annotations

from typing import Any

import torch

from topologies.training.configs import (
    OptimizerConfig,
)
from topologies.training.errors import TrainingError


class _OptimizerFactoryMixin:
    """Trainer mixin (split out of engine.py; state lives on the Trainer)."""

    def _build_optimizers(self, optimizer: Any, optimizers: dict | None, scheduler: Any) -> None:
        if optimizers is not None and optimizer not in ("auto", None):
            # Both forms given: 'optimizer' must be default-compatible; treat
            # 'optimizer' as the default entry override.
            merged: dict[str, Any] = dict(optimizers)
            if "default" not in merged:
                merged["default"] = optimizer
            optimizers = merged
            optimizer = "auto"
        if optimizers is not None:
            if not isinstance(optimizers, dict) or not optimizers:
                raise TrainingError(
                    "optimizers must be a non-empty mapping name -> config.",
                    stage="optimizer",
                )
            for name, entry in optimizers.items():
                self._add_optimizer(name, entry, scheduler if name == "default" else None)
        else:
            entry = optimizer if optimizer not in ("auto", None) else "auto"
            self._add_optimizer("default", entry, scheduler)

    def _add_optimizer(self, name: str, entry: Any, scheduler_override: Any) -> None:
        params = self._params_for(name)
        kind, cfg, instance, factory = self._normalize_entry(entry)
        opt: Any
        if kind == "instance":
            if not isinstance(instance, torch.optim.Optimizer):
                raise TrainingError(
                    f"Optimizer {name!r} instance is not a torch Optimizer.",
                    stage="optimizer",
                    optimizer_name=name,
                    received=type(instance).__name__,
                )
            opt = instance
            self._optimizer_configs[name] = None
        elif kind == "factory":
            try:
                opt = factory(params)
            except Exception as e:
                raise TrainingError(
                    f"Optimizer factory for {name!r} failed: {e}",
                    stage="optimizer",
                    optimizer_name=name,
                ) from e
            self._optimizer_configs[name] = None
        else:
            assert cfg is not None
            if scheduler_override is not None:
                cfg = self._apply_scheduler_override(cfg, scheduler_override, name)
            if name == "default" and self._optimizer_kwargs:
                cfg = cfg.model_copy(update={"kwargs": {**cfg.kwargs, **self._optimizer_kwargs}})
            try:
                opt = self._create_optimizer(cfg, params, cls=instance if kind == "class" else None)
            except TrainingError:
                raise
            except Exception as e:
                raise TrainingError(
                    f"Failed to create optimizer {name!r}: {e}",
                    stage="optimizer",
                    optimizer_name=name,
                ) from e
            self._optimizer_configs[name] = cfg
        self.optimizers[name] = opt
        sched = self._create_scheduler(
            name, opt, cfg if kind in ("config", "class") else None, scheduler_override
        )
        self.schedulers[name] = sched

    def _params_for(self, name: str):
        # Never fall back to *all* parameters: that would silently train frozen /
        # teacher weights (distillation, GAN) when the recipe has a bug.
        try:
            return self.recipe.optimizer_params(name, self.model, self.components)
        except TrainingError:
            raise
        except Exception as e:  # recipe code
            raise TrainingError(f"Recipe optimizer_params({name!r}) failed: {e}",
                                recipe=self.recipe.name, stage="optimizer",
                                optimizer_name=name) from e

    def _normalize_entry(self, entry: Any) -> tuple[str, OptimizerConfig | None, Any, Any]:
        # instance > class > factory > named configuration > recipe default.
        if isinstance(entry, torch.optim.Optimizer):
            return "instance", None, entry, None
        if isinstance(entry, type) and issubclass(entry, torch.optim.Optimizer):
            # Optimizer class: hyperparameters come from the policy (lr=, weight_decay=)
            # plus optimizer_kwargs=; the recipe's scheduler still applies.
            return "class", self._policy_to_config(self._policy), entry, None
        if callable(entry) and not isinstance(entry, (str, dict, OptimizerConfig)):
            return "factory", None, None, entry
        if entry in ("auto", None):
            cfg = self._policy_to_config(self._policy)
            return "config", cfg, None, None
        if isinstance(entry, str):
            base = dict(self._policy)
            base["optimizer"] = entry
            return "config", self._policy_to_config(base), None, None
        if isinstance(entry, OptimizerConfig):
            return "config", entry, None, None
        if isinstance(entry, dict):
            d = dict(entry)
            # Detect instance/factory embedded in dict.
            if "instance" in d and isinstance(d["instance"], torch.optim.Optimizer):
                if any(k in d for k in ("name", "factory", "lr", "scheduler")):
                    raise TrainingError(
                        "Conflicting optimizer specification: 'instance' combined with "
                        "name/factory/hyperparameters.",
                        stage="optimizer",
                        hint="Use either instance=... alone or name-based config, not both.",
                    )
                return "instance", None, d["instance"], None
            if "factory" in d and callable(d["factory"]):
                if any(k in d for k in ("name", "instance", "lr", "scheduler")):
                    raise TrainingError(
                        "Conflicting optimizer specification: 'factory' combined with "
                        "name/instance/hyperparameters.",
                        stage="optimizer",
                        hint="Use either factory=... alone or name-based config, not both.",
                    )
                return "factory", None, None, d["factory"]
            base = dict(self._policy)
            name_map = {"optimizer": "optimizer", "name": "optimizer"}
            allowed = set(OptimizerConfig.model_fields) | {
                "optimizer",
                "instance",
                "factory",
                "scheduler_instance",
                "scheduler_factory",
            }
            unknown = [k for k in d if k not in allowed]
            if unknown:
                import difflib

                sug = difflib.get_close_matches(str(unknown[0]), sorted(allowed), n=1)
                raise TrainingError(
                    f"Unknown optimizer option {unknown[0]!r}.",
                    stage="optimizer",
                    hint=(f"Did you mean {sug[0]!r}? " if sug else "")
                    + "Optimizer-specific extras go in kwargs={...}.",
                    available=sorted(OptimizerConfig.model_fields),
                )
            for k, v in d.items():
                if k in ("instance", "factory"):
                    continue
                base[name_map.get(k, k)] = v
            # Scheduler instance/factory inside dict handled at scheduler stage.
            return "config", self._policy_to_config(base), None, None
        raise TrainingError(
            f"Unsupported optimizer entry {entry!r}.",
            stage="optimizer",
            hint="Use a name ('adamw'), OptimizerConfig, instance, or factory.",
        )

    def _policy_to_config(self, policy: dict) -> OptimizerConfig:
        data: dict[str, Any] = {
            "name": str(policy.get("optimizer", policy.get("name", "adamw"))),
            "lr": float(policy.get("lr", 3e-4)),
            "weight_decay": float(policy.get("weight_decay", 0.0)),
        }
        if policy.get("scheduler") is not None:
            data["scheduler"] = policy.get("scheduler")
        if policy.get("momentum") is not None:
            data["momentum"] = float(policy["momentum"])
        if policy.get("betas") is not None:
            data["betas"] = tuple(policy["betas"])  # type: ignore[assignment]
        extra_sched = policy.get("scheduler_kwargs", {})
        if extra_sched:
            data["scheduler_kwargs"] = dict(extra_sched)
        # Remaining OptimizerConfig fields (eps, kwargs, warmup, ...) pass through.
        for k in ("eps", "clip_grad_norm", "kwargs", "warmup_steps", "warmup_ratio",
                  "decay_policy"):
            if policy.get(k) is not None:
                data[k] = policy[k]
        # Carry scheduler instance/factory keys through for scheduler stage.
        for k in ("scheduler_instance", "scheduler_factory"):
            if k in policy:
                data.setdefault("scheduler_kwargs", {})[k] = policy[k]
        try:
            return OptimizerConfig(**data)
        except Exception as e:
            raise TrainingError(f"Invalid optimizer policy {data}: {e}", stage="optimizer") from e

    def _apply_scheduler_override(
        self, cfg: OptimizerConfig, scheduler: Any, name: str
    ) -> OptimizerConfig:
        d = cfg.model_dump(mode="python")
        if isinstance(scheduler, str):
            d["scheduler"] = scheduler
        elif scheduler is None:
            d["scheduler"] = None
        else:
            d.setdefault("scheduler_kwargs", {})["__override__"] = scheduler
        try:
            return OptimizerConfig(**d)
        except Exception as e:
            raise TrainingError(
                f"Invalid scheduler override: {e}", stage="scheduler", optimizer_name=name
            ) from e

    def _create_optimizer(
        self, cfg: OptimizerConfig, params: Any, cls: type | None = None
    ) -> torch.optim.Optimizer:
        from topologies.training.registries import build_optimizer

        try:
            plist = list(params)
        except TypeError:  # not an iterable of parameters
            plist = []
        if not plist:
            raise TrainingError(
                "Model has no trainable parameters for the optimizer.",
                stage="optimizer",
                hint="Pass a model with nn.Parameters, or provide an optimizer "
                "instance bound to the right parameters.",
            )
        return build_optimizer(cls if cls is not None else cfg.name, plist, cfg, model=self.model)

    def _create_scheduler(
        self, name: str, opt: Any, cfg: OptimizerConfig | None, override: Any
    ) -> Any:
        sched = self._create_base_scheduler(name, opt, cfg, override)
        warmup = self._warmup_steps(cfg)
        if warmup <= 0:
            return sched
        from topologies.training.registries import WarmupScheduler

        return WarmupScheduler(opt, warmup, sched)

    def _warmup_steps(self, cfg: OptimizerConfig | None) -> int:
        if cfg is None:
            return 0
        if cfg.warmup_steps:
            return int(cfg.warmup_steps)
        if cfg.warmup_ratio:
            return round(cfg.warmup_ratio * self._estimate_total_steps())
        return 0

    def _create_base_scheduler(
        self, name: str, opt: Any, cfg: OptimizerConfig | None, override: Any
    ) -> Any:
        from topologies.training.registries import (
            build_scheduler,
            call_scheduler_factory,
            canonical_scheduler,
            list_schedulers,
        )

        # Precedence: instance > factory > named configuration.
        sched_entry = override
        if cfg is not None:
            extra = cfg.scheduler_kwargs or {}
            if "__override__" in extra:
                sched_entry = extra["__override__"]
            elif "scheduler_instance" in extra:
                sched_entry = extra["scheduler_instance"]
            elif "scheduler_factory" in extra:
                sched_entry = extra["scheduler_factory"]
            elif isinstance(override, dict):
                if "scheduler_instance" in override:
                    sched_entry = override["scheduler_instance"]
                elif "scheduler_factory" in override:
                    sched_entry = override["scheduler_factory"]
        # Schedules span the steps that remain after warmup.
        total = max(1, self._estimate_total_steps() - self._warmup_steps(cfg))
        if sched_entry is not None and not isinstance(sched_entry, str):
            if callable(sched_entry) and not hasattr(sched_entry, "step"):
                try:
                    return call_scheduler_factory(sched_entry, opt, total)
                except Exception as e:
                    raise TrainingError(
                        f"Scheduler factory for {name!r} failed: {e}",
                        stage="scheduler",
                        optimizer_name=name,
                    ) from e
            if hasattr(sched_entry, "step"):
                return sched_entry
        sched_name = None
        sched_kwargs: dict[str, Any] = {}
        if isinstance(sched_entry, str):
            sched_name = sched_entry.lower()
        elif cfg is not None and cfg.scheduler:
            sched_name = cfg.scheduler.lower()
            sched_kwargs = dict(cfg.scheduler_kwargs or {})
            sched_kwargs.pop("__override__", None)
            sched_kwargs.pop("scheduler_instance", None)
            sched_kwargs.pop("scheduler_factory", None)
        if not sched_name or sched_name == "none":
            return None
        if canonical_scheduler(sched_name) not in list_schedulers():
            raise TrainingError(
                f"Unknown scheduler {sched_name!r}.",
                stage="scheduler",
                optimizer_name=name,
                available=list_schedulers(),
            )
        try:
            return build_scheduler(sched_name, opt, total, sched_kwargs)
        except TrainingError:
            raise
        except Exception as e:
            raise TrainingError(
                f"Failed to create scheduler {sched_name!r} for {name!r}: {e}",
                stage="scheduler",
                optimizer_name=name,
            ) from e

    def _estimate_total_steps(self) -> int:
        """Optimizer steps the run will take (``max_steps`` caps epochs x steps per epoch)."""
        try:
            n = len(self.train_loader) if self.train_loader is not None else 10
        except TypeError:
            n = 10
        per_epoch = self._optimizer_steps_per_epoch(int(n))
        total = max(1, per_epoch * max(1, self.config.max_epochs))
        if self.config.max_steps is not None:
            total = max(1, min(total, int(self.config.max_steps)))
        return total

    def _optimizer_steps_per_epoch(self, n_batches: int) -> int:
        """Optimizer updates per epoch over ``n_batches`` (partial windows are flushed)."""
        if self.recipe.level != "loss":
            return int(n_batches)
        acc = max(1, int(self.config.accumulation_steps))
        return -(-int(n_batches) // acc)
