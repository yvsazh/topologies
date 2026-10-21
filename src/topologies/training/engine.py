"""Generic Trainer execution engine (no task-specific knowledge)."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from topologies._execution import model_execution
from topologies._util.seed import seed_everything
from topologies.training._checkpointing import _CheckpointMixin
from topologies.training._loops import _LoopsMixin
from topologies.training._observability import _VISUALIZERS_UNSET, _ObservabilityMixin
from topologies.training._optim_factory import _OptimizerFactoryMixin
from topologies.training._run_info import _RunInfoMixin
from topologies.training._step_metrics import (
    _batch_size_of,
    _concat_outputs,
    _to_cpu,
)
from topologies.training._steps import _StepsMixin
from topologies.training._trainer_config import config_input, merge_trainer_config
from topologies.training.adapters import ModelAdapter, ModelSpec
from topologies.training.callbacks import Callback, CallbackList
from topologies.training.configs import (
    OptimizerConfig,
    TrainerConfig,
    validate_with_hints,
)
from topologies.training.device import (
    move_batch_to_device,
    resolve_device,
    resolve_precision,
)
from topologies.training.distributed import (
    DistState,
    get_dist_state,
    maybe_wrap_sampler,
    resolve_strategy,
    wrap_model_ddp,
)
from topologies.training.errors import TrainingError
from topologies.training.logging import BaseLogger
from topologies.training.monitoring import BaseMonitor
from topologies.training.recipe_base import (
    FitContext,
    Recipe,
    RecipeConfig,
)


def _set_seed(seed: int | None) -> None:
    if seed is not None:
        seed_everything(seed)


def _yaml_training_defaults(model: nn.Module) -> dict:
    """Registry ``training.defaults`` for a catalog model, from the packaged projection.

    ``training/model_recipes.json`` ships in the wheel (registry YAML does not),
    so this never reads YAML or depends on the current directory.
    """
    arch_id = getattr(model, "architecture_id", "") or ""
    if not arch_id:
        return {}
    from topologies.training.resolve import _packaged_model_spec

    spec = _packaged_model_spec(str(arch_id)) or {}
    return dict(spec.get("defaults") or {})


def _is_one_shot(data: Any) -> bool:
    """True for iterators/generators: iterating them consumes them.

    Checked by type, never by calling ``iter()``: creating a shuffled DataLoader
    iterator draws from the global RNG and would shift seeded runs.
    """
    return isinstance(data, Iterator)


class _PeekedIterator:
    """A one-shot iterator whose first item was already taken; yields it first."""

    def __init__(self, first: Any, rest: Any) -> None:
        self._pending = [first]
        self._rest = rest

    def __iter__(self) -> _PeekedIterator:
        return self

    def __next__(self) -> Any:
        if self._pending:
            return self._pending.pop()
        return next(self._rest)


def _peek_batch(loader: Any) -> dict | None:
    """First batch of a re-iterable loader (dicts only); never consumes iterators."""
    if loader is None or _is_one_shot(loader):
        return None
    try:
        it = iter(loader)
        b = next(it)
        return b if isinstance(b, dict) else None
    except Exception:  # noqa: BLE001 - user loaders may raise anything; resolution falls back
        return None


def _peek_one_shot(loader: Any) -> tuple[dict | None, Any]:
    """Peek a one-shot iterator: ``(batch_or_None, iterator_that_still_yields_it)``."""
    try:
        first = next(loader)
    except StopIteration:
        return None, iter(())
    return (first if isinstance(first, dict) else None), _PeekedIterator(first, loader)


HookEvent = Literal["before_optimizer_step", "after_optimizer_step"]
HOOK_EVENTS: tuple[str, ...] = ("before_optimizer_step", "after_optimizer_step")


class HookHandle:
    """Returned by ``Trainer.add_hook``; ``remove()`` unregisters the hook."""

    def __init__(self, hooks: list[Any], fn: Any) -> None:
        self._hooks, self.fn = hooks, fn

    def remove(self) -> None:
        if self.fn in self._hooks:
            self._hooks.remove(self.fn)














class Trainer(_ObservabilityMixin, _OptimizerFactoryMixin, _RunInfoMixin, _StepsMixin, _CheckpointMixin, _LoopsMixin):
    """Generic execution engine. All algorithm knowledge lives in Recipes.

    ``Trainer(model, train, val=None, *, recipe=None, **config)``

    - ``model``: any ``nn.Module``.
    - ``train_loader`` / ``val_loader``: DataLoaders, iterables of batches, or
      map-style datasets (wrapped using ``batch_size`` / ``num_workers``).
      Batches may be dicts, ``(x, y)`` tuples or tensors (mapped onto the
      recipe's batch contract).
    - ``recipe``: name, ``Recipe`` instance, or ``None`` to resolve from
      ``ModelSpec`` / registry metadata / batch keys.
    - ``recipe_config``: strict per-recipe options (typo hints on errors).
    - ``tokenizer``: checked against the model/recipe vocabulary and bundled
      by ``save_pretrained``.
    - ``optimizer``: registered name (``list_optimizers()``), ``OptimizerConfig``
      / dict, ``torch.optim.Optimizer`` subclass (with ``optimizer_kwargs=``),
      instance, or factory ``fn(params)``. ``scheduler``: registered name or
      factory ``fn(optimizer)`` / ``fn(optimizer, total_steps)``.
    - ``**config``: ``TrainerConfig`` fields, e.g. ``max_epochs``,
      ``max_steps``, ``batch_size``, ``lr``, ``weight_decay``, ``device``,
      ``precision``, ``checkpoint_dir``, ``resume``, ``clip_grad_norm``,
      ``early_stopping_patience``, ``monitor``, ``logger``, ``log_dir``.

    After ``fit()``: ``evaluate(data)``, ``predict(data)``,
    ``save_pretrained(path)``, ``history``, ``stop_reason``, ``samples_seen``,
    ``tokens_seen``. Callbacks may call ``trainer.request_stop(reason)`` to
    end training cleanly after the current step.
    """

    def analyze(
        self,
        analysis: Any,
        context: Any = None,
        *,
        include_history: bool = True,
        include_model: bool = False,
    ) -> Any:
        """Analyze a read-only snapshot; model input is included only when explicitly selected."""
        from topologies.analysis import AnalysisContext, analyze
        from topologies.analysis.integrations import from_trainer, trainer_provenance

        base = context or AnalysisContext()
        selected = from_trainer(self, include_history=include_history, include_model=include_model)
        # A model has no provenance of its own: describe the Trainer that holds it.
        source = getattr(selected, "source_provenance", None) if not include_model else None
        source = source or trainer_provenance(self)
        ctx = AnalysisContext(
            parameters=dict(base.parameters), metadata={**dict(base.metadata), "trainer": source}
        )
        return analyze(selected, analysis, context=ctx)

    def visualize(self, *, visualizer: Any = "matplotlib", **options: Any) -> Any:
        """Explicitly render a detached snapshot of this Trainer's history."""
        from copy import deepcopy

        from topologies.visualization import visualize

        return visualize(deepcopy(self.history), visualizer=visualizer, **options)

    def __init__(
        self,
        model: nn.Module,
        train_loader: Any = None,
        val_loader: Any = None,
        *,
        recipe: Any = None,
        recipe_config: dict | RecipeConfig | None = None,
        components: dict | Any | None = None,
        optimizer: Any = "auto",
        optimizers: dict[str, Any] | None = None,
        scheduler: Any = None,
        config: TrainerConfig | dict | None = None,
        adapter: ModelAdapter | None = None,
        input_map: dict[str, str] | None = None,
        forward_map: Callable | None = None,
        output_map: Callable | None = None,
        model_spec: ModelSpec | dict | None = None,
        callbacks: list[Callback] | None = None,
        logger: BaseLogger | str | None = None,
        monitor: BaseMonitor | str | None = None,
        visualizer: Any = _VISUALIZERS_UNSET,
        visualizers: Any = _VISUALIZERS_UNSET,
        monitoring: Any | None = None,
        logging_cfg: Any | None = None,
        allow_ambiguous_recipe: bool = False,
        tokenizer: Any = None,
        optimizer_kwargs: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        **trainer_kwargs: Any,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TrainingError(
                "Trainer model must be a torch.nn.Module.",
                stage="config",
                received=type(model).__name__,
            )
        self.model = model
        self._hooks: dict[str, list[Any]] = {event: [] for event in HOOK_EVENTS}
        self.metrics = dict(metrics or {})
        self.train_loader = train_loader
        self.val_loader = val_loader
        # -- TrainerConfig merge: explicit kwargs > monitoring/logging_cfg > config.
        self.config = merge_trainer_config(
            config, trainer_kwargs, allow_ambiguous_recipe=allow_ambiguous_recipe,
            monitoring=monitoring, logging_cfg=logging_cfg)
        # Remember explicitness for monitor resolution.
        self._config_input_dict = config_input(config)
        self._explicit_kw_keys = set(trainer_kwargs)
        self._monitor_arg = monitor
        self._logger_arg = logger
        if visualizer is not _VISUALIZERS_UNSET and visualizers is not _VISUALIZERS_UNSET:
            raise TrainingError("Pass visualizer or visualizers, not both.", stage="config")
        self._visualizer_selection = (
            visualizers if visualizers is not _VISUALIZERS_UNSET else visualizer
        )
        self._check_checkpoint_policy()
        _set_seed(self.config.seed)
        # -- device / precision.
        try:
            self.device = resolve_device(self.config.device)
        except Exception as e:
            raise TrainingError(f"Bad device: {e}", stage="config") from e
        # -- distributed state (torchrun is the only launch path).
        self.dist: DistState = get_dist_state()
        self.strategy = resolve_strategy(self.config.strategy, self.config.devices)
        if self.strategy == "ddp" and self.dist.world_size <= 1:
            raise TrainingError(
                "Distributed training was requested but no process group is initialized.",
                stage="config", config_key="devices",
                hint=f"launch with: torchrun --nproc_per_node=N your_script.py and call "
                "torch.distributed.init_process_group(...) before Trainer(...). "
                f"(strategy={self.config.strategy!r}, devices={self.config.devices!r})")
        if (self.dist.world_size > 1 and self.config.device == "auto"
                and torch.cuda.is_available()):
            self.device = torch.device("cuda", self.dist.local_rank)
            torch.cuda.set_device(self.device)
        self.precision = resolve_precision(self.config.precision, self.device)
        self._pin_memory = self._pin_memory_enabled()
        # -- datasets -> loaders (batch_size / num_workers from config).
        self.tokenizer = tokenizer
        self.train_loader = train_loader = self._as_loader(train_loader, shuffle=True)
        self.val_loader = val_loader = self._as_loader(val_loader, shuffle=False)
        # -- recipe resolution.
        from topologies.training.resolve import resolve_recipe

        if _is_one_shot(train_loader):
            warnings.warn(
                "train data is a one-shot iterator/generator: it can only train for one "
                "epoch. Pass a DataLoader, list or dataset to train for several epochs.",
                UserWarning, stacklevel=2)
            peek, train_loader = _peek_one_shot(train_loader)
            self.train_loader = train_loader
        else:
            peek = _peek_batch(train_loader)
        # Resolution always attempts metadata/task/batch inference when no
        # explicit recipe is given; low confidence raises (never silent guess).
        resolved = resolve_recipe(
            recipe=recipe,
            model=model,
            model_spec=model_spec,
            train_batch=peek,
            allow_auto=True,
        )
        self.recipe: Recipe = resolved.recipe
        self.recipe_explanation = resolved.explanation
        # -- recipe config (strict, with typo hints).
        if isinstance(recipe_config, RecipeConfig):
            self.recipe_config = recipe_config
        else:
            rdict = dict(recipe_config or {})
            # Merge YAML training defaults that belong to the recipe schema.
            for k, v in _yaml_training_defaults(model).items():
                lk = str(k).lower()
                if lk in ("learning_rate", "lr") and "lr" not in rdict:
                    continue  # lr belongs to optimizer policy, handled below
                if lk not in rdict and lk in type(self.recipe).ConfigSchema.model_fields:
                    rdict[lk] = v
            schema = type(self.recipe).ConfigSchema
            if ("loss" in rdict or "loss_kwargs" in rdict) and "loss" not in schema.model_fields:
                from topologies.training.losses import recipes_with_loss_override

                raise TrainingError(
                    f"A loss override is not supported by recipe {self.recipe.name!r}.",
                    recipe=self.recipe.name,
                    stage="config",
                    config_key="loss",
                    hint="Remove recipe_config['loss'] or use a recipe that supports it: "
                    + ", ".join(recipes_with_loss_override())
                    + ".",
                )
            try:
                self.recipe_config = validate_with_hints(schema, rdict)
            except TrainingError as e:
                # Enrich with recipe name + available params.
                raise TrainingError(
                    str(e),
                    recipe=self.recipe.name,
                    stage="config",
                ) from e
        # -- components (typed).
        self.components = self._build_components(components)
        self._check_tokenizer()
        self._warn_if_validation_unsupported()
        self._preflight_done = False
        self._batch_to_loss_takes_adapter: bool | None = None
        self._last_grad_norm: float | None = None
        # -- adapter.
        if adapter is not None:
            self.adapter = adapter
        elif input_map is not None or forward_map is not None or output_map is not None:
            self.adapter = ModelAdapter(
                input_map=input_map, forward_map=forward_map, output_map=output_map
            )
        else:
            self.adapter = ModelAdapter()
        # Recipes treat adapter=None as "use my own default call path"; pass the adapter to
        # batch_to_loss only when the user configured one.
        self._user_adapter = adapter is not None or any(
            m is not None for m in (input_map, forward_map, output_map))
        # -- model placement / compile.
        optimization_session = getattr(self.model, "__topologies_optimization_session__", None)
        if optimization_session is not None:
            required_device = str(getattr(optimization_session, "target", "cpu"))
            if required_device != str(self.device):
                raise TrainingError(
                    f"Prepared optimization session requires {required_device}; Trainer resolved {self.device}. "
                    "Set device explicitly to the session target; optimized backends are not moved automatically.",
                    stage="config", device=str(self.device))
        try:
            self.model.to(self.device)
        except Exception as e:
            raise TrainingError(
                f"Failed to move model to {self.device}: {e}",
                stage="config",
                device=str(self.device),
            ) from e
        if self.dist.world_size > 1:
            if optimization_session is not None:
                raise TrainingError("Prepared optimization sessions cannot be DDP-wrapped; "
                                    "train them single-process.", stage="config")
            self.model = wrap_model_ddp(self.model, self.device)
        if self.config.compile:
            if optimization_session is not None:
                raise TrainingError("Prepared optimization sessions cannot be silently torch.compile-wrapped; "
                                    "use an explicitly tested compiled backend integration.", stage="config")
            eager = self.model
            try:
                self.model = torch.compile(self.model, mode=self.config.compile_mode or "default")
            except Exception as e:  # noqa: BLE001 - any compile failure falls back to eager
                warnings.warn(f"torch.compile failed ({e}); continuing uncompiled.", stacklevel=2)
            else:
                self._compile_warmup(peek, eager)
        # -- optimizer policy precedence:
        # generic defaults -> recipe defaults -> YAML defaults -> user overrides.
        policy = {"optimizer": "adamw", "lr": 3e-4, "weight_decay": 0.0, "scheduler": None}
        try:
            policy.update(self.recipe.default_training_policy())
        except Exception as e:  # recipe code: report it instead of training with other defaults
            raise TrainingError(f"Recipe default_training_policy() failed: {e}",
                                recipe=self.recipe.name, stage="config") from e
        yaml_d = _yaml_training_defaults(model)
        if "learning_rate" in yaml_d and "lr" not in policy:
            policy["lr"] = yaml_d["learning_rate"]
        for k in ("lr", "learning_rate", "weight_decay", "optimizer", "scheduler"):
            if k in yaml_d:
                policy["lr" if k == "learning_rate" else k] = yaml_d[k]
        # User shortcuts (Trainer(..., lr=1e-3, weight_decay=0.0)) always win.
        if self.config.lr is not None:
            policy["lr"] = float(self.config.lr)
        if self.config.weight_decay is not None:
            policy["weight_decay"] = float(self.config.weight_decay)
        self._policy = policy
        self._optimizer_kwargs = dict(optimizer_kwargs or {})
        self.optimizers: dict[str, Any] = {}
        self.schedulers: dict[str, Any] = {}
        self._optimizer_configs: dict[str, OptimizerConfig | None] = {}
        self._build_optimizers(optimizer, optimizers, scheduler)
        # -- amp scaler (fp16 on CUDA only).
        self.scaler: Any = None
        if self.precision == "fp16" and self.device.type == "cuda":
            amp_scaler = getattr(torch.amp, "GradScaler", None)  # torch >= 2.3
            if amp_scaler is not None:
                self.scaler = amp_scaler("cuda")
            else:  # pragma: no cover - torch 2.2
                from torch.cuda.amp import GradScaler as _GS

                self.scaler = _GS()
        # -- logging / monitoring / callbacks / state.
        self._resolve_logger_and_monitor(logger, monitor)
        self.callbacks = CallbackList(callbacks or [])
        # global_step counts optimizer updates (as in HF/Lightning); micro_step counts
        # micro-batches. They differ only with gradient accumulation.
        self.global_step = 0
        self.micro_step = 0
        self._optimizer_stepped = False
        self.current_epoch = 0
        self.history: list[dict] = []
        # Best-checkpoint and early-stopping state are tracked separately: sharing
        # one value made an improvement look like "no change" to early stopping.
        self._ckpt_best: float | None = None
        self._ckpt_best_epoch: int | None = None
        self._es_best: float | None = None
        self._es_best_epoch: int | None = None
        self._no_improve = 0
        self._accum_count = 0
        self._stop_requested = False
        self._early_stopped = False
        self.stop_reason: str | None = None
        self.samples_seen = 0
        self._tokens_seen = 0
        self._tokens_seen_acc: torch.Tensor | None = None  # masked counts, synced lazily
        self._fit_started = False
        self._train_t0: float | None = None
        self._epoch_t0: float | None = None
        self._last_checkpoint_path: str | None = None
        # -- resume.
        if self.config.resume:
            self._do_resume(self.config.resume)

    # -- data / contract helpers --------------------------------------------
    def add_hook(self, event: HookEvent, fn: Callable[..., Any]) -> HookHandle:
        """Register an optimizer-step hook; returns a handle with ``remove()``.

        - ``"before_optimizer_step"``: ``fn(global_step)`` at the start of every training
          step (before its forward/backward), e.g. QAT observer/fake-quant schedules.
        - ``"after_optimizer_step"``: ``fn(trainer, optimizer_name)`` right after each
          ``optimizer.step()``, e.g. re-applying pruning masks.

        Registering the same function twice for one event is a no-op.
        """
        if event not in self._hooks:
            raise ValueError(f"Unknown hook event {event!r}; expected one of {HOOK_EVENTS}.")
        if not callable(fn):
            raise TypeError("hook must be callable")
        hooks = self._hooks[event]
        if fn not in hooks:
            hooks.append(fn)
        return HookHandle(hooks, fn)

    def register_optimization_step_hook(self, hook: Any) -> None:
        """Deprecated alias of ``add_hook("before_optimizer_step", hook)``."""
        warnings.warn("Trainer.register_optimization_step_hook is deprecated; use "
                      "Trainer.add_hook('before_optimizer_step', fn).", DeprecationWarning,
                      stacklevel=2)
        self.add_hook("before_optimizer_step", hook)

    def _run_optimization_step_hooks(self, step: int) -> None:
        for hook in tuple(self._hooks["before_optimizer_step"]):
            hook(int(step))

    def _as_loader(self, data: Any, *, shuffle: bool) -> Any:
        """Wrap a map-style dataset in a DataLoader; pass loaders/iterables through."""
        if data is None:
            return None
        from torch.utils.data import IterableDataset

        from topologies.data.loaders import dataset_collate_fn, is_loader

        if isinstance(data, IterableDataset):
            # Samples, not batches: batch them (order is the dataset's; no shuffle).
            return DataLoader(
                data,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                collate_fn=dataset_collate_fn(data),
                pin_memory=self._pin_memory,
            )
        if is_loader(data) or not hasattr(data, "__len__"):
            return data
        workers = int(self.config.num_workers)
        kwargs: dict[str, Any] = {
            "batch_size": self.config.batch_size,
            "shuffle": shuffle,
            "num_workers": workers,
            "collate_fn": dataset_collate_fn(data),
            "pin_memory": self._pin_memory_enabled(),
        }
        # prefetch_factor and persistent_workers are illegal at num_workers=0.
        if workers > 0:
            if self.config.persistent_workers:
                kwargs["persistent_workers"] = True
            if self.config.prefetch_factor is not None:
                kwargs["prefetch_factor"] = int(self.config.prefetch_factor)
        return DataLoader(data, **kwargs)

    def _pin_memory_enabled(self) -> bool:
        """Whether batches should be pinned. ``"auto"`` follows the device."""
        flag = self.config.pin_memory
        if flag is None or flag == "auto":
            return self.device.type == "cuda"
        return bool(flag)

    def _move_batch(self, batch: dict) -> dict:
        non_blocking = self._pin_memory_enabled() and self.device.type == "cuda"
        return move_batch_to_device(batch, self.device, non_blocking=non_blocking)

    @staticmethod
    def _propagate_set_epoch(loader: Any, epoch: int) -> None:
        """Call ``set_epoch`` on a loader's sampler, batch sampler, and dataset.

        PyTorch uses this hook so distributed and reshuffling samplers, and
        datasets that draw a fresh window each epoch, see the epoch index.
        Objects without the method are left alone. The same object is called
        once even if it is reachable through more than one attribute.
        """
        if loader is None:
            return
        seen: set[int] = set()
        for obj in (
            getattr(loader, "sampler", None),
            getattr(loader, "batch_sampler", None),
            getattr(loader, "dataset", None),
        ):
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            fn = getattr(obj, "set_epoch", None)
            if callable(fn):
                fn(int(epoch))

    def _check_tokenizer(self) -> None:
        """One rule for every token model: tokenizer, model and recipe agree on vocab."""
        from topologies.build.contract import model_vocab_size

        sizes: dict[str, int] = {}
        if self.tokenizer is not None:
            tv = getattr(self.tokenizer, "vocab_size", None)
            if tv is None:
                raise TrainingError(
                    "tokenizer has no vocab_size attribute.",
                    stage="config",
                    received=type(self.tokenizer).__name__,
                    hint="Pass a topologies.tokenizers.Tokenizer (or any object exposing "
                    "vocab_size / encode / decode).",
                )
            sizes["tokenizer.vocab_size"] = int(tv)
        mv = model_vocab_size(self.model)
        if mv is not None and (
            sizes or "vocab_size" in getattr(type(self.recipe_config), "model_fields", {})
        ):
            sizes["model.vocab_size"] = int(mv)
        rv = getattr(self.recipe_config, "vocab_size", None)
        if isinstance(rv, int):
            sizes["recipe_config.vocab_size"] = rv
        if len(set(sizes.values())) > 1:
            raise TrainingError(
                "Vocabulary sizes disagree.",
                recipe=self.recipe.name,
                stage="config",
                received="\n".join(f"    {k} = {v}" for k, v in sizes.items()),
                hint="Build the model with vocab_size=tokenizer.vocab_size and leave "
                "recipe_config.vocab_size unset (or set it to the same value).",
            )

    def _recipe_supports_validation(self) -> bool:
        if self.recipe.level == "loss":
            return True
        from topologies.training.recipe_base import Recipe as _Base

        return type(self.recipe).validation_step is not _Base.validation_step

    def _warn_if_validation_unsupported(self) -> None:
        if self.val_loader is not None and not self._recipe_supports_validation():
            warnings.warn(
                f"Recipe {self.recipe.name!r} (level={self.recipe.level}) defines no "
                "validation_step(); val_loader will be ignored. Implement "
                "validation_step() in the recipe to get validation metrics.",
                UserWarning,
                stacklevel=3,
            )

    def _compile_warmup(self, batch: dict | None, eager: nn.Module) -> None:
        """Compilation is lazy: run one no-grad forward now so failures surface here.

        On failure, warn and fall back to the eager model (only possible when a
        dict batch could be peeked; otherwise errors surface at the first step).
        """
        if batch is None:
            return
        try:
            prepared = self._prepare_batch(batch)
            with model_execution(self.model, training=True, grad_mode="no_grad"):
                self.recipe.predict_step(self.model, prepared, self.recipe_config,
                                         self.components, self.adapter)
        except Exception as e:  # noqa: BLE001 - compiler errors have no common base class
            warnings.warn(f"torch.compile failed on the first batch ({type(e).__name__}: {e}); "
                          "continuing uncompiled.", RuntimeWarning, stacklevel=3)
            self.model = eager

    # -- config helpers -----------------------------------------------------
    def _build_components(self, components: dict | Any | None) -> Any:
        schema = getattr(type(self.recipe), "ComponentsSchema", None)
        if schema is None:
            return components
        if components is None:
            # Components with all-optional fields may construct empty.
            try:
                return schema()
            except Exception as e:
                raise TrainingError(
                    f"Recipe {self.recipe.name!r} requires components: {e}",
                    recipe=self.recipe.name,
                    stage="component",
                    hint=f"Pass components={{...}} matching {schema.__name__}.",
                ) from e
        if isinstance(components, schema):
            return components
        if isinstance(components, dict):
            try:
                return schema(**components)
            except Exception as e:
                raise TrainingError(
                    f"Invalid components for recipe {self.recipe.name!r}: {e}",
                    recipe=self.recipe.name,
                    stage="component",
                ) from e
        raise TrainingError(
            f"Invalid components for recipe {self.recipe.name!r}.",
            recipe=self.recipe.name,
            stage="component",
            received=type(components).__name__,
        )

    # -- optimizer construction ---------------------------------------------
    # -- public API ---------------------------------------------------------
    def request_stop(self, reason: str | None = None) -> None:
        """Stop training cleanly: the current step finishes, the epoch is closed
        (validation, history, checkpoint, events), end-of-training events run and
        ``fit()`` returns normally. ``stop_reason`` records the first reason given.
        Safe to call from any callback hook.
        """
        if not self._stop_requested:
            self.stop_reason = str(reason) if reason else "requested"
        self._stop_requested = True

    @property
    def best_metric(self) -> float | None:
        """Best monitored value: early stopping's when it is enabled, else the best checkpoint's."""
        if self.config.early_stopping_patience is not None and self._es_best is not None:
            return self._es_best
        return self._ckpt_best

    @property
    def _best_epoch(self) -> int | None:
        if self.config.early_stopping_patience is not None and self._es_best is not None:
            return self._es_best_epoch
        return self._ckpt_best_epoch

    @property
    def stop_requested(self) -> bool:
        """True once ``request_stop`` was called (epoch/fit-level recipes may poll it)."""
        return bool(self._stop_requested)

    @property
    def tokens_seen(self) -> int:
        """Non-padding training tokens processed (``tokens`` batches; ``mask`` honored)."""
        extra = int(self._tokens_seen_acc) if self._tokens_seen_acc is not None else 0
        return int(self._tokens_seen) + extra

    @tokens_seen.setter
    def tokens_seen(self, value: int) -> None:
        self._tokens_seen = int(value)
        self._tokens_seen_acc = None

    def _final_stop_reason(self) -> str:
        if self._stop_requested:
            return self.stop_reason or "requested"
        if self._early_stopped:
            return "early_stopping"
        if self.config.max_steps is not None and self.global_step >= self.config.max_steps:
            return "max_steps"
        return "max_epochs"

    def _fit_result(self, epochs: int | None) -> dict:
        return {
            "global_step": self.global_step,
            "micro_step": self.micro_step,
            "history": self.history,
            "epochs": epochs,
            "stop_reason": self.stop_reason,
            "samples_seen": self.samples_seen,
            "tokens_seen": self.tokens_seen,
        }

    def fit(self) -> dict:
        # The process group belongs to the caller (torchrun): never destroyed here.
        return self._fit_single()

    def _session_info(self, info: Any) -> Any:
        from dataclasses import asdict, is_dataclass

        from topologies.visualization import SessionInfo

        run_info = asdict(info) if is_dataclass(info) else None
        mode = getattr(getattr(self.config, "monitoring", None), "mode", None)
        return SessionInfo(
            source="training",
            title=(run_info or {}).get("model_name"),
            run_info=run_info,
            request_stop=self.request_stop,
            options={"monitor": self._monitor_arg if isinstance(self._monitor_arg, str)
                     else mode},
            extra={"history": []},
        )

    def _fit_single(self) -> dict:
        import time as _time

        self._stop_requested = False
        self._early_stopped = False
        self.stop_reason = None
        visualizer = getattr(self.reporter, "visualizer", None)
        self._fit_started = True
        succeeded = False
        setup_complete = False
        self._train_t0 = _time.monotonic()
        _spe = self._safe_steps_per_epoch()
        _info = self._build_run_info(_spe, self._safe_batch_size())
        _total = self._compute_total_steps(_spe)
        try:
            if visualizer is not None:
                visualizer.prepare(source="training", context=self._session_info(_info))
            if _info is not None:
                from topologies.training.reporting.events import TrainBeginEvent

                try:
                    self.reporter.on_train_begin(TrainBeginEvent(info=_info, total_steps=_total))
                except Exception:  # noqa: BLE001, S110
                    pass
            self.callbacks.on_train_begin(self)
            setup_complete = True
            level = self.recipe.level
            if level == "fit":
                ctx = FitContext(
                    model=self.model,
                    train_loader=self.train_loader,
                    val_loader=self.val_loader,
                    device=self.device,
                    precision=self.precision,
                    rank=self.dist.rank,
                    world_size=self.dist.world_size,
                    config=self.config,
                    recipe_config=self.recipe_config,
                    components=self.components,
                    adapter=self.adapter,
                    logger=self.logger,
                    trainer=self,
                )
                try:
                    result = self.recipe.fit_loop(ctx)
                except NotImplementedError:
                    raise
                except TrainingError:
                    raise
                except Exception as e:
                    raise TrainingError(
                        f"Recipe fit_loop failed: {e}", recipe=self.recipe.name, stage="recipe-loop"
                    ) from e
                self.global_step = result.global_step
                self.stop_reason = self._final_stop_reason()
                # Train end for fit-level (boundaries owned by Trainer).
                from topologies.training.reporting.events import TrainEndEvent

                _dur = _time.monotonic() - self._train_t0 if self._train_t0 is not None else 0.0
                self.reporter.on_train_end(
                    TrainEndEvent(
                        total_epochs=int(result.epochs or 0),
                        global_step=int(self.global_step),
                        duration=float(_dur),
                        best_metric=self.best_metric,
                        best_epoch=self._best_epoch,
                        history=list(self.history),
                    )
                )
                self.callbacks.on_train_end(self)
                succeeded = True
                return self._fit_result(result.epochs)
            if level == "epoch":
                result = self._fit_epoch_level()
            elif level == "step":
                result = self._fit_step_level()
            else:
                result = self._fit_loss_level()
            succeeded = True
            return result
        except KeyboardInterrupt as error:
            self.reporter.on_error(error)
            raise
        except TrainingError as error:
            self.reporter.on_error(error)
            raise
        except Exception as e:
            self.reporter.on_error(e)
            self.callbacks.on_exception(self, e)
            if not setup_complete:
                raise
            raise TrainingError(
                f"Training failed: {e}", stage="recipe-loop", recipe=self.recipe.name
            ) from e
        finally:
            if visualizer is not None and succeeded:
                try:
                    from copy import deepcopy

                    from topologies.visualization import adapt_result

                    visualizer.render_safe(adapt_result(deepcopy(self.history)))
                except Exception as exc:  # noqa: BLE001 - policy (b): sink isolation
                    self.reporter._warn_once("visualizer", exc)
            try:
                self.reporter.close()
            except Exception:  # noqa: BLE001, S110 - policy (b): closing sinks is best-effort
                pass
            try:
                self.logger.close()
            except Exception:  # noqa: BLE001, S110 - policy (b): closing sinks is best-effort
                pass

    # -- level loops ----------------------------------------------------------
    # -- steps ------------------------------------------------------------------
    # -- optimizer services (also used by Level 2/3 contexts) ----------------------
    # -- validation / logging / checkpoint / early stopping ---------------------------
    def validate(self) -> dict[str, float]:
        """Evaluate ``val_loader``; metrics are prefixed ``val_`` (e.g. ``val_acc``)."""
        if self.val_loader is None or not self._recipe_supports_validation():
            return {}
        return self.evaluate(self.val_loader, prefix="val_")

    @torch.no_grad()
    def evaluate(self, data: Any = None, *, prefix: str = "", metrics: dict[str, Any] | None = None,
                 detailed: bool = False) -> dict[str, float] | Any:
        """Recipe-scored metrics over ``data`` (loader or dataset; default val_loader).

        No gradients, model in eval mode, batches moved to the trainer device.
        Batch metrics are averaged weighted by batch size (or by the recipe's
        ``_weight``, e.g. token counts), then ``recipe.summarize_metrics`` adds
        derived values such as perplexity.
        """
        loader = self._as_loader(data, shuffle=False) if data is not None else self.val_loader
        loader = maybe_wrap_sampler(loader, strategy=self.dist.strategy, evaluation=True)
        extra_metrics = self.metrics if metrics is None else metrics
        if loader is None:
            raise TrainingError(
                "evaluate() needs data (or a val_loader).",
                stage="validation",
                recipe=self.recipe.name,
                hint="Call trainer.evaluate(test_loader_or_dataset).",
            )
        from topologies.training.distributed import get_dist_state
        eval_dist = get_dist_state()
        if eval_dist.world_size > 1:
            sampler = getattr(loader, "sampler", None)
            if not bool(getattr(sampler, "nonpadding", False) or getattr(loader, "nonpadding", False)):
                raise TrainingError(
                    "Distributed evaluation cannot prove the input partition is non-padding/unique.",
                    recipe=self.recipe.name, stage="validation",
                    hint="Supply a sampler/loader declaring nonpadding=True; legacy scalar reduction is not globally exact.",
                )
        sums: dict[str, float] = {}
        total_w = 0.0
        from topologies.evaluation.api import (
            EvaluationResult,
            fresh_metric_states,
        )
        metric_states = fresh_metric_states(extra_metrics) if extra_metrics else {}
        callable_totals: dict[str, list[float]] = {}
        world_size = int(eval_dist.world_size)
        if world_size > 1:
            unsupported = [name for name, state in metric_states.items()
                           if not callable(getattr(state, "all_reduce", None))
                           and not callable(getattr(state, "merge_state", None))]
            if unsupported:
                raise TrainingError(
                    f"Metrics {unsupported} do not declare distributed merge support.",
                    recipe=self.recipe.name, stage="validation",
                    hint="Implement all_reduce() or merge_state() for distributed evaluation.",
                )
        evaluation_records = 0
        with model_execution(self.model, training=False, grad_mode="no_grad", preserve_rng=True):
            for batch in loader:
                prepared = self._prepare_batch(batch)
                try:
                    if extra_metrics:
                        hook = getattr(type(self.recipe), "evaluation_step", None)
                        from topologies.training.recipe_base import Recipe as _BaseRecipe
                        if hook is None or hook is _BaseRecipe.evaluation_step:
                            raise TrainingError(
                                f"Recipe {self.recipe.name!r} has no evaluation_step() record hook.",
                                recipe=self.recipe.name, stage="validation",
                                hint="Implement evaluation_step() returning EvaluationStepResult; an extra forward is never run.",
                            )
                        step = self.recipe.evaluation_step(
                            self.model, prepared, self.recipe_config, self.components, self.adapter
                        )
                        if not hasattr(step, "record") or not hasattr(step, "metrics"):
                            raise TypeError("evaluation_step() must return EvaluationStepResult(metrics, record)")
                        metrics = step.metrics
                        for name, state in metric_states.items():
                            if callable(getattr(state, "update", None)):
                                state.update(step.record)
                            elif callable(state):
                                reducer = getattr(state, "reducer", None)
                                if reducer not in ("mean", "sum"):
                                    raise TypeError(f"Callable metric {name!r} must declare reducer='mean' or 'sum'")
                                weight = float(torch.as_tensor(step.record.weights).sum()) if step.record.weights is not None else 1.0
                                value = float(state(step.record))
                                sums_ = callable_totals.setdefault(name, [0.0, 0.0])
                                sums_[0] += value * weight if reducer == "mean" else value
                                sums_[1] += weight
                            else:
                                raise TypeError(f"Metric {name!r} must implement update(record)/compute() or be a declared callable reducer")
                        evaluation_records += 1
                    else:
                        metrics = self.recipe.validation_step(
                            self.model, prepared, self.recipe_config, self.components, self.adapter
                        )
                except NotImplementedError as e:
                    raise TrainingError(
                        str(e),
                        recipe=self.recipe.name,
                        stage="validation",
                        hint="Implement validation_step() in the recipe.",
                    ) from e
                except TrainingError:
                    raise
                except Exception as e:
                    raise TrainingError(
                        f"Validation forward failed: {e}",
                        recipe=self.recipe.name,
                        stage="validation",
                    ) from e
                from topologies.nn.aux import collect_aux_losses

                collect_aux_losses(self.model)  # drop aux; validation loss stays task loss
                metrics = dict(metrics)
                w = float(metrics.pop("_weight", _batch_size_of(prepared)))
                for k, v in metrics.items():
                    sums[k] = sums.get(k, 0.0) + float(v) * w
                total_w += w
        if world_size > 1:
            packed = torch.tensor([*sums.values(), total_w], dtype=torch.float64, device=self.device)
            torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
            n_sums = len(sums)
            sums = {key: float(packed[i]) for i, key in enumerate(sums)}
            total_w = float(packed[n_sums])
        avg = {k: v / max(total_w, 1e-12) for k, v in sums.items()}
        avg = self.recipe.summarize_metrics(avg)
        flat = {f"{prefix}{k}": float(v) for k, v in avg.items()}
        if not detailed:
            return flat
        values = {}
        for k, state in metric_states.items():
            if callable(state) and not callable(getattr(state, "update", None)):
                total, weight = callable_totals.get(k, (0.0, 0.0))
                values[k] = total / weight if getattr(state, "reducer", None) == "mean" and weight else total
                continue
            if world_size > 1:
                if callable(getattr(state, "all_reduce", None)):
                    state.all_reduce()
                else:
                    state.merge_state()
            values[k] = state.compute()
        return EvaluationResult(values, evaluation_records, slices={"scalar": EvaluationResult(flat, int(total_w))})

    @torch.no_grad()
    def predict(self, data: Any, *, batch_size: int | None = None) -> Any:
        """Model outputs for ``data``, on CPU, concatenated over batches.

        ``data``: a DataLoader/iterable, a dataset, a dict batch, or a raw
        input tensor (mapped to the recipe's primary input, e.g. ``pixels``).
        Targets are not needed. Device placement is handled for you.
        """
        if isinstance(data, torch.Tensor):
            keys = self.recipe.input_keys() or self.recipe.required_batch_keys()
            batches: Any = [{keys[0]: data}]
        elif isinstance(data, dict):
            batches = [data]
        elif (
            batch_size is not None
            and not isinstance(data, DataLoader)
            and hasattr(data, "__getitem__")
        ):
            from topologies.data.loaders import make_loader

            batches = make_loader(data, batch_size=batch_size)
        else:
            batches = self._as_loader(data, shuffle=False)
        outs = []
        with model_execution(self.model, training=False, grad_mode="no_grad", preserve_rng=True):
            for batch in batches:
                b = self._batch_to_dict(batch) if not isinstance(batch, dict) else batch
                b = self._move_batch(self.adapter.adapted_batch(b))
                out = self.recipe.predict_step(
                    self.model, b, self.recipe_config, self.components, self.adapter
                )
                outs.append(_to_cpu(out))
        return _concat_outputs(outs)

    def save_pretrained(self, path: str | Path, **metadata: Any) -> Path:
        """Write model weights + constructor config + tokenizer + recipe to ``path``.

        Reload with ``topologies.load_pretrained(path)``.
        """
        from topologies.pretrained import save_pretrained

        meta = {"history": list(self.history), "global_step": self.global_step, **metadata}
        return save_pretrained(
            self.model,
            path,
            tokenizer=self.tokenizer,
            recipe=self.recipe.name,
            recipe_config=self.recipe_config.model_dump(mode="json"),
            metadata=meta,
        )

    def _should_stop(self, total_epochs: int, epoch: int) -> bool:
        if self.config.max_steps is not None and self.global_step >= self.config.max_steps:
            return True
        return (epoch + 1) >= total_epochs

    # -- convenience ---------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "global_step": self.global_step,
            "micro_step": self.micro_step,
            "current_epoch": self.current_epoch,
            "history": self.history,
            "samples_seen": self.samples_seen,
            "tokens_seen": self.tokens_seen,
            "stop_reason": self.stop_reason,
        }



__all__ = ["HookHandle", "OptimizerConfig", "Trainer", "TrainerConfig"]
