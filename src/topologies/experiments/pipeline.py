"""Default trial pipeline: resolve -> seed -> tokenizer -> data -> model -> trainer
-> fit -> evaluate -> costs -> artifacts. Each stage is a function so it can be
overridden (``Experiment(build_model=..., build_data=..., make_trainer=...,
evaluate=...)``) and tested on its own.
"""

from __future__ import annotations

import inspect
import math
import time
import warnings
from typing import Any

from topologies._util.seed import seed_everything as _seed_everything
from topologies.experiments.errors import ExperimentError
from topologies.experiments.routing import (
    constructor_params,
    is_ctx_factory,
    is_tokenizer_path,
    parse_tokenizer_spec,
    resolve_factory,
)
from topologies.experiments.spec import canonical, warn_once

# ---------------------------------------------------------------- seed


def seed_everything(seed: int) -> None:
    """Seed python, numpy (if imported), torch and CUDA before the model is built."""
    _seed_everything(seed)


# ---------------------------------------------------------------- tokenizer


def _tokenizer_value(ctx: Any) -> tuple[Any, dict[str, Any]] | None:
    task = ctx.task
    consumes = "tokenizer" in getattr(task, "consumes", frozenset()) if task is not None else False
    value = ctx.params.get("tokenizer")
    if value is None and consumes:
        value = getattr(task, "default_tokenizer", None)
    if value is None:
        return None
    kw = {k.split(".", 1)[1]: v for k, v in ctx.params.items() if k.startswith("tokenizer.")}
    return value, kw


def get_tokenizer(ctx: Any) -> tuple[Any, str | None]:
    """``(tokenizer, cache key)`` for the trial; trained once per spec + corpus and cached."""
    from topologies.experiments.cache import stable_hash

    found = _tokenizer_value(ctx)
    if found is None:
        return None, None
    value, kw = found
    task = ctx.task
    corpus_fp = task.corpus_fingerprint(ctx) if hasattr(task, "corpus_fingerprint") else None

    def corpus() -> Any:
        if task is None or not hasattr(task, "tokenizer_corpus"):
            raise ExperimentError("Training a tokenizer needs a task with a text corpus.",
                                  stage="tokenizer",
                                  hint="Use LanguageModeling(...) or pass a trained Tokenizer.")
        return task.tokenizer_corpus(ctx)

    if hasattr(value, "encode") and hasattr(value, "vocab_size") and not isinstance(value, type):
        return value, stable_hash({"instance": canonical(value, warn=False)})
    if is_tokenizer_path(value):
        from topologies.tokenizers import load_tokenizer

        return ctx.cache.get_or_create(f"tokpath:{value}", lambda: load_tokenizer(value)), \
            stable_hash({"path": str(value)})
    spec = parse_tokenizer_spec(value) if isinstance(value, (str, dict)) else None
    if spec is not None:
        spec.update(kw)
        key = stable_hash({"spec": canonical(spec, warn=False), "data": corpus_fp})
        kind = spec.pop("kind")
        if kind == "char":
            spec.pop("vocab_size", None)  # derived from the text

        def train() -> Any:
            from topologies.tokenizers import train_tokenizer

            try:
                return train_tokenizer(corpus(), kind=kind, **spec)
            except ImportError:
                raise
            except Exception as e:
                raise ExperimentError(f"Training tokenizer {value!r} failed: {e}",
                                      stage="tokenizer") from e

        tok, meta = ctx.cache.tokenizer(key, train)
        ctx.info["tokenizer_train_s"] = meta.get("train_s")
        return tok, key
    if callable(value):
        key = stable_hash({"fn": canonical(value, warn=False), "kw": canonical(kw, warn=False),
                           "data": corpus_fp})

        def train_fn() -> Any:
            from topologies.experiments.routing import model_accepts

            extra = {k: v for k, v in kw.items() if model_accepts(value, k)}
            return value(corpus(), **extra)

        tok, meta = ctx.cache.tokenizer(key, train_fn)
        ctx.info["tokenizer_train_s"] = meta.get("train_s")
        return tok, key
    from topologies.tokenizers import tokenizer_types

    raise ExperimentError(
        f"Cannot interpret tokenizer {value!r}.", stage="tokenizer", key="tokenizer",
        hint="Use a spec such as 'char', 'word:8k', 'bpe:4k', 'hf-bpe:4k', "
             "'hf-wordpiece:8k', a dict {'kind': 'bpe', 'vocab_size': 4096}, a Tokenizer, a "
             f"saved tokenizer directory, or fn(train_texts). Kinds: {tokenizer_types()}.")


# ---------------------------------------------------------------- data


def build_data(ctx: Any) -> Any:
    from topologies.experiments.tasks import Datasets

    override = ctx.runtime.build_data
    if override is not None:
        ds = override(ctx)
    elif ctx.task is not None:
        ds = ctx.task.build_data(ctx)
    else:
        raise ExperimentError("No data: pass task=... or build_data=fn(ctx).", stage="data")
    if isinstance(ds, (tuple, list)) and not isinstance(ds, Datasets):
        ds = Datasets(*ds)
    if not isinstance(ds, Datasets):
        raise ExperimentError("build_data must return Datasets(train, val, test) or a tuple.",
                              stage="data")
    return ds


# ---------------------------------------------------------------- model


def model_kwargs(ctx: Any, factory: Any) -> dict[str, Any]:
    """Geometry injected by alias (first accepted name), then explicit ``model.*`` kwargs."""
    from topologies.experiments.tasks import GEOMETRY_ALIASES

    explicit = {k.split(".", 1)[1]: v for k, v in ctx.params.items() if k.startswith("model.")}
    out: dict[str, Any] = {}
    if ctx.task is not None:
        names, _ = constructor_params(factory)
        for canon, value in ctx.task.geometry(ctx).items():
            aliases = GEOMETRY_ALIASES.get(canon, (canon,))
            if any(a in explicit for a in aliases):
                continue
            alias = next((a for a in aliases if a in names), None)
            if alias is not None and alias not in out:
                out[alias] = value
    out.update(explicit)
    return out


def missing_knobs_error(factory: Any) -> ExperimentError:
    name = getattr(factory, "__name__", repr(factory))
    return ExperimentError(
        f"{name} declares no budget knobs, so target_params cannot be matched.",
        stage="model", key="target_params",
        hint=f"register_budget_knobs({name}, dim=range(...), layers=range(...)), or "
             f"Variant(..., search={{'dim': range(32, 1025, 32)}}), or "
             f"Variant(..., target_params=None) to exempt it.")


def build_model(ctx: Any, *, meta: bool = False) -> Any:
    """Construct the trial's model (on the ``meta`` device when ``meta=True``)."""
    import torch

    if ctx.runtime.build_model is not None:
        if meta:
            with torch.device("meta"):
                return ctx.runtime.build_model(ctx)
        return ctx.runtime.build_model(ctx)
    arch = ctx.params.get("model")
    if arch is None:
        raise ExperimentError("No model to build.", stage="model",
                              hint="Pass model=MyModel, a 'model' axis, or build_model=fn(ctx).")
    factory = resolve_factory(arch)
    target = ctx.params.get("target_params")
    if is_ctx_factory(factory):
        if target is not None:
            ctx.info["param_matching"] = "skipped (factory(ctx) model)"
        if meta:
            with torch.device("meta"):
                return factory(ctx)
        return factory(ctx)
    kwargs = model_kwargs(ctx, factory)
    search = ctx.params.get("search")
    if target is not None and search is None:
        from topologies.build import get_budget_knobs

        if not get_budget_knobs(factory):
            raise missing_knobs_error(factory)
    from topologies.build import build

    size = ctx.params.get("size")
    build_kw: dict[str, Any] = {"target_params": target, "search": search,
                                "param_count": ctx.params.get("param_count", "total")}
    if "tolerance" in ctx.params:
        build_kw["tolerance"] = ctx.params["tolerance"]
    try:
        if meta:
            with torch.device("meta"):
                return build(factory, size, **build_kw, **kwargs)
        return build(factory, size, **build_kw, **kwargs)
    except ExperimentError:
        raise
    except TypeError as e:
        name = getattr(factory, "__name__", repr(factory))
        raise ExperimentError(
            f"Constructing {name} failed: {e}", stage="model",
            hint="Pass the missing constructor arguments as model.<arg> keys (or a Variant), "
                 "or use a task that injects them (vocab_size, max_seq, num_classes, ...).",
        ) from e
    except ValueError as e:
        raise ExperimentError(f"Building {getattr(factory, '__name__', factory)} failed: {e}",
                              stage="model", hint=getattr(e, "hint", None)) from e


def _hidden_position_limit(model: Any, vocab: int | None) -> tuple[str, int] | None:
    import torch.nn as nn

    from topologies.build.contract import unwrap

    best = None
    for name, mod in unwrap(model).named_modules():
        if isinstance(mod, nn.Embedding) and "pos" in name.lower() and \
                mod.num_embeddings != vocab and (best is None or mod.num_embeddings < best[1]):
            best = (name, int(mod.num_embeddings))
    return best


def check_compat(ctx: Any, model: Any) -> None:
    """Model vs data: vocabulary covers the tokenizer, context covers ``block_size``."""
    from topologies.build.contract import embedding_sizes, model_context_length, model_vocab_size

    name = type(model).__name__
    tok = ctx.tokenizer
    if tok is not None:
        tv = int(tok.vocab_size)
        mv = model_vocab_size(model)
        if mv is None:
            sizes = [n for _, n in embedding_sizes(model)]
            mv = max(sizes) if sizes else None
        if mv is not None and mv < tv:
            raise ExperimentError(
                f"{name} has a vocabulary of {mv} but the tokenizer has {tv} tokens.",
                stage="model",
                hint="Let the experiment inject vocab_size (do not set model.vocab_size), or use "
                     "a smaller tokenizer.")
    block = ctx.task_params.get("block_size")
    if block:
        cl = model_context_length(model)
        if cl is not None and cl < int(block):
            raise ExperimentError(
                f"sequence length {block} > model context {cl} ({name}).", stage="model",
                hint=f"Use task.block_size<={cl}, or build the model with a longer context "
                     "(max_seq / context_length).")
        if cl is None:
            hidden = _hidden_position_limit(model, int(tok.vocab_size) if tok else None)
            if hidden is not None and hidden[1] < int(block):
                raise ExperimentError(
                    f"sequence length {block} > model context {hidden[1]} ({name} has a hidden "
                    f"{hidden[1]}-slot position table '{hidden[0]}').", stage="model",
                    hint=f"Use task.block_size<={hidden[1]} for this model (e.g. a Variant with "
                         f"**{{'task.block_size': {hidden[1]}}}), or exclude it.")


# ---------------------------------------------------------------- trainer


def _steps_per_epoch(train: Any, batch_size: int) -> int | None:
    try:
        from topologies.data.loaders import is_loader

        if is_loader(train):
            return len(train)
        return max(1, math.ceil(len(train) / max(1, batch_size)))
    except Exception:  # noqa: BLE001
        return None


def budget_limits(ctx: Any, datasets: Any, batch_size: int) -> dict[str, Any]:
    """``train_for`` -> ``max_epochs`` / ``max_steps`` (time budgets are callbacks)."""
    b = ctx.budget
    spe = _steps_per_epoch(datasets.train, batch_size)
    steps = int(b["steps"]) if b.get("steps") else None
    task = ctx.task
    for unit in ("tokens", "bytes"):
        if not b.get(unit):
            continue
        tps = task.tokens_per_sample(ctx) if task is not None else None
        if not tps:
            raise ExperimentError(f"train_for in {unit} needs a token task (LanguageModeling).",
                                  stage="config", key="train_for")
        tokens = float(b[unit])
        if unit == "bytes":
            bpt = task.bytes_per_token(ctx)
            if not bpt:
                raise ExperimentError("Cannot convert bytes to tokens for this task.",
                                      stage="config", key="train_for")
            tokens = tokens / bpt
        s = max(1, math.ceil(tokens / (batch_size * tps)))
        steps = s if steps is None else min(steps, s)
    out: dict[str, Any] = {}
    if b.get("epochs"):
        out["max_epochs"] = int(b["epochs"])
    elif steps is not None and spe:
        out["max_epochs"] = max(1, math.ceil(steps / spe))
    else:
        out["max_epochs"] = 10**6  # time / step budgets bound the run instead
    if steps is not None:
        out["max_steps"] = steps
    if ctx.runtime.rung and ctx.runtime.rung.get("final_epochs"):
        out["max_epochs"] = int(ctx.runtime.rung["final_epochs"])
    return out


def optimizer_args(ctx: Any) -> dict[str, Any]:
    """``optimizer`` / ``optimizer.*`` / ``scheduler`` / ``scheduler.*`` -> Trainer kwargs."""
    import torch

    p = ctx.params
    fields = {k.split(".", 1)[1]: v for k, v in p.items() if k.startswith("optimizer.")}
    sched = p.get("scheduler")
    sched_kw = {k.split(".", 1)[1]: v for k, v in p.items() if k.startswith("scheduler.")}
    opt = p.get("optimizer")
    out: dict[str, Any] = {}
    from topologies.training.configs import OptimizerConfig

    known = set(OptimizerConfig.model_fields)
    cfg_fields = {k: v for k, v in fields.items() if k in known}
    extra = {k: v for k, v in fields.items() if k not in known}
    if isinstance(opt, type) and issubclass(opt, torch.optim.Optimizer):
        out["optimizer"] = opt
        out["optimizer_kwargs"] = {**{k: v for k, v in cfg_fields.items()
                                      if k not in ("lr", "weight_decay")}, **extra}
        for k in ("lr", "weight_decay"):
            if k in cfg_fields:
                out[k] = cfg_fields[k]
    elif callable(opt) and not isinstance(opt, (str, dict)):
        if fields:
            raise ExperimentError("optimizer.* fields cannot apply to an optimizer factory.",
                                  stage="trainer", key="optimizer",
                                  hint="Bake hyperparameters into the factory, or register it "
                                       "with register_optimizer(name, factory).")
        out["optimizer"] = opt
    elif opt is not None or fields or sched is not None:
        entry: dict[str, Any] = {}
        if isinstance(opt, dict):
            entry.update(opt)
        elif isinstance(opt, str):
            entry["name"] = opt
        entry.update(cfg_fields)
        if extra:
            entry["kwargs"] = {**entry.get("kwargs", {}), **extra}
        if isinstance(sched, str) or (sched is None and sched_kw):
            if sched is not None:
                entry["scheduler"] = sched
            if sched_kw:
                entry["scheduler_kwargs"] = {**entry.get("scheduler_kwargs", {}), **sched_kw}
        elif callable(sched):
            out["scheduler"] = sched
        out["optimizer"] = entry
        if not entry:
            out.pop("optimizer")
    if "optimizer" in out and not isinstance(out["optimizer"], dict) and sched is not None:
        if isinstance(sched, str):
            from topologies.training.registries import build_scheduler

            name, kw = sched, dict(sched_kw)
            out["scheduler"] = lambda o, total_steps: build_scheduler(name, o, total_steps, kw)
        else:
            out["scheduler"] = sched
    return out


def recipe_config(ctx: Any) -> dict[str, Any]:
    p = ctx.params
    cfg = {k.split(".", 1)[1]: v for k, v in p.items() if k.startswith("recipe.")}
    if p.get("loss") is not None:
        cfg["loss"] = p["loss"]
        loss_kw = {k.split(".", 1)[1]: v for k, v in p.items() if k.startswith("loss.")}
        if loss_kw:
            cfg["loss_kwargs"] = {**cfg.get("loss_kwargs", {}), **loss_kw}
    return cfg


def trainer_kwargs(ctx: Any, model: Any, datasets: Any) -> dict[str, Any]:
    """Everything ``Trainer(...)`` receives in the default pipeline."""
    from topologies.build.contract import model_vocab_size

    p = ctx.params
    kw = {k.split(".", 1)[1]: v for k, v in p.items() if k.startswith("trainer.")}
    kw.pop("device", None)
    batch_size = int(kw.get("batch_size", 32))
    kw.update(budget_limits(ctx, datasets, batch_size))
    kw.update(optimizer_args(ctx))
    recipe = p.get("recipe") or (ctx.task.recipe if ctx.task is not None else None)
    tok = ctx.tokenizer
    if tok is not None:
        mv = model_vocab_size(model)
        if mv is None or mv == int(tok.vocab_size):
            kw["tokenizer"] = tok
    # Every trial keeps last.pt so a killed run continues from the last finished
    # epoch. A halving rung resumes only after the first rung, as before.
    ckpt_dir = ctx.dir / "ckpt"
    kw.setdefault("checkpoint_dir", str(ckpt_dir))
    last = ckpt_dir / "last.pt"
    rung = ctx.runtime.rung or {}
    if "resume" not in kw and last.is_file() and (not rung or rung.get("resume")):
        kw["resume"] = str(last)
    out = {"recipe": recipe, "recipe_config": recipe_config(ctx) or None,
           "callbacks": ctx.callbacks(), "logger": "jsonl", "log_dir": str(ctx.dir),
           "monitor": "none", "seed": ctx.seed, "device": ctx.device}
    out.update(kw)
    return out


def _show_plan(trainer: Any, ctx: Any) -> None:
    """Tell the monitor which trial this is, before the first frame is drawn."""
    trial = getattr(ctx, "trial", None)
    total = getattr(trial, "plan_total", None)
    if trial is None or not total:
        return
    monitor = getattr(trainer, "monitor", None)
    show = getattr(monitor, "set_plan", None)
    if not callable(show):
        return
    done = getattr(trial, "plan_done", None)
    show(int(trial.index) + 1, int(total), done=None if done is None else int(done))


def make_trainer(ctx: Any, model: Any, datasets: Any) -> Any:
    from topologies.training import Trainer

    if ctx.runtime.make_trainer is not None:
        trainer = ctx.runtime.make_trainer(ctx, model, datasets)
        _warn_if_callbacks_missing(ctx, trainer)
        return trainer
    return Trainer(model, datasets.train, datasets.val, **trainer_kwargs(ctx, model, datasets))


def _warn_if_callbacks_missing(ctx: Any, trainer: Any) -> None:
    from topologies.experiments.callbacks import ProgressBridge

    try:
        cbs = list(getattr(trainer.callbacks, "callbacks", trainer.callbacks) or [])
    except Exception:  # noqa: BLE001
        cbs = []
    if not any(isinstance(c, ProgressBridge) for c in cbs):
        warn_once(f"make_trainer for {ctx.experiment!r} built a Trainer without "
                  "ctx.callbacks(): pruning, timeouts, divergence detection and live progress "
                  "are disabled. Pass Trainer(..., callbacks=ctx.callbacks()).")


# ---------------------------------------------------------------- evaluate


def evaluate(ctx: Any, trainer: Any) -> dict[str, float]:
    if ctx.runtime.evaluate is not None:
        return dict(ctx.runtime.evaluate(ctx, trainer) or {})
    if ctx.task is not None:
        return dict(ctx.task.evaluate(ctx, trainer) or {})
    ds = ctx.datasets
    return dict(trainer.evaluate(ds.val, prefix="val_")) if ds.val is not None else {}


def _derive(ctx: Any, metrics: dict) -> dict:
    """``task.derive`` (user/task code); a failure warns instead of vanishing silently."""
    try:
        return dict(ctx.task.derive(ctx, metrics) or {})
    except Exception as e:  # noqa: BLE001 - derived metrics are optional; the trial goes on
        warnings.warn(f"{type(ctx.task).__name__}.derive failed ({type(e).__name__}); "
                      "derived metrics are skipped.", RuntimeWarning, stacklevel=3)
        return {}


def derived_history(ctx: Any, history: list[dict]) -> list[dict]:
    out = []
    for row in history or []:
        r = dict(row)
        if ctx.task is not None:
            r.update(_derive(ctx, r))
        out.append(r)
    return out


def build_report(model: Any) -> dict[str, Any] | None:
    rep = getattr(model, "build_report", None)
    if rep is None:
        return None
    try:
        return {"target": int(rep.target), "params": int(rep.params),
                "rel_error": float(rep.rel_error), "knobs": dict(rep.kwargs),
                "count": getattr(rep, "count", "total")}
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- one attempt


def call_trial_fn(ctx: Any) -> dict[str, Any]:
    """Level 3: call ``trial_fn(ctx, **params)`` and normalize its return value."""
    from topologies.experiments.trial import TrialOutput

    fn = ctx.runtime.trial_fn
    try:
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())[1:]
        var_kw = any(q.kind is q.VAR_KEYWORD for q in params)
        names = {q.name for q in params if q.kind in (q.POSITIONAL_OR_KEYWORD, q.KEYWORD_ONLY)}
    except (TypeError, ValueError):
        var_kw, names = True, set()
    kwargs = {k: v for k, v in ctx.params.items()
              if k.isidentifier() and (var_kw or k in names)}
    t0 = time.monotonic()
    out = fn(ctx, **kwargs)
    train_s = time.monotonic() - t0
    if out is None:
        out = TrialOutput(metrics={})
    elif isinstance(out, dict):
        out = TrialOutput(metrics=dict(out))
    elif not isinstance(out, TrialOutput):
        raise ExperimentError(
            f"Trial function returned {type(out).__name__}; expected a dict of metrics or "
            "TrialOutput.", stage="trial")
    metrics = {k: v for k, v in out.metrics.items()}
    if ctx.task is not None:
        metrics.update({k: v for k, v in _derive(ctx, metrics).items() if k not in metrics})
    history = derived_history(ctx, out.history) if out.history is not None else ctx.history
    ctx.extras.update(out.extras or {})
    if out.artifacts:
        ctx.extras["artifacts"] = {k: str(v) for k, v in out.artifacts.items()}
    return {"final": metrics, "history": history, "train_s": train_s, "eval_s": None,
            "trainer": None, "model": None, "fit_out": None, "build": None}


def run_pipeline(ctx: Any) -> dict[str, Any]:
    """Levels 1-2: one attempt of the default pipeline (exceptions propagate)."""
    seed_everything(ctx.seed)
    _ = ctx.tokenizer
    datasets = ctx.datasets
    if ctx.task is not None:
        try:
            ctx.info["bytes_per_token_train"] = ctx.task.bytes_per_token(ctx)
        except Exception:  # noqa: BLE001, S110 - only used for cost.bytes_seen
            pass
    model = build_model(ctx)
    if ctx.runtime.build_model is None:
        check_compat(ctx, model)
    trainer = make_trainer(ctx, model, datasets)
    _show_plan(trainer, ctx)
    ctx.info["precision"] = str(getattr(trainer, "precision", None))
    t0 = time.monotonic()
    fit_out = trainer.fit()
    train_s = time.monotonic() - t0
    if ctx.info.get("rung_paused"):
        return {"final": {}, "history": derived_history(ctx, trainer.history),
                "train_s": train_s, "eval_s": None, "trainer": trainer, "model": model,
                "fit_out": fit_out, "build": build_report(model), "paused": True}
    t1 = time.monotonic()
    final = evaluate(ctx, trainer)
    eval_s = time.monotonic() - t1
    return {"final": final, "history": derived_history(ctx, trainer.history),
            "train_s": train_s, "eval_s": eval_s, "trainer": trainer, "model": model,
            "fit_out": fit_out, "build": build_report(model)}


__all__ = [
    "budget_limits",
    "build_data",
    "build_model",
    "call_trial_fn",
    "check_compat",
    "evaluate",
    "get_tokenizer",
    "make_trainer",
    "model_kwargs",
    "optimizer_args",
    "recipe_config",
    "run_pipeline",
    "seed_everything",
    "trainer_kwargs",
]
