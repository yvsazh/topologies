"""YAML (or JSON) experiment specs for ``topologies exp run spec.yaml``.

    name: arch-x-tokenizer
    task:
      type: language-modeling
      train: data/tinystories/tinystories_mini_train.txt
      val_fraction: 0.1
      block_size: 128
    axes:
      model:
        gpt: TransformerLM
        rope-rms-swiglu: {arch: TransformerLM, pos: rope, norm: rms, activation: swiglu}
        mine: my_pkg.nets:MyLM
      tokenizer: [char, "bpe:1k", "bpe:4k"]
    target_params: 1M
    train_for: 3 epochs
    train: {batch_size: 32, lr: 1.0e-3}
    seeds: 2

Classes and callables are named by import path (``pkg.module:obj``), toolbox
name or catalog id. ``{arch: ..., **overrides}`` makes a ``Variant``. The
schema is strict: unknown keys fail with suggestions. Python specs remain the
primary interface; YAML covers the common cases.
"""

from __future__ import annotations

import difflib
import inspect
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from topologies.experiments.errors import ExperimentError

_IMPORT_RE = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_][\w.]*$")
TASK_TYPES = {"language-modeling": "topologies.experiments.tasks:LanguageModeling",
              "image-classification": "topologies.experiments.tasks:ImageClassification",
              "supervised": "topologies.experiments.tasks:Supervised"}
#: Keys whose string values may be import paths of callables.
_CALLABLE_KEYS = ("optimizer", "scheduler", "loss")


class TaskYAML(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str


class ExperimentYAML(BaseModel):
    """Strict schema of an experiment YAML file."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    tags: dict[str, str] = Field(default_factory=dict)
    task: TaskYAML | None = None
    axes: dict[str, Any] = Field(default_factory=dict)
    fixed: dict[str, Any] = Field(default_factory=dict)
    model: Any = None
    size: str | None = None
    target_params: str | int | None = None
    param_count: str = "total"
    train_for: Any = None
    train: dict[str, Any] = Field(default_factory=dict)
    seeds: int | list[int] = 1
    goal: str | list[str] | None = None
    select: str = "last"
    space: str = "grid"
    n_trials: int | None = None
    baseline: dict[str, Any] | None = None
    exclude: list[dict[str, Any]] = Field(default_factory=list)
    include: list[dict[str, Any]] = Field(default_factory=list)
    strategy: str = "full"
    eta: int = 3
    min_budget: Any = None
    prune: str | None = None
    trial_timeout: Any = None
    workers: int = 1
    devices: list[str] | None = None
    save: str = "none"
    root: str | None = None
    on_error: str = "continue"
    on_oom: str = "skip"
    monitor: str = "live"


def _import(path: str) -> Any:
    from topologies.build.config import import_path

    try:
        return import_path(path)
    except Exception as e:
        raise ExperimentError(f"Cannot import {path!r}: {e}", stage="spec") from e


def _resolve_value(key: str, value: Any) -> Any:
    base = key.split(".", 1)[0]
    if isinstance(value, str) and base in _CALLABLE_KEYS and _IMPORT_RE.match(value):
        return _import(value)
    return value


def _axis_values(key: str, values: Any) -> Any:
    from topologies.experiments.space import Variant

    def one(label: str | None, v: Any) -> Any:
        if isinstance(v, dict) and "arch" in v:
            rest = {k: x for k, x in v.items() if k != "arch"}
            return Variant(label or str(v["arch"]), v["arch"], **rest)
        if isinstance(v, dict) and "value" in v and label is not None:
            rest = {k: x for k, x in v.items() if k != "value"}
            return Variant(label, _resolve_value(key, v["value"]), **rest)
        return _resolve_value(key, v)

    if isinstance(values, dict):
        return {str(k): one(str(k), v) for k, v in values.items()}
    if isinstance(values, list):
        return [one(None, v) for v in values]
    raise ExperimentError(f"Axis {key!r} must be a list or a mapping in YAML.", stage="spec",
                          key=key)


def _task(spec: TaskYAML, base: Path) -> Any:
    kind = spec.type
    target = TASK_TYPES.get(kind)
    if target is None:
        if _IMPORT_RE.match(kind):
            target = kind
        else:
            sug = difflib.get_close_matches(kind, list(TASK_TYPES), n=1)
            raise ExperimentError(f"Unknown task type {kind!r}.", stage="spec", key="task.type",
                                  hint=(f"Did you mean {sug[0]!r}? " if sug else "")
                                  + "Use one of " + ", ".join(TASK_TYPES)
                                  + " or an import path 'pkg.module:MyTask'.")
    cls = _import(target)
    kwargs = dict(spec.model_extra or {})
    for k in ("train", "val", "test"):
        v = kwargs.get(k)
        if isinstance(v, str) and _IMPORT_RE.match(v) and not Path(v).exists():
            obj = _import(v)
            kwargs[k] = obj() if callable(obj) and not hasattr(obj, "__getitem__") else obj
        elif isinstance(v, str) and not Path(v).is_absolute() and (base / v).exists():
            kwargs[k] = str(base / v)
    try:
        params = inspect.signature(cls).parameters
        accepts_any = any(p.kind is p.VAR_KEYWORD for p in params.values())
        unknown = [k for k in kwargs if k not in params and not accepts_any]
    except (TypeError, ValueError):
        unknown = []
    if unknown:
        sug = difflib.get_close_matches(unknown[0], list(params), n=1)
        raise ExperimentError(f"Unknown task option {unknown[0]!r} for {cls.__name__}.",
                              stage="spec", key=f"task.{unknown[0]}",
                              hint=f"Did you mean {sug[0]!r}?" if sug else None,
                              available=[p for p in params if p != "self"])
    if "goal" in kwargs and isinstance(kwargs["goal"], list):
        kwargs["goal"] = tuple(kwargs["goal"])
    return cls(**kwargs)


def experiment_from_dict(data: dict[str, Any], *, base: Path | None = None) -> Any:
    from pydantic import ValidationError

    from topologies.experiments.core import Experiment

    try:
        spec = ExperimentYAML(**data)
    except ValidationError as e:
        notes = []
        fields = list(ExperimentYAML.model_fields)
        for err in e.errors():
            loc = ".".join(str(x) for x in err.get("loc", ()))
            if err.get("type") == "extra_forbidden":
                sug = difflib.get_close_matches(loc, fields, n=1)
                notes.append(f"unknown key {loc!r}" + (f" (did you mean {sug[0]!r}?)"
                                                       if sug else ""))
            else:
                notes.append(f"{loc}: {err.get('msg')}")
        raise ExperimentError("Invalid experiment spec: " + "; ".join(notes), stage="spec",
                              available=fields) from e
    base = base or Path(".")
    task = _task(spec.task, base) if spec.task is not None else None
    goal: Any = spec.goal
    if isinstance(goal, list):
        goal = tuple(goal)
    return Experiment(
        spec.name, task=task,
        axes={k: _axis_values(k, v) for k, v in spec.axes.items()},
        fixed={k: _resolve_value(k, v) for k, v in spec.fixed.items()},
        model=spec.model, size=spec.size, target_params=spec.target_params,
        param_count=spec.param_count, train_for=spec.train_for, train=spec.train,
        seeds=spec.seeds, goal=goal, select=spec.select, space=spec.space,
        n_trials=spec.n_trials, baseline=spec.baseline, exclude=spec.exclude,
        include=spec.include, strategy=spec.strategy, eta=spec.eta,
        min_budget=spec.min_budget, prune=spec.prune, trial_timeout=spec.trial_timeout,
        workers=spec.workers, devices=spec.devices, save=spec.save,
        root=str(base / spec.root) if spec.root and not Path(spec.root).is_absolute()
        else spec.root, on_error=spec.on_error, on_oom=spec.on_oom, monitor=spec.monitor,
        tags=spec.tags, description=spec.description)


def experiment_from_yaml(path: str | Path) -> Any:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".json":
        data = json.loads(text)
    else:
        import yaml

        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ExperimentError(f"{p} must contain a mapping.", stage="spec")
    return experiment_from_dict(data, base=p.parent)


__all__ = ["ExperimentYAML", "experiment_from_dict", "experiment_from_yaml"]
