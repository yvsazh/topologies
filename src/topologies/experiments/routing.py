"""Route axis / fixed keys to pipeline stages, with typo hints.

Dotted keys name their stage (``model.dim``, ``trainer.batch_size``,
``optimizer.betas``, ``recipe.label_smoothing``, ``task.block_size``,
``tokenizer.min_frequency``, ``scheduler.T_max``, ``loss.gamma``). Bare keys
resolve by priority:

1. ``TrainerConfig`` fields (``lr`` / ``weight_decay`` are its shortcuts),
2. ``OptimizerConfig`` fields,
3. the recipe's config fields,
4. model constructor kwargs accepted by every model the key applies to.

A bare key found in several of 2-4 raises with the dotted options; an
unknown key raises with ``difflib`` suggestions. Level-3 experiments
(``trial_fn``) keep keys free-form.
"""

from __future__ import annotations

import difflib
import inspect
import math
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from topologies.experiments.errors import ExperimentError

#: Top-level routing keys (not namespaced).
TOP_LEVEL = ("model", "size", "target_params", "search", "tolerance", "param_count",
             "tokenizer", "optimizer", "scheduler", "loss", "recipe", "train_for", "seed")
#: Prefixes of dotted keys.
NAMESPACES = ("model", "tokenizer", "optimizer", "scheduler", "loss", "recipe", "trainer",
              "task", "synth")
#: Trainer kwargs the experiment manages itself.
MANAGED_TRAINER_KEYS = frozenset({"logger", "log_dir", "monitor", "callbacks", "tokenizer",
                                  "recipe", "recipe_config", "optimizer", "scheduler",
                                  "optimizer_kwargs", "config", "seed", "max_epochs",
                                  "max_steps", "train_loader", "val_loader", "model"})


# ---------------------------------------------------------------- field tables


def trainer_fields() -> set[str]:
    from topologies.training.configs import TrainerConfig
    from topologies.training.engine import Trainer

    names = set(TrainerConfig.model_fields)
    try:
        for p in inspect.signature(Trainer.__init__).parameters.values():
            if p.kind is p.KEYWORD_ONLY or p.kind is p.POSITIONAL_OR_KEYWORD:
                names.add(p.name)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        pass
    names.discard("self")
    return names


def optimizer_fields() -> set[str]:
    from topologies.training.configs import OptimizerConfig

    return set(OptimizerConfig.model_fields) - {"scheduler", "scheduler_kwargs"}


def recipe_fields(recipe: str | None) -> set[str]:
    if not recipe:
        return set()
    try:
        import topologies.training  # noqa: F401  (registers built-in recipes)
        from topologies.training.recipe_base import _RECIPE_REGISTRY

        cls = _RECIPE_REGISTRY.get(str(recipe))
        return set(cls.ConfigSchema.model_fields) if cls is not None else set()
    except Exception:  # noqa: BLE001
        return set()


def is_ctx_factory(fn: Any) -> bool:
    """A model factory ``fn(ctx) -> nn.Module`` (first parameter named ``ctx``)."""
    if isinstance(fn, type) or not callable(fn):
        return False
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return False
    return bool(params) and params[0].name == "ctx"


def resolve_factory(arch: Any) -> Any:
    """Class / callable for a model value (import path, catalog id, toolbox name)."""
    if isinstance(arch, str):
        from topologies.build import resolve_architecture

        try:
            return resolve_architecture(arch)
        except Exception as e:
            raise ExperimentError(
                f"Cannot resolve model {arch!r}: {e}", stage="model", key="model",
                hint="Use a class, an import path 'pkg.module:Class', a toolbox name "
                     "('TransformerLM') or a catalog id ('llama-decoder' — see "
                     "`topologies models list`).") from e
    return arch


def constructor_params(factory: Any) -> tuple[set[str], bool]:
    """``(named parameters, accepts **kwargs)``; unintrospectable -> ``(set(), True)``."""
    if is_ctx_factory(factory):
        return set(), True
    try:
        params = inspect.signature(factory).parameters.values()
    except (TypeError, ValueError):
        return set(), True
    names = {p.name for p in params if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    var_kw = any(p.kind is p.VAR_KEYWORD for p in params)
    return names, var_kw


def model_accepts(factory: Any, key: str) -> bool:
    names, var_kw = constructor_params(factory)
    return var_kw or key in names


# ---------------------------------------------------------------- tokenizer specs

_TOK_RE = re.compile(r"^\s*([a-zA-Z][\w-]*)\s*(?::\s*([0-9.]+)\s*([kKmM]?))?\s*$")


def parse_tokenizer_spec(spec: Any) -> dict[str, Any] | None:
    """``"bpe:4k"`` -> ``{"kind": "bpe", "vocab_size": 4096}``; ``None`` if not a spec.

    ``k`` means 1024 for vocabularies (``bpe:1k`` = 1024). ``hf-bpe:4k`` and
    ``hf-wordpiece:8k`` select the Rust backend. Dicts pass through (``kind``
    required). Instances, paths and callables are not specs.
    """
    if isinstance(spec, dict):
        if "kind" not in spec:
            raise ExperimentError(f"Tokenizer dict {spec} needs a 'kind'.", stage="tokenizer",
                                  hint="{'kind': 'bpe', 'vocab_size': 4096}")
        d = dict(spec)
        kind = str(d["kind"]).lower()
        if kind.startswith("hf-"):
            d["kind"], d["model"] = "hf", kind[3:]
        return d
    if not isinstance(spec, str):
        return None
    m = _TOK_RE.match(spec)
    if not m:
        return None
    kind = m.group(1).lower()
    from topologies.tokenizers import tokenizer_types

    known = set(tokenizer_types()) | {"hf-bpe", "hf-wordpiece"}
    if kind not in known:
        return None
    out: dict[str, Any] = {"kind": kind}
    if kind.startswith("hf-"):
        out = {"kind": "hf", "model": kind[3:]}
    if m.group(2):
        mult = {"": 1, "k": 1024, "m": 1024 * 1024}[(m.group(3) or "").lower()]
        out["vocab_size"] = int(float(m.group(2)) * mult)
    return out


def is_tokenizer_path(value: Any) -> bool:
    if not isinstance(value, (str, Path)):
        return False
    try:
        p = Path(value)
        return (p.is_dir() and (p / "tokenizer.json").is_file()) or (
            p.is_file() and p.suffix == ".json")
    except (OSError, ValueError):
        return False


# ---------------------------------------------------------------- budgets / durations

_DUR_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*(s|sec|secs|second|seconds|m|min|mins|minute|"
                     r"minutes|h|hr|hrs|hour|hours|d|day|days)\s*$", re.IGNORECASE)
_DUR_MULT = {"s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1, "m": 60, "min": 60,
             "mins": 60, "minute": 60, "minutes": 60, "h": 3600, "hr": 3600, "hrs": 3600,
             "hour": 3600, "hours": 3600, "d": 86400, "day": 86400, "days": 86400}
_COUNT_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([kKmMbBgG]?)\s*([a-zA-Z]+)?\s*$")
_BYTES_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([kKmMgG]?)[bB]\s*$")
_SI = {"": 1, "k": 1e3, "m": 1e6, "b": 1e9, "g": 1e9}


def parse_duration(value: Any) -> float:
    """``"15m"`` -> 900.0, ``"90s"``, ``"1.5h"``, numbers are seconds."""
    if isinstance(value, bool):
        raise ExperimentError(f"Invalid duration {value!r}.", stage="config")
    if isinstance(value, (int, float)):
        if value <= 0:
            raise ExperimentError(f"Duration must be > 0; got {value!r}.", stage="config")
        return float(value)
    m = _DUR_RE.match(str(value))
    if not m:
        raise ExperimentError(f"Invalid duration {value!r}.", stage="config",
                              hint="Use e.g. '90s', '15m', '1.5h' or a number of seconds.")
    return float(m.group(1)) * _DUR_MULT[m.group(2).lower()]


def _parse_amount(v: Any, key: str) -> float:
    """``5_000``, ``"5M"``, ``"5M tokens"``, ``"20MB"`` -> a positive number."""
    if isinstance(v, bool):
        raise ExperimentError(f"Invalid train_for {key}={v!r}.", stage="config")
    if isinstance(v, (int, float)):
        n = float(v)
    else:
        text = str(v).strip()
        m = _BYTES_RE.match(text) if key == "bytes" else None
        m = m or _COUNT_RE.match(text)
        if not m:
            raise ExperimentError(f"Invalid train_for {key}={v!r}.", stage="config",
                                  key="train_for", hint="Use a number such as 2000 or '5M'.")
        n = float(m.group(1)) * _SI[m.group(2).lower()]
    if not n > 0:
        raise ExperimentError(f"train_for {key} must be > 0; got {v!r}.", stage="config")
    return n


_UNITS = {"epoch": "epochs", "epochs": "epochs", "ep": "epochs", "step": "steps",
          "steps": "steps", "token": "tokens", "tokens": "tokens", "tok": "tokens"}


def parse_train_for(value: Any) -> dict[str, float]:
    """Training budget -> ``{"epochs"|"steps"|"time"|"tokens"|"bytes": number, ...}``.

    ``3`` / ``"3 epochs"``, ``"2000 steps"``, ``"10m"`` / ``"90s"`` (wall clock),
    ``"5M tokens"``, ``"20MB"`` (raw text, decimal units), or a dict combining
    them (whichever limit is reached first stops the trial).
    """
    if value is None:
        return {"epochs": 1}
    if isinstance(value, dict):
        out: dict[str, float] = {}
        for k, v in value.items():
            key = {"epoch": "epochs", "step": "steps", "token": "tokens", "byte": "bytes",
                   "seconds": "time", "duration": "time"}.get(str(k), str(k))
            if key not in ("epochs", "steps", "time", "tokens", "bytes"):
                sug = difflib.get_close_matches(key, ["epochs", "steps", "time", "tokens",
                                                      "bytes"], n=1)
                raise ExperimentError(
                    f"Unknown train_for limit {k!r}.", stage="config", key="train_for",
                    hint=(f"Did you mean {sug[0]!r}? " if sug else "")
                    + "Use epochs / steps / time / tokens / bytes.")
            if key == "time":
                out["time"] = parse_duration(v)
                continue
            n = _parse_amount(v, key)
            out[key] = math.ceil(n) if key in ("epochs", "steps") else n
        if not out:
            raise ExperimentError("train_for dict is empty.", stage="config", key="train_for")
        return out
    if isinstance(value, bool):
        raise ExperimentError(f"Invalid train_for {value!r}.", stage="config")
    if isinstance(value, (int, float)):
        if value <= 0:
            raise ExperimentError("train_for must be positive.", stage="config")
        return {"epochs": int(value)}
    text = str(value).strip()
    if _DUR_RE.match(text):
        return {"time": parse_duration(text)}
    m = _BYTES_RE.match(text)
    if m:
        return {"bytes": float(m.group(1)) * _SI[m.group(2).lower()]}
    m = _COUNT_RE.match(text)
    if m:
        unit = _UNITS.get((m.group(3) or "epochs").lower())
        if unit is not None:
            n = float(m.group(1)) * _SI[m.group(2).lower()]
            if unit in ("epochs", "steps"):
                return {unit: math.ceil(n)}
            return {unit: n}
    raise ExperimentError(
        f"Invalid train_for {value!r}.", stage="config", key="train_for",
        hint="Use '3 epochs', '2000 steps', '10m', '5M tokens', '20MB' or "
             "{'epochs': 3, 'time': '10m'}.")


# ---------------------------------------------------------------- the router


def _suggest(key: str, options: Iterable[str]) -> str | None:
    sug = difflib.get_close_matches(key, sorted(set(options)), n=1, cutoff=0.6)
    return sug[0] if sug else None


class Router:
    """Resolve keys to canonical dotted form for one experiment.

    ``recipes``: recipe names trials may use; ``models``: model values the
    bare keys must apply to; ``task_params``: names accepted by ``task.*``.
    """

    def __init__(self, *, recipes: Iterable[str | None] = (), models: Iterable[Any] = (),
                 task_params: Iterable[str] = (), free_form: bool = False,
                 synth: bool = False) -> None:
        self.free_form = free_form
        self.synth = synth
        self.trainer = trainer_fields()
        self.optimizer = optimizer_fields()
        self.recipes = [r for r in recipes if r]
        self.recipe = set().union(*(recipe_fields(r) for r in self.recipes)) \
            if self.recipes else set()
        self.task = set(task_params)
        self.models = list(models)
        self._factories: list[Any] | None = None
        self._scope: list[Any] | None = None
        self._memo: dict[tuple, str] = {}

    # -- helpers -------------------------------------------------------
    @staticmethod
    def _resolve_all(models: list[Any]) -> list[Any]:
        out = []
        for m in models:
            if m is None or isinstance(m, dict):
                continue
            try:
                out.append(resolve_factory(m))
            except ExperimentError:
                continue
        return out

    def factories(self) -> list[Any]:
        """Models the current key applies to (one trial's model, or all of them)."""
        if self._scope is not None:
            return self._scope
        if self._factories is None:
            self._factories = self._resolve_all(self.models)
        return self._factories

    def model_kwargs(self) -> set[str]:
        names: set[str] = set()
        for f in self.factories():
            names |= constructor_params(f)[0]
        return names

    def known_keys(self) -> list[str]:
        keys = set(TOP_LEVEL) | self.trainer | self.optimizer | self.recipe
        keys |= self.model_kwargs()
        keys |= {f"task.{k}" for k in self.task}
        keys |= {f"trainer.{k}" for k in self.trainer}
        keys |= {f"optimizer.{k}" for k in self.optimizer}
        keys |= {f"recipe.{k}" for k in self.recipe}
        keys |= {f"model.{k}" for k in self.model_kwargs()}
        return sorted(keys)

    # -- resolution ----------------------------------------------------
    def resolve(self, key: str, models: list[Any] | None = None) -> str:
        """Canonical dotted key (raises ``ExperimentError`` with suggestions).

        ``models``: the model value(s) this key applies to (default: every model of
        the experiment); model kwargs must be accepted by all of them.
        """
        if self.free_form:
            return key
        memo = (key, tuple(id(m) for m in models) if models is not None else None)
        if memo in self._memo:
            return self._memo[memo]
        prev = self._scope
        self._scope = self._resolve_all(models) if models is not None else None
        try:
            out = self._resolve(key)
        finally:
            self._scope = prev
        self._memo[memo] = out
        return out

    def _resolve(self, key: str) -> str:
        if not isinstance(key, str) or not key:
            raise ExperimentError(f"Invalid key {key!r}.", stage="routing")
        if key.startswith("data."):
            key = "task." + key[5:]
        if key in TOP_LEVEL:
            return key
        if "." in key:
            ns, _, sub = key.partition(".")
            if ns not in NAMESPACES:
                sug = _suggest(ns, NAMESPACES)
                raise ExperimentError(
                    f"Unknown key namespace {ns!r} in {key!r}.", stage="routing", key=key,
                    hint=(f"Did you mean '{sug}.{sub}'? " if sug else "")
                    + "Namespaces: " + ", ".join(f"{n}.*" for n in NAMESPACES) + ".")
            self._check_dotted(ns, sub, key)
            return key
        return self._resolve_bare(key)

    def _check_dotted(self, ns: str, sub: str, key: str) -> None:
        if not sub:
            raise ExperimentError(f"Key {key!r} names no field.", stage="routing", key=key)
        if ns == "trainer":
            if sub in MANAGED_TRAINER_KEYS:
                raise ExperimentError(
                    f"{key} is managed by the experiment.", stage="routing", key=key,
                    hint={"max_epochs": "Use train_for='3 epochs'.",
                          "max_steps": "Use train_for='2000 steps'.",
                          "seed": "Use seeds=... (or a 'seed' axis).",
                          "optimizer": "Use the 'optimizer' key.",
                          "scheduler": "Use the 'scheduler' key.",
                          "recipe": "Use the 'recipe' key."}.get(
                        sub, "Experiments set it per trial (trial directory, jsonl log)."))
            if sub not in self.trainer:
                self._unknown(key, sub, self.trainer, "trainer")
        elif ns == "optimizer":
            if sub in ("scheduler", "scheduler_kwargs"):
                raise ExperimentError(f"{key}: use the 'scheduler' / 'scheduler.<kw>' keys.",
                                      stage="routing", key=key)
            if sub not in self.optimizer:
                sug = difflib.get_close_matches(sub, sorted(self.optimizer), n=1, cutoff=0.8)
                if sug:  # near-miss of a real field: a typo, not an optimizer extra
                    self._unknown(key, sub, self.optimizer, "optimizer")
        elif ns == "recipe":
            if self.recipes and sub not in self.recipe:
                self._unknown(key, sub, self.recipe, "recipe")
        elif ns == "task":
            if sub not in self.task:
                self._unknown(key, sub, self.task, "task")
        elif ns == "synth":
            if not self.synth:
                raise ExperimentError(
                    f"{key}: synth.* keys vary the synthetic data and need a SynthTask.",
                    stage="routing", key=key,
                    hint="task=synth.SynthTask(spec, eval=real_eval); keys are spec overrides "
                         "(synth.size, synth.generator.<node path>, "
                         "synth.ops.dedup.near.threshold, synth.mix.real).")
        elif ns == "model":
            bad = [f for f in self.factories() if not model_accepts(f, sub)]
            if bad:
                names = sorted(constructor_params(bad[0])[0])
                sug = _suggest(sub, names)
                raise ExperimentError(
                    f"{getattr(bad[0], '__name__', bad[0])} does not accept {sub!r}.",
                    stage="routing", key=key,
                    hint=(f"Did you mean 'model.{sug}'? " if sug else "")
                    + "Restrict the key to some models with Variant(label, Model, "
                      f"{sub}=...) instead of a global key.",
                    available=names)

    def _unknown(self, key: str, sub: str, options: set[str], ns: str) -> None:
        sug = _suggest(sub, options)
        raise ExperimentError(
            f"Unknown key {key!r}.", stage="routing", key=key,
            hint=(f"Did you mean '{ns}.{sug}'?" if sug else f"No such {ns} field."),
            available=sorted(f"{ns}.{o}" for o in options))

    def _resolve_bare(self, key: str) -> str:
        if key in self.trainer and key not in MANAGED_TRAINER_KEYS:
            return f"trainer.{key}"
        hits: list[str] = []
        if key in self.optimizer:
            hits.append(f"optimizer.{key}")
        if key in self.recipe:
            hits.append(f"recipe.{key}")
        facs = self.factories()
        if facs and all(model_accepts(f, key) for f in facs) and any(
                key in constructor_params(f)[0] for f in facs):
            hits.append(f"model.{key}")
        if key in self.task:
            hits.append(f"task.{key}")
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise ExperimentError(
                f"Key {key!r} is ambiguous: it matches {', '.join(hits)}.", stage="routing",
                key=key, hint="Use the dotted form: " + " or ".join(repr(h) for h in hits) + ".")
        if key in MANAGED_TRAINER_KEYS:
            self._check_dotted("trainer", key, key)
        partial = [f for f in facs if key in constructor_params(f)[0]]
        if partial:
            missing = [getattr(f, "__name__", repr(f)) for f in facs if
                       not model_accepts(f, key)]
            raise ExperimentError(
                f"{key!r} is a constructor argument of some models but not of "
                f"{', '.join(missing)}.", stage="routing", key=key,
                hint=f"Use Variant(label, Model, {key}=...) for the models that take it.")
        sug = _suggest(key, self.known_keys())
        raise ExperimentError(
            f"Unknown key {key!r}.", stage="routing", key=key,
            hint=(f"Did you mean {sug!r}? " if sug else "")
            + "Bare keys resolve to trainer, optimizer, recipe or model-constructor fields; "
              "dotted keys name the stage (model.dim, trainer.batch_size, task.block_size).")

    def route(self, params: dict[str, Any], models: list[Any] | None = None) -> dict[str, Any]:
        """Resolve every key; two spellings of one key with different values raise."""
        out: dict[str, Any] = {}
        origin: dict[str, str] = {}
        for k, v in params.items():
            ck = self.resolve(k, models)
            if ck in out and origin[ck] != k:
                raise ExperimentError(f"{k!r} and {origin[ck]!r} both set {ck!r}.",
                                      stage="routing", key=ck)
            out[ck] = v
            origin[ck] = k
        return out


__all__ = [
    "NAMESPACES",
    "TOP_LEVEL",
    "Router",
    "constructor_params",
    "is_ctx_factory",
    "is_tokenizer_path",
    "parse_duration",
    "parse_tokenizer_spec",
    "parse_train_for",
    "resolve_factory",
]
