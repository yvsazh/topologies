"""Attribute space: what varies between records, and how much.

    space = AttributeSpace({
        "label": ["refund", "shipping_delay", "warranty"],          # uniform categorical
        "language": {"en": 0.6, "de": 0.2, "uk": 0.2},             # weighted categorical
        "length": intuniform(4, 60),                                 # continuous (LHS-able)
        "persona": synth.personas(200),                              # pool
        "tone": synth.when("label", {"refund": ["angry", "polite"]}, default=["neutral"]),
        "invoice": synth.from_program(make_invoice),                 # derived value
        "max_tokens": 256,                                           # constant
    }, where=lambda a: not (a["language"] == "uk" and a["tone"] == "terse"))

Values: constants, lists (uniform), ``{value: weight}`` dicts (wrap a constant
dict in ``const({...})``), ``experiments`` distributions (``choice`` becomes
categorical; ``uniform`` / ``loguniform`` / ``intuniform`` / ``intloguniform``
are continuous), pools, ``when(...)`` conditionals, and callables
``fn(attrs)`` / ``fn(attrs, rng)`` / ``from_program(fn)`` evaluated after the
sampled attributes (in declaration order). Samplers (``sampling.py``) control
how categorical / continuous attributes are combined (coverage guarantees);
every record stores its full assignment.
"""

from __future__ import annotations

import inspect
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from topologies.experiments.distributions import (
    Distribution,
    choice,
    intloguniform,
    intuniform,
    loguniform,
    uniform,
)
from topologies.synth.ctor import Recorded
from topologies.synth.errors import SynthError, unknown
from topologies.synth.rng import SynthRandom

# ---------------------------------------------------------------- attribute kinds


class Attr(Recorded):
    """Base attribute. ``kind``: ``const`` | ``categorical`` | ``continuous`` | ``derived``."""

    kind = "const"
    name: str = "?"
    depends: tuple[str, ...] = ()

    def sample(self, rng: SynthRandom, attrs: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    def describe(self) -> Any:
        """JSON-able description (part of the generation fingerprint)."""
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.describe()!r})"


class Const(Attr):
    kind = "const"

    def __init__(self, value: Any) -> None:
        self.value = value

    def sample(self, rng: SynthRandom, attrs: Mapping[str, Any]) -> Any:
        return self.value

    def describe(self) -> Any:
        from topologies.experiments.spec import canonical

        return {"const": canonical(self.value, warn=False)}


def const(value: Any) -> Const:
    """A constant attribute value (needed for dict constants: ``const({"a": 1})``)."""
    return Const(value)


class Categorical(Attr):
    """Finite values with weights (uniform by default)."""

    kind = "categorical"

    def __init__(self, values: Sequence[Any], weights: Sequence[float] | None = None) -> None:
        values = list(values)
        if not values:
            raise SynthError("A categorical attribute needs at least one value.",
                             stage="attributes")
        if weights is None:
            weights = [1.0] * len(values)
        weights = [float(w) for w in weights]
        if len(weights) != len(values) or any(w < 0 or not math.isfinite(w) for w in weights) \
                or sum(weights) <= 0:
            raise SynthError("Attribute weights must be finite, >= 0, one per value and not all "
                             "zero.", stage="attributes")
        total = sum(weights)
        self._values = values
        self._probs = [w / total for w in weights]

    @property
    def values(self) -> list[Any]:
        return list(self._values)

    @property
    def probs(self) -> list[float]:
        return list(self._probs)

    def prob_of(self, value: Any) -> float:
        for v, p in zip(self._values, self._probs):
            if v == value:
                return p
        return 0.0

    def sample(self, rng: SynthRandom, attrs: Mapping[str, Any]) -> Any:
        if len(set(self._probs)) == 1:
            return self._values[rng.randrange(len(self._values))]
        return rng.choices(self._values, weights=self._probs, k=1)[0]

    def reweighted(self, factors: Mapping[Any, float]) -> Categorical:
        """Copy with weights multiplied by ``factors[value]`` (adaptive sampling)."""
        w = [p * float(factors.get(_key(v), factors.get(v, 1.0)) if _hashable(v) else p)
             for v, p in zip(self._values, self._probs)]
        out = type(self).__new__(type(self))
        out.__dict__.update(self.__dict__)
        total = sum(w) or 1.0
        out._probs = [x / total for x in w]
        out.__dict__["_ctor"] = (Categorical, (list(self._values),), {"weights": out._probs}) \
            if type(self) is Categorical else None
        if out.__dict__["_ctor"] is None:
            out.__dict__["_ctor_stale"] = "reweighted"
        return out

    def describe(self) -> Any:
        from topologies.experiments.spec import canonical

        uniform_ = len(set(self._probs)) == 1
        vals = canonical(self._values, warn=False)
        if uniform_:
            return {"values": vals}
        return {"values": vals, "probs": [round(p, 12) for p in self._probs]}


def _hashable(v: Any) -> bool:
    try:
        hash(v)
        return True
    except TypeError:
        return False


def _key(v: Any) -> Any:
    return v if isinstance(v, (str, int, float, bool)) or v is None else json.dumps(
        v, sort_keys=True, default=str)


class Pool(Categorical):
    """A (large) list of values, optionally weighted, from a list or a file.

    Files: ``.txt`` (one value per line), ``.json`` (list), ``.jsonl`` (one
    value per line), ``.yaml`` (list). Loaded lazily; the content hash is part
    of the fingerprint.
    """

    kind = "categorical"

    def __init__(self, source: str | Path | Sequence[Any], weights: Sequence[float] | None = None,
                 *, name: str | None = None) -> None:
        self.source = source
        self._weights_arg = list(weights) if weights is not None else None
        self.pool_name = name
        self._loaded = False
        if not isinstance(source, (str, Path)):
            self._load()

    def _load(self) -> None:
        if self._loaded:
            return
        values = load_pool(self.source) if isinstance(self.source, (str, Path)) else \
            list(self.source)
        Categorical.__init__(self, values, self._weights_arg)
        self._loaded = True

    @property
    def values(self) -> list[Any]:
        self._load()
        return list(self._values)

    @property
    def probs(self) -> list[float]:
        self._load()
        return list(self._probs)

    def sample(self, rng: SynthRandom, attrs: Mapping[str, Any]) -> Any:
        self._load()
        return super().sample(rng, attrs)

    def prob_of(self, value: Any) -> float:
        self._load()
        return super().prob_of(value)

    def describe(self) -> Any:
        from topologies.experiments.cache import stable_hash

        self._load()
        return {"pool": self.pool_name or (str(self.source) if isinstance(
            self.source, (str, Path)) else None), "n": len(self._values),
            "sha": stable_hash([self._values, self._probs])}


def load_pool(path: str | Path) -> list[Any]:
    p = Path(path)
    if not p.is_file():
        raise SynthError(f"Pool file {p} does not exist.", stage="attributes",
                         hint="Use a .txt (one value per line), .json/.yaml (a list) or .jsonl "
                              "file, or pass a Python list.")
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".json":
        data = json.loads(text)
    elif p.suffix == ".jsonl":
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    elif p.suffix in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text)
    else:
        data = [line.strip() for line in text.splitlines() if line.strip()]
    if isinstance(data, dict) and "values" in data:
        data = data["values"]
    if not isinstance(data, list) or not data:
        raise SynthError(f"Pool file {p} holds no list of values.", stage="attributes")
    return data


def pool(source: str | Path | Sequence[Any], weights: Sequence[float] | None = None, *,
         name: str | None = None) -> Pool:
    """``synth.pool("personas.txt")`` / ``synth.pool([...], weights=[...])``."""
    return Pool(source, weights, name=name)


class Continuous(Attr):
    """Numeric attribute from an ``experiments`` distribution (LHS uses ``ppf``)."""

    kind = "continuous"

    def __init__(self, dist: Distribution) -> None:
        self.dist = dist

    def sample(self, rng: SynthRandom, attrs: Mapping[str, Any]) -> Any:
        return self.dist.sample(rng)

    def ppf(self, u: float) -> Any:
        """Inverse CDF (``u`` in ``[0, 1)``) for stratified / Latin-hypercube sampling."""
        d = self.dist
        u = min(max(float(u), 0.0), 1.0 - 1e-12)
        if isinstance(d, uniform):
            return d.low + (d.high - d.low) * u
        if isinstance(d, loguniform):
            lo, hi = math.log(d.low), math.log(d.high)
            return math.exp(lo + (hi - lo) * u)
        if isinstance(d, intuniform):
            return int(d.low) + min(int(u * (int(d.high) - int(d.low) + 1)),
                                    int(d.high) - int(d.low))
        if isinstance(d, intloguniform):
            v = math.exp(math.log(d.low) + (math.log(d.high + 1) - math.log(d.low)) * u)
            return min(int(d.high), max(int(d.low), int(v)))
        ppf = getattr(d, "ppf", None)
        if callable(ppf):
            return ppf(u)
        raise SynthError(f"{d!r} has no inverse CDF; LHS cannot stratify it.",
                         stage="attributes", hint="Use sampler='random' for this attribute, or "
                                                  "give the distribution a ppf(u) method.")

    def describe(self) -> Any:
        return {"distribution": repr(self.dist)}


class Derived(Attr):
    """Value computed from the other attributes: ``fn(attrs)`` / ``fn(attrs, rng)`` /
    ``fn(rng)`` / ``fn(ctx)`` (``ctx.rng``, ``ctx.attrs``, ``ctx.record_index``)."""

    kind = "derived"

    def __init__(self, fn: Callable[..., Any], *, depends: Sequence[str] = ()) -> None:
        if not callable(fn):
            raise SynthError(f"Derived attribute needs a callable; got {fn!r}.",
                             stage="attributes")
        self.fn = fn
        self.depends = tuple(depends)
        self._mode = _call_mode(fn)

    def sample(self, rng: SynthRandom, attrs: Mapping[str, Any], index: int = -1) -> Any:
        mode = self._mode
        if mode == "ctx":
            return self.fn(_AttrCtx(rng, dict(attrs), index))
        if mode == "rng":
            return self.fn(rng)
        if mode == "rng_attrs":
            return self.fn(rng, dict(attrs))
        if mode == "attrs_rng":
            return self.fn(dict(attrs), rng)
        if mode == "none":
            return self.fn()
        return self.fn(dict(attrs))

    def describe(self) -> Any:
        from topologies.synth.fingerprint import callable_fingerprint

        return {"fn": callable_fingerprint(self.fn)}


class _AttrCtx:
    """Minimal context handed to ``fn(ctx)`` attribute producers."""

    def __init__(self, rng: SynthRandom, attrs: dict, index: int) -> None:
        self.rng, self.attrs, self.record_index = rng, attrs, index

    def __getitem__(self, key: str) -> Any:
        return self.attrs[key]


def _call_mode(fn: Callable[..., Any]) -> str:
    try:
        params = [p for p in inspect.signature(fn).parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                  and p.default is p.empty]
    except (TypeError, ValueError):
        return "attrs"
    names = [p.name for p in params]
    if not names:
        return "none"
    if names[0] == "ctx":
        return "ctx"
    if names[0] in ("rng", "random", "r"):
        return "rng_attrs" if len(names) >= 2 else "rng"
    return "attrs_rng" if len(names) >= 2 else "attrs"


class When(Attr):
    """Conditional attribute: its distribution depends on another attribute's value.

    ``when("label", {"refund": ["angry", "polite"], "warranty": "neutral"}, default=[...])``
    """

    kind = "derived"

    def __init__(self, on: str, mapping: Mapping[Any, Any], default: Any = None) -> None:
        self.on = str(on)
        self.mapping = {k: as_attr(v) for k, v in mapping.items()}
        self.default = as_attr(default) if default is not None else None
        self.depends = (self.on,)

    def sample(self, rng: SynthRandom, attrs: Mapping[str, Any], index: int = -1) -> Any:
        if self.on not in attrs:
            raise SynthError(f"when({self.on!r}, ...) needs attribute {self.on!r} declared "
                             "before it.", stage="attributes")
        key = attrs[self.on]
        spec = self.mapping.get(key, self.mapping.get(str(key), self.default))
        if spec is None:
            raise SynthError(f"when({self.on!r}, ...) has no entry for {key!r} and no default.",
                             stage="attributes",
                             hint=f"Add {key!r} to the mapping or pass default=[...].")
        if isinstance(spec, Derived):
            return spec.sample(rng, attrs, index)
        return spec.sample(rng, attrs)

    def describe(self) -> Any:
        return {"when": self.on, "cases": {str(k): v.describe() for k, v in self.mapping.items()},
                "default": self.default.describe() if self.default is not None else None}


def when(on: str, mapping: Mapping[Any, Any] | Callable[[Any], Any], *,
         default: Any = None) -> Attr:
    """Conditional attribute (see ``When``); ``when("label", fn)`` calls ``fn(label_value)``."""
    if callable(mapping) and not isinstance(mapping, Mapping):
        fn = mapping
        return Derived(lambda attrs: fn(attrs[on]), depends=(on,))
    return When(on, mapping, default)


# ---------------------------------------------------------------- normalization


def _is_weight_dict(spec: Mapping) -> bool:
    return bool(spec) and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                              for v in spec.values())


def as_attr(spec: Any) -> Attr:
    """Normalize a user value into an ``Attr`` (see module docstring for the rules)."""
    if isinstance(spec, Attr):
        return spec
    to_attr = getattr(spec, "as_attr", None)
    if callable(to_attr) and not isinstance(spec, type):
        return to_attr()
    if isinstance(spec, choice):
        return Categorical(list(spec.values))
    if isinstance(spec, Distribution):
        return Continuous(spec)
    if isinstance(spec, (list, tuple, range)):
        return Categorical(list(spec))
    if isinstance(spec, Mapping):
        if _is_weight_dict(spec):
            return Categorical(list(spec), list(spec.values()))
        raise SynthError(f"Ambiguous dict attribute {dict(spec)!r}.", stage="attributes",
                         hint="Weighted choices need numeric weights ({'en': 0.6, 'de': 0.4}); "
                              "wrap a constant dict with synth.const({...}).")
    if callable(spec) and not isinstance(spec, type):
        return Derived(spec)
    return Const(spec)


# ---------------------------------------------------------------- the space


class AttributeSpace(Recorded):
    """Named attributes + constraints (``where`` forbids, ``require`` forces combinations)."""

    def __init__(self, attrs: Mapping[str, Any] | AttributeSpace | None = None, *,
                 where: Callable[[dict], bool] | None = None,
                 require: Iterable[Mapping[str, Any]] | None = None) -> None:
        if isinstance(attrs, AttributeSpace):
            self.attrs: dict[str, Attr] = dict(attrs.attrs)
            where = where or attrs.where
            require = require if require is not None else attrs.require
        else:
            self.attrs = {}
            for name, spec in (attrs or {}).items():
                if not isinstance(name, str) or not name.isidentifier():
                    raise SynthError(f"Attribute name {name!r} must be an identifier.",
                                     stage="attributes")
                a = as_attr(spec)
                self.attrs[name] = a
        for name, a in self.attrs.items():
            a.name = name
        self.where = where
        self.require = [dict(r) for r in (require or [])]
        for r in self.require:
            for k, v in r.items():
                a = self.attrs.get(k)
                if a is None:
                    raise unknown("attribute", k, self.attrs, stage="attributes",
                                  extra="require=[...] names attributes of the space.")
                if a.kind == "categorical" and v not in a.values:  # type: ignore[attr-defined]
                    raise SynthError(f"require=[...] uses {k}={v!r}, which is not a value of "
                                     f"{k!r}.", stage="attributes",
                                     available=[str(x) for x in a.values])  # type: ignore
        for name, a in self.attrs.items():
            for dep in a.depends:
                if dep not in self.attrs:
                    raise unknown("attribute", dep, self.attrs, stage="attributes",
                                  extra=f"{name!r} depends on it.")
                if list(self.attrs).index(dep) > list(self.attrs).index(name) and \
                        self.attrs[dep].kind == "derived":
                    raise SynthError(f"{name!r} depends on {dep!r}, which is declared after it.",
                                     stage="attributes", hint=f"Declare {dep!r} first.")

    # -- views -------------------------------------------------------------
    def __contains__(self, name: object) -> bool:
        return name in self.attrs

    def __getitem__(self, name: str) -> Attr:
        try:
            return self.attrs[name]
        except KeyError:
            raise unknown("attribute", name, self.attrs, stage="attributes") from None

    def __iter__(self):
        return iter(self.attrs)

    def __len__(self) -> int:
        return len(self.attrs)

    @property
    def names(self) -> list[str]:
        return list(self.attrs)

    def of_kind(self, kind: str) -> list[str]:
        return [n for n, a in self.attrs.items() if a.kind == kind]

    @property
    def categorical(self) -> list[str]:
        return self.of_kind("categorical")

    @property
    def continuous(self) -> list[str]:
        return self.of_kind("continuous")

    def values(self, name: str) -> list[Any]:
        a = self[name]
        if a.kind != "categorical":
            raise SynthError(f"Attribute {name!r} is {a.kind}, not categorical.",
                             stage="attributes")
        return a.values  # type: ignore[attr-defined]

    # -- sampling ------------------------------------------------------------
    def sample_free(self, rng: SynthRandom) -> dict[str, Any]:
        """Independent draw of every categorical / continuous attribute."""
        return {n: a.sample(rng, {}) for n, a in self.attrs.items()
                if a.kind in ("categorical", "continuous")}

    def complete(self, base: Mapping[str, Any], rng: SynthRandom, index: int = -1) -> dict:
        """Constants and derived attributes on top of sampled ``base`` (declaration order)."""
        out: dict[str, Any] = {}
        for name, a in self.attrs.items():
            if name in base:
                out[name] = base[name]
            elif a.kind == "derived":
                out[name] = a.sample(rng, out, index)  # type: ignore[call-arg]
            else:
                out[name] = a.sample(rng, out)
        return out

    def allowed(self, attrs: Mapping[str, Any]) -> bool:
        if self.where is None:
            return True
        try:
            return bool(self.where(dict(attrs)))
        except KeyError:
            return True

    def with_attrs(self, **changes: Any) -> AttributeSpace:
        """Copy with some attributes replaced (``None`` removes one)."""
        attrs: dict[str, Any] = dict(self.attrs)
        for k, v in changes.items():
            if v is None:
                attrs.pop(k, None)
            else:
                attrs[k] = v
        return AttributeSpace(attrs, where=self.where, require=self.require)

    def reweighted(self, factors: Mapping[str, Mapping[Any, float]]) -> AttributeSpace:
        """Copy with categorical weights multiplied (``{"tone": {"angry": 2.0}}``)."""
        attrs: dict[str, Any] = dict(self.attrs)
        for name, f in factors.items():
            a = self[name]
            if a.kind == "categorical":
                attrs[name] = a.reweighted(f)  # type: ignore[attr-defined]
        return AttributeSpace(attrs, where=self.where, require=self.require)

    def describe(self) -> dict[str, Any]:
        from topologies.synth.fingerprint import callable_fingerprint

        return {"attrs": {n: {"kind": a.kind, **(_as_dict(a.describe()))}
                          for n, a in self.attrs.items()},
                "where": callable_fingerprint(self.where) if self.where else None,
                "require": self.require}

    def expected(self, name: str) -> dict[str, float]:
        """Target distribution of a categorical attribute (``{value: p}``; for coverage)."""
        a = self[name]
        if a.kind != "categorical":
            return {}
        return {str(_key(v)): p for v, p in zip(a.values, a.probs)}  # type: ignore[attr-defined]

    def __repr__(self) -> str:
        return f"AttributeSpace({', '.join(f'{n}:{a.kind}' for n, a in self.attrs.items())})"


def _as_dict(d: Any) -> dict:
    return d if isinstance(d, dict) else {"value": d}


__all__ = ["Attr", "AttributeSpace", "Categorical", "Const", "Continuous", "Derived", "Pool",
           "When", "as_attr", "const", "load_pool", "pool", "when"]
