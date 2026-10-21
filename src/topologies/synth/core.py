"""Nodes: reproducible producers of values, composable into generators of any data.

A *node* produces one value per record. Distributions, constants, records,
renderers, transforms, processes and whole generators are all nodes, and
composing nodes yields nodes. Three ways to write one:

    f0 = synth.uniform(80, 800)                                  # 1. compose
    gen = synth.record(audio=synth.audio.sine(f0=f0), f0=f0)

    @synth.node(type=synth.Audio)                                # 2. a function
    def chirp(r, *, f0: float, f1: float, seconds: float = 1.0, sr: int = 16_000):
        t = torch.arange(int(seconds * sr)) / sr
        return torch.sin(2 * torch.pi * (f0 + (f1 - f0) * t / seconds / 2) * t)

    class RandomWalk(synth.Node):                                # 3. a class
        def __init__(self, steps=128, scale=1.0):
            self.steps, self.scale = steps, scale
        def sample(self, ctx):
            n, s = ctx.resolve(self.steps), ctx.resolve(self.scale)
            return (torch.randn(n, generator=ctx.rng.torch) * s).cumsum(0)

Parameters may be constants, nodes, ``experiments`` distributions or containers
holding nodes; they are resolved per record. Within one record a node object is
sampled once wherever it is referenced (sharing by identity); ``repeat``,
``iid`` and ``node.fresh()`` give independent draws. Every value depends only on
``(spec seed, record index, node path)`` (see ``synth.evaluate``).
"""

from __future__ import annotations

import copy
import functools
import inspect
import operator
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from topologies.synth.ctor import Recorded, factory
from topologies.synth.errors import SynthError, unknown

_RNG_NAMES = ("r", "rng")


# ---------------------------------------------------------------- helpers


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def is_node(x: Any) -> bool:
    return isinstance(x, Node)


def as_param(value: Any) -> Any:
    """Normalize a parameter value: ``experiments`` distributions become nodes; containers
    keep their structure (nodes inside them are resolved per record)."""
    from topologies.experiments.distributions import Distribution

    if isinstance(value, Node):
        return value
    if isinstance(value, Distribution):
        from topologies.synth.dists import from_distribution

        return from_distribution(value)
    if isinstance(value, list):
        return [as_param(v) for v in value]
    if isinstance(value, tuple) and type(value) is tuple:
        return tuple(as_param(v) for v in value)
    if isinstance(value, dict) and type(value) is dict:
        return {k: as_param(v) for k, v in value.items()}
    return value


def has_nodes(value: Any) -> bool:
    """Does ``value`` (a parameter) hold a node anywhere inside?"""
    if isinstance(value, Node):
        return True
    if isinstance(value, (list, tuple)) and not isinstance(value, Const):
        return any(has_nodes(v) for v in value)
    if isinstance(value, dict):
        return any(has_nodes(v) for v in value.values())
    return False


def iter_children(value: Any, label: str) -> list[tuple[str, Node]]:
    """``(label, node)`` for every node inside a parameter value (containers traversed)."""
    if isinstance(value, Node):
        return [(label, value)]
    out: list[tuple[str, Node]] = []
    if isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            out += iter_children(v, f"{label}[{i}]")
    elif isinstance(value, dict):
        for k, v in value.items():
            out += iter_children(v, f"{label}.{k}")
    return out


def _ctor_params(obj: Any) -> dict[str, Any]:
    """Constructor arguments by parameter name (from the ``Recorded`` record)."""
    ctor = getattr(obj, "__dict__", {}).get("_ctor")
    if ctor is None:
        return {}
    target, args, kwargs = ctor
    try:
        fn = target.__init__ if isinstance(target, type) else target
        sig = inspect.signature(fn)
        plist = [p for p in sig.parameters.values()][1 if isinstance(target, type) else 0:]
    except (TypeError, ValueError):
        return dict(kwargs)
    out: dict[str, Any] = {}
    i = 0
    for p in plist:
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and i < len(args):
            out[p.name] = args[i]
            i += 1
        elif p.kind is p.VAR_POSITIONAL:
            out[p.name] = list(args[i:])
            i = len(args)
    out.update(kwargs)
    for p in plist:  # defaults left out == defaults written out
        if p.name not in out and p.default is not p.empty and \
                p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            out[p.name] = p.default
    return out


# ---------------------------------------------------------------- the node


class Node(Recorded):
    """Base node (see module docstring).

    Hooks: ``sample(ctx)`` (required), ``render_batch(ctxs, params_list)`` (optional
    vectorized rendering over many records; set ``batched = True``), ``setup(env)``
    (once per run), ``params()`` (child values; default: constructor arguments),
    ``describe()``, ``fingerprint()``.
    """

    #: Value type (``synth.Audio``, ``synth.Text``, ... or an instance with parameters).
    type: Any = None
    #: Short tags for ``synth.lib()`` (``["audio", "ddsp"]``).
    tags: tuple[str, ...] = ()
    #: ``True`` when ``render_batch`` renders many records at once.
    batched: bool = False

    # -- protocol ------------------------------------------------------------------
    def sample(self, ctx: Any) -> Any:
        if self.batched:
            return self.render_batch([ctx], [ctx.resolve_params(self)])[0]
        raise NotImplementedError(f"{type(self).__name__}.sample(ctx)")

    def render_batch(self, ctxs: Sequence[Any], params_list: Sequence[dict[str, Any]]
                     ) -> list[Any]:
        """Values for many records; ``params_list[i]`` holds the resolved ``params()``."""
        return [self.sample(c) for c in ctxs]

    def setup(self, env: Any) -> None:
        """Called once per run (load data packs, build tables)."""

    def params(self) -> dict[str, Any]:
        """Parameter values (constants or nodes); default: the constructor arguments."""
        return _ctor_params(self)

    def node_kind(self) -> str:
        """Short name used in structural paths (``audio/chirp.f1``)."""
        return _snake(type(self).__name__)

    @property
    def node_name(self) -> str | None:
        """Explicit path given with ``.named(...)`` / ``name=``."""
        return self.__dict__.get("_name")

    def describe(self) -> str:
        from topologies.synth.introspect import describe

        return describe(self, echo=False)

    def fingerprint(self) -> str:
        from topologies.synth.evaluate import graph_fingerprint

        return graph_fingerprint(self)

    def to_yaml(self, path: Any = None) -> str:
        from topologies.synth.yaml_spec import node_to_yaml

        return node_to_yaml(self, path)

    # -- copies ---------------------------------------------------------------------
    def named(self, name: str) -> Node:
        """A copy with an explicit path ``name`` (pins its randomness to that name)."""
        if not isinstance(name, str) or not name:
            raise SynthError("A node name must be a non-empty string.", stage="node")
        new = copy.copy(self)
        new.__dict__["_name"] = name
        calls = [c for c in new.__dict__.get("_calls", []) if c[0] != "named"]
        new.__dict__["_calls"] = [*calls, ("named", (name,), {})]
        return new

    def fresh(self) -> Node:
        """An independent copy: same parameters, its own randomness (breaks sharing)."""
        new = copy.copy(self)
        new.__dict__.pop("_name", None)
        new.__dict__["_calls"] = [c for c in new.__dict__.get("_calls", [])
                                  if c[0] != "named"]
        return new

    def with_(self, **changes: Any) -> Node:
        """Copy with parameters replaced by path (``gen.with_(**{"audio.f1": 440})``)."""
        from topologies.synth.overrides import override_node

        return override_node(self, changes)

    def replace_params(self, **changes: Any) -> Node:
        """A new node with some constructor parameters changed (used by overrides)."""
        ctor = self.__dict__.get("_ctor")
        if ctor is None:
            raise SynthError(f"{type(self).__name__} cannot be copied with new parameters "
                             "(no constructor record).", stage="node")
        target, args, kwargs = ctor
        params = _ctor_params(self)
        bad = [k for k in changes if k not in params and not _accepts(target, k)]
        if bad:
            raise unknown(f"parameter of {self.node_kind()}", bad[0], list(params), stage="node")
        try:
            fn = target.__init__ if isinstance(target, type) else target
            names = [p.name for p in inspect.signature(fn).parameters.values()][
                1 if isinstance(target, type) else 0:]
        except (TypeError, ValueError):
            names = []
        args = list(args)
        kwargs = dict(kwargs)
        for k, v in changes.items():
            if k in names and names.index(k) < len(args):
                args[names.index(k)] = v
            else:
                kwargs[k] = v
        new = target(*args, **kwargs)
        for name, a, kw in self.__dict__.get("_calls", []):
            getattr(new, name)(*a, **kw)
        return new

    # -- sugar --------------------------------------------------------------------------
    def map(self, fn: Callable[[Any], Any]) -> Node:
        """``fn(value)`` of this node's value (a deterministic transform)."""
        return apply(fn, self)

    def get(self, key: Any) -> Node:
        """Attribute or key of the value (``node.get("shape")``, ``node.get("text")``)."""
        return apply(_get, self, key)

    def eq(self, other: Any) -> Node:
        """``value == other`` as a node (``==`` itself keeps identity semantics)."""
        return apply(operator.eq, self, other)

    def ne(self, other: Any) -> Node:
        return apply(operator.ne, self, other)

    def __getitem__(self, key: Any) -> Node:
        return apply(operator.getitem, self, key)

    def __iter__(self) -> Any:
        raise TypeError(f"A node is not iterable; use synth.repeat(...) for lists of draws "
                        f"or node[i] for one element ({self!r}).")

    def __or__(self, transform: Any) -> Node:
        return chain(self, transform)

    def __add__(self, o: Any) -> Node:
        return apply(operator.add, self, o)

    def __radd__(self, o: Any) -> Node:
        return apply(operator.add, o, self)

    def __sub__(self, o: Any) -> Node:
        return apply(operator.sub, self, o)

    def __rsub__(self, o: Any) -> Node:
        return apply(operator.sub, o, self)

    def __mul__(self, o: Any) -> Node:
        return apply(operator.mul, self, o)

    def __rmul__(self, o: Any) -> Node:
        return apply(operator.mul, o, self)

    def __truediv__(self, o: Any) -> Node:
        return apply(operator.truediv, self, o)

    def __rtruediv__(self, o: Any) -> Node:
        return apply(operator.truediv, o, self)

    def __floordiv__(self, o: Any) -> Node:
        return apply(operator.floordiv, self, o)

    def __rfloordiv__(self, o: Any) -> Node:
        return apply(operator.floordiv, o, self)

    def __mod__(self, o: Any) -> Node:
        return apply(operator.mod, self, o)

    def __rmod__(self, o: Any) -> Node:
        return apply(operator.mod, o, self)

    def __pow__(self, o: Any) -> Node:
        return apply(operator.pow, self, o)

    def __rpow__(self, o: Any) -> Node:
        return apply(operator.pow, o, self)

    def __neg__(self) -> Node:
        return apply(operator.neg, self)

    def __abs__(self) -> Node:
        return apply(operator.abs, self)

    def __lt__(self, o: Any) -> Node:
        return apply(operator.lt, self, o)

    def __le__(self, o: Any) -> Node:
        return apply(operator.le, self, o)

    def __gt__(self, o: Any) -> Node:
        return apply(operator.gt, self, o)

    def __ge__(self, o: Any) -> Node:
        return apply(operator.ge, self, o)

    __hash__ = object.__hash__

    def __eq__(self, other: object) -> bool:
        return self is other

    def __repr__(self) -> str:
        name = self.node_name
        return f"<node {self.node_kind()}{f' {name!r}' if name else ''}>"


def _accepts(target: Any, key: str) -> bool:
    try:
        fn = target.__init__ if isinstance(target, type) else target
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return True
    return any(p.kind is p.VAR_KEYWORD for p in params)


def _get(value: Any, key: Any) -> Any:
    if isinstance(value, Mapping):
        return value[key]
    if isinstance(key, str):
        return getattr(value, key)
    return value[key]


# ---------------------------------------------------------------- function nodes


class FnNode(Node):
    """A node from a ``@synth.node`` function: bound arguments + the function."""

    def __init__(self, factory: NodeFactory, bound: dict[str, Any], *,
                 name: str | None = None) -> None:
        self.factory = factory
        self.bound = bound
        self.missing_input: str | None = None
        self.batched = factory.batch is not None
        self.type = factory.type
        self.tags = tuple(factory.tags)
        if name is not None:
            self.__dict__["_name"] = name

    def params(self) -> dict[str, Any]:
        """Bound arguments plus the function's defaults (so a default written out
        explicitly and a default left out are the same node)."""
        return {**self.factory.defaults, **self.bound}

    def node_kind(self) -> str:
        return self.factory.label

    def sample(self, ctx: Any) -> Any:
        if self.missing_input is not None:
            raise SynthError(f"{self.node_kind()}(...) is a transform without its input "
                             f"{self.missing_input!r}.", stage="node",
                             hint=f"Chain it: synth.chain(source, {self.node_kind()}(...)) or "
                                  f"source | {self.node_kind()}(...).")
        values = ctx.resolve_params(self)
        if self.batched:
            return self.factory.call_batch([ctx], [values])[0]
        return self.factory.call(ctx, values)

    def render_batch(self, ctxs: Sequence[Any], params_list: Sequence[dict[str, Any]]
                     ) -> list[Any]:
        return self.factory.call_batch(ctxs, params_list)

    def bind_input(self, source: Any) -> FnNode:
        """A copy with the missing input parameter set to ``source`` (chains)."""
        if self.missing_input is None:
            raise SynthError(f"{self.node_kind()}(...) already has all its inputs; it cannot be "
                             "chained after another node.", stage="node",
                             hint=f"Create it without its first argument to use it as a "
                                  f"transform: {self.node_kind()}(<other params>).")
        new = copy.copy(self)
        new.bound = {self.missing_input: as_param(source), **self.bound}
        new.missing_input = None
        new.__dict__.pop("_name", None)
        new.__dict__["_ctor"] = (self.factory, (source,), dict(self._ctor_kwargs()))
        return new

    def _ctor_kwargs(self) -> dict[str, Any]:
        ctor = self.__dict__.get("_ctor")
        return dict(ctor[2]) if ctor else {}

    def replace_params(self, **changes: Any) -> Node:
        f = self.factory
        names = f.param_names
        bad = [k for k in changes if k not in names]
        if bad:
            raise unknown(f"parameter of {self.node_kind()}", bad[0], names, stage="node")
        target, args, kwargs = self.__dict__["_ctor"]
        args = list(args)
        kwargs = dict(kwargs)
        positional = f.positional_names
        for k, v in changes.items():
            if k in positional and positional.index(k) < len(args):
                args[positional.index(k)] = v
            else:
                kwargs[k] = v
        new = target(*args, **kwargs)
        if self.node_name is not None:
            new = new.named(self.node_name)
        return new


class NodeFactory:
    """What ``@synth.node`` returns: calling it builds a node (it does not run)."""

    def __init__(self, fn: Callable[..., Any], *, type: Any = None, name: str | None = None,
                 tags: Sequence[str] = (), example: Mapping[str, Any] | None = None,
                 batch: Any = None, check_batch: bool = False) -> None:
        self.fn = fn
        self.type = type
        self.label = name or fn.__name__
        self.tags = list(tags)
        self.example = dict(example) if example is not None else None
        self.check_batch = bool(check_batch)
        if batch is True:
            self.batch: Callable[..., Any] | None = fn
            self.per_record: Callable[..., Any] | None = None
        elif callable(batch):
            self.batch, self.per_record = batch, fn
        elif batch in (None, False):
            self.batch, self.per_record = None, fn
        else:
            raise SynthError(f"batch={batch!r} must be True, False or a function.",
                             stage="node")
        sig = inspect.signature(fn)
        plist = list(sig.parameters.values())
        self.rng_param = plist[0].name if plist and plist[0].name in _RNG_NAMES else None
        self.ctx_param = next((p.name for p in plist if p.name == "ctx"), None)
        user = [p for p in plist if p.name not in (self.rng_param, self.ctx_param)]
        self.signature = sig.replace(parameters=user)
        self.param_names = [p.name for p in user if p.kind not in (p.VAR_POSITIONAL,
                                                                    p.VAR_KEYWORD)]
        self.positional_names = [p.name for p in user
                                 if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        self.accepts_name = "name" in self.param_names
        self.defaults = {p.name: p.default for p in user if p.default is not p.empty
                         and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}
        self._checked = False
        functools.update_wrapper(self, fn)
        _REGISTRY[f"{fn.__module__}:{fn.__qualname__}"] = self

    # -- building nodes ---------------------------------------------------------------
    def __call__(self, *args: Any, **kwargs: Any) -> FnNode:
        name = None
        if "name" in kwargs and not self.accepts_name:
            name = kwargs.pop("name")
        try:
            ba = self.signature.bind_partial(*args, **kwargs)
        except TypeError as e:
            bad = [k for k in kwargs if k not in self.param_names]
            if bad:
                raise unknown(f"parameter of {self.label}()", bad[0], self.param_names,
                              stage="node") from None
            raise SynthError(f"{self.label}(): {e}", stage="node") from None
        bound = {k: as_param(v) for k, v in ba.arguments.items()}
        missing = [p.name for p in self.signature.parameters.values()
                   if p.name not in bound and p.default is p.empty
                   and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
        node = FnNode(self, bound, name=name)
        if missing:
            first = next((p.name for p in self.signature.parameters.values()
                          if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                          and p.default is p.empty), None)
            if missing == [first]:
                node.missing_input = first
            else:
                raise SynthError(f"{self.label}() is missing {missing}.", stage="node",
                                 hint=f"Pass them as constants or nodes: {self.label}("
                                      + ", ".join(f"{m}=..." for m in missing) + ").")
        node.__dict__["_ctor"] = (self, args, {**kwargs, **({"name": name} if name else {})})
        return node

    # -- calling the function ------------------------------------------------------------
    def _args(self, ctx: Any, values: Mapping[str, Any], rng: Any) -> tuple[list, dict]:
        pos: list[Any] = []
        kw: dict[str, Any] = {}
        for p in self.signature.parameters.values():
            if p.name not in values:
                continue
            if p.kind is p.VAR_POSITIONAL:
                pos += list(values[p.name])
            elif p.kind is p.VAR_KEYWORD:
                kw.update(values[p.name])
            elif p.kind is p.POSITIONAL_ONLY:
                pos.append(values[p.name])
            else:
                kw[p.name] = values[p.name]
        if self.ctx_param is not None:
            kw[self.ctx_param] = ctx
        if self.rng_param is not None:
            pos.insert(0, rng)
        return pos, kw

    def call(self, ctx: Any, values: Mapping[str, Any]) -> Any:
        """Per-record call with resolved values."""
        if self.per_record is None:
            return self.call_batch([ctx], [values])[0]
        pos, kw = self._args(ctx, values, ctx.rng if self.rng_param else None)
        return self.per_record(*pos, **kw)

    def call_batch(self, ctxs: Sequence[Any], values: Sequence[Mapping[str, Any]]
                   ) -> list[Any]:
        """Batched call: arguments stacked across records (see ``stack``)."""
        if self.batch is None:
            return [self.call(c, v) for c, v in zip(ctxs, values)]
        names = list(values[0]) if values else []
        stacked = {k: stack([v[k] for v in values]) for k in names}
        rngs = [c.rng for c in ctxs] if self.rng_param else None
        pos, kw = self._args(list(ctxs), stacked, rngs)
        out = self.batch(*pos, **kw)
        result = unstack(out, len(ctxs), self.label)
        if self.check_batch and not self._checked and len(ctxs) > 1:
            self._checked = True
            single = [self.call(c, v) if self.per_record is not None else
                      self.call_batch([c], [v])[0] for c, v in zip(ctxs, values)]
            for a, b in zip(result, single):
                if not values_equal(a, b):
                    raise SynthError(f"{self.label}: batched rendering differs from per-record "
                                     "rendering.", stage="node",
                                     hint="A batched implementation must give exactly the "
                                          "per-record values (no cross-record reductions, no "
                                          "shared random draws).")
        return result

    # -- metadata -------------------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        return {"name": self.label, "path": f"{self.fn.__module__}:{self.fn.__qualname__}",
                "tags": list(self.tags), "type": type_name(self.type),
                "params": {p.name: (None if p.default is p.empty else p.default)
                           for p in self.signature.parameters.values()},
                "doc": inspect.getdoc(self.fn) or "", "example": self.example,
                "batched": self.batch is not None}

    def __reduce__(self) -> Any:
        return (_resolve_factory, (f"{self.fn.__module__}:{self.fn.__qualname__}",))

    def __repr__(self) -> str:
        return f"<node factory {self.label}>"


_REGISTRY: dict[str, NodeFactory] = {}


def _resolve_factory(path: str) -> Any:
    from topologies.experiments.spec import import_name  # noqa: F401
    from topologies.synth.yaml_spec import resolve

    return resolve(path)


def type_name(t: Any) -> str | None:
    if t is None:
        return None
    return t.__name__ if isinstance(t, type) else type(t).__name__


def registry() -> dict[str, NodeFactory]:
    """Every ``@synth.node`` factory imported so far, by import path."""
    return dict(_REGISTRY)


def node(_fn: Callable[..., Any] | None = None, *, type: Any = None, name: str | None = None,
         tags: Sequence[str] = (), example: Mapping[str, Any] | None = None,
         batch: Any = None, check_batch: bool = False) -> Any:
    """Decorator: a plain function becomes a node factory (see module docstring).

    The first parameter named ``r`` / ``rng`` receives the node's ``Rand``; a parameter
    named ``ctx`` receives the ``Ctx``. ``batch=True`` declares a vectorized function
    (arguments arrive stacked across records, the result has a leading record
    dimension); ``batch=fn`` gives a separate batched implementation.

    >>> @synth.node(tags=["toy"])
    ... def noisy(r, *, x: float, scale: float = 0.1):
    ...     return x + r.normal(0, scale)
    >>> noisy(x=synth.uniform(0, 1))       # a node
    """
    def deco(fn: Callable[..., Any]) -> NodeFactory:
        return NodeFactory(fn, type=type, name=name, tags=tags, example=example, batch=batch,
                           check_batch=check_batch)

    if _fn is not None and callable(_fn):
        return deco(_fn)
    return deco


def fn(f: Callable[..., Any], **meta: Any) -> NodeFactory:
    """Wrap an existing function as a node factory without decorating its definition."""
    return NodeFactory(f, **meta)


# ---------------------------------------------------------------- batching helpers


def stack(values: Sequence[Any]) -> Any:
    """Stack per-record argument values for a batched function: numbers -> a tensor
    ``[B]``, same-shape tensors -> ``[B, ...]``, anything else -> the list."""
    import torch

    if not values:
        return []
    first = values[0]
    if all(isinstance(v, bool) for v in values):
        return torch.tensor(list(values), dtype=torch.bool)
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        if all(isinstance(v, int) for v in values):
            return torch.tensor(list(values), dtype=torch.int64)
        return torch.tensor([float(v) for v in values], dtype=torch.float32)
    if isinstance(first, torch.Tensor) and all(
            isinstance(v, torch.Tensor) and v.shape == first.shape and v.dtype == first.dtype
            for v in values):
        return torch.stack(list(values))
    return list(values)


def unstack(out: Any, n: int, label: str) -> list[Any]:
    """Per-record values from a batched result (a list, or a tensor with a leading dim)."""
    import torch

    if isinstance(out, torch.Tensor):
        if out.dim() == 0 or out.shape[0] != n:
            raise SynthError(f"{label}: a batched function must return a tensor with a leading "
                             f"record dimension of {n}; got shape {tuple(out.shape)}.",
                             stage="node")
        return [out[i].clone() for i in range(n)]
    if isinstance(out, (list, tuple)) and len(out) == n:
        return list(out)
    raise SynthError(f"{label}: a batched function must return a list of {n} values or a tensor "
                     f"[{n}, ...]; got {type(out).__name__}.", stage="node")


def values_equal(a: Any, b: Any) -> bool:
    """Exact equality for nested values with tensors."""
    import torch

    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and \
            a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(a, b))
    if type(a).__name__ == "Raw" and type(b).__name__ == "Raw":
        return values_equal(a.value, b.value) and a.lineage == b.lineage
    data_a, data_b = getattr(a, "data", None), getattr(b, "data", None)
    if data_a is not None and type(a) is type(b) and hasattr(a, "meta"):
        return values_equal(data_a, data_b) and a.meta() == b.meta()
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(values_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(values_equal(x, y) for x, y in zip(a, b))
    return bool(a == b)


# ---------------------------------------------------------------- built-in combinators


class Const(Node):
    """A constant (``synth.const(x)``); needed only to stop container traversal."""

    def __init__(self, value: Any, *, name: str | None = None) -> None:
        self.value = value
        if name:
            self.__dict__["_name"] = name

    def params(self) -> dict[str, Any]:
        return {}

    def sample(self, ctx: Any) -> Any:
        return self.value

    def node_kind(self) -> str:
        return "const"

    def as_attr(self) -> Any:
        from topologies.synth.attributes import Const as ConstAttr

        return ConstAttr(self.value)

    def config_value(self) -> Any:
        return self.value


@factory
def const(value: Any, *, name: str | None = None) -> Const:
    """A constant node (containers inside are not traversed)."""
    return Const(value, name=name)


class Apply(Node):
    """``fn(*args, **kwargs)`` of resolved values (deterministic; ``synth.apply``)."""

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        if not callable(fn):
            raise SynthError(f"apply needs a callable; got {fn!r}.", stage="node")
        self.fn = fn
        self.args = [as_param(a) for a in args]
        self.kwargs = {k: as_param(v) for k, v in kwargs.items()}

    def params(self) -> dict[str, Any]:
        out: dict[str, Any] = {f"arg{i}": a for i, a in enumerate(self.args)}
        out.update(self.kwargs)
        return out

    def node_kind(self) -> str:
        return getattr(self.fn, "__name__", "apply").strip("<>").replace("lambda", "fn")

    def sample(self, ctx: Any) -> Any:
        args = [ctx.resolve(a) for a in self.args]
        kwargs = {k: ctx.resolve(v) for k, v in self.kwargs.items()}
        return self.fn(*args, **kwargs)

    def replace_params(self, **changes: Any) -> Node:
        args = list(self.args)
        kwargs = dict(self.kwargs)
        for k, v in changes.items():
            if k.startswith("arg") and k[3:].isdigit() and int(k[3:]) < len(args):
                args[int(k[3:])] = v
            elif k in kwargs:
                kwargs[k] = v
            else:
                raise unknown(f"parameter of {self.node_kind()}", k, list(self.params()),
                              stage="node")
        return Apply(self.fn, *args, **kwargs)


@factory
def apply(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Apply:
    """A deterministic function of sampled values: ``synth.apply(math.hypot, x, y)``."""
    return Apply(fn, *args, **kwargs)


class Record(Node):
    """A record: a dict of fields (nodes or constants); nested dicts are nested records."""

    def __init__(self, fields: Mapping[str, Any] | None = None, /, **kw: Any) -> None:
        merged = {**dict(fields or {}), **kw}
        self.fields: dict[str, Any] = {}
        for k, v in merged.items():
            if not isinstance(k, str) or not k:
                raise SynthError(f"Record field names must be strings; got {k!r}.",
                                 stage="node")
            if isinstance(v, dict) and type(v) is dict and has_nodes(v):
                v = Record(v)
            self.fields[k] = as_param(v)

    def params(self) -> dict[str, Any]:
        return dict(self.fields)

    def node_kind(self) -> str:
        return "record"

    def sample(self, ctx: Any) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in self.fields.items():
            if isinstance(v, Latent):
                ctx.sample(v)
                continue
            out[k] = ctx.resolve(v, defer=True)
        return out

    def replace_params(self, **changes: Any) -> Node:
        fields = dict(self.fields)
        for k, v in changes.items():
            if k not in fields:
                raise unknown("record field", k, list(fields), stage="node")
            if v is None:
                fields.pop(k)
            else:
                fields[k] = v
        out = Record(fields)
        if self.node_name:
            out = out.named(self.node_name)
        return out


@factory
def record(fields: Mapping[str, Any] | None = None, /, **kw: Any) -> Record:
    """A record node: ``synth.record(audio=..., f0=f0)`` (nested dicts nest records)."""
    return Record(fields, **kw)


class Latent(Node):
    """Values kept in lineage (``latent``) but not in the output record."""

    def __init__(self, **nodes: Any) -> None:
        self.nodes = {k: as_param(v) for k, v in nodes.items()}

    def params(self) -> dict[str, Any]:
        return dict(self.nodes)

    def sample(self, ctx: Any) -> dict[str, Any]:
        values = {k: ctx.resolve(v) for k, v in self.nodes.items()}
        ctx.latent(**values)
        return values


@factory
def latent(**nodes: Any) -> Latent:
    """``synth.record(audio=a, hidden=synth.latent(f0=f0))``: f0 goes to lineage only."""
    return Latent(**nodes)


class Attr(Node):
    """The record's design-sampled attribute ``name`` (``synth.attr("tone")``)."""

    def __init__(self, name: str, default: Any = None) -> None:
        self.attr_name = str(name)
        self.default = default

    def params(self) -> dict[str, Any]:
        return {}

    def node_kind(self) -> str:
        return f"attr_{self.attr_name}"

    def sample(self, ctx: Any) -> Any:
        if self.attr_name in ctx.attrs:
            return ctx.attrs[self.attr_name]
        if self.default is not None:
            return self.default
        raise unknown("attribute", self.attr_name, list(ctx.attrs), stage="node",
                      extra="Declare it in Spec(attributes={...}) or give a default.")


@factory
def attr(name: str, default: Any = None) -> Attr:
    """The record's design-sampled attribute (attributes drive slices, loops, axes)."""
    return Attr(name, default)


class Switch(Node):
    """``cases[key]`` (only the chosen branch is sampled)."""

    def __init__(self, key: Any, cases: Mapping[Any, Any], default: Any = None) -> None:
        self.key = as_param(key)
        self.cases = {k: as_param(v) for k, v in dict(cases).items()}
        self.default = as_param(default)

    def params(self) -> dict[str, Any]:
        return {"key": self.key, "cases": self.cases, "default": self.default}

    def sample(self, ctx: Any) -> Any:
        k = ctx.resolve(self.key)
        if k in self.cases:
            return ctx.resolve(self.cases[k])
        if str(k) in self.cases:
            return ctx.resolve(self.cases[str(k)])
        if self.default is not None:
            return ctx.resolve(self.default)
        raise SynthError(f"switch: no case for {k!r} and no default.", stage="node",
                         available=[str(x) for x in self.cases])


@factory
def switch(key: Any, cases: Mapping[Any, Any], default: Any = None) -> Switch:
    """``synth.switch(synth.attr("kind"), {"sine": sine(...), "saw": saw(...)})``."""
    return Switch(key, cases, default)


class When(Node):
    """``then`` if ``cond`` is true else ``else_`` (only one branch is sampled)."""

    def __init__(self, cond: Any, then: Any, else_: Any = None) -> None:
        self.cond, self.then, self.else_ = as_param(cond), as_param(then), as_param(else_)

    def params(self) -> dict[str, Any]:
        return {"cond": self.cond, "then": self.then, "else_": self.else_}

    def node_kind(self) -> str:
        return "when"

    def sample(self, ctx: Any) -> Any:
        return ctx.resolve(self.then if ctx.resolve(self.cond) else self.else_)


@factory
def when(cond: Any, then: Any = None, else_: Any = None, *, default: Any = None) -> Any:
    """Conditional node ``synth.when(x > 0.5, a, b)``.

    With a string first argument and a mapping (``when("label", {...})``) it is the
    conditional *attribute* of ``synth.attributes`` instead.
    """
    if isinstance(cond, str) and (isinstance(then, Mapping) or (callable(then) and
                                                                 not is_node(then))):
        from topologies.synth.attributes import when as attr_when

        return attr_when(cond, then, default=default)
    return When(cond, then, else_)


class Repeat(Node):
    """A list of ``n`` independent draws of ``node`` (draw ``k`` keeps its value as ``n``
    grows; ``n`` may be a node)."""

    def __init__(self, node: Any, n: Any) -> None:
        self.node = as_param(node)
        self.n = as_param(n)

    def params(self) -> dict[str, Any]:
        return {"node": self.node, "n": self.n}

    def sample(self, ctx: Any) -> list[Any]:
        n = int(ctx.resolve(self.n))
        if not isinstance(self.node, Node):
            return [self.node] * n
        return [ctx.sample_scoped(self.node, f"[{k}]") for k in range(n)]


@factory
def repeat(node: Any, n: Any) -> Repeat:
    """``synth.repeat(note(...), synth.intuniform(2, 8))``: independent draws as a list."""
    return Repeat(node, n)


def iid(node: Node) -> Node:
    """An independent copy of ``node`` (same parameters, its own randomness)."""
    if not isinstance(node, Node):
        raise SynthError("iid() needs a node.", stage="node")
    return node.fresh()


class Where(Node):
    """Rejection sampling: draws of ``node`` until ``predicate(value)`` holds."""

    def __init__(self, node: Any, predicate: Callable[[Any], bool], max_tries: int = 100
                 ) -> None:
        if not callable(predicate):
            raise SynthError("where() needs predicate=fn(value) -> bool.", stage="node")
        self.node = as_param(node)
        self.predicate = predicate
        self.max_tries = int(max_tries)

    def params(self) -> dict[str, Any]:
        return {"node": self.node}

    def sample(self, ctx: Any) -> Any:
        from topologies.synth.ops.base import Reject

        for k in range(self.max_tries):
            value = ctx.sample_scoped(self.node, f"~{k}") if isinstance(self.node, Node) \
                else self.node
            if self.predicate(value):
                ctx.stat("where.tries", k + 1)
                return value
        ctx.stat("where.rejected", 1)
        raise Reject(f"where: no draw of {ctx.path} passed the predicate in "
                     f"{self.max_tries} tries")


@factory
def where(node: Any, predicate: Callable[[Any], bool], max_tries: int = 100) -> Where:
    """``synth.where(synth.normal(0, 1), lambda x: x > 0)`` (tries are counted in stats)."""
    return Where(node, predicate, max_tries)


class Chain(Node):
    """``source`` then transforms in order (``synth.chain(src, t1, t2)`` / ``src | t1``)."""

    def __init__(self, source: Any, *transforms: Any) -> None:
        if isinstance(source, Chain):
            transforms = (*source.transforms, *transforms)
            source = source.source
        self.source = as_param(source)
        self.transforms = [t() if isinstance(t, NodeFactory) else t for t in transforms]
        steps: list[Node] = []
        cur: Any = self.source
        for t in self.transforms:
            bind = getattr(t, "bind_input", None)
            if not callable(bind):
                raise SynthError(f"chain: {t!r} is not a transform.", stage="node",
                                 hint="Transforms are nodes created without their input, "
                                      "e.g. synth.audio.reverb(decay=0.5).")
            cur = bind(cur)
            steps.append(cur)
        self.steps = steps
        last = steps[-1] if steps else source
        self.type = getattr(last, "type", None)
        self.batched = False

    def params(self) -> dict[str, Any]:
        return {"source": self.source, "steps": list(self.steps)}

    def node_kind(self) -> str:
        return "chain"

    def sample(self, ctx: Any) -> Any:
        return ctx.resolve(self.steps[-1] if self.steps else self.source, defer=True)

    def replace_params(self, **changes: Any) -> Node:
        source = changes.pop("source", self.source)
        new_steps = list(changes.pop("steps", self.steps))
        if changes:
            raise unknown("parameter of chain", next(iter(changes)), ["source", "steps"],
                          stage="overrides")
        transforms = list(self.transforms)
        for k, (old, new) in enumerate(zip(self.steps, new_steps)):
            if new is old:
                continue
            ctor = getattr(new, "__dict__", {}).get("_ctor")
            if not isinstance(new, FnNode) or ctor is None:
                raise SynthError("chain: only @synth.node transforms can be overridden by path.",
                                 stage="overrides")
            transforms[k] = ctor[0](**ctor[2])
        out = chain(source, *transforms)
        return out.named(self.node_name) if self.node_name else out


@factory
def chain(source: Any, *transforms: Any) -> Chain:
    """Apply transforms in order: ``synth.chain(osc, synth.audio.lowpass(800), reverb)``."""
    return Chain(source, *transforms)


class Mixture(Node):
    """One component per record, chosen by weight: ``synth.mixture([(a, 0.7), (b, 0.3)])``."""

    def __init__(self, parts: Any) -> None:
        items = list(parts.items()) if isinstance(parts, Mapping) else list(parts)
        if not items:
            raise SynthError("mixture needs at least one component.", stage="node")
        self.parts: list[Any] = []
        self.weights: list[float] = []
        self.names: list[str] = []
        for item in items:
            node_, w = item if isinstance(item, tuple) and len(item) == 2 else (item, 1.0)
            node_ = as_generator(node_) if not isinstance(node_, Node) else node_
            self.parts.append(node_)
            self.weights.append(float(w))
            base = node_.node_name or getattr(node_, "name", None)
            base = base if isinstance(base, str) and base else node_.node_kind()
            nm = base if base not in self.names else f"{base}_{len(self.names)}"
            self.names.append(nm)
        if any(w < 0 for w in self.weights) or sum(self.weights) <= 0:
            raise SynthError("mixture weights must be >= 0 and not all zero.", stage="node")
        self.produces = next((getattr(p, "produces", None) for p in self.parts
                              if getattr(p, "produces", None) is not None), None)

    def params(self) -> dict[str, Any]:
        return {"parts": list(self.parts)}

    def sample(self, ctx: Any) -> Any:
        i = ctx.rng.choices(range(len(self.parts)), weights=self.weights, k=1)[0]
        ctx.log(generator=self.names[i])
        return ctx.resolve(self.parts[i], defer=True)

    def with_weights(self, weights: Mapping[str, float]) -> Mixture:
        bad = [k for k in weights if k not in self.names]
        if bad:
            raise unknown("mixture component", bad[0], self.names, stage="generator")
        return Mixture([(p, float(weights.get(n, w)))
                        for p, n, w in zip(self.parts, self.names, self.weights)])

    def replace_params(self, **changes: Any) -> Node:
        parts = list(changes.pop("parts", self.parts))
        if len(parts) != len(self.parts):
            raise SynthError("mixture: replace parts one by one (parts[i]).", stage="overrides")
        weights = {k: float(v) for k, v in changes.items()}
        bad = [k for k in weights if k not in self.names]
        if bad:
            raise unknown("mixture component", bad[0], self.names, stage="overrides")
        return Mixture([(p, weights.get(n, w))
                        for p, n, w in zip(parts, self.names, self.weights)])


@factory
def mixture(parts: Any) -> Mixture:
    """A weighted choice of nodes (or generators) per record."""
    return Mixture(parts)


# ---------------------------------------------------------------- generators


class GeneratorFn(Node):
    """A record generator from a function (``@synth.generator``).

    ``fn(r, *, label, tone)``: the first parameter (``r`` / ``rng`` / ``ctx``) gets the
    record's ``Rand`` (or ``Ctx``); keyword parameters named like attributes receive
    the record's attribute values; other parameters keep their defaults unless set
    with ``gen.with_(noise=0.2)`` (constants or nodes). Return a record dict, a list of
    records (fan-out), ``None`` (no record) or yield records.
    """

    def __init__(self, fn: Callable[..., Any], *, schema: Any = None, name: str | None = None,
                 bound: Mapping[str, Any] | None = None) -> None:
        if not callable(fn):
            raise SynthError(f"generator needs a callable; got {fn!r}.", stage="generator")
        self.fn = fn
        self.produces = schema
        self.name = name or getattr(fn, "__name__", "generator")
        self.bound = {k: as_param(v) for k, v in dict(bound or {}).items()}
        try:
            plist = list(inspect.signature(fn).parameters.values())
        except (TypeError, ValueError):
            plist = []
        self._first = plist[0].name if plist and plist[0].kind in (
            plist[0].POSITIONAL_ONLY, plist[0].POSITIONAL_OR_KEYWORD) else None
        rest = plist[1:] if self._first is not None else plist
        self._names = {p.name for p in rest if p.kind in (p.POSITIONAL_OR_KEYWORD,
                                                          p.KEYWORD_ONLY)}
        self._var_kw = any(p.kind is p.VAR_KEYWORD for p in rest)
        self._attrs_dict = self._names == {"attrs"} and not self._var_kw
        self._required = [p.name for p in rest if p.default is p.empty and p.kind in (
            p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
        bad = [k for k in self.bound if k not in self._names and not self._var_kw]
        if bad:
            raise unknown(f"parameter of {self.name}()", bad[0], sorted(self._names),
                          stage="generator")
        self.__wrapped__ = fn
        self.__doc__ = getattr(fn, "__doc__", None)

    def params(self) -> dict[str, Any]:
        return dict(self.bound)

    def node_kind(self) -> str:
        return self.name

    def sample(self, ctx: Any) -> Any:
        first: Any = ()
        if self._first is not None:
            first = (ctx if self._first == "ctx" else ctx.rng,)
        if self._attrs_dict:
            return self.fn(*first, dict(ctx.attrs))
        kwargs: dict[str, Any] = {}
        for k in self._names:
            if k in self.bound:
                kwargs[k] = ctx.resolve(self.bound[k])
            elif k in ctx.attrs:
                kwargs[k] = ctx.attrs[k]
        if self._var_kw:
            for k, v in ctx.attrs.items():
                kwargs.setdefault(k, v)
            for k, v in self.bound.items():
                kwargs.setdefault(k, ctx.resolve(v))
        missing = [k for k in self._required if k not in kwargs]
        if missing:
            raise SynthError(f"Generator {self.name} needs attributes {missing} that the "
                             "spec does not define.", stage="generator",
                             hint=f"Add them to attributes={{...}} (e.g. {missing[0]!r}: "
                                  "[...]), give the parameters defaults, or bind them with "
                                  f"{self.name}.with_({missing[0]}=...).")
        return self.fn(*first, **kwargs)

    def replace_params(self, **changes: Any) -> Node:
        return GeneratorFn(self.fn, schema=self.produces, name=self.name,
                           bound={**self.bound, **changes})

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    def __reduce_ex__(self, protocol: Any) -> Any:
        # The module attribute named like the function is this object (the decorator
        # replaced the function), so pickle by reference to it (worker processes).
        qual = f"{getattr(self.fn, '__module__', '')}:{getattr(self.fn, '__qualname__', '')}"
        try:
            base = _resolve_factory(qual)
        except Exception:  # noqa: BLE001
            base = None
        if base is self:
            return (_resolve_factory, (qual,))
        if isinstance(base, GeneratorFn) and base.fn is self.fn:
            return (_rebind_generator, (qual, dict(self.bound)))
        return super().__reduce_ex__(protocol)

    def __repr__(self) -> str:
        return f"generator({self.name})"


class Program(GeneratorFn):
    """Compatibility adapter for plain functions used as attributes or generators.

    Generator programs accept the historical positional ``ctx``/``rng``/``attrs``
    forms and are represented as core nodes so they compose with the node evaluator.
    """

    def __init__(self, fn: Callable[..., Any], *, schema: Any = None,
                 name: str | None = None) -> None:
        super().__init__(fn, schema=schema, name=name)

    def sample(self, ctx: Any) -> Any:
        try:
            params = [p for p in inspect.signature(self.fn).parameters.values()
                      if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
        except (TypeError, ValueError):
            params = []
        first = params[0].name if params else None
        if first == "ctx":
            return self.fn(ctx)
        if first in ("rng", "random", "r"):
            return self.fn(ctx.rng, dict(ctx.attrs)) if len(params) >= 2 and \
                params[1].default is params[1].empty else self.fn(ctx.rng)
        if first in ("attrs", "a"):
            return self.fn(dict(ctx.attrs), ctx.rng) if len(params) >= 2 and \
                params[1].default is params[1].empty else self.fn(dict(ctx.attrs))
        if not params:
            return self.fn()
        return self.fn(ctx)

    def as_attr(self) -> Any:
        from topologies.synth.attributes import Derived

        return Derived(self.fn)

    def config(self) -> dict[str, Any]:
        from topologies.synth.fingerprint import callable_fingerprint

        return {"type": "program", "fn": callable_fingerprint(self.fn),
                "schema": self.produces.to_config() if self.produces is not None else None}

    def describe(self) -> str:
        return f"program({self.name})"


def from_program(fn: Callable[..., Any], *, name: str | None = None,
                 schema: Any = None) -> Program:
    """Adapt a plain callable to a core node (also usable with ``as_attr()``)."""
    return Program(fn, name=name, schema=schema)


def correlated_values(rng: Any, means: Mapping[str, float], stds: Mapping[str, float],
                      corr: Mapping[tuple[str, str], float]) -> dict[str, float]:
    """Sample jointly normal values from pairwise correlations using Cholesky."""
    names = list(means)
    n = len(names)
    cov = [[1.0 if i == j else float(corr.get((a, b), corr.get((b, a), 0.0)))
            for j, b in enumerate(names)] for i, a in enumerate(names)]
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            residual = cov[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            lower[i][j] = max(residual, 1e-12) ** 0.5 if i == j else residual / lower[j][j]
    draws = [rng.gauss(0, 1) for _ in names]
    return {key: means[key] + stds.get(key, 1.0) * sum(
        lower[i][k] * draws[k] for k in range(i + 1)) for i, key in enumerate(names)}


def _rebind_generator(qual: str, bound: dict[str, Any]) -> Any:
    return _resolve_factory(qual).replace_params(**bound)


def generator(_fn: Callable[..., Any] | None = None, *, schema: Any = None,
              name: str | None = None) -> Any:
    """Decorator: ``fn(r, **attrs) -> record`` becomes a generator node.

    >>> @synth.generator(schema=synth.Schema(text=synth.Text(), label=synth.Label(["a", "b"])))
    ... def toy(r, *, label):
    ...     return {"text": f"example of {label} #{r.randint(0, 9)}", "label": label}
    """
    def deco(f: Callable[..., Any]) -> GeneratorFn:
        return GeneratorFn(f, schema=schema, name=name)

    if _fn is not None and callable(_fn):
        return deco(_fn)
    return deco


def as_generator(obj: Any) -> Node:
    """The root node of a generator: a node, a ``{field: node}`` dict, a plain function
    ``fn(r, **attrs) -> record``, a ``{generator: weight}`` mixture or an object with
    ``as_generator()``."""
    if isinstance(obj, Node):
        return obj
    to_gen = getattr(obj, "as_generator", None)
    if callable(to_gen) and not isinstance(obj, type):
        return as_generator(to_gen())
    if isinstance(obj, Mapping):
        if obj and all(isinstance(k, Node) or callable(k) for k in obj) and \
                all(isinstance(v, (int, float)) for v in obj.values()):
            return Mixture(obj)
        return Record(obj)
    if isinstance(obj, NodeFactory):
        return obj()
    if callable(obj) and not isinstance(obj, type):
        return GeneratorFn(obj)
    raise SynthError(f"Cannot use {obj!r} as a generator.", stage="generator",
                     hint="Pass synth.record(...), a {field: node} dict, a node, or a "
                          "function fn(r, **attrs) -> record (@synth.generator).")


def iter_nodes(root: Any) -> list[Node]:
    """Every node reachable from ``root`` (each once, declaration order)."""
    seen: dict[int, Node] = {}
    stack_: list[Any] = [root]
    while stack_:
        x = stack_.pop()
        if not isinstance(x, Node) or id(x) in seen:
            continue
        seen[id(x)] = x
        kids = [c for k, v in x.params().items() for _, c in iter_children(v, k)]
        stack_.extend(reversed(kids))
    return list(seen.values())


__all__ = [
    "Apply", "Attr", "Chain", "Const", "FnNode", "GeneratorFn", "Latent", "Mixture",
    "Node", "NodeFactory", "Program", "Record", "Repeat", "Switch", "When", "Where",
    "apply", "as_generator", "as_param", "attr", "chain", "const", "correlated_values",
    "fn", "from_program", "generator", "has_nodes", "iid", "is_node", "iter_children",
    "iter_nodes", "latent", "mixture", "node", "record", "registry", "repeat", "stack",
    "switch", "unstack", "values_equal", "when", "where",
]
