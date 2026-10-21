"""Evaluating node graphs: paths, seeding, sharing, scopes and batched rendering.

**Paths.** ``bind(root, seed)`` gives every node a path, in priority order:
explicit ``name=`` / ``.named()`` > record field path (``audio``, ``meta.lang``)
> structural position (``audio/chirp.f1``: parameter ``f1`` of the ``chirp`` node at
``audio``). A shared node takes the path of its first occurrence (declaration
order) within the best priority.

**Seeding.** The ``Rand`` of a node for record ``i`` is seeded from
``(spec seed, i, path)``; adding, removing or reordering *other* fields never
changes a field's values, and ``repeat`` draw ``k`` lives at ``path[k]``.

**Sharing.** Within one record a node object is sampled once (memoized by
identity and scope). ``repeat`` / ``where`` open *scopes*: nodes whose path lies
under the scoped node are re-sampled per draw (``path[k]`` / ``path~k``); nodes
bound elsewhere stay shared.

**Batching.** ``Evaluator.evaluate_many`` samples records one by one, but a
node with ``batched = True`` returns a *pending* value; when every record of the
chunk is sampled, pending values are rendered per node with one
``render_batch`` call (in dependency order) and substituted. A consumer that
needs a concrete value (any non-batched node) forces it per record, so results
never depend on batching.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from topologies.synth.core import (
    Attr,
    Chain,
    Const,
    GeneratorFn,
    Latent,
    Mixture,
    Node,
    Record,
    has_nodes,
    iter_children,
)
from topologies.synth.errors import SynthError
from topologies.synth.rng import Rand, path_base, record_seed

_BOUNDARY = "/.[~#"


# ---------------------------------------------------------------- binding


class Binding:
    """Paths and seed bases of every node reachable from ``root`` (see module docstring)."""

    def __init__(self, root: Node, seed: int) -> None:
        self.root = root
        self.seed = int(seed)
        self.paths: dict[int, str] = {}
        self.kinds: dict[int, str] = {}
        self.nodes: dict[int, Node] = {}
        self._assign()
        self.bases = {k: path_base(self.seed, p) for k, p in self.paths.items()}
        self.dyn_params: dict[int, list[str]] = {
            k: [name for name, v in n.params().items() if has_nodes(v)]
            for k, n in self.nodes.items()}

    def _set(self, node: Node, path: str, kind: str) -> bool:
        if id(node) in self.paths:
            return False
        self.paths[id(node)] = path
        self.kinds[id(node)] = kind
        self.nodes[id(node)] = node
        return True

    def _assign(self) -> None:
        from topologies.synth.core import iter_nodes

        everything = iter_nodes(self.root)
        by_name: dict[str, Node] = {}
        for n in everything:
            nm = n.node_name
            if nm:
                if nm in by_name and by_name[nm] is not n:
                    raise SynthError(f"Two different nodes are named {nm!r}.", stage="node",
                                     hint="Names are paths and must be unique; rename one "
                                          "(or reuse the same node object to share it).")
                by_name[nm] = n
                self._set(n, nm, "name")
        root_path = self.root.node_name or ""
        self._set(self.root, root_path, "root")
        self._fields(self.root, self.paths[id(self.root)])
        self._structure(self.root, set())
        seen: dict[str, int] = {}
        for k, p in self.paths.items():
            if p in seen and seen[p] != k and p:
                raise SynthError(f"Two different nodes share the path {p!r}.", stage="node",
                                 hint="Give one of them another name (.named(...)).")
            seen[p] = k

    def _fields(self, rec: Node, path: str) -> None:
        """Record field paths (priority 1), recursively through nested records."""
        if not isinstance(rec, Record):
            return
        for fname, value in rec.fields.items():
            base = f"{path}.{fname}" if path else fname
            for label, child in iter_children(value, base):
                if isinstance(child, Latent):
                    self._set(child, label, "field")
                    for k, v in child.nodes.items():
                        kp = f"{path}.{k}" if path else k
                        for lab2, c2 in iter_children(v, kp):
                            self._set(c2, lab2, "latent")
                    continue
                self._set(child, label, "field")
                if isinstance(child, Record):
                    self._fields(child, self.paths[id(child)])

    def _structure(self, node: Node, visited: set[int]) -> None:
        """Structural paths (priority 2): pre-order DFS in declaration order."""
        if id(node) in visited:
            return
        visited.add(id(node))
        path = self.paths[id(node)]
        for pname, value in node.params().items():
            for label, child in iter_children(value, pname):
                if isinstance(node, Record):
                    cp = f"{path}.{label}" if path else label
                else:
                    head = f"{path}/" if path else ""
                    cp = f"{head}{node.node_kind()}.{label}"
                self._set(child, cp, "param")
                self._structure(child, visited)

    def path(self, node: Node) -> str | None:
        return self.paths.get(id(node))

    def describe(self) -> list[tuple[str, Node]]:
        return sorted(((p, self.nodes[k]) for k, p in self.paths.items()), key=lambda t: t[0])


# ---------------------------------------------------------------- pending values


class Pending:
    """A batched node's value for one record, rendered later (see module docstring)."""

    __slots__ = ("ctx", "done", "error", "node", "params", "state", "value")

    def __init__(self, node: Node, ctx: Ctx, params: dict[str, Any]) -> None:
        self.node, self.ctx, self.params = node, ctx, params
        self.state = ctx.state
        self.done = False
        self.value: Any = None
        self.error: BaseException | None = None


def _pendings_in(value: Any, out: list[Pending]) -> list[Pending]:
    if isinstance(value, Pending):
        if not value.done:
            out.append(value)
    elif isinstance(value, Mapping):
        for v in value.values():
            _pendings_in(v, out)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _pendings_in(v, out)
    return out


def substitute(value: Any) -> Any:
    """``value`` with every (rendered) pending replaced by its value."""
    if isinstance(value, Pending):
        if value.error is not None:
            raise value.error
        return substitute(value.value) if value.done else value
    if isinstance(value, dict):
        if not _has_pending(value):
            return value
        return {k: substitute(v) for k, v in value.items()}
    if isinstance(value, list):
        if not _has_pending(value):
            return value
        return [substitute(v) for v in value]
    if isinstance(value, tuple) and type(value) is tuple:
        if not _has_pending(value):
            return value
        return tuple(substitute(v) for v in value)
    return value


def _has_pending(value: Any) -> bool:
    if isinstance(value, Pending):
        return True
    if isinstance(value, Mapping):
        return any(_has_pending(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_pending(v) for v in value)
    return False


# ---------------------------------------------------------------- per-record state


class RecordState:
    """Everything one record's evaluation shares: memo, latents, logs, scopes, pins."""

    def __init__(self, ev: Evaluator, index: int, attrs: Mapping[str, Any],
                 pins: Mapping[str, Any] | None = None, *, defer: bool = False) -> None:
        self.ev = ev
        self.index = int(index)
        self.attrs = dict(attrs)
        self.pins = dict(pins or {})
        self.memo: dict[Any, Any] = {}
        # Dynamic nodes can be created and discarded while a record is sampled.
        # Keep their identities alive for the lifetime of this record: memo and
        # path bookkeeping are keyed by id(node), which Python may otherwise reuse.
        self.memo_nodes: dict[int, Node] = {}
        self.latents: dict[str, Any] = {}
        self.logs: dict[str, Any] = {}
        self.stats: dict[str, float] = defaultdict(float)
        self.scopes: list[tuple[str, str]] = []
        self.dyn: dict[str, int] = defaultdict(int)
        self.dyn_paths: dict[int, str] = {}
        self.defer = defer
        self.pending: list[Pending] = []

    # -- paths ----------------------------------------------------------------------
    def bound_path(self, node: Node, parent: Ctx | None) -> str:
        p = self.ev.binding.paths.get(id(node))
        if p is not None:
            return p
        p = self.dyn_paths.get(id(node))
        if p is None:
            head = parent.path if parent is not None else ""
            key = f"{head}/~{node.node_kind()}"
            k = self.dyn[key]
            self.dyn[key] += 1
            p = f"{key}#{k}"
            self.dyn_paths[id(node)] = p
        return p

    def scoped(self, bound: str) -> tuple[str, tuple[tuple[str, str], ...]]:
        """The effective path of a node bound at ``bound`` and the scopes that apply."""
        applied = tuple((pre, suf) for pre, suf in self.scopes
                        if bound == pre or (bound.startswith(pre) and
                                            bound[len(pre):len(pre) + 1] in _BOUNDARY))
        if not applied:
            return bound, ()
        cuts = sorted({len(pre) for pre, _ in applied}, reverse=True)
        out = bound
        for cut in cuts:
            sufs = "".join(suf for pre, suf in applied if len(pre) == cut)
            out = out[:cut] + sufs + out[cut:]
        return out, applied


class Ctx:
    """What a node sees for one record (``ctx.rng``, ``ctx.index``, ``ctx.attrs``,
    ``ctx.path``, ``ctx.sample(node)``, ``ctx.resolve(value)``, ``ctx.latent(**v)``,
    ``ctx.log(**info)``)."""

    __slots__ = ("_rng", "_seed", "node", "path", "state", "used_rng")

    def __init__(self, state: RecordState, node: Node | None, path: str, seed: int) -> None:
        self.state = state
        self.node = node
        self.path = path
        self._seed = seed
        self._rng: Rand | None = None
        self.used_rng = False

    # -- identity ---------------------------------------------------------------------
    @property
    def rng(self) -> Rand:
        """This node's ``Rand`` for this record (``rng.torch`` for torch draws)."""
        self.used_rng = True
        if self._rng is None:
            self._rng = Rand(self._seed)
        return self._rng

    @property
    def index(self) -> int:
        return self.state.index

    record_index = index

    @property
    def attrs(self) -> dict[str, Any]:
        return self.state.attrs

    @property
    def spec_seed(self) -> int:
        return self.state.ev.seed

    @property
    def env(self) -> Any:
        return self.state.ev.env

    runtime = env

    @property
    def dir(self) -> Any:
        return getattr(self.state.ev.env, "dir", None)

    # -- sampling -----------------------------------------------------------------------
    def sample(self, node: Any, *, defer: bool = False) -> Any:
        """The value of ``node`` for this record (memoized: shared nodes agree)."""
        if not isinstance(node, Node):
            return self.resolve(node, defer=defer)
        value = self.state.ev.sample_node(self.state, node, self)
        if isinstance(value, Pending) and not defer:
            value = self.state.ev.force(value)
        return value

    def resolve(self, value: Any, *, defer: bool = False) -> Any:
        """Constants as they are; nodes (also inside lists / tuples / dicts) sampled."""
        if isinstance(value, Node):
            return self.sample(value, defer=defer)
        if isinstance(value, list):
            return [self.resolve(v, defer=defer) for v in value]
        if isinstance(value, tuple) and type(value) is tuple:
            return tuple(self.resolve(v, defer=defer) for v in value)
        if isinstance(value, dict) and type(value) is dict:
            return {k: self.resolve(v, defer=defer) for k, v in value.items()}
        return value

    def resolve_params(self, node: Node, *, defer: bool = False) -> dict[str, Any]:
        """``node.params()`` resolved for this record."""
        params = node.params()
        dyn = self.state.ev.binding.dyn_params.get(id(node))
        if dyn is None:
            dyn = [k for k, v in params.items() if has_nodes(v)]
        out = dict(params)
        for k in dyn:
            out[k] = self.resolve(params[k], defer=defer)
        return out

    def sample_scoped(self, node: Node, suffix: str, *, defer: bool = False) -> Any:
        """Sample ``node`` (and the nodes under its path) as an independent draw ``suffix``."""
        st = self.state
        bound = st.bound_path(node, self)
        st.scopes.append((bound, suffix))
        try:
            return self.sample(node, defer=defer)
        finally:
            st.scopes.pop()

    def child(self, name: str) -> Ctx:
        """A sub-context with its own stable randomness (``path/name``)."""
        p = f"{self.path}/{name}"
        return Ctx(self.state, self.node, p, record_seed(path_base(self.state.ev.seed, p),
                                                         self.state.index))

    # -- records of what happened ------------------------------------------------------
    def latent(self, **values: Any) -> None:
        """Keep values in the record's lineage (``latent``) without adding fields."""
        for k, v in values.items():
            self.state.latents[k] = self.state.ev.force_all(v)

    def log(self, **info: Any) -> None:
        """Free-form values stored in the record's lineage (``logs``)."""
        self.state.logs.update(info)

    def stat(self, key: str, n: float = 1) -> None:
        self.state.stats[key] += n

    def verify(self, record: Mapping[str, Any], verifier: Any) -> Any:
        """Run a verifier (callable or ``ops.verify`` op) on a record -> ``Verdict``."""
        from topologies.synth.verdict import as_verdict

        check = getattr(verifier, "check_sync", None)
        if callable(check):
            return check(dict(record), self)
        return as_verdict(verifier(dict(record), self))

    def __repr__(self) -> str:
        return f"Ctx(index={self.index}, path={self.path!r})"


# ---------------------------------------------------------------- results


@dataclass
class Sample:
    """One record's evaluation: the value plus what lineage keeps."""

    index: int
    value: Any = None
    latents: dict[str, Any] = field(default_factory=dict)
    logs: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, float] = field(default_factory=dict)
    error: BaseException | None = None
    attrs: dict[str, Any] = field(default_factory=dict)


#: Node kinds whose values are not recorded as automatic latents.
_NO_LATENT = (Record, Const, Attr, Latent, Chain, Mixture, GeneratorFn)


def _small(v: Any) -> bool:
    if v is None or isinstance(v, (bool, int, str)):
        return not isinstance(v, str) or len(v) <= 200
    if isinstance(v, float):
        return math.isfinite(v)
    if isinstance(v, (list, tuple)) and len(v) <= 16:
        return all(isinstance(x, (bool, int, str)) or (isinstance(x, float) and math.isfinite(x))
                   for x in v)
    return False


class Evaluator:
    """Samples records of one generator graph (see module docstring)."""

    def __init__(self, root: Node, *, seed: int = 0, env: Any = None,
                 auto_latents: bool = True) -> None:
        self.root = root
        self.seed = int(seed)
        self.env = env
        self.binding = Binding(root, self.seed)
        self.auto_latents = bool(auto_latents)
        self.timings: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)

    # -- core ---------------------------------------------------------------------------
    def sample_node(self, state: RecordState, node: Node, parent: Ctx | None) -> Any:
        state.memo_nodes[id(node)] = node
        bound = state.bound_path(node, parent)
        path, applied = state.scoped(bound) if state.scopes else (bound, ())
        key = (id(node), applied) if applied else id(node)
        if key in state.memo:
            return state.memo[key]
        if path in state.pins:
            value = state.pins[path]
            state.memo[key] = value
            return value
        base = self.binding.bases.get(id(node))
        if base is None or applied:
            base = path_base(self.seed, path)
        ctx = Ctx(state, node, path, record_seed(base, state.index))
        t0 = time.perf_counter()
        if node.batched and state.defer:
            params = ctx.resolve_params(node, defer=True)
            value: Any = Pending(node, ctx, params)
            state.pending.append(value)
        else:
            value = node.sample(ctx)
        self.timings[bound] += time.perf_counter() - t0
        self.calls[bound] += 1
        state.memo[key] = value
        if self.auto_latents and ctx.used_rng and not isinstance(node, _NO_LATENT) and \
                self.binding.kinds.get(id(node)) != "field" and _small(value):
            state.latents.setdefault(path, list(value) if isinstance(value, tuple) else value)
        return value

    def force(self, p: Pending) -> Any:
        """Render one pending value now (per record)."""
        if p.done:
            if p.error is not None:
                raise p.error
            return p.value
        params = {k: self.force_all(v) for k, v in p.params.items()}
        t0 = time.perf_counter()
        try:
            p.value = p.node.render_batch([p.ctx], [params])[0]
        except BaseException as e:
            p.error, p.done = e, True
            raise
        finally:
            path = self.binding.paths.get(id(p.node), p.ctx.path)
            self.timings[path] += time.perf_counter() - t0
        p.done = True
        return p.value

    def force_all(self, value: Any) -> Any:
        for p in _pendings_in(value, []):
            self.force(p)
        return substitute(value)

    def flush(self, pendings: Sequence[Pending]) -> None:
        """Render every pending value, grouped per node, in dependency order."""
        todo = [p for p in pendings if not p.done]
        while todo:
            ready = [p for p in todo if not _pendings_in(p.params, [])]
            if not ready:  # cannot happen in a DAG; render the rest one by one
                for p in todo:
                    self._force_safe(p)
                break
            groups: dict[int, list[Pending]] = {}
            for p in ready:
                groups.setdefault(id(p.node), []).append(p)
            for group in groups.values():
                node_ = group[0].node
                params = [{k: substitute(v) for k, v in p.params.items()} for p in group]
                t0 = time.perf_counter()
                try:
                    out = node_.render_batch([p.ctx for p in group], params)
                    if len(out) != len(group):
                        raise SynthError(f"{node_.node_kind()}.render_batch returned {len(out)} "
                                         f"values for {len(group)} records.", stage="node")
                    for p, v in zip(group, out):
                        p.value, p.done = v, True
                except SynthError:
                    raise
                except Exception:  # noqa: BLE001 - isolate the failing records
                    for p in group:
                        self._force_safe(p)
                path = self.binding.paths.get(id(node_), group[0].ctx.path)
                self.timings[path] += time.perf_counter() - t0
                self.calls[path] += len(group)
            todo = [p for p in todo if not p.done]

    def _force_safe(self, p: Pending) -> None:
        try:
            self.force(p)
        except Exception as e:  # noqa: BLE001
            p.error, p.done = e, True

    # -- public API ----------------------------------------------------------------------
    def _start(self, index: int, attrs: Mapping[str, Any], pins: Mapping[str, Any] | None,
               defer: bool) -> tuple[RecordState, Sample]:
        state = RecordState(self, index, attrs, pins, defer=defer)
        out = Sample(index=index, attrs=state.attrs)
        root_path = self.binding.paths[id(self.root)]
        ctx = Ctx(state, None, root_path, 0)
        try:
            out.value = ctx.sample(self.root, defer=defer)
        except Exception as e:  # noqa: BLE001 - reported per record
            out.error = e
        return state, out

    @staticmethod
    def _finish(state: RecordState, out: Sample) -> Sample:
        if out.error is None:
            try:
                out.value = substitute(out.value)
                out.latents = {k: substitute(v) for k, v in state.latents.items()}
            except Exception as e:  # noqa: BLE001
                out.error = e
        else:
            out.latents = dict(state.latents)
        out.logs = dict(state.logs)
        out.stats = dict(state.stats)
        return out

    def evaluate(self, index: int, attrs: Mapping[str, Any] | None = None, *,
                 pins: Mapping[str, Any] | None = None) -> Sample:
        """Record ``index`` alone (no batching)."""
        state, out = self._start(index, attrs or {}, pins, defer=False)
        return self._finish(state, out)

    def evaluate_many(self, indices: Sequence[int], attrs: Sequence[Mapping[str, Any]], *,
                      batch: bool = True,
                      pins: Sequence[Mapping[str, Any] | None] | None = None) -> list[Sample]:
        """Several records; batched nodes render once per node when ``batch``."""
        if not batch:
            return [self.evaluate(i, a, pins=p) for i, a, p in
                    zip(indices, attrs, pins or [None] * len(indices))]
        started = [self._start(i, a, p, defer=True) for i, a, p in
                   zip(indices, attrs, pins or [None] * len(indices))]
        self.flush([p for st, _ in started for p in st.pending])
        return [self._finish(st, out) for st, out in started]

    def take_timings(self) -> tuple[dict[str, float], dict[str, int]]:
        t, c = dict(self.timings), dict(self.calls)
        self.timings.clear()
        self.calls.clear()
        return t, c


# ---------------------------------------------------------------- fingerprints


def _canon(value: Any, refs: dict[int, int], order: list[Node]) -> Any:
    import hashlib

    from topologies.synth.fingerprint import fp_canonical

    if isinstance(value, Node):
        if id(value) not in refs:
            refs[id(value)] = len(order)
            order.append(value)
        return {"node": refs[id(value)]}
    if isinstance(value, (list, tuple)) and type(value) in (list, tuple):
        return [_canon(v, refs, order) for v in value]
    if isinstance(value, dict) and type(value) is dict:
        return {str(k): _canon(v, refs, order) for k, v in sorted(value.items(),
                                                                  key=lambda kv: str(kv[0]))}
    try:
        import torch

        if isinstance(value, torch.Tensor):
            t = value.detach().cpu().contiguous()
            return {"tensor": list(t.shape), "dtype": str(t.dtype),
                    "sha": hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16]}
    except ImportError:  # pragma: no cover
        pass
    return fp_canonical(value)


def node_identity(node: Node) -> Any:
    """What defines a node's behaviour apart from its parameters (code + builder calls)."""
    from topologies.synth.core import FnNode
    from topologies.synth.fingerprint import callable_fingerprint

    if isinstance(node, FnNode):
        f = node.factory
        return {"fn": callable_fingerprint(f.fn), "batch": callable_fingerprint(f.batch)
                if f.batch is not None and f.batch is not f.fn else None,
                "missing": node.missing_input}
    custom = getattr(node, "fingerprint_config", None)
    if callable(custom):
        return {"cls": callable_fingerprint(type(node)), "config": custom()}
    ident: dict[str, Any] = {"cls": callable_fingerprint(type(node))}
    for attr in ("fn", "predicate"):
        v = node.__dict__.get(attr)
        if callable(v):
            ident[attr] = callable_fingerprint(v)
    if isinstance(node, Where):
        ident["max_tries"] = node.max_tries
    if isinstance(node, Mixture):
        ident["weights"] = list(node.weights)
    if isinstance(node, Attr):
        ident["attr"] = node.attr_name
    if isinstance(node, Const):
        ident["value"] = None
    return ident


def graph_fingerprint(root: Node) -> str:
    """Hash of the whole graph under ``root``: node code, parameters, names, builder
    calls and *sharing* (a node referenced twice differs from two equal nodes)."""
    from topologies.experiments.cache import stable_hash
    from topologies.synth.fingerprint import fp_canonical

    refs: dict[int, int] = {id(root): 0}
    order: list[Node] = [root]
    forms = []
    i = 0
    while i < len(order):
        n = order[i]
        params = n.params()
        if isinstance(n, Const):
            params = {"value": n.value}
        forms.append({"id": node_identity(n), "name": n.node_name,
                      "params": _canon(params, refs, order),
                      "calls": fp_canonical([c for c in n.__dict__.get("_calls", [])
                                             if c[0] != "named"])})
        i += 1
    return stable_hash(forms)


from topologies.synth.core import Where

__all__ = ["Binding", "Ctx", "Evaluator", "Pending", "RecordState", "Sample", "graph_fingerprint",
           "node_identity", "substitute"]
