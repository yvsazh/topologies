"""YAML (or JSON) synth specs: ``Spec.to_yaml()`` / ``Spec.from_yaml()`` and the CLI.

    name: fraud
    blueprint:                         # start from a blueprint ...
      name: tabular
    size: 2000                         # ... and override any Spec argument

    name: invoices
    attributes:
      format: [email, ocr-scan, chat]
      language: {en: 0.6, de: 0.2, uk: 0.2}
    generator: my_pkg.invoices:make_invoice
    pipeline:
      - filters.length: {min_chars: 40}
      - dedup.near: {threshold: 0.85}
    size: 50000
    max_time: 30m

Nodes: ``{call: target, args: [...], then: [...], **kwargs}`` builds an object
(``target`` is ``synth.<attr path>`` or an import path ``pkg.mod:obj``; ``then``
lists builder method calls such as SCM nodes); ``{ref: pkg.mod:obj}`` is an
importable object (functions, classes, ``@synth.generator`` functions);
``{path: file}`` is a file path relative to the YAML file. Pipeline items may
use the short form ``op.name: kwargs``; top-level ``generator``, ``assemble`` and
``where`` accept a plain import path. Unknown keys fail with suggestions.

``to_yaml`` writes every part by its recorded constructor arguments and checks the
round trip: the reloaded spec must have the same fingerprint, or it raises with
the part that cannot be written (typically a lambda: define it at module level).
"""

from __future__ import annotations

import functools
import importlib
import inspect
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from topologies.synth.errors import SynthError, suggest, unknown

_IMPORT_RE = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_][\w.]*$")
_RESERVED = ("call", "args", "kwargs", "then", "ref", "path")
_CALLABLE_KEYS = ("generator", "assemble", "where")


def _spec_params() -> dict[str, inspect.Parameter]:
    from topologies.synth.spec import Spec

    return {k: p for k, p in inspect.signature(Spec.__init__).parameters.items()
            if k != "self"}


# -- resolving names ------------------------------------------------------------------------
def resolve(target: str) -> Any:
    """``synth.ops.dedup.exact`` / ``pkg.mod:obj.attr`` -> the object."""
    if not isinstance(target, str) or not target:
        raise SynthError(f"Invalid call target {target!r}.", stage="yaml")
    if ":" in target:
        mod_name, _, qual = target.partition(":")
        try:
            obj: Any = importlib.import_module(mod_name)
        except ImportError as e:
            raise SynthError(f"Cannot import {mod_name!r} (from {target!r}): {e}", stage="yaml",
                             hint="Import paths look like pkg.module:name; the module must be "
                                  "importable (installed or on PYTHONPATH).") from None
        path = qual.split(".")
    else:
        parts = target.split(".")
        if parts[0] != "synth":
            raise SynthError(f"Cannot resolve {target!r}.", stage="yaml",
                             hint="Use synth.<name> (synth.uniform, synth.ops.dedup.exact) or an "
                                  "import path pkg.module:name.")
        from topologies import synth

        obj, path = synth, parts[1:]
    for i, attr in enumerate(path):
        try:
            obj = getattr(obj, attr)
        except AttributeError:
            options = [n for n in dir(obj) if not n.startswith("_")]
            options += list(getattr(obj, "__all__", []))
            where = ".".join([target.split(":")[0] if ":" in target else "synth", *path[:i]])
            raise unknown(f"name in {where}", attr, options, stage="yaml") from None
    return obj


def _import_path(obj: Any) -> str | None:
    """``module:qualname`` that imports back to ``obj`` itself, if any."""
    mod = getattr(obj, "__module__", None)
    qual = getattr(obj, "__qualname__", None)
    if not mod or not qual or "<" in qual:
        return None
    path = f"{mod}:{qual}"
    try:
        if resolve(path) is obj:
            return _short(path, obj)
    except SynthError:
        return None
    return None


def _short(path: str, obj: Any) -> str:
    """``topologies.synth.generators.scm:SCM`` -> ``synth.SCM`` (or
    ``synth.ops.filters.Length``) when that resolves to the same object."""
    mod, _, name = path.partition(":")
    cands = [f"synth.{name}"]
    if not path.startswith("topologies."):
        return path
    rest = mod[len("topologies.synth"):].lstrip(".") if mod.startswith("topologies.synth") \
        else ""
    if rest.startswith("lib."):  # library areas: synth.audio.sine, synth.text.template
        cands.append(f"synth.{rest.split('.')[1]}.{name}")
    if rest:
        cands.append(f"synth.{rest}.{name}")
    for c in cands:
        try:
            resolved = resolve(c)
        except Exception:  # noqa: BLE001, S112 - lazy modules may raise anything; try next
            continue
        if resolved is obj:
            return c
    return path


# -- building (YAML -> objects) --------------------------------------------------------------
class _Builder:
    def __init__(self, base: Path) -> None:
        self.base = base
        self._built: dict[int, Any] = {}

    def value(self, v: Any) -> Any:
        if isinstance(v, list):
            return [self.value(x) for x in v]
        if isinstance(v, Mapping):
            if "call" in v:
                # YAML aliases (*id001) are the same mapping object: the same node, shared.
                if id(v) not in self._built:
                    self._built[id(v)] = (v, self.call(v))
                return self._built[id(v)][1]
            if "ref" in v:
                self._only(v, "ref")
                return resolve(v["ref"])
            if "path" in v:
                self._only(v, "path")
                return self.path(v["path"])
            return {k: self.value(x) for k, x in v.items()}
        return v

    def path(self, p: Any) -> str:
        q = Path(str(p)).expanduser()
        return str(q if q.is_absolute() else (self.base / q))

    @staticmethod
    def _only(node: Mapping[str, Any], key: str) -> None:
        extra = [k for k in node if k != key]
        if extra:
            raise SynthError(f"A {{{key}: ...}} node takes no other keys; got {extra}.",
                             stage="yaml")

    def call(self, node: Mapping[str, Any]) -> Any:
        node = dict(node)
        target = node.pop("call")
        args = node.pop("args", [])
        then = node.pop("then", [])
        kwargs = {**node.pop("kwargs", {}), **node}
        if not isinstance(args, list):
            args = [args]
        fn = resolve(target)
        a = [self.value(x) for x in args]
        kw = {k: self.value(x) for k, x in kwargs.items()}
        obj = _invoke(fn, a, kw, str(target))
        for step in then or []:
            if not isinstance(step, Mapping):
                raise SynthError(f"'then' steps are {{method: ..., args, kwargs}}; got {step!r}.",
                                 stage="yaml")
            step = dict(step)
            if "method" in step:
                m = step.pop("method")
                sa = step.pop("args", [])
                skw = {**step.pop("kwargs", {}), **step}
            elif len(step) == 1:
                m, val = next(iter(step.items()))
                sa, skw = ([], dict(val)) if isinstance(val, Mapping) else (
                    list(val) if isinstance(val, list) else [val], {})
            else:
                raise SynthError(f"Invalid 'then' step {step!r}.", stage="yaml")
            meth = getattr(obj, m, None)
            if not callable(meth):
                raise unknown(f"method of {type(obj).__name__}", m,
                              [n for n in dir(obj) if not n.startswith("_")], stage="yaml")
            r = _invoke(meth, [self.value(x) for x in sa],
                        {k: self.value(x) for k, x in skw.items()}, f"{target}.{m}")
            obj = r if r is not None and type(r) is type(obj) else obj
        return obj

    def short(self, item: Any, namespace: str) -> Any:
        """``{op.name: kwargs}`` short form (pipeline items, generator)."""
        if isinstance(item, str) and (_IMPORT_RE.match(item) or item.startswith("synth.")):
            obj = resolve(item if ":" in item or item.startswith("synth.")
                          else f"{namespace}.{item}")
            return obj() if namespace == "synth.ops" and _is_factory(obj) else obj
        if isinstance(item, Mapping) and len(item) == 1 and not any(k in _RESERVED for k in item):
            key, val = next(iter(item.items()))
            target = key if ":" in key or key.startswith("synth.") else f"{namespace}.{key}"
            if isinstance(val, Mapping):
                return self.call({"call": target, **val}) if not any(
                    k in val for k in ("call",)) else self.call({"call": target, "kwargs": val})
            if isinstance(val, list):
                return self.call({"call": target, "args": val})
            return self.call({"call": target, "args": [] if val is None else [val]})
        return self.value(item)


def _is_factory(obj: Any) -> bool:
    return callable(obj) and not hasattr(obj, "apply") and not hasattr(obj, "check")


def _invoke(fn: Any, args: list[Any], kwargs: dict[str, Any], label: str) -> Any:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        sig = None
    if sig is not None:
        params = sig.parameters
        if not any(p.kind == p.VAR_KEYWORD for p in params.values()):
            for k in kwargs:
                if k not in params:
                    raise unknown(f"argument of {label}", k, [p for p in params
                                                             if p not in ("self", "cls")],
                                  stage="yaml")
    try:
        return fn(*args, **kwargs)
    except SynthError:
        raise
    except TypeError as e:
        raise SynthError(f"{label}(...) failed: {e}", stage="yaml",
                         hint=f"Signature: {label}{sig}" if sig is not None else None) from None


# -- dumping (objects -> YAML) ---------------------------------------------------------------
class _Dumper:
    def __init__(self, base: Path | None) -> None:
        self.base = base
        self._nodes: dict[int, Any] = {}

    def value(self, v: Any, where: str) -> Any:
        if v is None or isinstance(v, (bool, int, float, str)):
            return v
        from topologies.synth.core import Node

        if isinstance(v, Node):
            # The same node object twice -> the same dict twice -> a YAML anchor + alias, so
            # sharing survives the round trip.
            if id(v) not in self._nodes:
                self._nodes[id(v)] = (v, self._node(v, where))
            return self._nodes[id(v)][1]
        return self._value(v, where)

    def _node(self, v: Any, where: str) -> Any:
        from topologies.synth.core import GeneratorFn

        if isinstance(v, GeneratorFn):
            fn = v.fn
            qual = f"{getattr(fn, '__module__', '')}:{getattr(fn, '__qualname__', '')}"
            try:
                base = resolve(qual)
            except SynthError:
                base = None
            if base is v:
                return {"ref": _short(qual, v)}
            if isinstance(base, GeneratorFn) and base.fn is fn and v.bound:
                out = {"call": f"{_short(qual, base)}.with_"}
                out.update({k: self.value(x, f"{where}.{k}") for k, x in v.bound.items()})
                return out
        return self._value(v, where)

    def _value(self, v: Any, where: str) -> Any:
        if v is None or isinstance(v, (bool, int, float, str)):
            return v
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, (list, tuple)):
            return [self.value(x, f"{where}[{i}]") for i, x in enumerate(v)]
        if isinstance(v, (set, frozenset)):
            return [self.value(x, where) for x in sorted(v, key=repr)]
        if isinstance(v, Mapping):
            out = {k if isinstance(k, (str, int, float, bool)) else str(k):
                   self.value(x, f"{where}.{k}") for k, x in v.items()}
            if any(k in _RESERVED for k in out):
                return {"call": "builtins:dict", "args": [[[k, x] for k, x in out.items()]]}
            return out
        if isinstance(v, functools.partial):
            return {"call": "functools:partial",
                    "args": [self.value(v.func, where), *[self.value(a, where) for a in v.args]],
                    **{k: self.value(x, where) for k, x in v.keywords.items()}}
        if isinstance(v, type) or inspect.isfunction(v) or inspect.isbuiltin(v) or \
                inspect.ismethod(v):
            path = _import_path(v)
            if path is None:
                raise self._fail(v, where)
            return {"ref": path}
        stale = getattr(v, "__dict__", {}).get("_ctor_stale")
        ctor = getattr(v, "__dict__", {}).get("_ctor")
        if ctor is not None:
            first = ctor[1][0] if ctor[1] else None
            if first is not None and inspect.isfunction(first):
                p = _import_path(first)
                if p is None:
                    qual = f"{first.__module__}:{first.__qualname__}"
                    try:
                        if resolve(qual) is v:
                            return {"ref": qual}
                    except SynthError:
                        pass
        if hasattr(v, "model_dump") and hasattr(type(v), "model_fields"):
            path = _import_path(type(v))
            if path is None:
                raise self._fail(v, where)
            return {"call": path, **self.value(v.model_dump(exclude_defaults=True), where)}
        to_config = getattr(v, "to_config", None)
        from_config = getattr(type(v), "from_config", None)
        if callable(to_config) and isinstance(inspect.getattr_static(type(v), "from_config",
                                                                     None), classmethod):
            path = _import_path(type(v))
            if path is not None and from_config is not None:
                return {"call": f"{path}.from_config", "args": [self.value(to_config(), where)]}
        if stale:
            raise SynthError(f"{where} ({type(v).__name__}) was modified after construction "
                             f"({stale!r}) and cannot be written to YAML.", stage="yaml",
                             hint="Build it with the final arguments instead of mutating it.")
        if ctor is not None:
            cls, args, kwargs = ctor
            path = _import_path(cls)
            if path is None:
                raise self._fail(v, where)
            args, kwargs = _drop_defaults(cls, list(args), dict(kwargs))
            node: dict[str, Any] = {"call": path}
            if args:
                node["args"] = [self.value(a, f"{where}.{cls.__name__}[{i}]")
                                for i, a in enumerate(args)]
            kw = {k: self.value(x, f"{where}.{k}") for k, x in kwargs.items()}
            if any(k in _RESERVED for k in kw):
                node["kwargs"] = kw
            else:
                node.update(kw)
            calls = v.__dict__.get("_calls")
            if calls:
                node["then"] = [{"method": m,
                                 **({"args": self.value(list(a), where)} if a else {}),
                                 **({"kwargs": self.value(k, where)} if k else {})}
                                for m, a, k in calls]
            return node
        fallback = self._by_signature(v, where)
        if fallback is not None:
            return fallback
        to_records = getattr(v, "to_records", None)
        if callable(to_records):
            return self.value(list(to_records()), where)
        raise self._fail(v, where)

    def _by_signature(self, v: Any, where: str) -> dict[str, Any] | None:
        """Objects whose ``__init__`` arguments are all stored as same-named attributes
        (experiments distributions and similar small value classes)."""
        path = _import_path(type(v))
        if path is None:
            return None
        try:
            params = list(inspect.signature(type(v).__init__).parameters.values())[1:]
        except (TypeError, ValueError):
            return None
        node: dict[str, Any] = {"call": path}
        for p in params:
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                return None
            if p.name not in getattr(v, "__dict__", {}):
                return None
            val = v.__dict__[p.name]
            if p.default is not inspect.Parameter.empty and _same(val, p.default):
                continue
            node[p.name] = self.value(val, f"{where}.{p.name}")
        return node

    @staticmethod
    def _fail(v: Any, where: str) -> SynthError:
        what = getattr(v, "__qualname__", type(v).__name__)
        lam = "<lambda>" in str(what) or "<locals>" in str(what)
        return SynthError(f"Cannot write {where} ({what}) to YAML.", stage="yaml",
                          hint="Lambdas and local functions have no import path: define them "
                               "at module level (pkg.mod:fn)." if lam else
                          "Only importable functions/classes and synth parts built from "
                          "arguments can be written; build this part in a module-level "
                          "function and reference it as {call: pkg.mod:fn}.")


def _drop_defaults(cls: Any, args: list[Any], kwargs: dict[str, Any]
                   ) -> tuple[list[Any], dict[str, Any]]:
    """Leave out arguments equal to the signature defaults (shorter YAML, same object)."""
    try:
        sig = inspect.signature(cls.__init__ if isinstance(cls, type) else cls)
    except (TypeError, ValueError):
        return args, kwargs
    params = [p for p in sig.parameters.values() if p.name not in ("self", "cls")]
    kw = {k: v for k, v in kwargs.items()
          if not (k in sig.parameters and sig.parameters[k].default is not inspect._empty
                  and _same(v, sig.parameters[k].default))}
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    while args and len(args) <= len(positional):
        p = positional[len(args) - 1]
        if p.default is inspect._empty or not _same(args[-1], p.default):
            break
        args.pop()
    return args, kw


def _same(a: Any, b: Any) -> bool:
    try:
        return bool(a == b) and type(a) is type(b)
    except Exception:  # noqa: BLE001
        return False


def spec_to_dict(spec: Any, *, base: Path | None = None) -> dict[str, Any]:
    """The YAML-able dict of ``spec`` (unverified; ``spec_to_yaml`` checks the round trip)."""
    from topologies.synth.schema import Schema

    d = _Dumper(base)
    out: dict[str, Any] = {"name": spec.name}
    bp = getattr(spec, "_from_blueprint", None)
    if bp is not None:
        out["blueprint"] = d.value(dict(bp), "blueprint")
        items = dict(getattr(spec, "_bp_overrides", {}) or {})
    else:
        params = _spec_params()
        items = {}
        for k, v in spec._kwargs.items():
            if k in ("name", "blueprint") or v is None:
                continue
            default = params[k].default
            if default is not inspect.Parameter.empty and _same(v, default):
                continue
            if k == "pipeline" and not v:
                continue
            items[k] = v
    for k, v in items.items():
        if k == "name":
            continue
        if k == "schema" and isinstance(v, Schema):
            out[k] = d.value(v.to_config(), k)
        else:
            out[k] = d.value(v, k)
    return out


def spec_from_dict(data: Mapping[str, Any], *, base: str | Path | None = None) -> Any:
    """Build a ``Spec`` from a parsed YAML/JSON mapping (see module docstring)."""
    from topologies.synth.spec import Spec

    if not isinstance(data, Mapping):
        raise SynthError("A synth spec file must contain a mapping.", stage="yaml")
    params = _spec_params()
    for k in data:
        if k not in params:
            raise unknown("spec key", str(k), list(params), stage="yaml")
    if "name" not in data:
        raise SynthError("A synth spec needs a name.", stage="yaml", hint="name: my-dataset")
    b = _Builder(Path(base) if base is not None else Path.cwd())
    kwargs: dict[str, Any] = {}
    for k, v in data.items():
        if k == "name":
            kwargs[k] = str(v)
        elif k in _CALLABLE_KEYS and isinstance(v, str):
            kwargs[k] = resolve(v) if (_IMPORT_RE.match(v) or v.startswith("synth.")) else v
        elif k == "generator":
            kwargs[k] = b.short(v, "synth") if isinstance(v, Mapping) and len(v) == 1 and \
                not any(x in _RESERVED for x in v) else b.value(v)
        elif k == "pipeline":
            if not isinstance(v, list):
                raise SynthError("pipeline must be a list of ops.", stage="yaml")
            kwargs[k] = [b.short(item, "synth.ops") for item in v]
        elif k == "reference" and isinstance(v, str):
            kwargs[k] = b.path(v)
        elif k == "blueprint" and isinstance(v, Mapping):
            kwargs[k] = {kk: b.value(x) for kk, x in v.items()}
        elif k == "root" and isinstance(v, str):
            kwargs[k] = b.path(v)
        else:
            kwargs[k] = b.value(v)
    return Spec(**kwargs)


class _BlockDumper(yaml.SafeDumper):
    """Multi-line strings (templates) as ``|`` blocks."""


def _str_repr(dumper: yaml.SafeDumper, data: str) -> Any:
    if "\n" in data and all(not line.endswith(" ") for line in data.split("\n")):
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_BlockDumper.add_representer(str, _str_repr)


def _yaml_dump(data: Any) -> str:
    return yaml.dump(data, Dumper=_BlockDumper, sort_keys=False, allow_unicode=True,
                     width=100)


def spec_to_yaml(spec: Any, path: str | Path | None = None, *, check: bool = True) -> str:
    """YAML text of ``spec`` (also written to ``path``); ``check`` verifies the round trip."""
    base = Path(path).resolve().parent if path is not None else Path.cwd()
    data = spec_to_dict(spec, base=base)
    text = _yaml_dump(data)
    if check:
        back = spec_from_dict(yaml.safe_load(text), base=base)
        if back.fingerprint() != spec.fingerprint() or back.size != spec.size:
            raise SynthError(f"{spec.name}: the YAML form does not reproduce the spec "
                             f"({_diff_part(spec, back)} differs).", stage="yaml",
                             hint="A part was probably mutated after construction or holds state "
                                  "YAML cannot express; build it from arguments, or keep this "
                                  "spec in Python.")
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text


def _diff_part(a: Any, b: Any) -> str:
    if a.size != b.size:
        return "size"
    if a.generator.fingerprint() != b.generator.fingerprint():
        return "generator"
    if a.attributes.describe() != b.attributes.describe():
        return "attributes"
    if a.gen_fingerprint() != b.gen_fingerprint():
        return "schema / sampler / assemble"
    for x, y in zip(a.pipeline, b.pipeline):
        if x.fingerprint() != y.fingerprint():
            return f"op {x.name}"
    if len(a.pipeline) != len(b.pipeline):
        return "pipeline"
    return "splits / reference"


def load_text(path: str | Path) -> Any:
    p = Path(path)
    if not p.is_file():
        raise SynthError(f"Spec file {p} does not exist.", stage="yaml",
                         hint="topologies synth init tabular my-data -o spec.yaml")
    text = p.read_text(encoding="utf-8")
    try:
        return json.loads(text) if p.suffix == ".json" else yaml.safe_load(text)
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        raise SynthError(f"{p}: invalid {'JSON' if p.suffix == '.json' else 'YAML'}: {e}",
                         stage="yaml") from None


def spec_from_yaml(path: str | Path) -> Any:
    """Load a ``Spec`` from a YAML/JSON file (paths resolve relative to the file)."""
    p = Path(path)
    return spec_from_dict(load_text(p), base=p.resolve().parent)


def node_to_yaml(node: Any, path: str | Path | None = None, *, check: bool = True) -> str:
    """YAML text of one node graph (sharing kept as YAML anchors); ``check`` verifies that
    the reloaded graph has the same fingerprint."""
    base = Path(path).resolve().parent if path is not None else Path.cwd()
    text = _yaml_dump(_Dumper(base).value(node, "node"))
    if check:
        back = node_from_yaml(text, base=base)
        if back.fingerprint() != node.fingerprint():
            raise SynthError("The YAML form does not reproduce the node graph.", stage="yaml",
                             hint="A node was mutated after construction or holds state YAML "
                                  "cannot express; build it from arguments.")
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text


def node_from_yaml(text_or_path: str | Path, *, base: str | Path | None = None) -> Any:
    """A node graph from YAML text or a file (``{call: ...}`` / ``{ref: ...}`` nodes)."""
    from topologies.synth.core import as_generator

    p = Path(str(text_or_path)) if not str(text_or_path).lstrip().startswith(("{", "call", "ref")) \
        and "\n" not in str(text_or_path) else None
    if p is not None and p.is_file():
        data, base = load_text(p), base or p.resolve().parent
    else:
        data = yaml.safe_load(str(text_or_path))
    obj = _Builder(Path(base) if base is not None else Path.cwd()).value(data)
    return as_generator(obj) if isinstance(obj, Mapping) else obj


__all__ = ["node_from_yaml", "node_to_yaml", "resolve", "spec_from_dict", "spec_from_yaml",
           "spec_to_dict", "spec_to_yaml", "suggest"]
