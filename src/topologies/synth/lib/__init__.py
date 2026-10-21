"""The library: ready-made generators, built only on the public ``topologies.synth`` API.

    synth.lib()                      # every library node: area, tags, parameters
    synth.lib(tag="audio")           # filtered by tag
    synth.lib("harmonic")            # one entry with its docstring and example
    synth.audio.harmonic(f0=220.0)   # areas are attributes of synth: random, process, text,
                                     # audio, image, series, tabular, graph, seq, math, code

Every factory is a ``@synth.node`` function or a ``synth.Node`` subclass with
``tags`` and an ``example`` (the keyword arguments docs, tests and
``topologies synth lib NAME`` use). The library is a consumer of the system:
it imports nothing from ``topologies.synth`` except public names (a test checks).
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

from topologies import synth

AREAS = ("random", "process", "text", "audio", "image", "series", "tabular", "graph", "seq",
         "math", "code")


def _area(name: str) -> Any:
    return importlib.import_module(f"topologies.synth.lib.{name}")


def entries() -> list[dict[str, Any]]:
    """Every library node factory / class, with area, tags, parameters, doc and example."""
    import importlib.util

    out: list[dict[str, Any]] = []
    for area in AREAS:
        if importlib.util.find_spec(f"topologies.synth.lib.{area}") is None:
            continue
        mod = _area(area)
        for name in getattr(mod, "__all__", []):
            obj = getattr(mod, name, None)
            if isinstance(obj, synth.NodeFactory):
                d = obj.describe()
                d.update(area=area, name=name, kind="function")
                out.append(d)
            elif inspect.isclass(obj) and issubclass(obj, synth.Node) and \
                    getattr(obj, "example", None) is not None:
                sig = inspect.signature(obj.__init__)
                out.append({"area": area, "name": name, "kind": "class",
                            "path": f"{obj.__module__}:{obj.__qualname__}",
                            "tags": list(obj.tags), "doc": inspect.getdoc(obj) or "",
                            "example": dict(obj.example), "batched": bool(obj.batched),
                            "type": getattr(obj.type, "__name__", None) if obj.type else None,
                            "params": {p.name: (None if p.default is p.empty else p.default)
                                       for p in list(sig.parameters.values())[1:]
                                       if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}})
    return out


def lib(name: str | None = None, *, tag: str | None = None, area: str | None = None,
        echo: bool = True) -> Any:
    """List library nodes (``tag`` / ``area`` filters) or show one (``name``)."""
    rows = entries()
    if name is not None:
        hits = [r for r in rows if r["name"] == name or f"{r['area']}.{r['name']}" == name]
        if not hits:
            import difflib

            from topologies.synth import SynthError

            names = sorted({r["name"] for r in rows})
            sug = difflib.get_close_matches(name, names, n=1)
            raise SynthError(f"Unknown library node {name!r}.", stage="lib",
                             hint=(f"Did you mean {sug[0]!r}? " if sug else "") +
                             "synth.lib() lists them.")
        r = hits[0]
        text = (f"synth.{r['area']}.{r['name']}  [{', '.join(r['tags'])}]"
                f"{'  (batched)' if r.get('batched') else ''}\n{r['doc']}\n"
                f"parameters: {r['params']}\nexample: synth.{r['area']}.{r['name']}("
                + ", ".join(f"{k}={v!r}" for k, v in (r['example'] or {}).items()) + ")")
        if echo:
            print(text)
        return r
    if tag is not None:
        rows = [r for r in rows if tag in r["tags"]]
    if area is not None:
        rows = [r for r in rows if r["area"] == area]
    if echo:
        width = max((len(r["area"]) + len(r["name"]) + 1 for r in rows), default=10)
        for r in rows:
            first = (r["doc"].splitlines() or [""])[0]
            print(f"{r['area'] + '.' + r['name']:<{width}}  {', '.join(r['tags'])[:30]:<30}  "
                  f"{first[:70]}")
    return rows


def __getattr__(name: str) -> Any:
    if name in AREAS:
        return _area(name)
    raise AttributeError(f"module 'topologies.synth.lib' has no attribute {name!r}")


__all__ = ["AREAS", "entries", "lib"]

synth.callable_module(__name__, "lib")
