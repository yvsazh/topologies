"""``Spec.with_(**changes)``: copies with overrides, including dotted paths into parts.

    spec.with_(size=5000, max_time="5m")                   # constructor arguments
    spec.with_(**{"generator.audio.seconds": 2.0})          # a node parameter, by path
    spec.with_(**{"generator.audio.f1": synth.uniform(500, 900)})   # ... or a new node
    spec.with_(**{"ops.dedup.near.threshold": 0.9})         # an op setting (by op name)
    spec.with_(**{"ops.dedup.near": None})                  # remove an op (ablation)
    spec.with_(**{"attributes.tone": ["angry", "neutral"]}) # replace an attribute
    spec.with_(**{"mixture.shapes": 0.8})                   # a mixture weight
    spec.with_(**{"mix.real": 0.3})                         # real:synthetic mix (SynthTask)

These are exactly the ``synth.*`` axes of experiments (``synth.size``,
``synth.generator.audio.seconds``, ...). Node paths walk record fields and node
parameters (``audio.f1`` = parameter ``f1`` of the node in field ``audio``;
``notes[2].pitch`` indexes a list parameter); ``gen.with_(**{path: value})`` does
the same on a generator directly. Copies are made along the path only, so YAML
(constructor records) stays correct.
"""

from __future__ import annotations

import copy
import inspect
import re
from collections.abc import Mapping
from typing import Any

from topologies.synth.ctor import set_recorded
from topologies.synth.errors import SynthError, unknown


def _spec_params() -> list[str]:
    from topologies.synth.spec import Spec

    return [p for p in inspect.signature(Spec.__init__).parameters if p != "self"]


def _set_attr(obj: Any, path: list[str], value: Any, what: str) -> Any:
    """Copy of ``obj`` with ``obj.<path> = value`` (nested objects copied on the way)."""
    new = copy.copy(obj)
    head = path[0]
    if len(path) == 1:
        with_ = getattr(new, "with_", None)
        if callable(with_) and not hasattr(new, head) and head not in getattr(new, "__dict__", {}):
            return with_(**{head: value})
        if not hasattr(new, head):
            names = [k for k in vars(new) if not k.startswith("_")]
            raise unknown(f"{what} setting", head, names, stage="overrides")
        if isinstance(getattr(new, head), dict) and isinstance(value, Mapping) is False and \
                head in ("params",):
            raise SynthError(f"{what}.{head} is a dict; set a key: {what}.{head}.<key>.",
                             stage="overrides")
        setattr(new, head, value)
        set_recorded(new, head, value)
        refresh = getattr(new, "refresh", None)
        if callable(refresh):
            refresh()
        return new
    child = getattr(new, head, None)
    if child is None:
        raise unknown(f"{what} part", head, [k for k in vars(new) if not k.startswith("_")],
                      stage="overrides")
    if isinstance(child, dict):
        d = dict(child)
        if len(path) == 2:
            d[path[1]] = value
        else:
            d[path[1]] = _set_attr(d[path[1]], path[2:], value, f"{what}.{head}")
        setattr(new, head, d)
        set_recorded(new, head, d)
    else:
        setattr(new, head, _set_attr(child, path[1:], value, f"{what}.{head}"))
        set_recorded(new, head, getattr(new, head))
    refresh = getattr(new, "refresh", None)
    if callable(refresh):
        refresh()
    return new


_PART = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


def split_path(path: str) -> list[Any]:
    """``"audio.harmonics[2].amp"`` -> ``["audio", "harmonics", 2, "amp"]``."""
    out: list[Any] = []
    for m in _PART.finditer(path):
        out.append(m.group(1) if m.group(1) is not None else int(m.group(2)))
    if not out:
        raise SynthError(f"Empty node path {path!r}.", stage="overrides")
    return out


def override_node(root: Any, changes: Mapping[str, Any]) -> Any:
    """Copy of the node graph ``root`` with parameters replaced by path."""
    for path, value in changes.items():
        root = _override(root, split_path(path), value, path)
    return root


def _override(node: Any, parts: list[Any], value: Any, full: str) -> Any:
    from topologies.synth.core import Node

    if isinstance(node, Node):
        params = node.params()
        head = parts[0]
        if head not in params:
            if len(parts) == 1:  # a parameter left at its default
                return node.replace_params(**{head: value})
            raise unknown(f"parameter of {node.label()} (in {full!r})", str(head),
                          list(params), stage="overrides")
        if len(parts) == 1:
            return node.replace_params(**{head: value})
        return node.replace_params(**{head: _override(params[head], parts[1:], value, full)})
    if isinstance(node, list):
        i = parts[0]
        if not isinstance(i, int) or not -len(node) <= i < len(node):
            raise SynthError(f"{full!r}: index {i!r} is not in a list of {len(node)}.",
                             stage="overrides")
        new = list(node)
        new[i] = value if len(parts) == 1 else _override(new[i], parts[1:], value, full)
        return new
    if isinstance(node, dict):
        k = parts[0]
        if k not in node and len(parts) > 1:
            raise unknown(f"key (in {full!r})", str(k), list(node), stage="overrides")
        new_d = dict(node)
        new_d[k] = value if len(parts) == 1 else _override(new_d[k], parts[1:], value, full)
        return new_d
    if len(parts) == 1:
        return _set_attr(node, [str(parts[0])], value, "generator")
    raise SynthError(f"{full!r}: {type(node).__name__} has no parameters to override.",
                     stage="overrides")


def _find_op(pipeline: list[Any], name: str) -> int:
    names = [op.name for op in pipeline]
    for i, n in enumerate(names):
        if n == name:
            return i
    matches = [i for i, n in enumerate(names) if n.split("#")[0] == name or
               n.split(".")[0] == name or n.endswith("." + name)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise SynthError(f"ops.{name} matches several ops: {[names[i] for i in matches]}.",
                         stage="overrides", hint="Use the exact op name.")
    raise unknown("op", name, names, stage="overrides")


def apply_overrides(spec: Any, changes: Mapping[str, Any]) -> Any:
    """See module docstring."""
    params = _spec_params()
    plain: dict[str, Any] = {}
    gen = spec.generator
    pipeline = list(spec.pipeline)
    attr_changes: dict[str, Any] = {}
    gen_changed = ops_changed = False
    meta = dict(spec.meta)
    for key, value in changes.items():
        k = key.removeprefix("synth.")
        if "." not in k:
            if k in params:
                plain[k] = value
            else:
                raise unknown("spec setting", k, [*params, "generator.*", "ops.*",
                                                  "attributes.*", "mix.real"],
                              stage="overrides")
            continue
        head, _, rest = k.partition(".")
        if head == "generator":
            gen = override_node(gen, {rest: value})
            gen_changed = True
        elif head in ("mixture", "weights"):
            from topologies.synth.core import Mixture

            if not isinstance(gen, Mixture):
                raise SynthError("mixture.* overrides need a Mixture generator.",
                                 stage="overrides")
            gen = gen.with_weights({rest: float(value)})
            gen_changed = True
        elif head == "ops":
            parts = rest.split(".")
            names = [op.name for op in pipeline]
            i, field_parts = None, []
            for k in range(len(parts), 0, -1):  # longest exact op name first
                cand = ".".join(parts[:k])
                if cand in names:
                    i, field_parts = names.index(cand), parts[k:]
                    break
            if i is None:
                i, field_parts = _find_op(pipeline, parts[0]), parts[1:]
            opname = pipeline[i].name
            if not field_parts:
                if value is None or value is False:
                    pipeline.pop(i)
                else:
                    pipeline[i] = value
            else:
                pipeline[i] = _set_attr(pipeline[i], field_parts, value, f"ops.{opname}")
            ops_changed = True
        elif head == "attributes":
            attr_changes[rest] = value
        elif head == "mix":
            meta[f"mix.{rest}"] = value
        elif head == "sampler":
            plain["sampler"] = _set_attr(spec.sampler_spec, rest.split("."), value, "sampler")
        else:
            raise unknown("override namespace", head, ["generator", "ops", "attributes", "mix",
                                                       "mixture", "sampler"],
                          stage="overrides")
    if gen_changed:
        plain["generator"] = gen
    if ops_changed:
        for op in pipeline:
            if "#" in op.name:
                op.name = op.name.split("#")[0]
        plain["pipeline"] = pipeline
    if attr_changes:
        plain["attributes"] = spec.attributes.with_attrs(**attr_changes)
    if meta != spec.meta:
        plain["meta"] = meta
    return spec._copy_with(**plain)


__all__ = ["apply_overrides", "override_node", "split_path"]
