"""Deterministic path-seeded replay of generated records."""
from __future__ import annotations

from collections.abc import Mapping
from difflib import get_close_matches
from typing import Any

from topologies.synth.core import Attr, as_generator, iter_children
from topologies.synth.errors import SynthError
from topologies.synth.evaluate import Binding, Evaluator
from topologies.synth.generators.base import Raw, collect, default_assemble
from topologies.synth.schema import schema_of
from topologies.synth.valuetypes import decode_value, normalize_record


def _aliases(root: Any) -> dict[str, int]:
    """All field/parameter spellings for nodes, including noncanonical aliases."""
    from topologies.synth.core import Node

    found: dict[str, int] = {}
    visited: set[tuple[int, str]] = set()

    def walk(node: Any, path: str) -> None:
        if not isinstance(node, Node) or (id(node), path) in visited:
            return
        visited.add((id(node), path))
        found[path] = id(node)
        for key, value in node.params().items():
            label = f"{path}.{key}" if path else key
            for child_path, child in iter_children(value, label):
                walk(child, child_path)

    walk(root, "")
    return found


def counterfactual(gen: Any, record: Mapping[str, Any], **changes: Any) -> dict[str, Any]:
    """Replay one raw record; changes name attributes, bound node paths or parameters."""
    spec = gen if hasattr(gen, "generator") and hasattr(gen, "seed") else None
    root = as_generator(spec.generator if spec else gen)
    if spec is not None:
        store = spec.store()
        record = decode_value(dict(record), lambda ref: store.blob_path(dict(ref)).read_bytes())
    lineage = record.get("_lineage") or {}
    seed_value = record.get("_seed", lineage.get("seed", getattr(spec, "seed", None)))
    index_value = record.get("_index", lineage.get("raw_index", lineage.get("index")))
    if seed_value is None or index_value is None:
        raise SynthError("Counterfactual replay requires generation seed and raw index.",
                         stage="counterfactual",
                         hint="Pass a materialized/stream record or explicit _seed and _index metadata.")
    seed, index = int(seed_value), int(index_value)
    if spec is not None and record.get("_seed") is None and lineage.get("seed") is None:
        seed = int(spec.seed)
    attrs = dict(record.get("_attrs") or lineage.get("attrs") or {})
    space = getattr(spec, "attributes", None)
    binding = Binding(root, seed)
    aliases = _aliases(root)
    path_nodes = {p: binding.nodes[k] for k, p in binding.paths.items()}
    nodes_by_id = binding.nodes
    attr_changes: dict[str, Any] = {}
    values: dict[str, Any] = {}
    params: dict[str, Any] = {}
    suggestions = sorted(set(attrs) | set(aliases) | set(binding.paths.values()))
    for original, value in changes.items():
        key = original.removeprefix("node.")
        if key.startswith("attr.") or key in attrs or (space is not None and key in space):
            name = key.removeprefix("attr.")
            known_attrs = space.names if space is not None else sorted(attrs)
            if name not in known_attrs:
                near = get_close_matches(name, known_attrs, n=3)
                hint = (f"Did you mean {', '.join(repr(x) for x in near)}?" if near
                        else f"Available attributes: {', '.join(known_attrs)}.")
                raise SynthError(f"Unknown attribute {name!r}.", stage="counterfactual",
                                 hint=hint)
            attr_changes[name] = value
            continue
        if key in aliases:
            values[binding.paths.get(aliases[key], key)] = value
            continue
        if key in binding.paths.values():
            values[key] = value
            continue
        # Override syntax also addresses node parameters, e.g. audio.f0.low.
        owner = [(len(p), p) for p in aliases if key.startswith((p + ".", p + "["))]
        if owner:
            _n, alias = max(owner)
            canonical = binding.paths.get(aliases[alias], alias)
            params[canonical + key[len(alias):]] = value
            continue
        near = get_close_matches(key, suggestions, n=3)
        raise SynthError(f"Unknown counterfactual path or attribute {original!r}.",
                         stage="counterfactual",
                         hint=(f"Did you mean {', '.join(repr(x) for x in near)}?" if near else
                               "Use a declared attribute or bound node/parameter path."))
    if space is not None:
        for name, value in attr_changes.items():
            attr = space[name]
            if attr.kind == "categorical" and value not in attr.values:
                raise SynthError(f"{value!r} is not a value of attribute {name!r}.",
                                 stage="counterfactual",
                                 hint=f"Choose one of: {', '.join(map(repr, attr.values))}.")
    attrs.update(attr_changes)
    changed_ids = {aliases.get(p, 0) for p in values}
    changed_ids.discard(0)
    for p in params:
        owner = max((x for x in aliases if p.startswith((x + ".", x + "["))),
                    key=len, default="")
        if owner:
            changed_ids.add(aliases[owner])
    changed_ids.update(id(n) for n in nodes_by_id.values()
                       if isinstance(n, Attr) and n.attr_name in attr_changes)
    affected = set(changed_ids)
    while True:
        prior = len(affected)
        for node_id, node in nodes_by_id.items():
            deps = {id(child) for value in node.params().values()
                    for _, child in iter_children(value, "")}
            if deps & affected:
                affected.add(node_id)
        if len(affected) == prior:
            break
    pins = {}
    for path, value in (record.get("_latent") or lineage.get("latent") or {}).items():
        node = path_nodes.get(path)
        if node is None or id(node) not in affected:
            pins[path] = value
    pins.update(values)
    if params:
        from topologies.synth.overrides import override_node

        try:
            root = override_node(root, params)
        except SynthError as exc:
            raise SynthError(str(exc), stage="counterfactual", hint=exc.hint) from exc
    sample = Evaluator(root, seed=seed).evaluate(index, attrs, pins=pins)
    if sample.error:
        raise sample.error
    raw_outputs = collect(sample.value)
    fanout = record.get("_fanout_index", record.get("_lineage", {}).get("sub", 0))
    target = int(fanout[0] if isinstance(fanout, (list, tuple)) and fanout else fanout or 0)
    if not 0 <= target < len(raw_outputs):
        raise SynthError(f"Replay produced {len(raw_outputs)} outputs; fanout index {target} is absent.",
                         stage="counterfactual")
    raw = raw_outputs[target]
    raw = raw if isinstance(raw, Raw) else Raw(raw)
    schema = spec.schema if spec is not None else schema_of(root)
    assembled = default_assemble(sample.attrs, raw.value, schema)
    assembled = normalize_record(assembled, schema)
    lineage_out = dict(raw.lineage)
    latents = {**sample.latents, **(lineage_out.get("latent") or {})}
    if latents:
        lineage_out["latent"] = latents
    if raw.evidence:
        lineage_out["evidence"] = raw.evidence
    if raw.parents:
        lineage_out["parents"] = raw.parents
    return {**assembled, "_attrs": dict(sample.attrs), "_latent": latents,
            "_seed": seed, "_index": index, "_fanout_index": record.get("_fanout_index", 0),
            "_lineage": lineage_out}


__all__ = ["counterfactual"]
