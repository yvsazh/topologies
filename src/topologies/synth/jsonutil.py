"""JSON helpers without dependencies: extract JSON from free text, a JSON-schema subset
validator, schema-valid examples, and cheap syntactic repairs.

The validator covers what structured-output schemas use in practice: ``type``
(incl. unions), ``properties`` / ``required`` / ``additionalProperties``,
``items`` / ``minItems`` / ``maxItems`` / ``uniqueItems``, ``enum`` / ``const``,
``minimum`` / ``maximum``, ``minLength`` / ``maxLength`` / ``pattern``,
``anyOf`` / ``oneOf`` / ``allOf`` and local ``$ref`` (``#/$defs/...``).
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

_FENCE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)```", re.DOTALL)


def _balanced(text: str, start: int) -> str | None:
    """The balanced ``{...}`` / ``[...]`` starting at ``start`` (string-aware)."""
    open_ch = text[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
            if depth == 0:
                return text[start:i + 1] if c == close_ch else None
    return None


def repair_json(text: str) -> str:
    """Cheap syntactic fixes: trailing commas, Python literals, smart quotes."""
    t = text.strip()
    t = t.replace("“", '"').replace("”", '"').replace("‘", "'").replace(
        "’", "'")
    t = re.sub(r",\s*([}\]])", r"\1", t)
    t = re.sub(r"\bTrue\b", "true", t)
    t = re.sub(r"\bFalse\b", "false", t)
    t = re.sub(r"\bNone\b", "null", t)
    return t


def extract_json(text: str, *, expect: str | None = None) -> Any:
    """Parse the JSON value in ``text`` (fenced block, bare value, or the first balanced
    object/array). ``expect="array"|"object"`` prefers that kind. Raises ``ValueError``."""
    if not isinstance(text, str):
        return text
    candidates: list[str] = [m.group(1) for m in _FENCE.finditer(text)]
    candidates.append(text)
    for cand in candidates:
        cand = cand.strip()
        for attempt in (cand, repair_json(cand)):
            try:
                value = json.loads(attempt)
                if expect is None or _kind(value) == expect:
                    return value
            except ValueError:
                pass
        starts = [i for i, c in enumerate(cand) if c in "{["]
        if expect == "array":
            starts = [i for i in starts if cand[i] == "["] + [i for i in starts if cand[i] == "{"]
        for s in starts[:50]:
            frag = _balanced(cand, s)
            if frag is None:
                continue
            for attempt in (frag, repair_json(frag)):
                try:
                    value = json.loads(attempt)
                    if expect is None or _kind(value) == expect:
                        return value
                except ValueError:
                    continue
    raise ValueError("no JSON value found in the text")


def _kind(v: Any) -> str:
    return "array" if isinstance(v, list) else "object" if isinstance(v, dict) else "scalar"


# ---------------------------------------------------------------- validation

_TYPES = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def _resolve(schema: dict, root: dict) -> dict:
    ref = schema.get("$ref")
    if not ref or not isinstance(ref, str) or not ref.startswith("#/"):
        return schema
    node: Any = root
    for part in ref[2:].split("/"):
        node = node.get(part, {}) if isinstance(node, dict) else {}
    return _resolve(node, root) if isinstance(node, dict) else {}


def validate_json_schema(value: Any, schema: dict, *, path: str = "$",
                         root: dict | None = None, limit: int = 5) -> list[str]:
    """Problems with ``value`` under ``schema`` (subset of JSON Schema; see module doc)."""
    root = root if root is not None else schema
    schema = _resolve(schema, root)
    errs: list[str] = []
    if not schema:
        return errs
    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        if not any(_TYPES.get(tt, lambda v: True)(value) for tt in types):
            return [f"{path}: expected {'/'.join(types)}, got {type(value).__name__}"]
    if "const" in schema and value != schema["const"]:
        errs.append(f"{path}: must equal {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: {value!r} not in {schema['enum']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errs.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errs.append(f"{path}: {value} > maximum {schema['maximum']}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errs.append(f"{path}: shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errs.append(f"{path}: longer than {schema['maxLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errs.append(f"{path}: does not match {schema['pattern']!r}")
    if isinstance(value, dict):
        props = schema.get("properties", {}) or {}
        for req in schema.get("required", []) or []:
            if req not in value:
                errs.append(f"{path}: missing {req!r}")
        for k, v in value.items():
            if k in props:
                errs += validate_json_schema(v, props[k], path=f"{path}.{k}", root=root)
            elif schema.get("additionalProperties") is False:
                errs.append(f"{path}: unexpected key {k!r}")
            elif isinstance(schema.get("additionalProperties"), dict):
                errs += validate_json_schema(v, schema["additionalProperties"],
                                             path=f"{path}.{k}", root=root)
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errs.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errs.append(f"{path}: more than {schema['maxItems']} items")
        if schema.get("uniqueItems") and len({json.dumps(v, sort_keys=True) for v in value}) \
                != len(value):
            errs.append(f"{path}: items are not unique")
        items = schema.get("items")
        if isinstance(items, dict):
            for i, v in enumerate(value):
                errs += validate_json_schema(v, items, path=f"{path}[{i}]", root=root)
                if len(errs) >= limit:
                    break
    for key in ("anyOf", "oneOf"):
        if key in schema:
            ok = [not validate_json_schema(value, s, path=path, root=root) for s in schema[key]]
            if not any(ok):
                errs.append(f"{path}: matches none of {key}")
    for s in schema.get("allOf", []) or []:
        errs += validate_json_schema(value, s, path=path, root=root)
    return errs[:limit]


def example_from_schema(schema: dict, rng: random.Random | None = None, *,
                        root: dict | None = None, depth: int = 0) -> Any:
    """A schema-valid example value (random, reproducible with ``rng``)."""
    rng = rng or random.Random(0)
    root = root if root is not None else schema
    schema = _resolve(schema or {}, root)
    if depth > 8:
        return None
    if "const" in schema:
        return schema["const"]
    if schema.get("enum"):
        return rng.choice(schema["enum"])
    for key in ("anyOf", "oneOf"):
        if schema.get(key):
            opts = [s for s in schema[key] if _resolve(s, root).get("type") != "null"] or \
                schema[key]
            return example_from_schema(opts[0], rng, root=root, depth=depth + 1)
    if schema.get("allOf"):
        merged: dict = {}
        for s in schema["allOf"]:
            merged.update(_resolve(s, root))
        return example_from_schema(merged, rng, root=root, depth=depth + 1)
    t = schema.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), t[0] if t else None)
    if t == "object" or (t is None and "properties" in schema):
        props = schema.get("properties", {}) or {}
        return {k: example_from_schema(v, rng, root=root, depth=depth + 1)
                for k, v in props.items()}
    if t == "array":
        n = max(int(schema.get("minItems", 1)), 1)
        n = min(n, int(schema.get("maxItems", n)))
        items = schema.get("items", {}) if isinstance(schema.get("items"), dict) else {}
        return [example_from_schema(items, rng, root=root, depth=depth + 1) for _ in range(n)]
    if t == "integer":
        lo, hi = int(schema.get("minimum", 0)), int(schema.get("maximum", 100))
        return rng.randint(lo, max(lo, hi))
    if t == "number":
        lo, hi = float(schema.get("minimum", 0.0)), float(schema.get("maximum", 100.0))
        return round(rng.uniform(lo, max(lo, hi)), 2)
    if t == "boolean":
        return rng.random() < 0.5
    if t == "null":
        return None
    if t == "string" or t is None:
        fmt = schema.get("format")
        if fmt == "date":
            return f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        if fmt == "email":
            return f"user{rng.randint(1, 999)}@example.com"
        n = max(int(schema.get("minLength", 4)), 1)
        word = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(max(n, 5)))
        return word[: int(schema.get("maxLength", len(word)))]
    return None


__all__ = ["example_from_schema", "extract_json", "repair_json", "validate_json_schema"]
