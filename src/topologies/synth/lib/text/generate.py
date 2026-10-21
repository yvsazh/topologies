"""Algorithmic string and constrained JSON instance factories."""
from __future__ import annotations

import datetime as dt
import math
import re
import uuid
from typing import Any

try:
    from re import _constants as regex_constants
    from re import _parser as regex_parser
except ImportError:  # Python 3.10
    import sre_constants as regex_constants
    import sre_parse as regex_parser

from topologies import synth


@synth.node(type=synth.Text, tags=["text", "regex"], example={"pattern": "[A-Z]{2}[0-9]{4}"})
def regex(r, *, pattern: str, max_repeat: int = 64):
    """Generate a full match for the regular-expression subset implemented here."""
    try:
        parsed = regex_parser.parse(pattern, re.UNICODE)
    except re.error as exc:
        raise synth.SynthError(f"Invalid regular expression: {exc}", stage="text") from exc

    def emit(tokens: Any) -> str:
        out = []
        for op, arg in tokens:
            c = regex_constants
            if op is c.LITERAL:
                out.append(chr(arg))
            elif op is c.NOT_LITERAL:
                out.append(_choose(r, [chr(arg + 1) if arg < 0x10ffff else chr(arg - 1)]))
            elif op is c.ANY:
                out.append("x")
            elif op is c.IN:
                negate = bool(arg and arg[0][0] is c.NEGATE)
                choices = []
                for cop, carg in arg:
                    if cop is c.LITERAL:
                        choices.append(chr(carg))
                    elif cop is c.RANGE:
                        lo, hi = carg
                        choices.extend(chr(i) for i in range(lo, min(hi + 1, lo + 2048)))
                    elif cop is c.CATEGORY:
                        choices.extend(_category(carg))
                    elif cop is not c.NEGATE:
                        raise synth.SynthError(f"Unsupported regex class construct {cop}.", stage="text")
                choices = sorted(set(choices))
                if negate:
                    choices = [ch for ch in "a0_ΩЖ中 " if not _class_contains(arg[1:], ch)]
                out.append(_choose(r, choices))
            elif op is c.CATEGORY:
                out.append(_choose(r, _category(arg)))
            elif op is c.SUBPATTERN:
                out.append(emit(arg[-1]))
            elif op is c.BRANCH:
                out.append(emit(r.choice(arg[1])))
            elif op in (c.MAX_REPEAT, c.MIN_REPEAT, getattr(c, "POSSESSIVE_REPEAT", object())):
                lo, hi, child = arg
                cap = hi if hi != c.MAXREPEAT else lo + 3
                if lo > max_repeat or lo > 10000:
                    raise synth.SynthError("Regex repetition minimum exceeds configured bound.", stage="text")
                count = r.randint(lo, min(cap, max_repeat))
                out.append("".join(emit(child) for _ in range(count)))
            elif op is c.AT:
                continue
            else:
                raise synth.SynthError(f"Unsupported regex construct {op}.", stage="text")
        return "".join(out)

    for _ in range(100):
        value = emit(parsed)
        if re.fullmatch(pattern, value) is not None:
            return value
    raise synth.SynthError(f"Could not construct a full match for pattern {pattern!r}.", stage="text")


def _category(category: Any) -> list[str]:
    c = regex_constants
    if category in (c.CATEGORY_DIGIT, c.CATEGORY_UNI_DIGIT):
        return list("0123456789") + list("٠١٢٣٤٥٦٧٨٩")
    if category in (c.CATEGORY_NOT_DIGIT, c.CATEGORY_UNI_NOT_DIGIT):
        return list("abc_ ΩЖ中")
    if category in (c.CATEGORY_SPACE, c.CATEGORY_UNI_SPACE):
        return [" ", "\t", "\n"]
    if category in (c.CATEGORY_WORD, c.CATEGORY_UNI_WORD):
        return list("abcXYZ_012ΩЖ中é")
    if category in (c.CATEGORY_NOT_SPACE, c.CATEGORY_UNI_NOT_SPACE):
        return list("abc_012ΩЖ")
    if category in (c.CATEGORY_NOT_WORD, c.CATEGORY_UNI_NOT_WORD):
        return list(" -.!?\t")
    raise synth.SynthError(f"Unsupported regex category {category}.", stage="text")


def _class_contains(tokens: Any, value: str) -> bool:
    c = regex_constants
    code = ord(value)
    for op, arg in tokens:
        if op is c.LITERAL and arg == code:
            return True
        if op is c.RANGE and arg[0] <= code <= arg[1]:
            return True
        if op is c.CATEGORY and value in _category(arg):
            return True
    return False


def _choose(r: Any, values: list[str]) -> str:
    if not values:
        raise synth.SynthError("Regex character class has no constructible character.", stage="text")
    return r.choice(values)


def _instance(r: Any, schema: dict[str, Any], root: dict[str, Any], depth: int = 0,
              refs: tuple[str, ...] = ()) -> Any:
    if depth > 32:
        raise synth.SynthError("JSON schema nesting exceeds 32.", stage="text")
    if not isinstance(schema, dict):
        raise synth.SynthError("Schema must be an object.", stage="text")
    _check_keywords(schema)
    if "$ref" in schema:
        ref = schema["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/"):
            raise synth.SynthError(f"Only local JSON schema references are supported: {ref!r}.", stage="text")
        if ref in refs:
            raise synth.SynthError(f"Cyclic JSON schema reference: {ref}.", stage="text")
        target: Any = root
        try:
            for part in ref[2:].split("/"):
                part = part.replace("~1", "/").replace("~0", "~")
                target = target[part]
        except (KeyError, TypeError):
            raise synth.SynthError(f"Unresolved local JSON schema reference {ref!r}.", stage="text") from None
        if not isinstance(target, dict):
            raise synth.SynthError(f"Reference {ref!r} does not point to a schema object.", stage="text")
        merged = {k: v for k, v in schema.items() if k != "$ref"}
        if merged:
            return _instance(r, {**target, **merged}, root, depth + 1, refs + (ref,))
        return _instance(r, target, root, depth + 1, refs + (ref,))
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        values = schema["enum"]
        if not isinstance(values, list) or not values:
            raise synth.SynthError("Schema enum must be a non-empty array.", stage="text")
        return r.choice(values)
    for key in ("anyOf", "oneOf"):
        if key in schema:
            branches = schema[key]
            if not isinstance(branches, list) or not branches:
                raise synth.SynthError(f"{key} must contain at least one schema.", stage="text")
            order = list(range(len(branches)))
            r.shuffle(order)
            for index in order:
                try:
                    value = _instance(r, branches[index], root, depth + 1, refs)
                    if key == "oneOf":
                        matches = sum(_matches(value, b, root) for b in branches)
                        if matches != 1:
                            continue
                    return value
                except synth.SynthError:
                    continue
            raise synth.SynthError(f"No constructible valid branch for {key}.", stage="text")
    if "allOf" in schema:
        branches = schema["allOf"]
        if not isinstance(branches, list):
            raise synth.SynthError("allOf must be an array.", stage="text")
        combined: dict[str, Any] = {}
        for branch in branches:
            if not isinstance(branch, dict):
                raise synth.SynthError("allOf members must be schema objects.", stage="text")
            combined = _merge_schema(combined, branch)
        remainder = {k: v for k, v in schema.items() if k != "allOf"}
        return _instance(r, _merge_schema(combined, remainder), root, depth + 1, refs)
    typ = schema.get("type")
    types = typ if isinstance(typ, list) else [typ] if typ else []
    if types:
        if not all(t in {"object", "array", "string", "integer", "number", "boolean", "null"} for t in types):
            raise synth.SynthError(f"Unsupported JSON schema type {typ!r}.", stage="text")
        order = list(types)
        r.shuffle(order)
        for candidate in order:
            try:
                value = _typed_instance(r, schema, candidate, root, depth, refs)
                if _matches(value, schema, root):
                    return value
            except synth.SynthError:
                continue
        raise synth.SynthError(f"Schema constraints are impossible or unsupported for type {typ!r}.", stage="text")
    inferred = "object" if "properties" in schema or "required" in schema else "string"
    return _typed_instance(r, schema, inferred, root, depth, refs)


def _typed_instance(r: Any, schema: dict[str, Any], typ: str, root: dict[str, Any],
                    depth: int, refs: tuple[str, ...]) -> Any:
    if typ == "object":
        props = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(props, dict) or not isinstance(required, list):
            raise synth.SynthError("Object properties/required have invalid shape.", stage="text")
        unknown = set(required) - set(props)
        if unknown and schema.get("additionalProperties") is False:
            raise synth.SynthError(f"Required properties disallowed by additionalProperties: {sorted(unknown)}.", stage="text")
        out = {}
        for key in sorted(set(required) | set(props)):
            if key in props:
                out[key] = _instance(r, props[key], root, depth + 1, refs)
            elif isinstance(schema.get("additionalProperties"), dict):
                out[key] = _instance(r, schema["additionalProperties"], root, depth + 1, refs)
        minimum, maximum = schema.get("minProperties", len(required)), schema.get("maxProperties", max(len(out), len(required)))
        if minimum > maximum or len(out) < minimum:
            if schema.get("additionalProperties") is False or schema.get("additionalProperties") is None:
                raise synth.SynthError("Object property bounds cannot be satisfied.", stage="text")
            i = 0
            while len(out) < minimum:
                key = f"extra{i}"
                i += 1
                out[key] = _instance(r, schema["additionalProperties"], root, depth + 1, refs)
        return out
    if typ == "array":
        minimum, maximum = schema.get("minItems", 0), schema.get("maxItems", max(1, schema.get("minItems", 0)))
        if minimum < 0 or maximum < minimum:
            raise synth.SynthError("Array minItems/maxItems bounds are impossible.", stage="text")
        if maximum > 10000:
            raise synth.SynthError("Array maximum exceeds generation safety bound 10000.", stage="text")
        count = r.randint(minimum, maximum)
        items = schema.get("items", {})
        result = []
        attempts = 0
        while len(result) < count:
            attempts += 1
            if attempts > count * 20 + 20:
                raise synth.SynthError("Could not satisfy uniqueItems with generated array values.", stage="text")
            value = _instance(r, items, root, depth + 1, refs)
            if not schema.get("uniqueItems") or value not in result:
                result.append(value)
        return result
    if typ in ("integer", "number"):
        lo = schema.get("minimum", -100 if typ == "integer" else -100.0)
        hi = schema.get("maximum", 100 if typ == "integer" else 100.0)
        if "exclusiveMinimum" in schema:
            lo = max(lo, schema["exclusiveMinimum"] + (1 if typ == "integer" else 1e-7))
        if "exclusiveMaximum" in schema:
            hi = min(hi, schema["exclusiveMaximum"] - (1 if typ == "integer" else 1e-7))
        if lo > hi:
            raise synth.SynthError("Numeric bounds are impossible.", stage="text")
        multiple = schema.get("multipleOf")
        if multiple is not None:
            if multiple <= 0:
                raise synth.SynthError("multipleOf must be positive.", stage="text")
            if typ == "integer":
                step = int(multiple)
                if step != multiple:
                    raise synth.SynthError("Integer multipleOf must be an integer.", stage="text")
                first = math.ceil(lo / step) * step
                last = math.floor(hi / step) * step
                if first > last:
                    raise synth.SynthError("Numeric bounds and multipleOf are impossible.", stage="text")
                return r.randint(first // step, last // step) * step
            lo = math.ceil(lo / multiple - 1e-10) * multiple
            hi = math.floor(hi / multiple + 1e-10) * multiple
            if lo > hi:
                raise synth.SynthError("Numeric bounds and multipleOf are impossible.", stage="text")
            return r.randint(round(lo / multiple), round(hi / multiple)) * multiple
        value = r.randint(math.ceil(lo), math.floor(hi)) if typ == "integer" else r.uniform(lo, hi)
        return value
    if typ == "boolean":
        return bool(r.randrange(2))
    if typ == "null":
        return None
    if typ == "string":
        return _string(r, schema)
    raise synth.SynthError(f"Unsupported schema type {typ!r}.", stage="text")


def _string(r: Any, schema: dict[str, Any]) -> str:
    minimum, maximum = schema.get("minLength", 0), schema.get("maxLength", max(1000, schema.get("minLength", 0)))
    if minimum < 0 or maximum < minimum:
        raise synth.SynthError("String minLength/maxLength bounds are impossible.", stage="text")
    fmt = schema.get("format")
    if fmt:
        values = {
            "date": [dt.date(2024, 2, 29).isoformat()],
            "date-time": ["2024-02-29T12:34:56Z"], "time": ["12:34:56Z"],
            "email": ["person@example.test"], "uuid": [str(uuid.UUID("12345678-1234-5678-9234-567812345678"))],
            "hostname": ["example.test"], "ipv4": ["192.0.2.1"], "ipv6": ["2001:db8::1"],
            "uri": ["https://example.test/path"], "uri-reference": ["/path"],
        }
        if fmt not in values:
            raise synth.SynthError(f"Unsupported JSON schema format {fmt!r}.", stage="text")
        value = values[fmt][0]
        if len(value) < minimum or len(value) > maximum:
            raise synth.SynthError(f"Length bounds conflict with format {fmt!r}.", stage="text")
        return value
    value = "x" * minimum
    pattern = schema.get("pattern")
    if pattern:
        try:
            if re.fullmatch(pattern, value) is None:
                raise synth.SynthError("String pattern requires regex generation; compose synth.text.regex for this schema.", stage="text")
        except re.error as exc:
            raise synth.SynthError(f"Invalid schema pattern: {exc}.", stage="text") from exc
    return value


def _check_keywords(schema: dict[str, Any]) -> None:
    known = {"$defs", "definitions", "$ref", "type", "const", "enum", "anyOf", "oneOf", "allOf",
             "properties", "required", "minProperties", "maxProperties", "additionalProperties",
             "minItems", "maxItems", "items", "uniqueItems", "minimum", "maximum", "exclusiveMinimum",
             "exclusiveMaximum", "multipleOf", "minLength", "maxLength", "pattern", "format", "title", "description"}
    unknown = set(schema) - known
    if unknown:
        raise synth.SynthError(f"Unsupported JSON schema keyword(s): {', '.join(sorted(unknown))}.", stage="text")


def _merge_schema(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    result = dict(left)
    for key, value in right.items():
        if key in result and isinstance(value, dict) and isinstance(result[key], dict):
            result[key] = _merge_schema(result[key], value)
        elif key in result and key in ("required",):
            result[key] = sorted(set(result[key]) | set(value))
        elif key in result and key in ("minimum", "minLength", "minItems", "minProperties", "exclusiveMinimum"):
            result[key] = max(result[key], value)
        elif key in result and key in ("maximum", "maxLength", "maxItems", "maxProperties", "exclusiveMaximum"):
            result[key] = min(result[key], value)
        else:
            result[key] = value
    return result


def _matches(value: Any, schema: dict[str, Any], root: dict[str, Any]) -> bool:
    return not synth.validate_json_schema(value, schema, root=root)


@synth.node(type=synth.Json, tags=["text", "json"], example={"schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}}})
def json_instance(r, *, schema: dict[str, Any]):
    """Construct deterministic examples for the explicitly supported JSON Schema subset."""
    return _instance(r, schema, schema)


__all__ = ["json_instance", "regex"]
