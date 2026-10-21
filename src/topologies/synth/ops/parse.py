"""Parsing raw generator text into record fields.

    synth.ops.parse.json()                         # record["text"] -> merged JSON keys
    synth.ops.parse.json(field="raw", into="output")
    synth.ops.parse.regex(r"Q: (?P<question>.*)\\nA: (?P<answer>.*)", field="text")
    synth.ops.parse.delimited("|", ["text", "label"])

Parse failures reject the record with the error (``ds.rejected(op="parse.json")``).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from topologies.synth.errors import SynthError
from topologies.synth.jsonutil import extract_json
from topologies.synth.ops.base import MapOp, Reject


class ParseJson(MapOp):
    def __init__(self, field: str = "text", *, into: str | None = None, keep: bool = False,
                 expect: str | None = "object", name: str = "parse.json") -> None:
        self.field, self.into, self.keep, self.expect, self.name = field, into, keep, expect, name

    def apply(self, record: dict, ctx: Any) -> dict:
        raw = record.get(self.field)
        if raw is None:
            raise Reject(f"no field {self.field!r} to parse")
        try:
            value = extract_json(raw, expect=self.expect) if isinstance(raw, str) else raw
        except ValueError as e:
            raise Reject(f"invalid JSON: {e}", raw=str(raw)[:300]) from None
        out = dict(record)
        if self.into is not None:
            out[self.into] = value
        elif isinstance(value, dict):
            if not self.keep:
                out.pop(self.field, None)
            out.update(value)
        else:
            raise Reject(f"expected a JSON object, got {type(value).__name__}")
        return out

    def config(self) -> dict[str, Any]:
        return {"op": self.name, "field": self.field, "into": self.into, "keep": self.keep,
                "expect": self.expect}


class ParseRegex(MapOp):
    def __init__(self, pattern: str, *, field: str = "text", flags: int = re.DOTALL,
                 keep: bool = False, name: str = "parse.regex") -> None:
        self.pattern, self.field, self.flags, self.keep, self.name = pattern, field, flags, \
            keep, name
        self._re = re.compile(pattern, flags)
        if not self._re.groupindex:
            raise SynthError("parse.regex needs named groups: (?P<field>...).", stage="parse")

    def apply(self, record: dict, ctx: Any) -> dict:
        raw = str(record.get(self.field, ""))
        m = self._re.search(raw)
        if not m:
            raise Reject(f"no match for {self.pattern[:40]!r}")
        out = dict(record)
        if not self.keep and self.field not in self._re.groupindex:
            out.pop(self.field, None)
        out.update({k: v.strip() if isinstance(v, str) else v for k, v in m.groupdict().items()})
        return out

    def config(self) -> dict[str, Any]:
        return {"op": self.name, "pattern": self.pattern, "field": self.field,
                "flags": self.flags, "keep": self.keep}


class ParseDelimited(MapOp):
    def __init__(self, sep: str, fields: Sequence[str], *, field: str = "text",
                 strict: bool = True, name: str = "parse.delimited") -> None:
        self.sep, self.fields, self.field, self.strict, self.name = sep, list(fields), field, \
            strict, name

    def apply(self, record: dict, ctx: Any) -> dict:
        raw = str(record.get(self.field, "")).strip()
        parts = [p.strip() for p in raw.split(self.sep, len(self.fields) - 1)]
        if len(parts) != len(self.fields) and self.strict:
            raise Reject(f"expected {len(self.fields)} parts separated by {self.sep!r}, "
                         f"got {len(parts)}")
        out = dict(record)
        if self.field not in self.fields:
            out.pop(self.field, None)
        out.update(dict(zip(self.fields, parts)))
        return out

    def config(self) -> dict[str, Any]:
        return {"op": self.name, "sep": self.sep, "fields": self.fields, "field": self.field,
                "strict": self.strict}


def json_(field: str = "text", **kw: Any) -> ParseJson:
    return ParseJson(field, **kw)


def regex(pattern: str, **kw: Any) -> ParseRegex:
    return ParseRegex(pattern, **kw)


def delimited(sep: str, fields: Sequence[str], **kw: Any) -> ParseDelimited:
    return ParseDelimited(sep, fields, **kw)


json = json_  # ``synth.ops.parse.json(...)`` (the stdlib module is imported as _json)

__all__ = ["ParseDelimited", "ParseJson", "ParseRegex", "delimited", "json", "json_", "regex"]


from topologies.synth.ops._callable import make_callable

make_callable(__name__, "json")
