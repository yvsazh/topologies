"""Structured renderers and matching parsers with lossless scalar and key handling."""
from __future__ import annotations

import base64
import csv
import io
import json
from typing import Any
from urllib.parse import parse_qsl, urlencode

import yaml

from topologies import synth

FORMATS = ("kv", "csv", "json", "yaml", "table", "email", "log", "form")


def render_value(value: Any, format: str = "json") -> str:
    """Render mappings, sequences or scalars in a reversible format."""
    if format not in FORMATS:
        raise ValueError(f"Unknown structured text format {format!r}; expected one of {FORMATS}.")
    if format == "json":
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if format == "yaml":
        return yaml.safe_dump(value, allow_unicode=True, sort_keys=False).rstrip()
    kind = "map" if isinstance(value, dict) else "list" if isinstance(value, (list, tuple)) else "scalar"
    rows = list(value.items()) if kind == "map" else list(enumerate(value)) if kind == "list" else [(None, value)]
    if format == "csv":
        stream = io.StringIO(newline="")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["kind", "key", "value"])
        writer.writerow(["container", "", json.dumps(kind)])
        for key, item in rows:
            writer.writerow(["entry", json.dumps(key, ensure_ascii=False),
                             json.dumps(item, ensure_ascii=False, separators=(",", ":"))])
        return stream.getvalue().rstrip("\n")
    if format == "form":
        pairs = [("__kind__", kind)]
        pairs.extend((json.dumps(key, ensure_ascii=False), json.dumps(item, ensure_ascii=False))
                     for key, item in rows)
        return urlencode(pairs)
    if format == "kv":
        return "\n".join(["@container: " + json.dumps(kind), *(
            json.dumps(key, ensure_ascii=False) + ": " + json.dumps(item, ensure_ascii=False)
            for key, item in rows)])
    if format == "log":
        entries = [{"container": kind}] + [{"key": key, "value": item} for key, item in rows]
        return "\n".join(json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
                         for entry in entries)
    if format == "email":
        enc = lambda item: base64.urlsafe_b64encode(item.encode("utf-8")).decode("ascii")
        lines = [f"X-Container: {kind}"]
        lines.extend(f"X-Field-{enc(json.dumps(key, ensure_ascii=False))}: "
                     f"{enc(json.dumps(item, ensure_ascii=False))}" for key, item in rows)
        return "\n".join(lines)
    # Markdown-like table; cells are JSON strings with delimiters escaped.
    esc = lambda item: json.dumps(item, ensure_ascii=False, separators=(",", ":")).replace("|", "\\u007c")
    lines = ["| kind | key | value |", "|---|---|---|"]
    lines.append(f"| container |  | {esc(kind)} |")
    lines.extend(f"| entry | {esc(key)} | {esc(item)} |" for key, item in rows)
    return "\n".join(lines)


def _rebuild(kind: str, entries: list[tuple[Any, Any]]) -> Any:
    if kind == "scalar":
        if len(entries) != 1:
            raise ValueError("Scalar encoding must contain exactly one value")
        return entries[0][1]
    if kind == "list":
        ordered = sorted(entries, key=lambda pair: pair[0])
        if [key for key, _ in ordered] != list(range(len(ordered))):
            raise ValueError("Sequence keys must be contiguous integers starting at zero")
        return [value for _, value in ordered]
    if kind == "map":
        out = {}
        for key, value in entries:
            try:
                duplicate = key in out
            except TypeError as exc:
                raise ValueError(f"Mapping key {key!r} is not hashable") from exc
            if duplicate:
                raise ValueError(f"Invalid or duplicate mapping key {key!r}")
            out[key] = value
        return out
    raise ValueError(f"Unknown container marker {kind!r}")


def parse(text: str, format: str = "json") -> Any:
    """Parse any output of :func:`render_value`; malformed and unknown formats fail."""
    if format not in FORMATS:
        raise ValueError(f"Unknown structured text format {format!r}; expected one of {FORMATS}.")
    if format == "json":
        return json.loads(text)
    if format == "yaml":
        return yaml.safe_load(text)
    entries = []
    if format == "csv":
        rows = list(csv.reader(io.StringIO(text, newline="")))
        if not rows or rows[0] != ["kind", "key", "value"] or len(rows) < 2 or rows[1][0] != "container":
            raise ValueError("Malformed structured CSV")
        kind = json.loads(rows[1][2])
        for row in rows[2:]:
            if len(row) != 3 or row[0] != "entry":
                raise ValueError("Malformed structured CSV row")
            entries.append((json.loads(row[1]), json.loads(row[2])))
    elif format == "form":
        pairs = parse_qsl(text, keep_blank_values=True, strict_parsing=True)
        if not pairs or pairs[0][0] != "__kind__":
            raise ValueError("Missing form container marker")
        kind = pairs[0][1]
        entries = [(json.loads(key), json.loads(value)) for key, value in pairs[1:]]
    elif format == "kv":
        lines = text.splitlines()
        if not lines or not lines[0].startswith("@container: "):
            raise ValueError("Missing key/value container marker")
        kind = json.loads(lines[0].partition(": ")[2])
        for line in lines[1:]:
            key, sep, value = line.partition(": ")
            if not sep:
                raise ValueError("Malformed key/value row")
            entries.append((json.loads(key), json.loads(value)))
    elif format == "log":
        lines = [json.loads(line) for line in text.splitlines()]
        if not lines or set(lines[0]) != {"container"}:
            raise ValueError("Missing log container record")
        kind = lines[0]["container"]
        for item in lines[1:]:
            if set(item) != {"key", "value"}:
                raise ValueError("Malformed log entry")
            entries.append((item["key"], item["value"]))
    elif format == "email":
        lines = text.splitlines()
        if not lines or not lines[0].startswith("X-Container: "):
            raise ValueError("Missing email container header")
        kind = lines[0].partition(": ")[2]
        for line in lines[1:]:
            name, sep, value = line.partition(": ")
            if not sep or not name.startswith("X-Field-"):
                raise ValueError("Malformed email field")
            decode = lambda part: base64.urlsafe_b64decode(part.encode("ascii")).decode("utf-8")
            entries.append((json.loads(decode(name[8:])), json.loads(decode(value))))
    else:
        lines = text.splitlines()
        if len(lines) < 3 or lines[0] != "| kind | key | value |" or lines[1] != "|---|---|---|":
            raise ValueError("Malformed table header")
        first = lines[2][2:-2].split(" | ")
        if len(first) != 3 or first[0] != "container":
            raise ValueError("Missing table container row")
        kind = json.loads(first[2])
        for line in lines[3:]:
            fields = line[2:-2].split(" | ")
            if len(fields) != 3 or fields[0] != "entry":
                raise ValueError("Malformed table row")
            entries.append((json.loads(fields[1]), json.loads(fields[2])))
    return _rebuild(kind, entries)


@synth.node(type=synth.Text, tags=["text", "render"],
            example={"value": {"name": "Olena", "score": 7}, "format": "json"})
def render(r, *, value: Any, format: str = "json"):
    """Node factory converting a structured value to reversible formatted text."""
    return render_value(value, format)


__all__ = ["FORMATS", "parse", "render", "render_value"]
