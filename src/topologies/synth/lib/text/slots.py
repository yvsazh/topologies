"""Templates with node-bound slots and exact span labels.

    doc = synth.text.template(
        "Переказ {{ amount:MONEY }} на рахунок {{ who:PER }} ({{ city }}).",
        amount=synth.text.fake("money", locale="uk"), who=synth.text.fake("name", locale="uk"),
        city=synth.text.fake("city", locale="uk"))
    # -> {"text": "...", "spans": [{"start", "end", "label": "MONEY", "text"}, ...]}

``{{ slot:LABEL }}`` marks a labeled slot: its character offsets in the rendered text
are recorded exactly. A slot whose value is itself a document ``{"text", "spans"}``
(another template, a grammar) is inserted with its spans shifted. Without labeled
slots or documents the value is a plain string. Everything else is the safe
template engine (``{% if %}``, ``{% for %}``, filters; see ``Template``).
"""

from __future__ import annotations

import re
from typing import Any

from topologies import synth
from topologies.synth.lib.text.template import Template

_OPEN, _MID, _CLOSE = "", "", ""
_LABELED = re.compile(r"\{\{\s*([A-Za-z_][\w.]*)\s*:\s*([A-Za-z_][\w-]*)\s*((?:\|[^}]*)?)\}\}")
_CACHE: dict[str, Template] = {}


def _compile(text: str) -> Template:
    t = _CACHE.get(text)
    if t is None:
        marked = _LABELED.sub(lambda m: f"{_OPEN}L:{m.group(2)}{_MID}{{{{ {m.group(1)} "
                                        f"{m.group(3)}}}}}{_CLOSE}", text)
        t = _CACHE[text] = Template(marked)
    return t


def _is_doc(v: Any) -> bool:
    return isinstance(v, dict) and "text" in v and isinstance(v.get("spans"), list)


def render(text: str, slots: dict[str, Any]) -> Any:
    """Render ``text`` with ``slots`` (plain values / documents); see module docstring."""
    inner: list[list[dict[str, Any]]] = []
    ctx: dict[str, Any] = {}
    for k, v in slots.items():
        if _is_doc(v):
            inner.append(v["spans"])
            ctx[k] = f"{_OPEN}D:{len(inner) - 1}{_MID}{v['text']}{_CLOSE}"
        else:
            ctx[k] = v
    raw = _compile(text).render(**ctx)
    if _OPEN not in raw:
        return raw
    out: list[str] = []
    spans: list[dict[str, Any]] = []
    stack: list[tuple[str, int]] = []
    i = 0
    n = 0
    while i < len(raw):
        ch = raw[i]
        if ch == _OPEN:
            j = raw.index(_MID, i)
            stack.append((raw[i + 1:j], n))
            i = j + 1
            continue
        if ch == _CLOSE and stack:
            tag, start = stack.pop()
            kind, _, val = tag.partition(":")
            if kind == "L":
                spans.append({"start": start, "end": n, "label": val})
            else:
                for s in inner[int(val)]:
                    spans.append({**s, "start": s["start"] + start, "end": s["end"] + start})
            i += 1
            continue
        out.append(ch)
        n += 1
        i += 1
    text_out = "".join(out)
    spans = [{**s, "text": text_out[s["start"]:s["end"]]} for s in spans if s["end"] > s["start"]]
    spans.sort(key=lambda s: (s["start"], s["end"]))
    return {"text": text_out, "spans": spans}


@synth.node(tags=["text", "template", "ner"],
            example={"text": "Hello {{ name:PER }}!", "name": "Olena"})
def template(text: str, **slots: Any):
    """Render a safe template with node-bound slots; ``{{ slot:LABEL }}`` records exact spans
    (the value is then ``{"text", "spans"}``)."""
    return render(text, slots)


__all__ = ["render", "template"]
