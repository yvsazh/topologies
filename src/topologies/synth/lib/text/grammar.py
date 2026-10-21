"""Weighted PCFG text nodes with data-driven agreement and span-safe finishing."""
from __future__ import annotations

import re
from typing import Any

from topologies import synth
from topologies.synth.lib.text.noise import Editor
from topologies.synth.lib.text.packs import pack
from topologies.synth.lib.text.slots import render as render_slots

_REF = re.compile(r"<([A-Za-z_][\w-]*)(?:\s+([^>]+))?>")
_SLOT = re.compile(r"\{([A-Za-z_][\w]*)(?::([A-Za-z_][\w-]*))?\}")
_ATTR = re.compile(r"([\w-]+)=([\w$-]+)")
_FEATURES = {"gender", "case", "number", "person"}


def _alternatives(value: Any) -> list[tuple[str, float]]:
    values = value if isinstance(value, list) else [value]
    out = []
    for item in values:
        text, weight = item if isinstance(item, (tuple, list)) else (item, 1.0)
        if not isinstance(text, str) or float(weight) <= 0:
            raise synth.SynthError("Grammar alternatives need text and positive weights.", stage="text")
        out.append((text, float(weight)))
    if not out:
        raise synth.SynthError("Grammar rules need at least one alternative.", stage="text")
    return out


def _features(text: str) -> dict[str, str]:
    return dict(_ATTR.findall(text))


def _postprocess(doc: dict[str, Any], processors: list[str]) -> dict[str, Any]:
    for processor in processors:
        text = doc["text"]
        spans = doc["spans"]
        if processor in {"upper", "lower", "capitalize"}:
            editor = Editor(text)
            for i, char in enumerate(text):
                replacement = char.upper() if processor == "upper" or (
                    processor == "capitalize" and i == 0) else char.lower() if processor == "lower" \
                    else char
                if replacement:
                    editor.keep(i, replacement[0])
                    for extra in replacement[1:]:
                        editor.insert(i, extra)
            new = editor.text()
            doc = {**doc, "text": new, "spans": editor.remap(spans)}
        elif processor == "sentence_punctuation":
            if text and text[-1] not in ".!?。！？":
                doc = {**doc, "text": text + "."}
        elif processor in {"collapse_spaces", "trim"}:
            ed = Editor(text)
            if processor == "collapse_spaces":
                i = 0
                while i < len(text):
                    ed.keep(i)
                    if text[i].isspace():
                        j = i + 1
                        while j < len(text) and text[j].isspace():
                            j += 1
                        i = j
                    else:
                        i += 1
            else:
                left = len(text) - len(text.lstrip())
                right = len(text.rstrip())
                for i in range(left, right):
                    ed.keep(i)
            new = ed.text()
            remapped = ed.remap(spans)
            doc = {**doc, "text": new, "spans": remapped}
        else:
            raise synth.SynthError(f"Unknown grammar postprocessor {processor!r}.", stage="text",
                                   available=["capitalize", "upper", "lower", "sentence_punctuation",
                                              "collapse_spaces", "trim"])
    return doc


@synth.node(type=synth.Text, tags=["text", "grammar"],
            example={"rules": {"S": ["Hello <name>.", ("Hi <name>!", 2)]}, "locale": "en"})
def grammar(r, *, rules: dict[str, Any], start: str = "S", max_depth: int = 8,
            locale: str = "en", slots: dict[str, Any] | None = None,
            features: dict[str, str] | None = None, postprocessors: list[str] | None = None):
    """Expand a weighted PCFG. Feature attrs inherit and unify through nonterminals;
    ``$name`` binds/unifies a feature. Lexical case/number forms come from pack paradigms.
    Slot markers ``{key}`` / ``{key:LABEL}`` produce text / exact-span documents."""
    if start not in rules or max_depth < 0:
        raise synth.SynthError("Grammar start must exist and max_depth must be nonnegative.", stage="text")
    alternatives = {name: _alternatives(value) for name, value in rules.items()}
    lexicon = pack(locale).lexicon
    paradigms = pack(locale).paradigms
    slot_values = dict(slots or {})
    # Minimum derivation height identifies alternatives that can terminate before the bound.
    minimum: dict[str, int] = {}
    for _ in range(len(alternatives) + 1):
        changed = False
        for name, choices in alternatives.items():
            costs = []
            for text, _ in choices:
                refs = [m.group(1) for m in _REF.finditer(text) if m.group(1) in alternatives]
                if all(ref in minimum for ref in refs):
                    costs.append(1 + max((minimum[x] for x in refs), default=0))
            if costs and (name not in minimum or min(costs) < minimum[name]):
                minimum[name] = min(costs)
                changed = True
        if not changed:
            break
    if start not in minimum:
        raise synth.SynthError(f"Grammar symbol {start!r} has no terminating derivation.", stage="text")

    def expand(symbol: str, depth: int, inherited: dict[str, str], bindings: dict[str, str]) -> str:
        if depth + minimum[symbol] - 1 > max_depth:
            raise synth.SynthError(f"Cannot terminate {symbol!r} by max_depth={max_depth}.",
                                   stage="text", hint="Raise max_depth or add a shorter terminating rule.")
        viable = []
        for text, weight in alternatives[symbol]:
            refs = [m.group(1) for m in _REF.finditer(text) if m.group(1) in alternatives]
            if all(depth + minimum.get(ref, max_depth + 1) <= max_depth for ref in refs):
                viable.append((text, weight))
        if not viable:
            raise synth.SynthError(f"No terminating alternative for {symbol!r} at depth {depth}.",
                                   stage="text", hint="Raise max_depth or add a terminal alternative.")
        text, _ = r.weighted(viable, [weight for _, weight in viable])
        out, pos = [], 0
        for match in _REF.finditer(text):
            out.append(text[pos:match.start()])
            name = match.group(1)
            local = _features(match.group(2) or "")
            requested = dict(inherited)
            requested.update({k: v for k, v in local.items() if k in _FEATURES})
            resolved = {}
            for key, value in requested.items():
                if value.startswith("$"):
                    var = value[1:]
                    if var in bindings:
                        resolved[key] = bindings[var]
                else:
                    resolved[key] = value
            if name in alternatives:
                for key, value in requested.items():
                    if value.startswith("$") and value[1:] in bindings:
                        requested[key] = bindings[value[1:]]
                out.append(expand(name, depth + 1, requested, bindings))
            else:
                entries = lexicon.get(name, [])
                eligible = [entry for entry in entries if all(
                    str(entry.get(k, "")) == v for k, v in resolved.items()
                    if k in {"gender", "person", "number"})]
                if not eligible:
                    raise synth.SynthError(f"No {name} lexicon entry matches {resolved or requested}.",
                                           stage="text")
                entry = r.choice(eligible)
                for key, value in requested.items():
                    if value.startswith("$"):
                        bindings[value[1:]] = str(entry.get(key, resolved.get(key, "")))
                lemma = str(entry.get("lemma", entry["word"]))
                case = resolved.get("case", entry.get("case"))
                number = resolved.get("number", entry.get("number"))
                form_keys = [f"{case}.{number}" if case and number else None, case, number]
                form_table = paradigms.get(lemma, {})
                word = str(entry["word"])
                for form_key in form_keys:
                    if form_key and form_key in form_table:
                        word = str(form_table[form_key])
                        break
                out.append(word)
            pos = match.end()
        out.append(text[pos:])
        return "".join(out)

    raw = expand(start, 0, dict(features or {}), {})
    if _SLOT.search(raw):
        missing = [m.group(1) for m in _SLOT.finditer(raw) if m.group(1) not in slot_values]
        if missing:
            raise synth.SynthError(f"Grammar references unbound slots {missing}.", stage="text")
        marked = _SLOT.sub(lambda m: "{{ " + m.group(1) +
                           (":" + m.group(2) if m.group(2) else "") + " }}", raw)
        rendered = render_slots(marked, slot_values)
        doc = rendered if isinstance(rendered, dict) else {"text": rendered, "spans": []}
        doc = _postprocess(doc, list(postprocessors or []))
        return doc if doc["spans"] else doc["text"]
    doc = _postprocess({"text": raw, "spans": []}, list(postprocessors or []))
    return doc["text"]


__all__ = ["grammar"]
