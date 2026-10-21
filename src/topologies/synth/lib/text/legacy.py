"""Compatibility adapter from the former class grammar to the public node grammar."""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from topologies import synth

_ATTR = re.compile(r"\{\{\s*attrs\.([A-Za-z_]\w*)\s*\}\}")


def _as_record(value: Any, field: str) -> dict[str, Any]:
    if isinstance(value, Mapping) and isinstance(value.get("text"), str):
        text = value["text"]
        metadata = {}
        marker = "\ue100"
        if marker in text:
            text, encoded = text.rsplit(marker, 1)
            metadata = json.loads(encoded.removesuffix("\ue101"))
        out = {field: text, **metadata}
        if value.get("spans"):
            out["spans"] = value["spans"]
        return out
    if isinstance(value, str) and "\ue100" in value:
        text, encoded = value.rsplit("\ue100", 1)
        return {field: text, **json.loads(encoded.removesuffix("\ue101"))}
    return {field: value}


class Grammar:
    """Legacy constructor yielding a node-composed grammar (strings and entity slots).

    Historical condition/tag alternatives are deliberately rejected; express those
    controls as explicit node attributes and record fields in the modern API.
    """

    def __init__(self, rules: Mapping[str, Any], *, start: str = "S",
                 weights: Mapping[str, Sequence[float]] | None = None,
                 entities: Mapping[str, str] | None = None, field: str = "text",
                 max_depth: int = 12, name: str | None = None) -> None:
        self.rules = dict(rules)
        self.start, self.weights = start, dict(weights or {})
        self.entities, self.field = dict(entities or {}), field
        self.max_depth, self.name = int(max_depth), name
        labels = {label for alternatives in rules.values()
                  for alt in ([alternatives] if isinstance(alternatives, str) else alternatives)
                  if isinstance(alt, Mapping) for label in (alt.get("tags") or {})}
        fields: dict[str, Any] = {field: synth.Text()}
        if self.entities:
            fields["spans"] = synth.Spans(text_field=field, labels=sorted(set(self.entities.values())))
        for key in labels:
            fields[key] = synth.Type()
        self.produces = synth.Schema(fields)

    def as_generator(self):
        base_rules: dict[str, Any] = {}
        for symbol, raw_alts in self.rules.items():
            alts = [raw_alts] if isinstance(raw_alts, str) else list(raw_alts)
            normalized = []
            for i, item in enumerate(alts):
                metadata = {}
                if isinstance(item, Mapping):
                    if set(item) - {"text", "weight", "tags"}:
                        raise synth.SynthError("Legacy grammar adapter supports only text/weight alternatives.",
                                               stage="grammar",
                                               hint="Use the public node grammar and explicit nodes for conditions/tags.")
                    text = str(item["text"])
                    weight = float(item.get("weight", 1.0))
                    metadata = dict(item.get("tags") or {})
                else:
                    text, weight = str(item), 1.0
                if symbol in self.weights:
                    weight = float(self.weights[symbol][i])
                if symbol == self.start and metadata:
                    text += "\ue100" + json.dumps(metadata, ensure_ascii=False) + "\ue101"
                normalized.append((text, weight))
            base_rules[symbol] = normalized
        slots: dict[str, Any] = {}
        for symbol, label in self.entities.items():
            if symbol not in base_rules:
                raise synth.SynthError(f"Entity symbol {symbol!r} is not a grammar rule.", stage="grammar")
            slot = f"__entity_{symbol}"
            slots[slot] = synth.text.grammar(rules={k: v for k, v in base_rules.items()}, start=symbol,
                                             max_depth=self.max_depth, locale="en")
        prepared = {}
        for key, choices in base_rules.items():
            entries = []
            for text, weight in choices:
                text = _ATTR.sub(lambda m: "{" + m.group(1) + "}", text)
                for symbol, label in self.entities.items():
                    text = text.replace(f"<{symbol}>", f"{{__entity_{symbol}:{label}}}")
                entries.append((text, weight))
            prepared[key] = entries
        # Attributes referenced by the legacy template are provided as node parameters.
        attrs = sorted({m.group(1) for textlist in self.rules.values()
                        for text in ([textlist] if isinstance(textlist, str) else
                                     [x.get("text", "") if isinstance(x, Mapping) else str(x)
                                      for x in textlist]) for m in _ATTR.finditer(text)})
        for attr in attrs:
            slots.setdefault(attr, synth.attr(attr))
        document = synth.text.grammar(rules=prepared, start=self.start,
                                      max_depth=self.max_depth, slots=slots)
        return synth.apply(_as_record, document, self.field)

    def generate(self, ctx: Any, attrs: dict[str, Any]):
        """Legacy call surface; generation delegates to the composed public node graph."""
        value = synth.sample(self.as_generator(), ctx.record_index, seed=ctx.spec_seed, attrs=attrs)
        return synth.Raw(value)

    def fingerprint(self) -> str:
        return synth.fingerprint_of({"rules": self.rules, "start": self.start,
                                     "weights": self.weights, "entities": self.entities,
                                     "field": self.field, "max_depth": self.max_depth})


__all__ = ["Grammar"]
