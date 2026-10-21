"""Record filters: length, language, heuristics, custom.

    synth.ops.filters.length(min_chars=40, max_chars=2000)       # or min_words / max_words
    synth.ops.filters.language("en")                             # built-in language ID
    synth.ops.filters.heuristic()                                # slot leaks, repetition
    synth.ops.filters.custom(fn)                                 # fn(record[, ctx]) -> bool|Verdict

Every filter takes ``field=`` (default: the record's text, see
``textutil.text_of``) and reports *why* it rejected a record.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

from topologies.synth.errors import SynthError
from topologies.synth.fingerprint import callable_fingerprint
from topologies.synth.ops.base import FilterOp, FnFilter
from topologies.synth.textutil import text_of, words
from topologies.synth.verdict import Verdict

_OK = Verdict(True)


class Length(FilterOp):
    evidence = "heuristic"

    def __init__(self, *, min_chars: int | None = None, max_chars: int | None = None,
                 min_words: int | None = None, max_words: int | None = None,
                 field: Any = None, name: str = "filters.length") -> None:
        if all(v is None for v in (min_chars, max_chars, min_words, max_words)):
            raise SynthError("filters.length needs at least one bound.", stage="filters",
                             hint="filters.length(min_chars=20, max_chars=2000)")
        self.min_chars, self.max_chars = min_chars, max_chars
        self.min_words, self.max_words = min_words, max_words
        self.field, self.name = field, name

    def check(self, record: dict, ctx: Any) -> Verdict:
        t = text_of(record, self.field)
        n = len(t)
        if self.min_chars is not None and n < self.min_chars:
            return Verdict(False, reason=f"{n} chars < {self.min_chars}")
        if self.max_chars is not None and n > self.max_chars:
            return Verdict(False, reason=f"{n} chars > {self.max_chars}")
        if self.min_words is not None or self.max_words is not None:
            w = len(words(t))
            if self.min_words is not None and w < self.min_words:
                return Verdict(False, reason=f"{w} words < {self.min_words}")
            if self.max_words is not None and w > self.max_words:
                return Verdict(False, reason=f"{w} words > {self.max_words}")
        return _OK

    def config(self) -> dict[str, Any]:
        return {"op": self.name, "min_chars": self.min_chars, "max_chars": self.max_chars,
                "min_words": self.min_words, "max_words": self.max_words,
                "field": self.field if not callable(self.field) else
                callable_fingerprint(self.field)}


class Language(FilterOp):
    evidence = "heuristic"

    def __init__(self, lang: str | Sequence[str] | None = None, *, attr: str | None = None,
                 min_confidence: float = 0.3, keep_unknown: bool = True, field: Any = None,
                 name: str = "filters.language") -> None:
        if lang is None and attr is None:
            raise SynthError("filters.language needs lang='en' (or a list) or attr='language' "
                             "(the expected language per record).", stage="filters")
        self.lang = [lang] if isinstance(lang, str) else (list(lang) if lang else None)
        self.attr = attr
        self.min_confidence, self.keep_unknown = float(min_confidence), bool(keep_unknown)
        self.field, self.name = field, name

    def check(self, record: dict, ctx: Any) -> Verdict:
        from topologies.synth.langid import detect

        allowed = self.lang
        if self.attr is not None:
            v = (record.get("_attrs") or {}).get(self.attr, record.get(self.attr))
            if v in (None, "mixed", "mixed-lang", "any"):
                return _OK
            allowed = [str(v)]
        got, conf = detect(text_of(record, self.field))
        if got == "unknown" or conf < self.min_confidence:
            return _OK if self.keep_unknown else Verdict(
                False, score=conf, reason=f"language unclear ({got}, {conf:.2f})")
        if allowed and got not in allowed:
            return Verdict(False, score=conf, reason=f"language {got} (conf {conf:.2f}), "
                                                     f"expected {'/'.join(allowed)}",
                           evidence={"detected": got})
        return Verdict(True, score=conf)

    def config(self) -> dict[str, Any]:
        return {"op": self.name, "lang": self.lang, "attr": self.attr,
                "min_confidence": self.min_confidence, "keep_unknown": self.keep_unknown,
                "field": self.field if not callable(self.field) else
                callable_fingerprint(self.field)}


#: Unfilled template slots and filler text (a template or grammar bug leaked through).
PLACEHOLDERS = [r"\{\{", r"\}\}", r"<\s*(label|text|input|output|name)\s*>", r"\[insert",
                r"\[your ", r"\blorem ipsum\b", r"\bxxx+\b", r"\{(text|label|input)\}"]

HEURISTIC_CHECKS = ("placeholder", "repetition", "alpha")


class Heuristic(FilterOp):
    """Cheap quality heuristics; ``checks=[...]`` picks which run (all by default)."""

    evidence = "heuristic"

    def __init__(self, *, checks: Sequence[str] | None = None, max_repetition: float = 0.3,
                 min_alpha: float = 0.3, field: Any = None, extra_patterns: Sequence[str] = (),
                 name: str = "filters.heuristic") -> None:
        chosen = list(checks) if checks is not None else list(HEURISTIC_CHECKS)
        bad = [c for c in chosen if c not in HEURISTIC_CHECKS]
        if bad:
            from topologies.synth.errors import unknown

            raise unknown("heuristic check", bad[0], HEURISTIC_CHECKS, stage="filters")
        self.checks = chosen
        self.max_repetition, self.min_alpha = float(max_repetition), float(min_alpha)
        self.field, self.name = field, name
        self.extra_patterns = list(extra_patterns)
        self._re = {k: [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in pats]
                    for k, pats in (("placeholder", PLACEHOLDERS + self.extra_patterns),)}

    def check(self, record: dict, ctx: Any) -> Verdict:
        t = text_of(record, self.field)
        for kind in ("placeholder",):
            if kind not in self.checks:
                continue
            for pat in self._re[kind]:
                m = pat.search(t)
                if m:
                    return Verdict(False, reason=f"{kind}: {m.group(0)[:40]!r}")
        if "repetition" in self.checks:
            w = words(t)
            if len(w) >= 9:
                tri = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
                rep = 1 - len(set(tri)) / len(tri)
                if rep > self.max_repetition:
                    return Verdict(False, score=rep, reason=f"repetitive ({rep:.0%} repeated "
                                                            "word 3-grams)")
            if re.search(r"(.)\1{19,}", t):
                return Verdict(False, reason="a character repeated 20+ times")
        if "alpha" in self.checks and t:
            alpha = sum(ch.isalpha() for ch in t) / len(t)
            if alpha < self.min_alpha:
                return Verdict(False, score=alpha, reason=f"only {alpha:.0%} letters")
        return _OK

    def config(self) -> dict[str, Any]:
        return {"op": self.name, "checks": self.checks, "max_repetition": self.max_repetition,
                "min_alpha": self.min_alpha, "extra": self.extra_patterns,
                "field": self.field if not callable(self.field) else
                callable_fingerprint(self.field)}


def length(**kw: Any) -> Length:
    return Length(**kw)


def language(lang: str | Sequence[str] | None = None, **kw: Any) -> Language:
    return Language(lang, **kw)


def heuristic(**kw: Any) -> Heuristic:
    return Heuristic(**kw)


def custom(fn: Callable[..., Any], *, name: str | None = None,
           evidence: str | None = None) -> FnFilter:
    """``fn(record[, ctx]) -> bool | Verdict`` as a filter op."""
    return FnFilter(fn, name=name, evidence=evidence)


__all__ = ["PLACEHOLDERS", "Heuristic", "Language", "Length",
           "custom", "heuristic", "language", "length"]


from topologies.synth.ops._callable import make_callable

make_callable(__name__, "custom")
