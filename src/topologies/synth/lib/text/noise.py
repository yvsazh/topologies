"""Text noise that keeps entity spans exact.

Every transform takes a text or a document ``{"text": ..., "spans": [...]}`` and
returns the same kind; span offsets are remapped through every edit (and a span's
``"text"`` is updated), so NER labels stay exact after noise:

    doc = synth.text.template("Pay {{ amount:MONEY }} to {{ who:PER }}", amount=..., who=...)
    noisy = doc | synth.text.typos(0.03, lang="uk") | synth.text.casing("random")
    synth.record(text=noisy["text"], spans=noisy["spans"])

Transforms: ``typos`` (keyboard neighbours of the pack's layout, deletions,
doubled keys, swaps), ``ocr`` (look-alike confusions), ``casing``,
``diacritics`` (drop accents), ``transliterate`` (pack table), ``eda`` (word
swap / drop / duplicate outside spans), ``whitespace`` and ``punctuation``
noise. Edits never cut through a span boundary in a way that would lose it:
word-level operations do not touch words inside spans.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Sequence
from typing import Any

from topologies import synth
from topologies.synth.lib.text.packs import pack
from topologies.synth.lib.text.packs import transliterate as _translit

# ---------------------------------------------------------------- documents


def as_doc(x: Any) -> tuple[str, list[dict[str, Any]], bool]:
    """``(text, spans, was_doc)`` from a string or a ``{"text", "spans"}`` document."""
    if isinstance(x, str):
        return x, [], False
    if isinstance(x, dict) and "text" in x:
        return str(x["text"]), [dict(s) for s in x.get("spans") or []], True
    raise synth.SynthError(f"Text noise needs a string or {{'text', 'spans'}}; got "
                           f"{type(x).__name__}.", stage="text")


def _out(text: str, spans: list[dict[str, Any]], was_doc: bool, src: Any) -> Any:
    if not was_doc:
        return text
    out = {k: v for k, v in src.items() if k not in ("text", "spans")}
    out.update(text=text, spans=spans)
    return out


class Editor:
    """Character edits with origin tracking: ``out`` chars carry their source position
    (inserted chars sit between sources), so any original boundary maps exactly."""

    def __init__(self, text: str) -> None:
        self.src = text
        self.chars: list[str] = []
        self.origin: list[float] = []
        self.pos: dict[int, int] = {}

    def keep(self, i: int, ch: str | None = None) -> None:
        self.pos.setdefault(i, len(self.chars))
        self.chars.append(self.src[i] if ch is None else ch)
        self.origin.append(float(i))

    def insert(self, after: int, ch: str) -> None:
        self.chars.append(ch)
        self.origin.append(after + 0.5)

    def text(self) -> str:
        return "".join(self.chars)

    def boundary(self, b: int) -> int:
        return sum(1 for o in self.origin if o < b)

    def remap(self, spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
        text = self.text()
        out = []
        for s in spans:
            first, last = s["start"], s["end"] - 1
            if first in self.pos and last in self.pos and self.pos[last] >= self.pos[first]:
                # both ends survived: exact even when words were reordered around them
                a, b = self.pos[first], self.pos[last] + 1
                while b < len(self.origin) and first < self.origin[b] < last + 1:
                    b += 1  # chars inserted inside the span's last character group
            else:
                a, b = self.boundary(s["start"]), self.boundary(s["end"])
            if b <= a:
                continue
            new = {**s, "start": a, "end": b}
            if "text" in s:
                new["text"] = text[a:b]
            out.append(new)
        return out


def _inside(i: int, spans: Sequence[dict[str, Any]]) -> bool:
    return any(s["start"] <= i < s["end"] for s in spans)


def _char_edit(x: Any, fn: Callable[[Editor, int, str], None]) -> Any:
    text, spans, was = as_doc(x)
    ed = Editor(text)
    for i, ch in enumerate(text):
        fn(ed, i, ch)
    return _out(ed.text(), ed.remap(spans), was, x)


# ---------------------------------------------------------------- character noise


def _neighbours(lang: str) -> dict[str, str]:
    rows = pack(lang).keyboard.get("rows") or pack("en").keyboard.get("rows") or []
    out: dict[str, str] = {}
    for row in rows:
        for i, c in enumerate(row):
            out[c] = (row[i - 1] if i else "") + (row[i + 1] if i + 1 < len(row) else "")
    return out


@synth.node(tags=["text", "noise"], example={"rate": 0.05})
def typos(r, x, *, rate: float = 0.03, lang: str = "en", protect_spans: bool = False):
    """Keyboard typos: neighbour keys (pack layout), deletions, doubled keys, swaps."""
    near = _neighbours(lang)
    text, spans, was = as_doc(x)
    ed = Editor(text)
    i = 0
    while i < len(text):
        c = text[i]
        if c.isalpha() and not (protect_spans and _inside(i, spans)) and r.random() < rate:
            kind = r.randrange(4)
            if kind == 0:  # deletion
                i += 1
                continue
            if kind == 1 and i + 1 < len(text) and text[i + 1].isalpha():  # swap
                ed.keep(i + 1)
                ed.keep(i)
                i += 2
                continue
            if kind == 2:  # doubled key
                ed.keep(i)
                ed.insert(i, c)
                i += 1
                continue
            nb = near.get(c.lower())
            if nb:
                ch = r.choice(nb)
                ed.keep(i, ch.upper() if c.isupper() else ch)
                i += 1
                continue
        ed.keep(i)
        i += 1
    return _out(ed.text(), ed.remap(spans), was, x)


_OCR = {"0": "O", "O": "0", "o": "0", "1": "l", "l": "1", "I": "l", "5": "S", "S": "5",
        "8": "B", "B": "8", "e": "c", "c": "e", "о": "0", "з": "3", "б": "6", "і": "i",
        "rn": "m", "m": "rn", "cl": "d"}


@synth.node(tags=["text", "noise", "ocr"], example={"rate": 0.1})
def ocr(r, x, *, rate: float = 0.05):
    """OCR look-alike confusions (0/O, 1/l, rn/m, Cyrillic о/0, ...)."""
    text, spans, was = as_doc(x)
    ed = Editor(text)
    i = 0
    while i < len(text):
        two = text[i:i + 2]
        if two in _OCR and r.random() < rate:
            for k, ch in enumerate(_OCR[two]):
                if k == 0:
                    ed.keep(i, ch)
                else:
                    ed.insert(i, ch)
            i += 2
            continue
        c = text[i]
        if c in _OCR and r.random() < rate:
            rep = _OCR[c]
            ed.keep(i, rep[0])
            for ch in rep[1:]:
                ed.insert(i, ch)
        else:
            ed.keep(i)
        i += 1
    return _out(ed.text(), ed.remap(spans), was, x)


@synth.node(tags=["text", "noise"], example={"mode": "random"})
def casing(r, x, *, mode: str = "random"):
    """Casing: ``lower`` / ``upper`` / ``title`` / ``random`` (whole text) / ``chaos``
    (per character). Offsets never change."""
    text, spans, was = as_doc(x)
    if mode == "lower":
        new = text.lower()
    elif mode == "upper":
        new = text.upper()
    elif mode == "title":
        new = text.title()
    elif mode == "chaos":
        new = "".join(c.upper() if r.random() < 0.5 else c.lower() for c in text)
    else:
        new = text.lower() if r.random() < 0.5 else text[:1].upper() + text[1:]
    if len(new) != len(text):  # rare Unicode case changes that alter length: keep source
        new = text
    return _out(new, [{**s, **({"text": new[s["start"]:s["end"]]} if "text" in s else {})}
                      for s in spans], was, x)


@synth.node(tags=["text", "noise"], example={})
def diacritics(x, *, keep: str = ""):
    """Drop accents / combining marks (``é`` -> ``e``); letters in ``keep`` stay."""
    def edit(ed: Editor, i: int, ch: str) -> None:
        if ch in keep:
            ed.keep(i)
            return
        base = "".join(c for c in unicodedata.normalize("NFKD", ch)
                       if not unicodedata.combining(c))
        base = unicodedata.normalize("NFC", base) or ch
        ed.keep(i, base[0])
        for extra in base[1:]:
            ed.insert(i, extra)
    return _char_edit(x, edit)


@synth.node(tags=["text", "noise", "transliteration"], example={"lang": "uk"})
def transliterate(x, *, lang: str = "uk"):
    """Latin transliteration with the language pack's table (offsets remapped)."""
    table = pack(lang).translit

    def edit(ed: Editor, i: int, ch: str) -> None:
        low = ch.lower()
        if low not in table:
            ed.keep(i)
            return
        t = table[low]
        if not t:
            return
        t = t.capitalize() if ch != low else t
        ed.keep(i, t[0])
        for extra in t[1:]:
            ed.insert(i, extra)
    return _char_edit(x, edit)


@synth.node(tags=["text", "noise"], example={"rate": 0.1})
def whitespace(r, x, *, rate: float = 0.05):
    """Doubled / dropped spaces outside spans."""
    text, spans, was = as_doc(x)
    ed = Editor(text)
    for i, ch in enumerate(text):
        if ch == " " and not _inside(i, spans) and r.random() < rate:
            if r.random() < 0.5:
                continue
            ed.keep(i)
            ed.insert(i, " ")
        else:
            ed.keep(i)
    return _out(ed.text(), ed.remap(spans), was, x)


@synth.node(tags=["text", "noise"], example={"rate": 0.3})
def punctuation(r, x, *, rate: float = 0.2):
    """Dropped or doubled punctuation outside spans."""
    text, spans, was = as_doc(x)
    ed = Editor(text)
    for i, ch in enumerate(text):
        if ch in ".,!?;:" and not _inside(i, spans) and r.random() < rate:
            if r.random() < 0.6:
                continue
            ed.keep(i)
            ed.insert(i, ch)
        else:
            ed.keep(i)
    return _out(ed.text(), ed.remap(spans), was, x)


# ---------------------------------------------------------------- word-level (EDA)


_WORD = re.compile(r"\S+")


@synth.node(tags=["text", "noise", "eda"], example={"alpha": 0.2})
def eda(r, x, *, alpha: float = 0.1):
    """Easy data augmentation: random word swaps, deletions and duplications; words inside
    spans are never moved or removed (their spans stay exact)."""
    text, spans, was = as_doc(x)
    words = [(m.start(), m.end()) for m in _WORD.finditer(text)]
    if len(words) < 3:
        return x
    free = [k for k, (a, b) in enumerate(words)
            if not any(s["start"] < b and a < s["end"] for s in spans)]
    order = list(range(len(words)))
    dup: dict[int, int] = {}
    dropped: set[int] = set()
    for _ in range(max(1, int(alpha * len(words)))):
        if len(free) < 2:
            break
        op = r.randrange(3)
        if op == 0:
            i, j = r.sample(free, 2)
            pi, pj = order.index(i), order.index(j)
            order[pi], order[pj] = order[pj], order[pi]
        elif op == 1 and len(free) > 3:
            k = r.choice(free)
            dropped.add(k)
            free.remove(k)
        else:
            k = r.choice(free)
            dup[k] = dup.get(k, 0) + 1
    ed = Editor(text)
    first = True
    for k in order:
        if k in dropped:
            continue
        a, b = words[k]
        if not first:
            ed.insert(a - 1, " ")
        first = False
        for i in range(a, b):
            ed.keep(i)
        for _ in range(dup.get(k, 0)):
            ed.insert(b - 1, " ")
            for i in range(a, b):
                ed.insert(b - 1, text[i])
    new = ed.text()
    # spans: their words kept order and position relative to neighbouring span words
    return _out(new, ed.remap(spans), was, x)


# ---------------------------------------------------------------- plain-string helpers


def typos_text(text: str, rng: Any, rate: float = 0.03, lang: str = "en") -> str:
    return typos.per_record(rng, text, rate=rate, lang=lang)


def casing_text(text: str, rng: Any, mode: str = "random") -> str:
    return casing.per_record(rng, text, mode=mode)


def drop_diacritics_text(text: str) -> str:
    return diacritics.per_record(text)


def eda_text(text: str, rng: Any, alpha: float = 0.1) -> str:
    return eda.per_record(rng, text, alpha=alpha)


def transliterate_text(text: str, lang: str = "uk") -> str:
    return _translit(text, lang)


__all__ = ["Editor", "as_doc", "casing", "casing_text", "diacritics", "drop_diacritics_text",
           "eda", "eda_text", "ocr", "punctuation", "transliterate", "transliterate_text",
           "typos", "typos_text", "whitespace"]
