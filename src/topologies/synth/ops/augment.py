"""Augmentations: label-preserving variants of records (pipeline ``flat_map`` ops).

    synth.augment.typos(rate=0.03)            # deletions, swaps, doubled / keyboard-adjacent keys
    synth.augment.casing("random")            # lower / upper / random / title
    synth.augment.drop_diacritics()
    synth.augment.eda(alpha=0.1)              # random word swap / deletion / duplication
    synth.augment.entity_swap({"PER": names, "PRODUCT": products})   # spans stay aligned
    synth.ops.augment(synth.augment.typos(0.02), synth.augment.casing("lower"), n=2)

Every variant keeps a ``parent`` link and the parent's group (so a group-aware
split keeps all variants of one source together) and inherits the labels.
``keep_original=True`` (default) keeps the source record as well. Randomness
comes from the per-record op RNG, so augmentation is reproducible.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from topologies.synth.errors import SynthError
from topologies.synth.fingerprint import callable_fingerprint, fp_canonical
from topologies.synth.lib.text.noise import (
    casing_text,
    drop_diacritics_text,
    eda_text,
    typos_text,
)
from topologies.synth.ops.base import FlatMapOp


class TextAugment(FlatMapOp):
    """Base for text augmentations over one text ``field``."""

    inherit = True

    def __init__(self, fn: Callable[[str, random.Random], str], *, field: str | None = None,
                 n: int = 1, keep_original: bool = True, name: str = "augment") -> None:
        self.fn, self.field, self.n = fn, field, int(n)
        self.keep_original, self.name = bool(keep_original), name

    def _field(self, record: Mapping[str, Any]) -> str:
        if self.field:
            return self.field
        for k, v in record.items():
            if not str(k).startswith("_") and isinstance(v, str):
                return k
        raise SynthError("augment needs a text field.", stage="augment",
                         hint="Pass field='text'.")

    def expand(self, record: dict, ctx: Any) -> list[dict]:
        f = self._field(record)
        rng = ctx.rng
        out = [record] if self.keep_original else []
        seen = {record.get(f)}
        for _ in range(self.n):
            new = self.fn(str(record.get(f, "")), rng)
            if new and new not in seen:
                seen.add(new)
                out.append({**record, f: new})
        return out

    def config(self) -> dict[str, Any]:
        return {"op": self.name, "fn": callable_fingerprint(self.fn), "field": self.field,
                "n": self.n, "keep_original": self.keep_original}


def typos(rate: float = 0.03, **kw: Any) -> TextAugment:
    return TextAugment(lambda t, rng: typos_text(t, rng, rate), name="augment.typos", **kw)


def casing(mode: str = "random", **kw: Any) -> TextAugment:
    return TextAugment(lambda t, rng: casing_text(t, rng, mode), name="augment.casing", **kw)


def drop_diacritics(**kw: Any) -> TextAugment:
    return TextAugment(lambda t, rng: drop_diacritics_text(t), name="augment.diacritics", **kw)


def eda(alpha: float = 0.1, **kw: Any) -> TextAugment:
    return TextAugment(lambda t, rng: eda_text(t, rng, alpha), name="augment.eda", **kw)


class Compose(FlatMapOp):
    """``n`` variants, each applying every transform in order (``synth.ops.augment``)."""

    def __init__(self, *transforms: TextAugment, n: int = 1, keep_original: bool = True,
                 field: str | None = None) -> None:
        if not transforms:
            raise SynthError("augment(...) needs at least one transform.", stage="augment")
        self.transforms, self.n = list(transforms), int(n)
        self.keep_original, self.field = bool(keep_original), field
        self.name = "augment"

    def expand(self, record: dict, ctx: Any) -> list[dict]:
        f = self.field or self.transforms[0]._field(record)
        out = [record] if self.keep_original else []
        seen = {record.get(f)}
        for _ in range(self.n):
            t = str(record.get(f, ""))
            for tr in self.transforms:
                t = tr.fn(t, ctx.rng)
            if t not in seen:
                seen.add(t)
                out.append({**record, f: t})
        return out

    def config(self) -> dict[str, Any]:
        return {"op": "augment", "transforms": [t.config() for t in self.transforms],
                "n": self.n, "keep_original": self.keep_original, "field": self.field}


def augment_op(*transforms: TextAugment, n: int = 1, keep_original: bool = True,
               field: str | None = None) -> Compose:
    return Compose(*transforms, n=n, keep_original=keep_original, field=field)


class EntitySwap(FlatMapOp):
    """Replace entity spans with other values of the same type; offsets are recomputed."""

    def __init__(self, pools: Mapping[str, Sequence[str]], *, text_field: str = "text",
                 spans_field: str = "spans", n: int = 1, keep_original: bool = True) -> None:
        self.pools = {k: list(v) for k, v in pools.items()}
        self.text_field, self.spans_field = text_field, spans_field
        self.n, self.keep_original = int(n), bool(keep_original)
        self.name = "augment.entity_swap"

    def swap(self, record: Mapping[str, Any], rng: random.Random) -> dict[str, Any]:
        text = str(record.get(self.text_field, ""))
        spans = sorted(record.get(self.spans_field) or [], key=lambda s: s["start"])
        out, new_spans, pos, shift = [], [], 0, 0
        for s in spans:
            a, b = int(s["start"]), int(s["end"])
            out.append(text[pos:a])
            pool = self.pools.get(s["label"])
            rep = rng.choice(pool) if pool else text[a:b]
            start = a + shift
            out.append(rep)
            new_spans.append({**s, "start": start, "end": start + len(rep), "text": rep})
            shift += len(rep) - (b - a)
            pos = b
        out.append(text[pos:])
        return {**record, self.text_field: "".join(out), self.spans_field: new_spans}

    def expand(self, record: dict, ctx: Any) -> list[dict]:
        out = [record] if self.keep_original else []
        for _ in range(self.n):
            out.append(self.swap(record, ctx.rng))
        return out

    def config(self) -> dict[str, Any]:
        return {"op": self.name, "pools": fp_canonical(self.pools), "text": self.text_field,
                "spans": self.spans_field, "n": self.n, "keep_original": self.keep_original}


def entity_swap(pools: Mapping[str, Sequence[str]], **kw: Any) -> EntitySwap:
    return EntitySwap(pools, **kw)


_WS = re.compile(r"\s+")

__all__ = ["Compose", "EntitySwap", "TextAugment", "augment_op", "casing", "casing_text",
           "drop_diacritics", "drop_diacritics_text", "eda", "eda_text", "entity_swap",
           "typos", "typos_text"]


from topologies.synth.ops._callable import make_callable

make_callable(__name__, "augment_op")
