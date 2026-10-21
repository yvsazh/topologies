"""Data-backed combinatorial language pools."""
from __future__ import annotations

import itertools
import random

from topologies import synth
from topologies.synth.lib.text.packs import pack


def _pool(name: str, locale: str) -> list[str]:
    return list(pack(locale).section("pools").get(name, []))


@synth.node(tags=["text", "pool"], example={"locale": "en"})
def personas(r, *, locale: str = "en"):
    """Sample a persona string from the locale pack."""
    values = _pool("personas", locale)
    if not values:
        raise synth.SynthError(f"Pack {locale!r} has no personas pool.", stage="text")
    return r.choice(values)


@synth.node(tags=["text", "pool"], example={"locale": "en"})
def topics(r, *, locale: str = "en"):
    """Sample a topic string from the locale pack."""
    values = _pool("topics", locale)
    if not values:
        raise synth.SynthError(f"Pack {locale!r} has no topics pool.", stage="text")
    return r.choice(values)


def persona_pool(n: int = 200, *, traits: dict[str, list[str]] | None = None,
                 seed: int = 0, locale: str = "en"):
    """Legacy attribute-pool factory backed by the locale pack and public pool API."""
    data = pack(locale).section("pools")
    defaults = data.get("traits", {})
    rows = {**defaults, **(traits or {})}
    if not rows:
        rows = {"persona": data.get("personas", [])}
    combos = list(itertools.product(*(rows[k] for k in rows)))
    random.Random(seed).shuffle(combos)
    values = [", ".join(map(str, combo)) for combo in combos[:int(n)]]
    return synth.pool(values, name=f"personas({n})")


def topic_pool(n: int = 50, *, seed_topics: list[str] | None = None, seed: int = 0,
               locale: str = "en"):
    """Legacy topic attribute pool backed by locale pack topic/facet data."""
    data = pack(locale).section("pools")
    seeds = list(seed_topics or data.get("topics", []))
    facets = data.get("facets", [])
    values = seeds + [f"{topic}: {facet}" for topic in seeds for facet in facets]
    random.Random(seed).shuffle(values)
    return synth.pool(values[:int(n)], name=f"topics({n})")


__all__ = ["persona_pool", "personas", "topic_pool", "topics"]
