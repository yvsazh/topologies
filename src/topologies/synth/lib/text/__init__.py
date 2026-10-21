"""``synth.text``: texts of any format and any language, with labels by construction.

    synth.text.template("Pay {{ amount:MONEY }} to {{ who:PER }}", amount=..., who=...)
    synth.text.fake("name", locale="uk"); synth.text.Faker("de", rng).iban()
    doc | synth.text.typos(0.03, lang="uk") | synth.text.casing("random")   # spans kept
    synth.text.pack("uk"); synth.text.languages(); synth.text.transliterate_text("Київ")

Language data lives in packs (``packs/<lang>/*.yaml``: faker data, keyboard
layouts, transliteration, lexicons, paradigms, number words); add a language by
adding a pack directory.
"""

from __future__ import annotations

from topologies.synth.lib.text.faker import KINDS, Faker, fake, locales, register_locale
from topologies.synth.lib.text.formats import parse, render, render_value
from topologies.synth.lib.text.generate import json_instance, regex
from topologies.synth.lib.text.grammar import grammar
from topologies.synth.lib.text.legacy import Grammar
from topologies.synth.lib.text.noise import (
    as_doc,
    casing,
    diacritics,
    eda,
    ocr,
    punctuation,
    transliterate,
    transliterate_text,
    typos,
    whitespace,
)
from topologies.synth.lib.text.numbers import (
    currency,
    date,
    format_currency,
    format_date,
    format_number,
    number,
    number_words,
)
from topologies.synth.lib.text.packs import Pack, languages, pack
from topologies.synth.lib.text.pools import persona_pool, personas, topic_pool, topics
from topologies.synth.lib.text.slots import template
from topologies.synth.lib.text.template import Template

__all__ = [
    "KINDS", "Faker", "Grammar", "Pack", "Template", "as_doc", "casing", "currency", "date",
    "diacritics", "eda", "fake", "format_currency", "format_date", "format_number", "grammar",
    "json_instance", "languages", "locales", "number", "number_words", "ocr", "pack", "parse",
    "persona_pool", "personas", "punctuation", "regex", "register_locale", "render",
    "render_value", "template", "topic_pool", "topics",
    "transliterate", "transliterate_text", "typos", "whitespace",
]
