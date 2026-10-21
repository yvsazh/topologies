"""Algorithmic text, audio, math and graph datasets with construction-time labels."""
from __future__ import annotations

import math
from typing import Any

from topologies import synth
from topologies.synth.blueprints._base import blueprint, default_splits, spec_kwargs


@synth.node(tags=["blueprint", "text"], example={"kind": "intent", "language": "en"})
def _text_record(r, *, ctx, blueprint: str, language: str, label: str,
                 noise: float, length: int, register: str,
                 intent_templates: Any = None):
    from topologies.synth.lib.text import Faker, template, typos
    faker = Faker(language, r)
    if blueprint == "text_classification":
        defaults = {
            "en": {"billing": "I need help with my bill", "shipping": "Where is my package",
                   "refund": "Please refund my order", "account": "I cannot sign in"},
            "uk": {"billing": "Допоможіть з моїм рахунком", "shipping": "Де моя посилка",
                   "refund": "Поверніть, будь ласка, моє замовлення", "account": "Не можу увійти в акаунт"},
        }
        table = dict(defaults.get(language, defaults["en"]))
        if intent_templates:
            custom = intent_templates.get(language, intent_templates)
            table.update(custom)
        phrase = table.get(label, f"Please help me with {label}")
        if language == "en":
            opening = r.choice(("Dear support", "Good afternoon")) if register == "formal" else r.choice(("Hello", "Hi"))
            closing = r.choice(("Sincerely", "Thank you for your assistance")) if register == "formal" else r.choice(("Thanks", "Please reply soon"))
        else:
            opening = r.choice(("Шановна службo підтримки", "Добрий день")) if register == "formal" else r.choice(("Привіт", "Вітаю"))
            closing = r.choice(("З повагою", "Щиро дякую")) if register == "formal" else r.choice(("Дякую", "Будь ласка, відповідайте"))
        text = f"{opening}, {faker.first_name()}! {phrase}. {closing}."
        if len(text) < length:
            text += " " + ("I would appreciate your assistance. " if language == "en" else "Буду вдячний за допомогу. ") * max(0, length // 36)
        doc = ctx.sample(typos(x=text, rate=noise, lang=language)) if noise else text
        return {"text": doc[:length], "label": label, "language": language, "register": register}
    if language == "uk":
        pattern = "Зв’яжіться з {{ person:PERSON }} у місті {{ city:LOCATION }} щодо замовлення {{ code:ORDER }}."
    else:
        pattern = "Contact {{ person:PERSON }} in {{ city:LOCATION }} about order {{ code:ORDER }}."
    doc = ctx.sample(template(text=pattern, person=faker.first_name(), city=faker.city(),
                              code=faker.id("ORD-", 6)))
    return ctx.sample(typos(x=doc, rate=noise, lang=language, protect_spans=True)) if noise else doc


@blueprint(example={"labels": ["billing", "shipping", "refund"], "language": "en", "size": 500})
def text_classification(name: str, *, labels: Any = ("billing", "shipping", "refund"),
                        language: str = "en", noise: float = 0.0, length: int = 96,
                        register: str = "neutral", templates: Any = None,
                        vary: Any = None, size: int = 1000,
                        splits: Any = None, **spec: Any):
    from topologies.synth.spec import Spec
    values = tuple(labels)
    attrs = {"label": list(values), "register": [register], "length": [length], "noise": [noise]}
    attrs.update(dict(vary or {}))
    return Spec(name, generator=_text_record(blueprint="text_classification", language=language,
                label=synth.attr("label"), noise=synth.attr("noise"), length=synth.attr("length"),
                register=synth.attr("register"), intent_templates=templates), attributes=attrs, size=size,
                splits=splits or default_splits("label"), **spec_kwargs(spec))


@blueprint(example={"language": "uk", "size": 500}, aliases=("span_extraction",))
def ner(name: str, *, language: str = "en", noise: float = 0.0, size: int = 1000,
        splits: Any = None, **spec: Any):
    from topologies.synth.spec import Spec
    return Spec(name, generator=_text_record(blueprint="ner", language=language,
                label="entity", noise=noise, length=96, register="neutral"), size=size,
                splits=splits or default_splits(), **spec_kwargs(spec))


@synth.node(tags=["blueprint", "text", "structured"], example={"format": "json"})
def _structured(r, *, ctx, format: str, noise: float, language: str):
    from topologies.synth.lib.text import Faker, parse, render_value, typos
    f = Faker(language, r)
    value = {"customer": f.first_name(), "invoice": f.id("INV-", 6),
             "amount": round(f.amount(100), 2), "currency": "USD"}
    text = render_value(value, format)
    # Whitespace doubling is safe only for JSON strings? Keep formatting intact and
    # use casing-free character errors only when the selected grammar tolerates it.
    if noise and format in ("kv", "csv", "yaml", "table"):
        text = ctx.sample(typos(x=text, rate=noise, lang=language))
    parsed = parse(text, format)
    if parsed != value:
        raise ValueError("Rendered record failed exact parser round trip")
    return {"text": text, "record": value, "parsed": parsed}


@blueprint(example={"format": "json", "size": 500})
def structured_extraction(name: str, *, format: str = "json", noise: float = 0.0,
                           language: str = "en", size: int = 1000, splits: Any = None, **spec: Any):
    from topologies.synth.spec import Spec
    return Spec(name, generator=_structured(format=format, noise=noise, language=language), size=size,
                splits=splits or default_splits(), **spec_kwargs(spec))


@synth.node(tags=["blueprint", "text", "transform"], example={"task": "transliteration"})
def _transform(r, *, task: str, language: str, source_format: str, target_format: str,
               input_unit: str, output_unit: str):
    from topologies.synth.lib.text import (
        format_currency,
        format_date,
        number_words,
        parse,
        render_value,
    )
    from topologies.synth.lib.text.packs import transliterate
    val = r.randint(12, 999)
    if task == "number_words":
        source, target = str(val), number_words(val, language)
    elif task == "date":
        source = "2024-03-17"
        target = format_date(source, language)
    elif task == "currency":
        amount = round(val + r.random(), 2)
        source, target = f"{amount:.2f} USD", format_currency(amount, "USD", language)
    elif task == "transliteration":
        source = "Київ та Львів" if language == "uk" else "sample words"
        target = transliterate(source, language) if language == "uk" else source
    elif task == "casing":
        source, target = "Synthetic Data", "SYNTHETIC DATA"
    elif task == "unit":
        source = f"{val} {input_unit}"
        factors = {"m": 1.0, "cm": .01, "km": 1000.0, "ft": .3048}
        if input_unit not in factors or output_unit not in factors:
            raise ValueError("unit conversion supports m, cm, km and ft")
        target = f"{val * factors[input_unit] / factors[output_unit]:g} {output_unit}"
    elif task == "format":
        obj = {"count": val, "language": language}
        source = render_value(obj, source_format)
        target = render_value(parse(source, source_format), target_format)
    else:
        raise ValueError("task must be number_words/date/currency/transliteration/casing/unit/format")
    return {"source": source, "target": target, "task": task, "language": language}


@blueprint(example={"task": "transliteration", "language": "uk", "size": 500})
def seq2seq_transform(name: str, *, task: str = "transliteration", language: str = "en",
                      source_format: str = "csv", target_format: str = "json",
                      input_unit: str = "km", output_unit: str = "m",
                      size: int = 1000, splits: Any = None, **spec: Any):
    from topologies.synth.spec import Spec
    return Spec(name, generator=_transform(task=task, language=language, source_format=source_format,
                target_format=target_format, input_unit=input_unit, output_unit=output_unit), size=size,
                splits=splits or default_splits(),
                **spec_kwargs(spec))


@synth.node(tags=["blueprint", "math", "text"], example={"language": "en"})
def _math_problem(r, *, ctx, language: str, kinds: tuple[str, ...]):
    from topologies.synth.lib.math import word_problem
    return ctx.sample(word_problem(language=language, kinds=kinds))


@blueprint(example={"language": "en", "size": 500})
def math_word_problems(name: str, *, language: str = "en", kinds: Any = ("arithmetic", "percent", "rate"),
                       size: int = 1000, splits: Any = None, **spec: Any):
    from topologies.synth.spec import Spec
    return Spec(name, generator=_math_problem(language=language, kinds=tuple(kinds)), size=size,
                splits=splits or default_splits(), **spec_kwargs(spec))


@synth.node(tags=["blueprint", "text", "corpus"], example={"language": "en"})
def _story(r, *, ctx, language: str, vocabulary: tuple[str, ...], sentences: int,
           depth: int, recursion_weight: float):
    from topologies.synth.lib.text import grammar
    if not vocabulary or sentences < 1 or depth < 0:
        raise synth.SynthError("Corpus vocabulary/sentences must be nonempty and depth nonnegative.",
                               stage="blueprint")
    verbs = ("found", "followed", "helped", "carried") if language == "en" else (
        "знайшов", "зустрів", "допоміг", "поніс")
    glue = ", then " if language == "en" else ", а потім "
    # A terminating clause and weighted recursive clause form a finite PCFG. Each
    # sentence has fresh lexical slots but the same editable grammar structure.
    rules: dict[str, Any] = {"S": " ".join(f"<C{i}>" for i in range(sentences))}
    slots = {}
    for i in range(sentences):
        slots.update({f"s{i}": r.choice(vocabulary), f"o{i}": r.choice(vocabulary),
                      f"v{i}": r.choice(verbs)})
        body = f"{{s{i}}} {{v{i}}} {{o{i}}}"
        alternatives: list[Any] = [body + "."]
        if depth:
            alternatives.append((body + glue + f"<C{i}>", recursion_weight))
        rules[f"C{i}"] = alternatives
    generated = ctx.sample(grammar(rules=rules, max_depth=depth + 1,
                                   locale=language, slots=slots))
    return {"text": str(generated).capitalize(), "language": language,
            "sentence_count": sentences, "grammar_depth": depth}


@blueprint(example={"language": "en", "depth": 2, "size": 500})
def lm_corpus(name: str, *, language: str = "en", vocabulary: Any = None, sentences: int = 3,
              depth: int = 2, recursion_weight: float = 0.4, size: int = 1000,
              splits: Any = None, **spec: Any):
    from topologies.synth.spec import Spec
    vocab = tuple(vocabulary or (("fox", "child", "key", "garden", "friend", "bird") if language == "en" else ("кіт", "дитина", "ключ", "сад", "друг", "птах")))
    if language not in ("en", "uk"):
        raise synth.unknown("corpus language", language, ["en", "uk"], stage="blueprint")
    return Spec(name, generator=_story(language=language, vocabulary=vocab, sentences=sentences,
                depth=depth, recursion_weight=recursion_weight), size=size,
                splits=splits or default_splits(), **spec_kwargs(spec))


@synth.node(tags=["blueprint", "audio", "ddsp"], example={"sr": 8000})
def _audio_record(r, *, ctx, task: str, sr: int, seconds: float, effects: bool,
                  curve: str = "constant", pitch_depth: float = 3.0):
    from topologies.synth.lib import audio
    f0 = float(r.choice((110, 146.83, 196, 220, 293.66, 392)))
    if task == "pitch":
        count = 32
        if curve == "constant":
            controls = [f0] * count
        elif curve == "glide":
            end = f0 * 2 ** (pitch_depth / 12.0)
            controls = [f0 + (end - f0) * i / (count - 1) for i in range(count)]
        elif curve == "vibrato":
            controls = [f0 * 2 ** (pitch_depth * math.sin(2 * math.pi * 4 * i / count) / 12.0)
                        for i in range(count)]
        else:
            raise synth.unknown("pitch curve", curve, ["constant", "glide", "vibrato"],
                                stage="blueprint")
        source = audio.ddsp(f0=controls, loudness=[0.8] * count, harmonics=8,
                            noise=0.0, seconds=seconds, sr=sr)
        value = ctx.sample(source)
        ctx.latent(f0_curve=controls, f0=f0, curve=curve)
        return {"audio": value, "f0_curve": controls, "f0": controls[0], "curve": curve}

    instrument = r.choice(("sine", "saw", "square"))
    controls = [f0] * 32
    source = getattr(audio, instrument)(f0=f0, seconds=seconds, sr=sr)
    value = ctx.sample(source)
    effect_name = "none"
    if effects:
        effect_name = r.choice(("distortion", "reverb", "none"))
        if effect_name == "distortion":
            effect = audio.distortion(x=value, drive=1.5)
            value = ctx.sample(effect)
        elif effect_name == "reverb":
            effect = audio.reverb(x=value, decay=.04, sr=sr)
            value = ctx.sample(effect)
    ctx.latent(f0_curve=controls, f0=f0, instrument=instrument)
    return {"audio": value, "instrument": instrument, "f0": f0,
            "pitch_class": round(12 * math.log2(f0 / 440.) + 69) % 12,
            "effect": effect_name, "label": instrument if task == "instrument" else
            round(12 * math.log2(f0 / 440.) + 69) % 12 if task == "pitch_class" else effect_name,
            "task": task}


def _audio_spec(name: str, task: str, sr: int, seconds: float, effects: bool, size: int, splits: Any, spec: Any):
    from topologies.synth.spec import Spec
    return Spec(name, generator=_audio_record(task=task, sr=sr, seconds=seconds, effects=effects), size=size,
                splits=splits or default_splits(), **spec_kwargs(spec))


@blueprint(example={"task": "instrument", "size": 120})
def audio_classification(name: str, *, task: str = "instrument", sr: int = 8000, seconds: float = .08,
                         effects: bool = True, size: int = 500, splits: Any = None, **spec: Any):
    if task not in ("instrument", "pitch_class", "effect"):
        raise synth.unknown("audio task", task, ["instrument", "pitch_class", "effect"],
                            stage="blueprint")
    return _audio_spec(name, task, sr, seconds, effects, size, splits, spec)


@blueprint(example={"sr": 8000, "curve": "glide", "size": 120})
def pitch_estimation(name: str, *, sr: int = 8000, seconds: float = .2, curve: str = "constant",
                     pitch_depth: float = 3.0, size: int = 500, splits: Any = None,
                     **spec: Any):
    from topologies.synth.spec import Spec
    if curve not in ("constant", "glide", "vibrato"):
        raise synth.unknown("pitch curve", curve, ["constant", "glide", "vibrato"],
                            stage="blueprint")
    return Spec(name, generator=_audio_record(task="pitch", sr=sr, seconds=seconds,
                effects=False, curve=curve, pitch_depth=pitch_depth),
                schema=synth.Schema(audio=synth.Audio(sr=sr), f0_curve=synth.Series(length=32),
                                    f0=synth.Number(min=1), curve=synth.Text()),
                size=size, splits=splits or default_splits(), **spec_kwargs(spec))


@synth.node(tags=["blueprint", "graph"], example={"n": 18})
def _graph_record(r, *, ctx, task: str, n: int, p: float):
    from topologies.synth.lib.graph import erdos_renyi, is_connected, sbm, shortest_paths
    if task == "community":
        graph = sbm(sizes=(n // 2, n - n // 2), probabilities=((.7, .08), (.08, .7)))
        value = ctx.sample(graph)
        parts = synth.Graph.parts(value)
        return {"graph": value, "task": task, "communities": parts["y"]}
    graph_node = erdos_renyi(n=n, p=p, directed=False)
    value = ctx.sample(graph_node)
    if task == "connectivity":
        return {"graph": value, "task": task, "label": ctx.sample(is_connected(x=value))}
    if task == "shortest_path":
        source = r.randrange(n)
        return {"graph": value, "task": task, "source": source,
                "distances": ctx.sample(shortest_paths(x=value, source=source))}
    raise ValueError("task must be community, connectivity or shortest_path")


@blueprint(example={"task": "community", "size": 200})
def graph_tasks(name: str, *, task: str = "community", n: int = 18, p: float = .15,
                size: int = 1000, splits: Any = None, **spec: Any):
    from topologies.synth.spec import Spec
    if task not in ("community", "connectivity", "shortest_path"):
        raise synth.unknown("graph task", task, ["community", "connectivity", "shortest_path"],
                            stage="blueprint")
    return Spec(name, generator=_graph_record(task=task, n=n, p=p), size=size,
                splits=splits or default_splits(), **spec_kwargs(spec))


__all__ = ["audio_classification", "graph_tasks", "lm_corpus", "math_word_problems", "ner",
           "pitch_estimation", "seq2seq_transform", "structured_extraction", "text_classification"]
