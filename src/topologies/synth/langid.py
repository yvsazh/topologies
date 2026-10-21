"""Lightweight, built-in language identification (no models, no downloads).

Three signals, cheapest first: Unicode script (Cyrillic, Greek, Arabic, Hebrew,
CJK, Hangul, Kana, Devanagari, Thai), distinctive letters (``ї є ґ`` → uk,
``ß`` → de, ``ñ`` → es, ``ł ś ż`` → pl, ...) and stopword hits, scored
together with character-trigram overlap with small per-language profiles.

    detect("Where is my parcel?")      # ("en", 0.9)
    detect("Де моя посилка?")           # ("uk", 0.95)

Short texts (a few words) may come back as ``("unknown", low confidence)``.
"""

from __future__ import annotations

import re
import unicodedata

_STOP_WORDS: dict[str, str] = {
    "en": ("the and is are was to of in it you i my for on with that this not be have please "
           "where when what can how do does will would your me we they a an"),
    "de": ("der die das und ist nicht ich du sie wir ein eine zu mit auf für von den dem "
           "bitte wo wann was kann wie mein meine ihr haben sind es noch schon"),
    "fr": ("le la les et est je tu il elle nous vous un une des de du pour avec sur pas ne "
           "mon ma mes où quand quoi comment est-ce que qui"),
    "es": ("el la los las y es en un una de del que por para con no mi mis yo tu donde "
           "cuando como qué está estoy pero muy"),
    "it": ("il lo la gli le e è di da in un una per con non mio mia io tu dove quando come "
           "che sono ho ma"),
    "pt": ("o a os as e é de do da em um uma para com não meu minha eu você onde quando como "
           "que está estou mas muito"),
    "nl": ("de het een en is ik je niet van op voor met dat dit mijn waar wanneer hoe wat "
           "zijn heb maar"),
    "pl": ("i w na nie to jest się że do z co jak mój moja gdzie kiedy proszę jestem mam ale "
           "czy"),
    "sv": "och är jag du det en ett att på för med inte min var när hur vad har",
    "cs": "a je to že na se v z do jak kde kdy můj moje prosím nejsem mám ale",
    "tr": "ve bir bu da de ne için ile değil benim nerede zaman nasıl lütfen var yok ama",
    "uk": ("і та що це не на як в у з до мій моя де коли будь ласка я ви ми він вона є але "
           "або"),
    "ru": "и в не на что это как с я ты мы он она мой моя где когда пожалуйста есть но или",
}
_STOP: dict[str, set[str]] = {k: set(v.split()) for k, v in _STOP_WORDS.items()}

_LETTERS: dict[str, str] = {
    "de": "ßäöü", "es": "ñ¿¡", "fr": "çèêëàâîôûœ", "pt": "ãõç", "pl": "ąęłńśźż",
    "cs": "řěůčšž", "tr": "ğşıİ", "sv": "åäö", "it": "àèìòù", "nl": "ij",
    "uk": "їєґі", "ru": "ыэъё",
}

_PROFILE_TEXT: dict[str, str] = {
    "en": "the order arrived late and the package was damaged please help me with a refund "
          "where is my parcel i would like to return this item thank you",
    "de": "die bestellung kam zu spät an und das paket war beschädigt bitte helfen sie mir "
          "wo ist mein paket ich möchte diesen artikel zurückgeben danke schön",
    "fr": "la commande est arrivée en retard et le colis était endommagé aidez moi s'il vous "
          "plaît où est mon colis je voudrais retourner cet article merci",
    "es": "el pedido llegó tarde y el paquete estaba dañado por favor ayúdenme dónde está "
          "mi paquete quiero devolver este artículo gracias",
    "it": "l'ordine è arrivato in ritardo e il pacco era danneggiato per favore aiutatemi "
          "dov'è il mio pacco vorrei restituire questo articolo grazie",
    "pt": "o pedido chegou atrasado e o pacote estava danificado por favor ajudem onde está "
          "o meu pacote quero devolver este artigo obrigado",
    "nl": "de bestelling kwam te laat aan en het pakket was beschadigd help me alstublieft "
          "waar is mijn pakket ik wil dit artikel terugsturen dank je",
    "pl": "zamówienie przyszło późno a paczka była uszkodzona proszę o pomoc gdzie jest moja "
          "paczka chcę zwrócić ten produkt dziękuję",
    "uk": "замовлення прийшло із запізненням і посилка була пошкоджена будь ласка допоможіть "
          "де моя посилка я хочу повернути цей товар дякую",
    "ru": "заказ пришёл с опозданием и посылка была повреждена пожалуйста помогите где моя "
          "посылка я хочу вернуть этот товар спасибо",
}


def _trigrams(text: str) -> set[str]:
    t = f" {text} "
    return {t[i:i + 3] for i in range(len(t) - 2)}


_PROFILES = {k: _trigrams(v) for k, v in _PROFILE_TEXT.items()}

_SCRIPTS = [
    ("CYRILLIC", "cyrillic"), ("GREEK", "el"), ("ARABIC", "ar"), ("HEBREW", "he"),
    ("HANGUL", "ko"), ("HIRAGANA", "ja"), ("KATAKANA", "ja"), ("CJK", "zh"),
    ("DEVANAGARI", "hi"), ("THAI", "th"), ("LATIN", "latin"),
]

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def script_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ch in text:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        for key, script in _SCRIPTS:
            if key in name:
                counts[script] = counts.get(script, 0) + 1
                break
    return counts


def detect(text: str) -> tuple[str, float]:
    """``(language code, confidence in [0, 1])`` or ``("unknown", c)``."""
    t = str(text).casefold()
    scripts = script_counts(t)
    total = sum(scripts.values())
    if total == 0:
        return "unknown", 0.0
    script, n = max(scripts.items(), key=lambda kv: kv[1])
    share = n / total
    if script not in ("latin", "cyrillic"):
        if script == "zh" and scripts.get("ja"):
            return "ja", share
        return script, share
    cands = [lang for lang in _STOP if (lang in ("uk", "ru")) == (script == "cyrillic")]
    words = _WORD.findall(t)
    grams = _trigrams(" ".join(words))
    scores: dict[str, float] = {}
    for lang in cands:
        stop = sum(1 for w in words if w in _STOP[lang])
        letters = sum(t.count(c) for c in _LETTERS.get(lang, "")) if lang != "nl" else \
            t.count("ij")
        prof = _PROFILES.get(lang)
        tri = len(grams & prof) / max(1, len(grams)) if prof else 0.0
        scores[lang] = 2.0 * stop / max(1, len(words)) + 0.6 * min(letters, 3) / 3 + tri
    if script == "cyrillic":
        # Distinctive letters are decisive between uk and ru.
        if any(c in t for c in "їєґ") or (t.count("і") > t.count("и") and "і" in t):
            scores["uk"] = scores.get("uk", 0) + 1.0
        if any(c in t for c in "ыэъё"):
            scores["ru"] = scores.get("ru", 0) + 1.0
    # English has no distinctive letters; reward plain ASCII text.
    if script == "latin" and "en" in scores and all(ord(c) < 128 for c in t if c.isalpha()):
        scores["en"] += 0.25
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best, s1 = ranked[0]
    s2 = ranked[1][1] if len(ranked) > 1 else 0.0
    conf = max(0.0, min(1.0, (s1 - s2) / max(s1, 1e-9))) * min(1.0, 0.4 + 0.15 * len(words))
    conf = conf * share
    if s1 < 0.15:
        return "unknown", conf
    return best, round(conf, 3)


__all__ = ["detect", "script_counts"]
