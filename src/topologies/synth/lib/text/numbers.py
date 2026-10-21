"""Locale-aware formatting and basic number words using pack data."""
from __future__ import annotations

import datetime as dt

from topologies import synth
from topologies.synth.lib.text.packs import pack


def number_words(value: int, locale: str = "en") -> str:
    """Spell integers through 999,999 using locale-specific cardinal morphology."""
    data = pack(locale).numbers
    words = data.get("words", {})
    value = int(value)
    if not -999999 <= value <= 999999:
        raise ValueError("number_words supports integers from -999999 through 999999")
    if value < 0:
        return words.get("minus", "minus") + " " + number_words(-value, locale)
    if value < 20 or (value < 100 and value % 10 == 0):
        return str(words[str(value)])
    if locale == "uk":
        def plural_form(n: int) -> str:
            last_two, last = n % 100, n % 10
            return "one" if last == 1 and last_two != 11 else \
                "few" if 2 <= last <= 4 and not 12 <= last_two <= 14 else "many"

        def group(n: int, gender: str = "m") -> str:
            if n == 0:
                return ""
            units = data["feminine_units"] if gender == "f" else data["words"]
            parts = []
            hundreds, rest = divmod(n, 100)
            if hundreds:
                parts.append(str(data.get("hundreds", words)[str(hundreds * 100)]))
            if 10 <= rest % 100 <= 19:
                parts.append(str(words[str(rest)]))
            else:
                tens, unit = divmod(rest, 10)
                if tens:
                    parts.append(str(words[str(tens * 10)]))
                if unit:
                    parts.append(str(units[str(unit)]))
            return " ".join(parts)

        parts = []
        millions, rem = divmod(value, 1_000_000)
        thousands, units = divmod(rem, 1000)
        if millions:
            parts.extend([group(millions), data["million_forms"][plural_form(millions)]])
        if thousands:
            parts.extend([group(thousands, "f"), data["thousand_forms"][plural_form(thousands)]])
        if units:
            parts.append(group(units))
        return " ".join(parts)

    def english(n: int) -> str:
        if n < 20:
            return str(words[str(n)])
        if n < 100:
            ten, one = divmod(n, 10)
            return str(words[str(ten * 10)]) + ("-" + str(words[str(one)]) if one else "")
        if n < 1000:
            hundred, rem = divmod(n, 100)
            return str(words[str(hundred)]) + " hundred" + (" " + english(rem) if rem else "")
        scale, rem = divmod(n, 1000)
        return english(scale) + " thousand" + (" " + english(rem) if rem else "")

    return english(value)


def format_number(value: float, locale: str = "en", decimals: int | None = None) -> str:
    """Format a number with locale decimal/group separators."""
    cfg = pack(locale).numbers.get("format", {})
    decimals = cfg.get("decimals", 2) if decimals is None else int(decimals)
    text = f"{value:,.{decimals}f}"
    return text.replace(",", "\0").replace(".", cfg.get("decimal", ".")).replace(
        "\0", cfg.get("group", ","))


def format_date(value: str | dt.date, locale: str = "en") -> str:
    """Format an ISO date or date object according to the locale pack."""
    d = dt.date.fromisoformat(value) if isinstance(value, str) else value
    return d.strftime(pack(locale).numbers.get("format", {}).get("date", "%Y-%m-%d"))


def format_currency(value: float, code: str | None = None, locale: str = "en") -> str:
    """Format a monetary value with locale separators and a currency label."""
    cfg = pack(locale).numbers
    code = code or cfg.get("currency", "USD")
    symbol = cfg.get("currencies", {}).get(code, code)
    amount = format_number(value, locale)
    return f"{symbol}{amount}" if cfg.get("format", {}).get("currency_position", "before") == "before" else \
        f"{amount} {symbol}"


@synth.node(type=synth.Text, tags=["text", "numbers"], example={"value": 1234, "locale": "uk"})
def number(r, *, value: float, locale: str = "en", decimals: int | None = None):
    """Locale number string node."""
    return format_number(value, locale, decimals)


@synth.node(type=synth.Text, tags=["text", "numbers"], example={"value": "2024-02-29", "locale": "uk"})
def date(r, *, value: str, locale: str = "en"):
    """Locale date string node; input is ISO date."""
    return format_date(value, locale)


@synth.node(type=synth.Text, tags=["text", "numbers"], example={"value": 1250.5, "code": "UAH", "locale": "uk"})
def currency(r, *, value: float, code: str | None = None, locale: str = "en"):
    """Locale currency string node."""
    return format_currency(value, code, locale)


__all__ = ["currency", "date", "format_currency", "format_date", "format_number", "number",
           "number_words"]
