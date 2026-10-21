"""A small, dependency-free faker: realistic names, addresses, dates, amounts and IDs.

    f = synth.text.Faker("uk", rng)          # rng: a synth.Rand / random.Random
    f.name(), f.email(), f.address(), f.date("2023-01-01", "2024-12-31"), f.iban()
    synth.text.fake("name", locale="uk")     # the same as a node (one value per record)

Locale data lives in language packs (``packs/<lang>/faker.yaml``: ``en``, ``uk``,
``de``); missing keys fall back to ``en``. ``register_locale("fr", {...})`` adds one
in code, or add a pack directory.
"""

from __future__ import annotations

import datetime as _dt
import math
import random
import string
from collections.abc import Mapping
from typing import Any

from topologies import synth
from topologies.synth.lib.text.packs import ROOT, pack, transliterate

_IBAN_LEN = {"GB": 22, "DE": 22, "UA": 29, "FR": 27, "PL": 28, "NL": 18}



_EXTRA: dict[str, dict[str, Any]] = {}


def register_locale(code: str, data: Mapping[str, Any]) -> None:
    """Add or extend a locale in code (keys as in ``packs/en/faker.yaml``)."""
    _EXTRA[str(code)] = {**_EXTRA.get(str(code), {}), **dict(data)}


def locales() -> list[str]:
    return sorted({p.name for p in ROOT.iterdir() if (p / "faker.yaml").is_file()} | set(_EXTRA))


def _data(locale: str) -> dict[str, Any]:
    if locale in _EXTRA and not (ROOT / locale).is_dir():
        return {**pack("en").faker, **_EXTRA[locale]}
    if not (ROOT / locale / "faker.yaml").is_file():
        raise synth.unknown("faker locale", locale, locales(), stage="faker",
                            extra="Add a pack (packs/<lang>/faker.yaml) or "
                                  "synth.text.register_locale(code, {...}).")
    return {**pack(locale).faker, **_EXTRA.get(locale, {})}


class Faker:
    """Locale-aware fake data from a seeded RNG (see module docstring)."""

    def __init__(self, locale: str = "en", rng: random.Random | None = None) -> None:
        self.locale = locale
        self.pack = _data(locale)
        self.rng = rng or random.Random(0)

    def _pick(self, key: str) -> Any:
        return self.rng.choice(self.pack[key])

    def _pattern(self, pattern: str) -> str:
        out = []
        for ch in pattern:
            if ch == "#":
                out.append(str(self.rng.randrange(10)))
            elif ch == "?":
                out.append(self.rng.choice(string.ascii_uppercase))
            else:
                out.append(ch)
        return "".join(out)

    # -- people ------------------------------------------------------------
    def first_name(self, gender: str | None = None) -> str:
        g = gender or self.rng.choice(["male", "female"])
        return self._pick("first_female" if g.startswith("f") else "first_male")

    def last_name(self) -> str:
        return self._pick("last")

    def name(self, gender: str | None = None) -> str:
        return f"{self.first_name(gender)} {self.last_name()}"

    def email(self, name: str | None = None) -> str:
        name = name or self.name()
        parts = transliterate(name.lower(), self.locale).split()
        sep = self.rng.choice([".", "_", ""])
        user = sep.join(p for p in parts if p.isascii()) or "user"
        if self.rng.random() < 0.4:
            user += str(self.rng.randint(1, 99))
        return f"{user}@{self._pick('domains')}"

    def phone(self) -> str:
        return self._pattern(self._pick("phone"))

    # -- places ---------------------------------------------------------------
    def city(self) -> str:
        return self._pick("cities")

    def street(self) -> str:
        return self._pick("streets")

    def postcode(self) -> str:
        return self._pattern(self._pick("postcode"))

    def address(self) -> str:
        return self.pack["address_format"].format(
            number=self.rng.randint(1, 220), street=self.street(), city=self.city(),
            postcode=self.postcode())

    def country(self) -> str:
        return self.pack["country"]

    # -- business ---------------------------------------------------------------
    def company(self) -> str:
        word = self._pick("company_words")
        suffix = self._pick("company_suffix")
        if self.locale == "uk":
            return f'{suffix} "{word}"'
        return f"{word} {suffix}"

    def product(self) -> str:
        return self._pick("products")

    def currency(self) -> str:
        return self.pack["currency"]

    def amount(self, median: float = 50.0, spread: float = 0.9, *, digits: int = 2) -> float:
        return round(math.exp(self.rng.gauss(math.log(max(median, 1e-6)), spread)), digits)

    def money(self, median: float = 50.0) -> str:
        v = self.amount(median)
        cur = self.currency()
        if self.locale == "en":
            sym = {"GBP": "£", "USD": "$", "EUR": "€"}.get(cur, cur + " ")
            return f"{sym}{v:,.2f}"
        text = f"{v:,.2f}".replace(",", " ").replace(".", ",")
        return f"{text} {'€' if cur == 'EUR' else 'грн' if cur == 'UAH' else cur}"

    def iban(self) -> str:
        cc = self.pack["iban_country"]
        n = _IBAN_LEN.get(cc, 22)
        digits = "".join(str(self.rng.randrange(10)) for _ in range(n - 4))
        return f"{cc}{self.rng.randint(10, 99)}{digits}"

    def id(self, prefix: str = "", digits: int = 8) -> str:
        return prefix + "".join(str(self.rng.randrange(10)) for _ in range(digits))

    def order_id(self) -> str:
        return self._pattern(self.rng.choice(["ORD-######", "#########", "??-#####-##"]))

    def uuid(self) -> str:
        h = "".join(self.rng.choice("0123456789abcdef") for _ in range(32))
        return f"{h[:8]}-{h[8:12]}-4{h[13:16]}-{h[16:20]}-{h[20:]}"

    # -- time ------------------------------------------------------------------------
    def date(self, start: str = "2022-01-01", end: str = "2025-12-31", *,
             fmt: str | None = None) -> str:
        """ISO date (or ``fmt="local"`` for the locale's format)."""
        a, b = _dt.date.fromisoformat(start), _dt.date.fromisoformat(end)
        d = a + _dt.timedelta(days=self.rng.randint(0, max(0, (b - a).days)))
        if fmt == "local":
            return d.strftime(self.pack["date_format"])
        return d.strftime(fmt) if fmt else d.isoformat()


def _shortcut(method: str):
    def fn(rng: random.Random | None = None, *args: Any, locale: str = "en", **kw: Any) -> Any:
        return getattr(Faker(locale, rng), method)(*args, **kw)

    fn.__name__ = method
    fn.__doc__ = f"``Faker(locale, rng).{method}(...)``."
    return fn


first_name = _shortcut("first_name")
last_name = _shortcut("last_name")
name = _shortcut("name")
email = _shortcut("email")
phone = _shortcut("phone")
city = _shortcut("city")
street = _shortcut("street")
address = _shortcut("address")
company = _shortcut("company")
product = _shortcut("product")
amount = _shortcut("amount")
money = _shortcut("money")
iban = _shortcut("iban")
order_id = _shortcut("order_id")
uuid = _shortcut("uuid")
date = _shortcut("date")

KINDS = ("first_name", "last_name", "name", "email", "phone", "city", "street", "postcode",
         "address", "country", "company", "product", "currency", "amount", "money", "iban",
         "id", "order_id", "uuid", "date")


@synth.node(type=synth.Text, tags=["text", "faker"], example={"kind": "name", "locale": "uk"})
def fake(r, *, kind: str = "name", locale: str = "en", **kw: Any):
    """One fake value per record: ``kind`` is a Faker method (name, email, city, iban, ...)."""
    if kind not in KINDS:
        raise synth.unknown("faker kind", kind, KINDS, stage="faker")
    return getattr(Faker(locale, r), kind)(**kw)


__all__ = [
    "KINDS",
    "Faker",
    "address",
    "amount",
    "city",
    "company",
    "date",
    "email",
    "fake",
    "first_name",
    "iban",
    "last_name",
    "locales",
    "money",
    "name",
    "order_id",
    "phone",
    "product",
    "register_locale",
    "street",
    "uuid",
]
