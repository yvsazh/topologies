"""Language packs: data, not code.

A pack is a directory of YAML files (``packs/<lang>/``): ``faker.yaml`` (names,
places, companies, formats), ``lexicon.yaml`` (words with features),
``paradigms.yaml`` (inflection tables), ``numbers.yaml`` (number words, number /
date / currency formats), ``keyboard.yaml`` (key neighbours for typos),
``translit.yaml`` (transliteration), ``intents.yaml`` / ``templates.yaml``
(sentence templates per task). Missing sections fall back to ``en`` where that
makes sense (faker data); a language is added by adding a directory:

    synth.text.pack("uk").faker["cities"]          # a built-in pack
    synth.text.pack("path/to/my_pack/")            # your own pack directory
    synth.text.languages()                          # ["de", "en", "uk"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from topologies import synth

ROOT = Path(__file__).resolve().parent / "packs"
_CACHE: dict[str, Pack] = {}


class Pack:
    """One language pack (see module docstring); sections load lazily."""

    def __init__(self, path: str | Path, code: str | None = None) -> None:
        self.path = Path(path)
        self.code = code or self.path.name
        self._sections: dict[str, Any] = {}

    def section(self, name: str) -> dict[str, Any]:
        if name not in self._sections:
            f = self.path / f"{name}.yaml"
            data = yaml.safe_load(f.read_text(encoding="utf-8")) if f.is_file() else {}
            self._sections[name] = data or {}
        return self._sections[name]

    def has(self, name: str) -> bool:
        return (self.path / f"{name}.yaml").is_file()

    @property
    def faker(self) -> dict[str, Any]:
        base = dict(pack("en").section("faker")) if self.code != "en" else {}
        base.update(self.section("faker"))
        return base

    @property
    def lexicon(self) -> dict[str, Any]:
        return self.section("lexicon")

    @property
    def paradigms(self) -> dict[str, Any]:
        return self.section("paradigms")

    @property
    def numbers(self) -> dict[str, Any]:
        return self.section("numbers")

    @property
    def keyboard(self) -> dict[str, Any]:
        return self.section("keyboard")

    @property
    def translit(self) -> dict[str, str]:
        return dict(self.section("translit").get("table") or {})

    def fingerprint(self) -> str:
        import hashlib

        h = hashlib.sha256()
        for f in sorted(self.path.glob("*.yaml")):
            h.update(f.name.encode())
            h.update(f.read_bytes())
        return h.hexdigest()[:16]

    def __repr__(self) -> str:
        return f"Pack({self.code!r}, {self.path})"


def pack(lang_or_path: str | Path) -> Pack:
    """The pack for a language code (built in) or a pack directory."""
    key = str(lang_or_path)
    if key in _CACHE:
        return _CACHE[key]
    p = Path(key)
    if (ROOT / key).is_dir():
        out = Pack(ROOT / key, key)
    elif p.is_dir():
        out = Pack(p)
    else:
        raise synth.unknown("language pack", key, languages(), stage="text",
                            extra="Add a directory with YAML files and pass its path.")
    _CACHE[key] = out
    return out


def languages() -> list[str]:
    """Built-in language codes."""
    return sorted(p.name for p in ROOT.iterdir() if p.is_dir())


def transliterate(text: str, lang: str = "uk") -> str:
    """Latin transliteration with the pack's table (case preserved for the first letter)."""
    table = pack(lang).translit
    out = []
    for ch in text:
        low = ch.lower()
        if low in table:
            t = table[low]
            out.append(t.capitalize() if ch != low and t else t)
        else:
            out.append(ch)
    return "".join(out)


__all__ = ["ROOT", "Pack", "languages", "pack", "transliterate"]
