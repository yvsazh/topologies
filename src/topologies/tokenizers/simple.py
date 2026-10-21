"""Dependency-free character- and word-level tokenizers."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from topologies.tokenizers.base import Tokenizer, iter_texts

_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_NO_SPACE_BEFORE = set(".,!?;:%)]}'”’…")
_NO_SPACE_AFTER = set("([{“‘")


def _strip_specials(text: str, specials: list[str]) -> str:
    for s in specials:
        text = text.replace(s, " ")
    return text


class CharTokenizer(Tokenizer):
    """One id per character. Tiny vocabularies; ideal for smoke tests and demos."""

    kind = "char"

    def __init__(self, chars: list[str] | str = (), **special_kw: Any) -> None:
        super().__init__(**special_kw)
        self.chars = sorted(set(chars))
        self._stoi = {c: i for i, c in enumerate(self.chars)}

    @classmethod
    def train(cls, source: Any, **special_kw: Any) -> CharTokenizer:
        probe = cls(**special_kw)
        seen: set[str] = set()
        for text in iter_texts(source):
            seen.update(_strip_specials(text, probe.all_special_tokens))
        return cls(sorted(seen), **special_kw)

    def _encode_plain(self, text: str) -> list[int]:
        return [self._stoi.get(c, -1) for c in text]

    def _decode_plain(self, ids: list[int]) -> str:
        return "".join(self.chars[i] for i in ids)

    def _content_size(self) -> int:
        return len(self.chars)

    def _state(self) -> dict:
        return {"chars": self.chars}

    @classmethod
    def _from_state(cls, state: dict, **kw: Any) -> CharTokenizer:
        kw.pop("_directory", None)
        return cls(state.get("chars", []), **kw)


class WordTokenizer(Tokenizer):
    """Whitespace/punctuation word splitting with a frequency-capped vocabulary.

    Out-of-vocabulary words map to ``unk``. Decoding re-joins words with
    simple punctuation spacing rules (lossy for whitespace).
    """

    kind = "word"

    def __init__(self, words: list[str] = (), *, lowercase: bool = False, **special_kw: Any):
        super().__init__(**special_kw)
        self.words = list(words)
        self.lowercase = bool(lowercase)
        self._stoi = {w: i for i, w in enumerate(self.words)}

    @classmethod
    def train(
        cls,
        source: Any,
        vocab_size: int | None = None,
        *,
        min_frequency: int = 1,
        lowercase: bool = False,
        **special_kw: Any,
    ) -> WordTokenizer:
        probe = cls(**special_kw)
        counts: Counter[str] = Counter()
        for text in iter_texts(source):
            text = _strip_specials(text, probe.all_special_tokens)
            if lowercase:
                text = text.lower()
            counts.update(_WORD_RE.findall(text))
        ranked = sorted(
            (w for w, c in counts.items() if c >= min_frequency),
            key=lambda w: (-counts[w], w),
        )
        if vocab_size is not None:
            room = int(vocab_size) - probe.num_special_tokens
            if room < 1:
                raise ValueError(
                    f"vocab_size={vocab_size} leaves no room after "
                    f"{probe.num_special_tokens} special tokens."
                )
            ranked = ranked[:room]
        return cls(ranked, lowercase=lowercase, **special_kw)

    def _encode_plain(self, text: str) -> list[int]:
        if self.lowercase:
            text = text.lower()
        return [self._stoi.get(w, -1) for w in _WORD_RE.findall(text)]

    def _decode_plain(self, ids: list[int]) -> str:
        out = ""
        prev = ""
        quote_open = False
        for i in ids:
            w = self.words[i]
            if w == '"':
                # Straight quotes: opening ones attach to the next word, closing to the previous.
                space = bool(out) and not quote_open and prev[-1:] not in _NO_SPACE_AFTER
                quote_open = not quote_open
            else:
                space = bool(out) and w[0] not in _NO_SPACE_BEFORE and not (
                    prev[-1:] in _NO_SPACE_AFTER or (prev == '"' and quote_open)
                )
            out += (" " if space else "") + w
            prev = w
        return out

    def _content_token(self, cid: int) -> str:
        return self.words[cid]

    def _content_size(self) -> int:
        return len(self.words)

    def _state(self) -> dict:
        return {"words": self.words, "lowercase": self.lowercase}

    @classmethod
    def _from_state(cls, state: dict, **kw: Any) -> WordTokenizer:
        kw.pop("_directory", None)
        return cls(state.get("words", []), lowercase=bool(state.get("lowercase", False)), **kw)
