"""Tokenizer contract shared by every tokenizer and every token-based model.

The contract is deliberately small so any tokenizer works with any
compatible model:

- ``vocab_size`` — pass it to the model constructor (``vocab_size=tok.vocab_size``);
- ``encode(text) -> list[int]`` / ``decode(ids) -> str``;
- ``pad_id`` / ``eos_id`` / ``bos_id`` / ``unk_id`` (``None`` when absent);
- ``save(dir)`` / ``Tokenizer.load(path)`` (one ``tokenizer.json``).

Built-in tokenizers reserve the first ids for special tokens and place their
own vocabulary after them, so special ids are stable across tokenizer types.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, ClassVar

import torch

TOKENIZER_FILE = "tokenizer.json"
FORMAT_VERSION = 1

#: Default special tokens (role -> token string). ``eos`` matches the
#: document separator used by GPT-2 style corpora such as TinyStories.
DEFAULT_SPECIAL_TOKENS: dict[str, str | None] = {
    "pad": "<|pad|>",
    "unk": "<|unk|>",
    "bos": None,
    "eos": "<|endoftext|>",
}
_ROLES = ("pad", "unk", "bos", "eos")

_TYPES: dict[str, type[Tokenizer]] = {}


def _normalize_specials(
    special_tokens: dict[str, str | None] | None,
    extra: Sequence[str] | None,
    *,
    needs_unk: bool,
) -> tuple[dict[str, str | None], list[str]]:
    roles = dict(DEFAULT_SPECIAL_TOKENS)
    if not needs_unk:
        roles["unk"] = None
    if special_tokens:
        unknown = set(special_tokens) - set(_ROLES)
        if unknown:
            raise ValueError(
                f"Unknown special-token roles {sorted(unknown)}; use {list(_ROLES)} "
                "or pass extra_special_tokens=[...] for additional tokens."
            )
        roles.update(special_tokens)
    ordered: list[str] = []
    for role in _ROLES:
        tok = roles.get(role)
        if tok and tok not in ordered:
            ordered.append(tok)
    for tok in extra or []:
        if tok and tok not in ordered:
            ordered.append(tok)
    return roles, ordered


def is_file_path(value: Any) -> bool:
    """True when ``value`` names an existing file (raw text never raises)."""
    if isinstance(value, Path):
        return value.is_file()
    if not isinstance(value, str) or "\n" in value or not value or len(value) > 1024:
        return False
    try:
        return Path(value).is_file()
    except (OSError, ValueError):
        return False


def iter_texts(source: Any) -> Iterable[str]:
    """Yield texts from a string, a file path, or an iterable of either."""
    if isinstance(source, Path):
        yield source.read_text(encoding="utf-8")
        return
    if isinstance(source, str):
        yield Path(source).read_text(encoding="utf-8") if is_file_path(source) else source
        return
    for item in source:
        yield from iter_texts(item)


class Tokenizer:
    """Base class: special-token handling, batching, persistence.

    Subclasses implement the content vocabulary through
    ``_encode_plain`` / ``_decode_plain`` / ``_content_size`` /
    ``_state`` / ``_from_state`` and set a unique ``kind``.
    """

    kind: ClassVar[str] = "base"
    needs_unk: ClassVar[bool] = True

    def __init__(
        self,
        special_tokens: dict[str, str | None] | None = None,
        extra_special_tokens: Sequence[str] | None = None,
    ) -> None:
        self._roles, self._specials = _normalize_specials(
            special_tokens, extra_special_tokens, needs_unk=self.needs_unk
        )
        self._special_to_id = {t: i for i, t in enumerate(self._specials)}
        self._special_re = (
            re.compile("(" + "|".join(re.escape(t) for t in sorted(self._specials,
                                                                  key=len, reverse=True)) + ")")
            if self._specials else None
        )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.kind and cls.kind != "base":
            _TYPES[cls.kind] = cls

    # -- content vocabulary (subclass API) -------------------------------
    def _encode_plain(self, text: str) -> list[int]:
        raise NotImplementedError

    def _decode_plain(self, ids: list[int]) -> str:
        raise NotImplementedError

    def _content_size(self) -> int:
        raise NotImplementedError

    def _content_token(self, cid: int) -> str:
        return self._decode_plain([cid])

    def _state(self) -> dict:
        raise NotImplementedError

    @classmethod
    def _from_state(cls, state: dict, **special_kw: Any) -> Tokenizer:
        raise NotImplementedError

    # -- public contract ---------------------------------------------------
    @property
    def num_special_tokens(self) -> int:
        return len(self._specials)

    @property
    def special_tokens(self) -> dict[str, str | None]:
        """Role -> token string (``None`` when the role is unused)."""
        return dict(self._roles)

    @property
    def all_special_tokens(self) -> list[str]:
        return list(self._specials)

    @property
    def vocab_size(self) -> int:
        return self.num_special_tokens + self._content_size()

    def __len__(self) -> int:
        return self.vocab_size

    def _role_id(self, role: str) -> int | None:
        tok = self._roles.get(role)
        return self._special_to_id.get(tok) if tok else None

    @property
    def pad_id(self) -> int | None:
        return self._role_id("pad")

    @property
    def unk_id(self) -> int | None:
        return self._role_id("unk")

    @property
    def bos_id(self) -> int | None:
        return self._role_id("bos")

    @property
    def eos_id(self) -> int | None:
        return self._role_id("eos")

    def token_to_id(self, token: str) -> int | None:
        if token in self._special_to_id:
            return self._special_to_id[token]
        ids = self._encode_plain(token)
        return ids[0] + self.num_special_tokens if len(ids) == 1 else None

    def id_to_token(self, idx: int) -> str:
        idx = int(idx)
        if 0 <= idx < self.num_special_tokens:
            return self._specials[idx]
        if not 0 <= idx < self.vocab_size:
            raise IndexError(f"Token id {idx} outside vocabulary of size {self.vocab_size}.")
        return self._content_token(idx - self.num_special_tokens)

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Text -> ids. Special-token strings inside ``text`` map to their ids."""
        out: list[int] = []
        if add_bos:
            if self.bos_id is None:
                raise ValueError("add_bos=True but this tokenizer has no bos token.")
            out.append(self.bos_id)
        off = self.num_special_tokens
        pieces = self._special_re.split(text) if self._special_re is not None else [text]
        for piece in pieces:
            if not piece:
                continue
            sid = self._special_to_id.get(piece)
            if sid is not None:
                out.append(sid)
                continue
            for i in self._encode_plain(piece):
                if i >= 0:
                    out.append(i + off)
                elif self.unk_id is not None:
                    out.append(self.unk_id)
                else:
                    raise ValueError(
                        f"{type(self).__name__} cannot encode part of {piece[:40]!r} and has "
                        "no unk token; retrain on more text or set special_tokens={'unk': ...}."
                    )
        if add_eos:
            if self.eos_id is None:
                raise ValueError("add_eos=True but this tokenizer has no eos token.")
            out.append(self.eos_id)
        return out

    def decode(self, ids: Iterable[int] | torch.Tensor, *, skip_special_tokens: bool = True) -> str:
        """Ids -> text. Accepts lists or 1-D tensors."""
        if isinstance(ids, torch.Tensor):
            ids = ids.reshape(-1).tolist()
        off = self.num_special_tokens
        parts: list[str] = []
        run: list[int] = []
        for i in ids:
            i = int(i)
            if i < 0:
                raise IndexError(
                    f"Token id {i} is negative (e.g. an ignore_index label such as -100); "
                    "filter such ids before decoding.")
            if i < off:
                if run:
                    parts.append(self._decode_plain(run))
                    run = []
                if not skip_special_tokens:
                    parts.append(self._specials[i])
            else:
                if i >= self.vocab_size:
                    raise IndexError(f"Token id {i} outside vocabulary of size {self.vocab_size}.")
                run.append(i - off)
        if run:
            parts.append(self._decode_plain(run))
        return "".join(parts)

    def encode_batch(self, texts: Sequence[str], **kw: Any) -> list[list[int]]:
        return [self.encode(t, **kw) for t in texts]

    def decode_batch(self, batch: Sequence[Sequence[int]] | torch.Tensor, **kw: Any) -> list[str]:
        if isinstance(batch, torch.Tensor):
            batch = batch.tolist()
        return [self.decode(ids, **kw) for ids in batch]

    def __call__(
        self, text: str | Sequence[str], *, return_tensors: bool = False, **kw: Any
    ) -> Any:
        """``tok("hi")`` -> ids; ``tok([...], return_tensors=True)`` -> padded batch dict."""
        if isinstance(text, str):
            ids = self.encode(text, **kw)
            return torch.tensor([ids], dtype=torch.long) if return_tensors else ids
        seqs = self.encode_batch(list(text), **kw)
        if not return_tensors:
            return seqs
        from topologies.data.collate import PadCollator

        pad = self.pad_id if self.pad_id is not None else 0
        return PadCollator(pad_id=pad)([{"tokens": s} for s in seqs])

    # -- persistence ---------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "format": "topologies-tokenizer",
            "version": FORMAT_VERSION,
            "type": self.kind,
            "special_tokens": {r: self._roles.get(r) for r in _ROLES},
            "extra_special_tokens": [
                t for t in self._specials if t not in self._roles.values()
            ],
            "vocab_size": self.vocab_size,
            "state": self._state(),
        }

    def save(self, path: str | Path) -> Path:
        """Save to ``path/tokenizer.json`` (directory) or to ``path`` (``.json`` file)."""
        p = Path(path)
        target = p if p.suffix == ".json" else p / TOKENIZER_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        self._save_extra(target.parent)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")
        return target

    def _save_extra(self, directory: Path) -> None:
        """Hook for tokenizers that store side files next to tokenizer.json."""

    @classmethod
    def load(cls, path: str | Path) -> Tokenizer:
        """Load any saved tokenizer (directory or ``tokenizer.json`` path)."""
        p = Path(path)
        target = p / TOKENIZER_FILE if p.is_dir() else p
        if not target.is_file():
            raise FileNotFoundError(f"No tokenizer found at {target}.")
        data = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("format") != "topologies-tokenizer":
            raise ValueError(f"{target} is not a topologies tokenizer file "
                             f"(format={data.get('format') if isinstance(data, dict) else None!r}).")
        version = data.get("version")
        if not isinstance(version, int) or version > FORMAT_VERSION:
            raise ValueError(
                f"{target} has tokenizer format version {version!r}; this topologies supports "
                f"up to {FORMAT_VERSION}. Upgrade topologies to load it.")
        kind = data.get("type")
        impl = _TYPES.get(str(kind))
        if impl is None:
            raise ValueError(
                f"Unknown tokenizer type {kind!r} in {target}. Known: {sorted(_TYPES)}."
            )
        if cls is not Tokenizer and not issubclass(impl, cls):
            raise TypeError(f"{target} holds a {impl.__name__}, not a {cls.__name__}.")
        tok = impl._from_state(
            data.get("state", {}),
            special_tokens=data.get("special_tokens"),
            extra_special_tokens=data.get("extra_special_tokens"),
            _directory=target.parent,
        )
        expected = data.get("vocab_size")
        if expected is not None and int(expected) != tok.vocab_size:
            raise ValueError(
                f"Tokenizer at {target} declares vocab_size={expected} but rebuilt "
                f"{tok.vocab_size}; the file is corrupt or from an incompatible version."
            )
        return tok

    def __repr__(self) -> str:
        return f"{type(self).__name__}(vocab_size={self.vocab_size})"


def tokenizer_types() -> list[str]:
    return sorted(_TYPES)
