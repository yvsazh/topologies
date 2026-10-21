"""Optional wrapper around the Rust ``tokenizers`` library (extra: ``nlp``).

Gives the fast HuggingFace BPE/WordPiece implementations the same
``Tokenizer`` contract, so they plug into ``lm_dataset``/``generate``/
``Trainer(tokenizer=...)`` unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from topologies.tokenizers.base import DEFAULT_SPECIAL_TOKENS, Tokenizer, iter_texts

HF_FILE = "hf_tokenizer.json"


def _require_tokenizers() -> Any:
    try:
        import tokenizers
    except ImportError as e:
        raise ImportError(
            "The 'tokenizers' package is required for HFTokenizer. "
            "Install it with: pip install 'topologies[nlp]'"
        ) from e
    return tokenizers


class HFTokenizer(Tokenizer):
    """Adapter for a ``tokenizers.Tokenizer`` (ids are the backend's own ids)."""

    kind = "hf"

    def __init__(self, backend: Any, special_tokens: dict[str, str | None] | None = None,
                 extra_special_tokens: Any = None) -> None:
        self.backend = backend
        roles = dict(DEFAULT_SPECIAL_TOKENS)
        roles["unk"] = None
        if special_tokens:
            roles.update(special_tokens)
        # Keep only roles the backend actually knows.
        self._roles = {r: (t if t and backend.token_to_id(t) is not None else None)
                       for r, t in roles.items()}
        self._specials = [t for t in self._roles.values() if t]
        self._special_to_id = {t: backend.token_to_id(t) for t in self._specials}
        self._special_re = None

    @classmethod
    def from_file(cls, path: str | Path, **kw: Any) -> HFTokenizer:
        tk = _require_tokenizers()
        return cls(tk.Tokenizer.from_file(str(path)), **kw)

    @classmethod
    def train(
        cls,
        source: Any,
        vocab_size: int = 4096,
        *,
        model: str = "bpe",
        min_frequency: int = 2,
        special_tokens: dict[str, str | None] | None = None,
        **_: Any,
    ) -> HFTokenizer:
        """Train a byte-level BPE (or WordPiece with ``model='wordpiece'``)."""
        tk = _require_tokenizers()
        roles = dict(DEFAULT_SPECIAL_TOKENS)
        roles["unk"] = None if model == "bpe" else roles["unk"]
        if special_tokens:
            roles.update(special_tokens)
        specials = [t for t in roles.values() if t]
        if model == "bpe":
            backend = tk.Tokenizer(tk.models.BPE())
            backend.pre_tokenizer = tk.pre_tokenizers.ByteLevel(add_prefix_space=False)
            backend.decoder = tk.decoders.ByteLevel()
            trainer = tk.trainers.BpeTrainer(
                vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=specials,
                initial_alphabet=tk.pre_tokenizers.ByteLevel.alphabet(),
            )
        elif model == "wordpiece":
            backend = tk.Tokenizer(tk.models.WordPiece(unk_token=roles["unk"] or "[UNK]"))
            backend.pre_tokenizer = tk.pre_tokenizers.Whitespace()
            backend.decoder = tk.decoders.WordPiece()
            trainer = tk.trainers.WordPieceTrainer(
                vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=specials,
            )
        else:
            raise ValueError(f"Unknown model {model!r}; use 'bpe' or 'wordpiece'.")
        backend.train_from_iterator(list(iter_texts(source)), trainer=trainer)
        return cls(backend, special_tokens=roles)

    @property
    def vocab_size(self) -> int:
        return int(self.backend.get_vocab_size())

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = list(self.backend.encode(text, add_special_tokens=False).ids)
        return self._with_boundaries(ids, add_bos=add_bos, add_eos=add_eos)

    def encode_batch(self, texts: Sequence[str], **kw: Any) -> list[list[int]]:
        """Encode with the Rust batch API. Ids match ``encode`` on each text."""
        add_bos = bool(kw.pop("add_bos", False))
        add_eos = bool(kw.pop("add_eos", False))
        if kw:
            unknown = ", ".join(sorted(kw))
            raise TypeError(f"unexpected arguments: {unknown}")
        rows = list(texts)
        if not rows:
            return []
        if add_bos and self.bos_id is None:
            raise ValueError("add_bos=True but this tokenizer has no bos token.")
        if add_eos and self.eos_id is None:
            raise ValueError("add_eos=True but this tokenizer has no eos token.")
        encodings = self.backend.encode_batch(rows, add_special_tokens=False)
        return [
            self._with_boundaries(list(encoding.ids), add_bos=add_bos, add_eos=add_eos)
            for encoding in encodings
        ]

    def _with_boundaries(self, ids: list[int], *, add_bos: bool, add_eos: bool) -> list[int]:
        if add_bos:
            if self.bos_id is None:
                raise ValueError("add_bos=True but this tokenizer has no bos token.")
            ids.insert(0, self.bos_id)
        if add_eos:
            if self.eos_id is None:
                raise ValueError("add_eos=True but this tokenizer has no eos token.")
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Any, *, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.reshape(-1).tolist()
        return self.backend.decode([int(i) for i in ids], skip_special_tokens=skip_special_tokens)

    def token_to_id(self, token: str) -> int | None:
        return self.backend.token_to_id(token)

    def id_to_token(self, idx: int) -> str:
        tok = self.backend.id_to_token(int(idx))
        if tok is None:
            raise IndexError(f"Token id {idx} outside vocabulary of size {self.vocab_size}.")
        return tok

    def _state(self) -> dict:
        return {"file": HF_FILE}

    def _save_extra(self, directory: Path) -> None:
        self.backend.save(str(directory / HF_FILE))

    @classmethod
    def _from_state(cls, state: dict, **kw: Any) -> HFTokenizer:
        directory = Path(kw.pop("_directory", "."))
        kw.pop("extra_special_tokens", None)
        return cls.from_file(directory / state.get("file", HF_FILE), **kw)
