"""Text / token-sequence datasets: documents, sliding blocks, packing, padding."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from topologies.data.collate import PadCollator

DEFAULT_SEPARATOR = "<|endoftext|>"


def read_text(path: str | Path, encoding: str = "utf-8") -> str:
    return Path(path).read_text(encoding=encoding)


def split_documents(
    text: str, separator: str | None = DEFAULT_SEPARATOR, *, strip: bool = True
) -> list[str]:
    """Split raw text into documents on ``separator`` (empty docs dropped)."""
    parts = [text] if not separator else text.split(separator)
    if strip:
        parts = [p.strip() for p in parts]
    return [p for p in parts if p]


def read_documents(
    paths: str | Path | Sequence[str | Path],
    separator: str | None = DEFAULT_SEPARATOR,
    *,
    strip: bool = True,
    limit: int | None = None,
    encoding: str = "utf-8",
) -> list[str]:
    """Read one or more text files and split them into documents."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    docs: list[str] = []
    for p in paths:
        docs.extend(split_documents(read_text(p, encoding), separator, strip=strip))
        if limit is not None and len(docs) >= limit:
            return docs[:limit]
    return docs


def _as_ids(ids: Any) -> torch.Tensor:
    t = torch.as_tensor(ids, dtype=torch.long)
    if t.dim() != 1:
        raise ValueError(f"Expected a 1-D sequence of token ids; got shape {tuple(t.shape)}.")
    return t


def sliding_blocks(
    ids: Any, block_size: int, stride: int | None = None, *, drop_last: bool = True
) -> torch.Tensor:
    """Cut a 1-D id sequence into ``[N, block_size]`` windows every ``stride`` tokens.

    ``stride`` defaults to ``block_size`` (no overlap). With ``drop_last=False``
    the tail window is aligned to the end of the sequence instead of dropped.
    """
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1; got {block_size}.")
    stride = int(stride or block_size)
    if stride < 1:
        raise ValueError(f"stride must be >= 1; got {stride}.")
    t = _as_ids(ids)
    n = int(t.shape[0])
    if n < block_size:
        raise ValueError(
            f"Sequence of {n} tokens is shorter than block_size={block_size}; "
            "use a smaller block_size or more text."
        )
    blocks = t.unfold(0, block_size, stride)  # [N, block_size] view, no Python loop
    last_start = (blocks.shape[0] - 1) * stride
    if not drop_last and last_start + block_size < n:
        return torch.cat([blocks, t[n - block_size:].unsqueeze(0)])
    return blocks.clone(memory_format=torch.contiguous_format)  # own storage, like before


def pack_sequences(
    seqs: Iterable[Sequence[int]],
    block_size: int,
    *,
    sep_id: int | None = None,
    stride: int | None = None,
    drop_last: bool = True,
) -> torch.Tensor:
    """Concatenate sequences (``sep_id`` between them) and cut into blocks."""
    flat: list[int] = []
    for s in seqs:
        flat.extend(int(x) for x in s)
        if sep_id is not None:
            flat.append(int(sep_id))
    return sliding_blocks(flat, block_size, stride, drop_last=drop_last)


class TokenBlockDataset(Dataset):
    """Fixed-length blocks ``{key: LongTensor[block_size]}`` from token ids."""

    def __init__(
        self,
        blocks: torch.Tensor,
        *,
        key: str = "tokens",
    ) -> None:
        if blocks.dim() != 2:
            raise ValueError("TokenBlockDataset expects a [N, block_size] tensor.")
        self.blocks = blocks.long()
        self.key = key

    @classmethod
    def from_ids(
        cls, ids: Any, block_size: int, stride: int | None = None, *, key: str = "tokens"
    ) -> TokenBlockDataset:
        return cls(sliding_blocks(ids, block_size, stride), key=key)

    @property
    def block_size(self) -> int:
        return int(self.blocks.shape[1])

    def __len__(self) -> int:
        return int(self.blocks.shape[0])

    def __getitem__(self, i: int) -> dict:
        return {self.key: self.blocks[i]}


class SequenceDataset(Dataset):
    """Variable-length sequences ``{key: LongTensor[T_i]}`` + a padding collate.

    ``collate_fn`` is exposed as an attribute; ``Trainer``/``make_loader`` pick
    it up automatically, so batches arrive as ``{"tokens", "mask"}``.
    """

    def __init__(self, seqs: Sequence[Sequence[int]], *, pad_id: int = 0, key: str = "tokens"):
        self.seqs = [torch.as_tensor(s, dtype=torch.long) for s in seqs]
        if not self.seqs:
            raise ValueError("SequenceDataset needs at least one sequence.")
        self.key = key
        self.collate_fn = PadCollator(pad_id=pad_id)

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, i: int) -> dict:
        return {self.key: self.seqs[i]}


def _encode(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text))
    if callable(tokenizer):
        return list(tokenizer(text))
    raise TypeError("tokenizer must have .encode(text) -> list[int] or be callable.")


def lm_dataset(
    source: str | Path | Sequence[str],
    tokenizer: Any,
    block_size: int,
    *,
    mode: str = "packed",
    stride: int | None = None,
    add_eos: bool = True,
    separator: str | None = DEFAULT_SEPARATOR,
) -> Dataset:
    """Language-modeling dataset straight from text.

    ``source``: a file path, raw text, or a list of documents.

    - ``mode="packed"`` (default): documents are joined with EOS and cut into
      ``block_size`` blocks — no padding, no wasted compute.
    - ``mode="padded"``: one document per sample (long ones split into
      ``block_size`` chunks), padded per batch with a ``mask``.

    Samples carry ``tokens`` only; causal-lm derives shifted labels itself.
    """
    from topologies.tokenizers.base import is_file_path

    if isinstance(source, Path) and not source.is_file():
        raise FileNotFoundError(f"lm_dataset: no such file {source}.")
    if is_file_path(source):
        docs = read_documents(source, separator)
    elif isinstance(source, str):
        docs = split_documents(source, separator)
    else:
        docs = [d for d in source if d]
    if not docs:
        raise ValueError("lm_dataset: no documents found in source.")
    eos = getattr(tokenizer, "eos_id", None) if add_eos else None
    encoded = []
    for d in docs:
        ids = _encode(tokenizer, d)
        if eos is not None:
            ids.append(int(eos))
        encoded.append(ids)
    if mode == "packed":
        return TokenBlockDataset(pack_sequences(encoded, block_size, stride=stride))
    if mode == "padded":
        chunks: list[list[int]] = []
        for ids in encoded:
            for s in range(0, len(ids), block_size):
                piece = ids[s : s + block_size]
                if len(piece) >= 2:
                    chunks.append(piece)
        pad_id = getattr(tokenizer, "pad_id", None)
        return SequenceDataset(chunks, pad_id=int(pad_id) if pad_id is not None else 0)
    raise ValueError(f"Unknown mode {mode!r}; use 'packed' or 'padded'.")
