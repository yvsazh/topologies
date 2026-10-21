from __future__ import annotations

from collections.abc import Sequence

import torch


PAD_ID = 0
EOS_ID = 1
BYTE_OFFSET = 2
VOCAB_SIZE = 258


def encode_bytes(text: str, append_eos: bool = True) -> list[int]:
    result = [BYTE_OFFSET + value for value in text.encode("utf-8")]
    if append_eos:
        result.append(EOS_ID)
    return result


class ByteSeq2SeqCollator:
    def __init__(self, pad_to_multiple: int = 1) -> None:
        if pad_to_multiple <= 0:
            raise ValueError("pad_to_multiple must be positive")
        self.pad_to_multiple = pad_to_multiple

    def __call__(self, batch: Sequence[tuple[str, str]]) -> dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("batch must not be empty")
        sources = [encode_bytes(source) for source, _ in batch]
        targets = [encode_bytes(target) for _, target in batch]
        source_width = self._width(max(map(len, sources)))
        target_width = self._width(max(map(len, targets)))
        src = torch.full((len(batch), source_width), PAD_ID, dtype=torch.long)
        tgt = torch.full((len(batch), target_width), PAD_ID, dtype=torch.long)
        labels = torch.full((len(batch), target_width), -100, dtype=torch.long)
        for index, (source, target) in enumerate(zip(sources, targets)):
            src[index, :len(source)] = torch.tensor(source)
            labels[index, :len(target)] = torch.tensor(target)
            tgt[index, 0] = EOS_ID
            if len(target) > 1:
                tgt[index, 1:len(target)] = torch.tensor(target[:-1])
        return {"src": src, "tgt": tgt, "labels": labels}

    def _width(self, length: int) -> int:
        return ((length + self.pad_to_multiple - 1) // self.pad_to_multiple) * self.pad_to_multiple


class DenoisingByteDataset:
    def __init__(self, texts: Sequence[str]) -> None:
        if not texts:
            raise ValueError("texts must not be empty")
        self.texts = tuple(texts)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[str, str]:
        text = self.texts[index]
        return text, text


encode_utf8_bytes = encode_bytes
CharformerSeq2SeqCollator = ByteSeq2SeqCollator
CharformerDenoisingDataset = DenoisingByteDataset
