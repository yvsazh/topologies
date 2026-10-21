"""Collate functions for variable-length sequence batches."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch.utils.data import default_collate

LABEL_KEYS = ("labels", "label", "targets", "target")


class PadCollator:
    """Pad variable-length 1-D sequences and emit an attention mask.

    - Every 1-D tensor/list field whose lengths differ across the batch (or that
      is listed in ``pad_keys``) is right-padded to the longest sample.
    - Label-like keys (``labels``, ``targets``, ...) are padded with
      ``label_pad_id`` (default ``-100``, ignored by cross-entropy); the rest
      with ``pad_id``.
    - ``mask_key`` (default ``"mask"``) holds ``1`` for real tokens and ``0``
      for padding, shaped like the padded sequences. Set ``mask_key=None`` to
      skip it.
    - Non-sequence fields fall back to ``default_collate``.
    """

    def __init__(
        self,
        pad_id: int = 0,
        *,
        pad_keys: Sequence[str] | None = None,
        label_pad_id: int = -100,
        mask_key: str | None = "mask",
        max_length: int | None = None,
        pad_to_multiple_of: int | None = None,
    ) -> None:
        self.pad_id = int(pad_id)
        self.pad_keys = tuple(pad_keys) if pad_keys is not None else None
        self.label_pad_id = int(label_pad_id)
        self.mask_key = mask_key
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def _is_seq(self, key: str, values: list) -> bool:
        if self.pad_keys is not None:
            return key in self.pad_keys
        first = values[0]
        if isinstance(first, (list, tuple)):
            return True
        if isinstance(first, torch.Tensor) and first.dim() == 1:
            return len({int(v.shape[0]) for v in values}) > 1 or key in ("tokens", *LABEL_KEYS)
        return False

    def __call__(self, samples: list[dict]) -> dict:
        if not samples:
            return {}
        if not isinstance(samples[0], dict):
            raise TypeError(
                "PadCollator expects dict samples, e.g. {'tokens': [...]}; got "
                f"{type(samples[0]).__name__}. Wrap the dataset with topologies.data.DictDataset."
            )
        out: dict[str, Any] = {}
        mask: torch.Tensor | None = None
        rest: dict[str, list] = {}
        for key in samples[0]:
            values = [s[key] for s in samples]
            if not self._is_seq(key, values):
                rest[key] = values
                continue
            seqs = [torch.as_tensor(v, dtype=torch.long) for v in values]
            if self.max_length is not None:
                seqs = [s[: self.max_length] for s in seqs]
            length = max(int(s.shape[0]) for s in seqs)
            if self.pad_to_multiple_of:
                m = int(self.pad_to_multiple_of)
                length = ((length + m - 1) // m) * m
            fill = self.label_pad_id if key in LABEL_KEYS else self.pad_id
            padded = torch.full((len(seqs), length), fill, dtype=torch.long)
            seq_mask = torch.zeros((len(seqs), length), dtype=torch.long)
            for i, s in enumerate(seqs):
                padded[i, : s.shape[0]] = s
                seq_mask[i, : s.shape[0]] = 1
            out[key] = padded
            if mask is None and key not in LABEL_KEYS:
                mask = seq_mask
        for key, values in rest.items():
            out[key] = default_collate(values)
        if self.mask_key and mask is not None and self.mask_key not in out:
            out[self.mask_key] = mask
        return out


def pad_collate(samples: list[dict]) -> dict:
    """Default ``PadCollator()``: pad id 0, labels -100, ``mask`` key."""
    return PadCollator()(samples)
