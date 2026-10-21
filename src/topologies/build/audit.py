"""Audit a constructor against the reusable-architecture rules.

Used by tests/CI for new architectures and as a checklist for contributors
(see docs/ADDING_ARCHITECTURES.md). Each rule is generic — no per-model code.

- naming:      token models take ``vocab_size`` (not ``vocab``/``n_vocab``).
- regularize:  a ``dropout``-style argument exists.
- in-channels: a model with an image/audio stem exposes ``in_channels``.
- seq-limit:   learned position tables are sized by an argument
               (``max_seq`` / ``max_len`` / ...), never a literal.
- img-geometry: patch position tables derive from ``img_size``/``patch_size``.
- fail-fast:   ``dim``/``heads`` mismatch is rejected in ``__init__``.
- presets:     size presets are registered (``topologies.build.presets``).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from topologies.build.presets import get_presets

SEQ_ARGS = ("max_seq", "max_seq_len", "max_len", "context_length", "block_size",
            "max_position_embeddings", "seq_len")
VOCAB_ALIASES = ("vocab", "n_vocab", "num_tokens", "vocab_len")


@dataclass(frozen=True)
class Finding:
    rule: str
    message: str

    def __str__(self) -> str:
        return f"[{self.rule}] {self.message}"


def audit_constructor(cls: Any, **kwargs: Any) -> list[Finding]:
    """Return rule violations for ``cls`` (empty list = compliant)."""
    out: list[Finding] = []
    try:
        params = inspect.signature(cls).parameters
    except (TypeError, ValueError):
        return [Finding("signature", "constructor signature is not introspectable")]
    names = set(params)
    is_character = getattr(cls, "input_geometry", None) == "unicode-codepoints"
    is_token = bool(names & {"vocab_size", *VOCAB_ALIASES}) or is_character
    for alias in VOCAB_ALIASES:
        if alias in names:
            out.append(Finding("naming", f"use 'vocab_size' instead of '{alias}'"))
    if not any("dropout" in n for n in names):
        out.append(Finding("regularize", "no dropout argument (add dropout: float = 0.0)"))
    if not get_presets(cls):
        out.append(Finding("presets", "no size presets registered"))
    try:
        with torch.device("meta"):
            model = cls(**kwargs)
    except Exception as e:  # noqa: BLE001
        out.append(Finding("construct", f"cannot construct with {kwargs}: {e}"))
        return out
    convs = [m for m in model.modules() if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d))]
    if convs and "in_channels" not in names and not is_token:
        out.append(Finding(
            "in-channels",
            f"input stem is fixed to {convs[0].in_channels} channels; add in_channels",
        ))
    has_seq_arg = bool(names & set(SEQ_ARGS))
    if is_token and not has_seq_arg:
        vocab = next((kwargs.get(n, params[n].default) for n in ("vocab_size", *VOCAB_ALIASES)
                      if n in params), None)
        tables = [m.num_embeddings for m in model.modules()
                  if isinstance(m, nn.Embedding) and m.num_embeddings != vocab]
        if tables:
            out.append(Finding(
                "seq-limit",
                f"hidden sequence limit: position table of {tables[0]} slots; add max_seq",
            ))
    if convs and not is_token and not ({"img_size", "image_size"} & names):
        slots = [p.shape[1] for n, p in model.named_parameters()
                 if p.dim() == 3 and p.shape[0] == 1 and "pos" in n.lower()]
        if slots:
            out.append(Finding(
                "img-geometry",
                f"position table fixed at {slots[0]} slots; derive it from img_size/patch_size",
            ))
    if {"dim", "heads"} <= names:
        heads = kwargs.get("heads", params["heads"].default)
        if isinstance(heads, int) and heads > 1:
            bad = {**kwargs, "dim": heads * 8 + 1}
            try:
                with torch.device("meta"):
                    cls(**bad)
                out.append(Finding("fail-fast",
                                   "dim not divisible by heads is accepted by __init__"))
            except ValueError:
                pass
            except Exception:  # noqa: BLE001
                out.append(Finding("fail-fast",
                                   "invalid dim/heads raises a non-ValueError error"))
    return out
