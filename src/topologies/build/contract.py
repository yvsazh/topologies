"""Generic model-contract introspection (no per-model code).

Token models are discovered through a small attribute convention that any
architecture can follow:

- ``vocab_size``      (or ``config.vocab_size`` / ``get_input_embeddings()``)
- ``context_length``  (or ``max_seq`` / ``max_seq_len`` / ``max_position_embeddings``
                       / ``block_size``, directly or on ``config``)
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

CONTEXT_ATTRS = ("context_length", "max_seq", "max_seq_len", "max_position_embeddings",
                 "block_size", "n_positions", "max_len")


def _wrapper_types() -> tuple[type, ...]:
    import torch

    types: list[type] = [nn.parallel.DistributedDataParallel, nn.DataParallel]
    dynamo = getattr(torch, "_dynamo", None)
    optimized = getattr(getattr(dynamo, "eval_frame", None), "OptimizedModule", None)
    if isinstance(optimized, type):
        types.append(optimized)
    return tuple(types)


def unwrap(model: nn.Module) -> nn.Module:
    """Strip DDP / DataParallel / torch.compile wrappers (and nothing else).

    A user sub-module that happens to be called ``module`` is not a wrapper.
    """
    while isinstance(model, _wrapper_types()):
        inner = getattr(model, "_orig_mod", None) or getattr(model, "module", None)
        if not isinstance(inner, nn.Module) or inner is model:
            break
        model = inner
    return model


def _int_attr(obj: Any, names: tuple[str, ...]) -> int | None:
    for name in names:
        try:
            v = getattr(obj, name, None)
        except Exception:  # noqa: BLE001
            v = None
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            return v
    return None


def model_vocab_size(model: nn.Module) -> int | None:
    """Vocabulary size a token model declares, or ``None`` if unknown."""
    m = unwrap(model)
    v = _int_attr(m, ("vocab_size",))
    if v is None:
        v = _int_attr(getattr(m, "config", None), ("vocab_size",))
    if v is None:
        get_emb = getattr(m, "get_input_embeddings", None)
        if callable(get_emb):
            try:
                emb = get_emb()
                v = int(emb.num_embeddings) if isinstance(emb, nn.Embedding) else None
            except Exception:  # noqa: BLE001
                v = None
    return v


def model_context_length(model: nn.Module) -> int | None:
    """Maximum sequence length a model declares, or ``None`` if unbounded/unknown."""
    m = unwrap(model)
    v = _int_attr(m, CONTEXT_ATTRS)
    if v is None:
        v = _int_attr(getattr(m, "config", None), CONTEXT_ATTRS)
    return v


def embedding_sizes(model: nn.Module) -> list[tuple[str, int]]:
    """``(name, num_embeddings)`` of every ``nn.Embedding`` (for diagnostics)."""
    return [(n, int(e.num_embeddings)) for n, e in unwrap(model).named_modules()
            if isinstance(e, nn.Embedding)]
