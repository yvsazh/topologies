"""Input normalisation: everything a user may pass becomes a stream of ``Batch`` objects.

Inputs are never mutated. Tensors move to the subject's device through
``topologies._execution.transfer``.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from topologies._execution import ModelCall, transfer
from topologies.introspection import _tree


@dataclass
class Batch:
    """One model invocation: a ``ModelCall``, a recipe batch, or a user callable."""

    call: ModelCall | None = None
    batch: dict | None = None
    fn: Callable[[nn.Module], Any] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def run(self, subject: Any, model: nn.Module) -> Any:
        if self.fn is not None:
            return self.fn(model)
        if self.batch is not None:
            recipe = subject.recipe
            return recipe.predict_step(model, self.batch, subject.recipe_cfg(), None)
        call = self.call or ModelCall()
        return model(*call.args, **call.kwargs)

    @property
    def size(self) -> int | None:
        value = self.batch if self.batch is not None else (
            (self.call.args, self.call.kwargs) if self.call is not None else None)
        found = _tree.first_tensor(value) if value is not None else None
        if found is None or found[1].ndim == 0:
            return None
        return int(found[1].shape[0])

    def tensors(self) -> Any:
        return self.batch if self.batch is not None else (
            (self.call.args, self.call.kwargs) if self.call is not None else None)

    def with_tensors(self, args: tuple, kwargs: dict) -> Batch:
        return Batch(ModelCall(tuple(args), dict(kwargs)), None, None, dict(self.meta))


def _is_dataset(value: Any) -> bool:
    try:
        from torch.utils.data import DataLoader, Dataset, IterableDataset
    except ImportError:  # pragma: no cover
        return False
    return isinstance(value, (DataLoader, Dataset, IterableDataset))


def is_collection(inputs: Any) -> bool:
    """True when ``inputs`` is a stream of batches rather than a single invocation."""
    if isinstance(inputs, (torch.Tensor, str, Mapping, ModelCall, Batch)):
        return False
    if _is_dataset(inputs):
        return True
    if isinstance(inputs, (list, tuple)):
        if inputs and all(isinstance(x, str) for x in inputs):
            return False
        if all(isinstance(x, torch.Tensor) for x in inputs):
            return False
        return bool(inputs) and all(isinstance(x, (Mapping, tuple, list, Batch, ModelCall))
                                    for x in inputs)
    if callable(inputs) and not isinstance(inputs, nn.Module):
        return False
    return isinstance(inputs, Iterable)


def normalize_inputs(subject: Any, inputs: Any, *, batch_size: int | None = None,
                     from_data: bool = False) -> Iterator[Batch]:
    """Yield ``Batch`` objects on the subject's device (see the table in §4.4)."""
    if inputs is None:
        raise ValueError("inputs are required: pass a tensor, string, dict, ModelCall, "
                         "callable(model) or an iterable of batches")
    batch_size = batch_size or getattr(subject, "batch_size", None)
    device = subject.device
    if isinstance(inputs, Batch):
        yield inputs
    elif isinstance(inputs, ModelCall):
        yield Batch(ModelCall(transfer(inputs.args, device), transfer(dict(inputs.kwargs), device)))
    elif isinstance(inputs, torch.Tensor):
        x = transfer(inputs, device)
        if batch_size and x.ndim and x.shape[0] > batch_size:
            for chunk in x.split(batch_size):
                yield Batch(ModelCall((chunk,)))
        else:
            yield Batch(ModelCall((x,)))
    elif isinstance(inputs, str) or (isinstance(inputs, (list, tuple)) and inputs
                                      and all(isinstance(x, str) for x in inputs)):
        yield tokenize(subject, [inputs] if isinstance(inputs, str) else list(inputs))
    elif isinstance(inputs, Mapping):
        yield _mapping_batch(subject, inputs, from_data=from_data)
    elif isinstance(inputs, (list, tuple)) and all(isinstance(x, torch.Tensor) for x in inputs) \
            and not from_data:
        yield Batch(ModelCall(tuple(transfer(list(inputs), device))))
    elif from_data and isinstance(inputs, (list, tuple)) and inputs:
        # (x, y) pairs from a dataset: first element = input, second = target.
        x = inputs[0]
        inner = next(normalize_inputs(subject, x, batch_size=None))
        if len(inputs) > 1:
            inner.meta["target"] = transfer(inputs[1], device)
            inner.meta.setdefault("assumptions", []).append(
                "dataset tuples split as (input, target)")
        yield inner
    elif callable(inputs) and not isinstance(inputs, nn.Module) and not is_collection(inputs):
        yield Batch(fn=inputs)
    elif is_collection(inputs):
        for item in inputs:
            yield from normalize_inputs(subject, item, batch_size=batch_size, from_data=True)
    else:
        raise TypeError(f"unsupported inputs of type {type(inputs).__name__}")


def _mapping_batch(subject: Any, inputs: Mapping, *, from_data: bool) -> Batch:
    data = transfer(dict(inputs), subject.device)
    if subject.recipe is not None:
        targets = {k: data[k] for k in subject.recipe.target_keys() if k in data}
        meta = {"targets": targets} if targets else {}
        return Batch(batch=data, meta=meta)
    return Batch(ModelCall((), data))


def tokenize(subject: Any, texts: list[str]) -> Batch:
    """Encode strings with the subject tokenizer; pad + mask for ragged batches."""
    tokenizer = subject.tokenizer
    if tokenizer is None:
        raise ValueError("string inputs need a tokenizer: load a bundle that has one or pass "
                         "introspect(model, tokenizer=...)")
    ids = [list(tokenizer.encode(text)) for text in texts]
    if any(not row for row in ids):
        raise ValueError("a string encoded to zero tokens")
    length = max(len(row) for row in ids)
    pad = getattr(tokenizer, "pad_id", None)
    pad = 0 if pad is None else int(pad)
    tokens = torch.full((len(ids), length), pad, dtype=torch.long)
    mask = torch.zeros((len(ids), length), dtype=torch.long)
    for i, row in enumerate(ids):
        tokens[i, :len(row)] = torch.tensor(row, dtype=torch.long)
        mask[i, :len(row)] = 1
    ragged = any(len(row) != length for row in ids)
    labels = [[_token_text(tokenizer, t) for t in row] for row in ids]
    meta = {"tokens": labels, "token_ids": ids, "texts": list(texts)}
    tokens, mask = tokens.to(subject.device), mask.to(subject.device)
    recipe = subject.recipe
    if recipe is not None:
        keys = recipe.input_keys() or ["tokens"]
        spec = recipe.expected_batch_spec()
        batch = {keys[0]: tokens}
        if ragged or ("mask" in spec and "mask" not in recipe.optional_batch_keys()):
            batch["mask"] = mask
        return Batch(batch=batch, meta=meta)
    kwargs = {"mask": mask} if ragged else {}
    return Batch(ModelCall((tokens,), kwargs), meta=meta)


def _token_text(tokenizer: Any, token_id: int) -> str:
    for name in ("id_to_token", "convert_ids_to_tokens"):
        fn = getattr(tokenizer, name, None)
        if callable(fn):
            try:
                text = fn(token_id)
                if isinstance(text, str):
                    return text
            except Exception:  # noqa: BLE001, S112
                continue
    try:
        return tokenizer.decode([token_id], skip_special_tokens=False)
    except TypeError:
        return tokenizer.decode([token_id])
