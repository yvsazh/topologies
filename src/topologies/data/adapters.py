"""Dataset adapters: turn any sample format into a recipe's dict-batch contract."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch.utils.data import Dataset, default_collate


def recipe_keys(recipe: Any, *, include_optional: bool = False) -> list[str]:
    """Batch keys a recipe expects, in declaration order.

    ``recipe`` is a recipe name (``"vision-classify"``) or a ``Recipe`` instance.
    The recipe's ``expected_batch_spec()`` is the single source of truth, so a
    new recipe automatically works with every adapter in this module.
    """
    from topologies.training.recipe_base import Recipe, get_recipe

    r = recipe if isinstance(recipe, Recipe) else get_recipe(str(recipe))
    keys = list(r.expected_batch_spec().keys())
    if include_optional:
        return keys
    optional = set(r.optional_batch_keys())
    return [k for k in keys if k not in optional]


def _as_dict(sample: Any, keys: Sequence[str]) -> dict:
    if isinstance(sample, dict):
        return sample
    if isinstance(sample, (tuple, list)):
        if len(sample) != len(keys):
            raise ValueError(
                f"Sample has {len(sample)} fields but keys={list(keys)} were requested. "
                "Pass keys=... with one name per field, e.g. keys=('pixels', 'labels')."
            )
        return dict(zip(keys, sample))
    if len(keys) != 1:
        raise ValueError(
            f"Sample of type {type(sample).__name__} is a single value but "
            f"keys={list(keys)} needs {len(keys)} fields."
        )
    return {keys[0]: sample}


class DictDataset(Dataset):
    """Wrap a dataset so every sample is a dict with the given ``keys``.

    Tuple/list samples are mapped positionally (``(img, label)`` ->
    ``{"pixels": img, "labels": label}``); single values map to ``keys[0]``;
    dict samples pass through unchanged. An optional ``transform`` receives
    the dict and returns a (possibly new) dict. Unknown attributes (e.g.
    ``classes``, ``targets``) are forwarded to the wrapped dataset.
    """

    def __init__(
        self,
        base: Any,
        keys: Sequence[str],
        transform: Callable[[dict], dict] | None = None,
    ) -> None:
        if not keys:
            raise ValueError("DictDataset needs at least one key.")
        self.base = base
        self.keys = tuple(keys)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> dict:
        out = _as_dict(self.base[i], self.keys)
        return self.transform(out) if self.transform is not None else out

    def __getattr__(self, name: str) -> Any:
        # Only called when normal lookup fails; never recurse on our own fields.
        if name in ("base", "keys", "transform") or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.base, name)


def as_dict_dataset(
    dataset: Any,
    keys: Sequence[str] | None = None,
    *,
    recipe: Any = None,
    transform: Callable[[dict], dict] | None = None,
) -> DictDataset:
    """Adapt ``dataset`` to dict samples, keys taken from ``keys`` or a recipe."""
    if keys is None:
        if recipe is None:
            raise ValueError("Pass keys=(...) or recipe='<name>' to name the sample fields.")
        keys = recipe_keys(recipe)
    return DictDataset(dataset, keys, transform=transform)


def for_recipe(dataset: Any, recipe: Any, *, transform: Callable[[dict], dict] | None = None):
    """Adapt ``dataset`` to ``recipe``'s batch contract (positional field mapping).

    >>> train = for_recipe(torchvision.datasets.MNIST(...), "vision-classify")
    >>> train[0].keys()  # dict_keys(['pixels', 'labels'])
    """
    return as_dict_dataset(dataset, recipe=recipe, transform=transform)


def tuple_to_dict(keys: Sequence[str]) -> Callable[[Any], dict]:
    """Return a function mapping one tuple sample to a dict with ``keys``."""
    keys = tuple(keys)
    return lambda sample: _as_dict(sample, keys)


def dict_collate(keys: Sequence[str] | None = None) -> Callable[[list], Any]:
    """Collate function that also converts tuple samples to dict batches.

    Use it as ``DataLoader(ds, collate_fn=dict_collate(("pixels", "labels")))``
    when you prefer not to wrap the dataset.
    """
    def _collate(samples: list) -> Any:
        if keys is not None:
            samples = [_as_dict(s, keys) for s in samples]
        return default_collate(samples)

    return _collate


class MapDataset(Dataset):
    """Apply ``fn`` lazily to every sample of ``base``."""

    def __init__(self, base: Any, fn: Callable[[Any], Any]) -> None:
        self.base = base
        self.fn = fn

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> Any:
        return self.fn(self.base[i])

    def __getattr__(self, name: str) -> Any:
        if name in ("base", "fn") or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.base, name)


class TensorDictDataset(Dataset):
    """Dict samples from equally-sized tensors: ``TensorDictDataset(pixels=x, labels=y)``."""

    def __init__(self, **tensors: torch.Tensor) -> None:
        if not tensors:
            raise ValueError("TensorDictDataset needs at least one tensor.")
        sizes = {k: int(v.shape[0]) for k, v in tensors.items()}
        if len(set(sizes.values())) != 1:
            raise ValueError(f"All tensors must share dim 0; got sizes {sizes}.")
        self.tensors = tensors
        self._n = next(iter(sizes.values()))

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int) -> dict:
        return {k: v[i] for k, v in self.tensors.items()}
