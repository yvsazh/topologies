"""DataLoader construction that honors dataset-declared collate functions."""

from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader, IterableDataset


def is_loader(obj: Any) -> bool:
    """True for objects that already yield batches (DataLoader or plain iterables).

    An ``IterableDataset`` yields *samples*, not batches, so it is not a loader.
    """
    if isinstance(obj, DataLoader):
        return True
    if isinstance(obj, IterableDataset):
        return False
    # A map-style dataset has __getitem__ + __len__; anything else iterable is a loader.
    return hasattr(obj, "__iter__") and not hasattr(obj, "__getitem__")


def dataset_collate_fn(dataset: Any) -> Any:
    """The collate function a dataset declares (looks through Subset/wrappers)."""
    seen = 0
    while dataset is not None and seen < 16:
        try:
            fn = getattr(dataset, "collate_fn", None)
        except Exception:  # noqa: BLE001
            fn = None
        if callable(fn):
            return fn
        dataset = getattr(dataset, "dataset", None) or getattr(dataset, "__dict__", {}).get(
            "base"
        )
        seen += 1
    return None


def make_loader(
    dataset: Any,
    batch_size: int = 32,
    *,
    shuffle: bool = False,
    collate_fn: Any = None,
    num_workers: int = 0,
    drop_last: bool = False,
    **kwargs: Any,
) -> DataLoader:
    """``DataLoader`` with ``dataset.collate_fn`` used when none is given."""
    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative")
    persistent = bool(kwargs.get("persistent_workers", False))
    prefetch = kwargs.get("prefetch_factor")
    if persistent and num_workers == 0:
        raise ValueError("persistent_workers requires num_workers > 0")
    if prefetch is not None and num_workers == 0:
        raise ValueError("prefetch_factor requires num_workers > 0")
    if prefetch is not None and (not isinstance(prefetch, int) or prefetch < 1):
        raise ValueError("prefetch_factor must be a positive integer")
    if collate_fn is None:
        collate_fn = dataset_collate_fn(dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=drop_last,
        **kwargs,
    )
