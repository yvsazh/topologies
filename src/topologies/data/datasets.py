"""Generic dataset loading (no dataset-specific wrappers).

Any map-style dataset works with ``Trainer`` as-is: dict samples, or
``(x, y)`` tuples mapped onto the recipe's keys. This module only adds
loaders for *formats*, not for named datasets:

- ``image_folder(root)``  — ``root/<class>/<image>`` trees (needs ``[vision]``);
- text: ``read_documents`` / ``lm_dataset`` in ``topologies.data.text``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from topologies.data.adapters import DictDataset, recipe_keys


def require(module: str, extra: str) -> Any:
    """Import an optional dependency or raise an ImportError naming the extra."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ImportError(
            f"{module!r} is required for this feature. Install it with: "
            f"pip install 'topologies[{extra}]'"
        ) from e


def image_folder(
    root: str | Path,
    *,
    image_size: int | tuple[int, int] | None = None,
    transform: Any = None,
) -> DictDataset:
    """Any ``root/<class>/<image>`` tree as ``{"pixels", "labels"}`` samples.

    ``classes`` / ``targets`` of the underlying ImageFolder are forwarded,
    so ``train_val_split(ds, stratify=True)`` and class names just work.
    """
    tv = require("torchvision", "vision")
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"image_folder: {root} is not a directory.")
    T = tv.transforms
    if transform is None:
        steps = [T.Resize(image_size)] if image_size else []
        steps.append(T.ToTensor())
        transform = T.Compose(steps)
    return DictDataset(tv.datasets.ImageFolder(str(root), transform=transform),
                       recipe_keys("vision-classify"))
