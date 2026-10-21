"""CIFAR-10 adapter for the official Python archive using restricted unpickling."""
from __future__ import annotations

import pickle
from pathlib import Path

from topologies.datasets.core import TopologiesDataset


class _DataOnlyUnpickler(pickle.Unpickler):
    """Allow only NumPy's array reconstruction primitives used by official batches."""
    def find_class(self, module, name):
        allowed = {
            ("numpy", "ndarray"): __import__("numpy").ndarray,
            ("numpy", "dtype"): __import__("numpy").dtype,
        }
        for module_name in ("numpy.core.multiarray", "numpy._core.multiarray"):
            try:
                import importlib
                allowed[(module_name, "_reconstruct")] = importlib.import_module(module_name)._reconstruct
            except (ImportError, AttributeError):
                pass
        if (module, name) in allowed:
            return allowed[(module, name)]
        raise pickle.UnpicklingError(f"global reference forbidden in CIFAR batch: {module}.{name}")


class Adapter(TopologiesDataset):
    @staticmethod
    def _partition(root: Path, names: list[str]):
        import numpy as np
        data, labels = [], []
        for name in names:
            candidates = list(root.rglob(name))
            if len(candidates) != 1 or not candidates[0].is_file():
                raise ValueError(f"expected exactly one extracted CIFAR provider file {name}")
            with candidates[0].open("rb") as stream:
                batch = _DataOnlyUnpickler(stream, encoding="bytes").load()
            if not isinstance(batch, dict) or b"data" not in batch:
                raise ValueError(f"invalid CIFAR provider batch {name}")
            pixels = np.asarray(batch[b"data"], dtype=np.uint8)
            batch_labels = batch.get(b"labels", batch.get(b"fine_labels"))
            if pixels.ndim != 2 or pixels.shape[1] != 3072 or not isinstance(batch_labels, list) or len(batch_labels) != len(pixels):
                raise ValueError(f"invalid CIFAR provider batch geometry/labels: {name}")
            data.append(pixels.reshape(-1, 3, 32, 32))
            labels.extend(batch_labels)
        return {"images": np.concatenate(data, axis=0), "labels": np.asarray(labels, dtype=np.int64)}

    def load_data(self, paths):
        root = paths[0]
        if self.split is None:
            return {"train": self._partition(root, [f"data_batch_{index}" for index in range(1, 6)]),
                    "test": self._partition(root, ["test_batch"])}
        names = [f"data_batch_{index}" for index in range(1, 6)] if self.split == "train" else ["test_batch"]
        return self._partition(root, names)
