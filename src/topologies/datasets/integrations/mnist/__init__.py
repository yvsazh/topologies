"""MNIST provider IDX adapter; optional NumPy is imported only on load."""
from __future__ import annotations

from topologies.datasets.core import TopologiesDataset


class Adapter(TopologiesDataset):
    def load_data(self, paths):
        arrays = super().load_data(paths)
        by_name = arrays if isinstance(arrays, dict) else dict(zip((resource["name"] for resource in self._resources()), arrays))
        partitions = ("train", "test") if self.split is None else (self.split,)
        result = {}
        for partition in partitions:
            images, labels = by_name[partition + "-images"], by_name[partition + "-labels"]
            if images.ndim != 3 or tuple(images.shape[1:]) != (28, 28) or labels.ndim != 1 or len(images) != len(labels):
                raise ValueError(f"MNIST {partition} IDX image/label geometry or count mismatch")
            result[partition] = {"images": images, "labels": labels}
        return result[partitions[0]] if self.split is not None else result
