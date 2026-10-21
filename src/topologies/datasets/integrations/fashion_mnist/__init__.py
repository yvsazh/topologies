"""Fashion-MNIST provider IDX adapter; validation/test are provider-owned."""
from __future__ import annotations

from topologies.datasets.core import TopologiesDataset


class Adapter(TopologiesDataset):
    def load_data(self, paths):
        arrays = super().load_data(paths)
        by_name = arrays if isinstance(arrays, dict) else dict(zip((resource["name"] for resource in self._resources()), arrays))
        partitions = ("train", "test") if self.split is None else (self.split,)
        result = {}
        for prefix in partitions:
            images, labels = by_name[prefix + "-images"], by_name[prefix + "-labels"]
            if images.ndim != 3 or tuple(images.shape[1:]) != (28, 28) or labels.ndim != 1 or len(images) != len(labels):
                raise ValueError(f"Fashion-MNIST {prefix} IDX image/label geometry or count mismatch")
            result[prefix] = {"images": images, "labels": labels}
        return result[partitions[0]] if self.split is not None else result
