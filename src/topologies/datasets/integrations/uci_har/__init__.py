"""UCI HAR whitespace matrices preserving the provider subject split."""
from pathlib import Path

from topologies.datasets.core import TopologiesDataset


def _numbers(path, cast=float):
    return [[cast(value) for value in line.split()] for line in Path(path).read_text().splitlines() if line.strip()]


class Adapter(TopologiesDataset):
    def load_data(self, paths):
        root = paths[0]
        base = next(root.rglob("X_train.txt")).parent.parent
        def read_partition(part):
            features = _numbers(base / part / f"X_{part}.txt")
            labels = _numbers(base / part / f"y_{part}.txt", int)
            subjects = _numbers(base / part / f"subject_{part}.txt", int)
            if not (len(features) == len(labels) == len(subjects)) or any(len(row) != 561 for row in features):
                raise ValueError("UCI HAR feature/label/subject count or 561-feature width mismatch")
            return {"features": features, "labels": [row[0] for row in labels], "subjects": [row[0] for row in subjects],
                    "sampling_rate_hz": 50, "feature_names": (base / "features.txt").read_text().splitlines()}
        if self.split is None:
            return {part: read_partition(part) for part in ("train", "test")}
        if self.split not in ("train", "test"):
            raise ValueError("UCI HAR uses the provider's train or test subject split")
        return read_partition(self.split)
