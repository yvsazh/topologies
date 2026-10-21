"""UCI Iris provider adapter and tabular preprocessing primitives."""
from __future__ import annotations

import csv
import math
from pathlib import Path

from topologies.datasets.core import TopologiesDataset


class Adapter(TopologiesDataset):
    """Parse provider CSV records without fetching resources or fitting on held-out data."""
    target = "class"

    def parse_csv(self, path, *, header=True, fieldnames=None, delimiter=","):
        with Path(path).open(newline="", encoding="utf-8") as stream:
            rows = csv.DictReader(stream, fieldnames=None if header else fieldnames, delimiter=delimiter)
            return list(rows)

    def load_data(self, paths):
        """Return the provider's original row representation, as dictionaries."""
        path = paths[0] / "iris.data" if paths[0].is_dir() else paths[0]
        return self.parse_csv(path, header=False, fieldnames=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])

    def fit(self, rows, numeric=(), categorical=()):
        """Fit means/scales and category vocabularies on train rows only."""
        for key in numeric:
            values = [float(r[key]) for r in rows if r.get(key) not in (None, "", "NA", "NaN")]
            if not values:
                self.numeric_stats[key] = (0.0, 1.0)
                continue
            mean = sum(values) / len(values)
            scale = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)) or 1.0
            self.numeric_stats[key] = (mean, scale)
        self.categories = {key: sorted({r[key] for r in rows if r.get(key) not in (None, "", "NA")})
                           for key in categorical}
        return self

    def transform(self, rows, numeric=(), categorical=()):
        output = []
        for row in rows:
            values = []
            for key in numeric:
                mean, scale = self.numeric_stats[key]
                raw = row.get(key)
                values.append((float(raw) if raw not in (None, "", "NA", "NaN") else mean) - mean)
                values[-1] /= scale
            for key in categorical:
                values.append(self.categories[key].index(row[key]) if row.get(key) in self.categories[key] else -1)
            output.append((values, row.get(self.target)))
        return output


def chronological_windows(values, train_end, validation_end, window, horizon=1):
    """Create windows wholly contained in their chronological partition."""
    if not (0 < train_end < validation_end < len(values)) or window < 1 or horizon < 1:
        raise ValueError("invalid split boundaries or window dimensions")
    partitions = ((0, train_end), (train_end, validation_end), (validation_end, len(values)))
    return [[(values[i:i + window], values[i + window:i + window + horizon])
             for i in range(start, end - window - horizon + 1)]
            for start, end in partitions]


def stratified_indices(labels, *, seed=0, fractions=(0.7, 0.15, 0.15)):
    """Deterministic class-stratified train/validation/test indices (not provider splits)."""
    import random
    if len(fractions) != 3 or any(value < 0 for value in fractions) or abs(sum(fractions) - 1) > 1e-8:
        raise ValueError("fractions must be three nonnegative values summing to one")
    by_label = {}
    for index, label in enumerate(labels):
        by_label.setdefault(label, []).append(index)
    rng = random.Random(seed)
    result = [[], [], []]
    for indices in by_label.values():
        rng.shuffle(indices)
        first = round(len(indices) * fractions[0])
        second = round(len(indices) * (fractions[0] + fractions[1]))
        for bucket, values in zip(result, (indices[:first], indices[first:second], indices[second:])):
            bucket.extend(values)
    return tuple(sorted(partition) for partition in result)
