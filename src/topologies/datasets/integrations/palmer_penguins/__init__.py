"""Palmer Penguins CSV adapter; fit transformations on training rows only."""
from __future__ import annotations

import csv
import math
from pathlib import Path

from topologies.datasets.core import TopologiesDataset


class Adapter(TopologiesDataset):
    target = "species"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.numeric_stats = {}
        self.categories = {}

    def load_data(self, paths):
        rows = []
        for path in paths:
            with Path(path).open(encoding=self.options.get("encoding", "utf-8"), newline="") as stream:
                rows.extend(csv.DictReader(stream))
        return rows

    def fit_preprocessor(self, train_rows, *, numeric, categorical):
        """Fit missing-value means/scales and category IDs using train rows only."""
        for name in numeric:
            values = [float(row[name]) for row in train_rows if row.get(name) not in (None, "", "NA", "NaN")]
            mean = sum(values) / len(values) if values else 0.0
            scale = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)) if values else 1.0
            self.numeric_stats[name] = (mean, scale or 1.0)
        self.categories = {name: {value: index for index, value in enumerate(sorted({row[name] for row in train_rows if row.get(name) not in (None, "", "NA")}))} for name in categorical}
        return self

    def transform(self, rows, *, numeric, categorical):
        result = []
        for row in rows:
            features = []
            for name in numeric:
                mean, scale = self.numeric_stats[name]
                value = row.get(name)
                features.append(((float(value) if value not in (None, "", "NA", "NaN") else mean) - mean) / scale)
            features.extend(self.categories[name].get(row.get(name), -1) for name in categorical)
            result.append((features, row.get(self.target)))
        return result
