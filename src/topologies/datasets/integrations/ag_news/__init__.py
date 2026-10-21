"""AG News CSV parser with explicit provider label mapping."""
import csv
from pathlib import Path

from topologies.datasets.core import TopologiesDataset


class Adapter(TopologiesDataset):
    LABELS = {1: "world", 2: "sports", 3: "business", 4: "science-technology"}

    def load_data(self, paths):
        results = {}
        for resource, path in zip(self._resources(), paths):
            rows = []
            with Path(path).open(encoding="utf-8", newline="") as source:
                for label, title, description in csv.reader(source):
                    source_id = int(label)
                    if source_id not in self.LABELS:
                        raise ValueError(f"unknown AG News source label {source_id}")
                    rows.append({"text": f"{title} {description}".strip(), "label": self.LABELS[source_id], "source_label": source_id})
            results[resource["name"]] = rows
        return next(iter(results.values())) if self.split is not None else results
