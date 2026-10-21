"""Jena climate chronology, boundary-safe windows, and train-only scaling."""
import csv

from topologies.datasets.core import TopologiesDataset


class Adapter(TopologiesDataset):
    def load_data(self, paths):
        csv_files = sorted(paths[0].rglob("*.csv")) if paths[0].is_dir() else paths
        if len(csv_files) != 1:
            raise ValueError("Jena archive must contain exactly one climate CSV")
        with csv_files[0].open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
        rows.sort(key=lambda row: row.get("Date Time", ""))
        return rows

    @staticmethod
    def chronological_partitions(rows, fractions=(0.7, 0.15, 0.15)):
        if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-8:
            raise ValueError("expected three chronological fractions summing to one")
        first, second = int(len(rows) * fractions[0]), int(len(rows) * (fractions[0] + fractions[1]))
        return rows[:first], rows[first:second], rows[second:]

    @staticmethod
    def fit_normalizer(train_rows, columns):
        stats = {}
        for column in columns:
            values = [float(row[column]) for row in train_rows if row.get(column) not in (None, "")]
            if not values:
                raise ValueError(f"training partition has no values for {column}")
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            stats[column] = (mean, variance ** 0.5 or 1.0)
        return stats

    @staticmethod
    def normalize(rows, columns, stats):
        """Apply statistics previously fitted exclusively to chronological training rows."""
        return [[(float(row[column]) - stats[column][0]) / stats[column][1] for column in columns]
                for row in rows]

    @classmethod
    def prepare_forecast_splits(cls, rows, columns, *, window, horizon, fractions=(0.7, 0.15, 0.15)):
        train_rows, validation_rows, test_rows = cls.chronological_partitions(rows, fractions)
        stats = cls.fit_normalizer(train_rows, columns)
        partitions = [cls.normalize(part, columns, stats) for part in (train_rows, validation_rows, test_rows)]
        # The caller receives independent windows from each partition; context never straddles boundaries.
        return {"train": cls.windows(partitions[0], columns=range(len(columns)), window=window, horizon=horizon),
                "validation": cls.windows(partitions[1], columns=range(len(columns)), window=window, horizon=horizon),
                "test": cls.windows(partitions[2], columns=range(len(columns)), window=window, horizon=horizon),
                "normalizer": stats}

    @staticmethod
    def windows(rows, columns, *, window, horizon):
        if window < 1 or horizon < 1:
            raise ValueError("window and horizon must be positive")
        values = [[float(row[column]) for column in columns] for row in rows]
        return [(values[index:index + window], values[index + window:index + window + horizon])
                for index in range(max(0, len(values) - window - horizon + 1))]
