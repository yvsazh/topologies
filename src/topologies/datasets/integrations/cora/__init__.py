"""LINQS Cora content/cites parser with reproducible transductive node masks."""
import random

from topologies.datasets.core import TopologiesDataset


class Adapter(TopologiesDataset):
    def load_data(self, paths):
        root = paths[0]
        content = next(root.rglob("cora.content"))
        cites = next(root.rglob("cora.cites"))
        node_ids, features, labels = [], [], []
        for line in content.read_text().splitlines():
            fields = line.split()
            if len(fields) < 3:
                raise ValueError("malformed Cora content row")
            node_ids.append(fields[0])
            features.append([float(value) for value in fields[1:-1]])
            labels.append(fields[-1])
        if len(set(node_ids)) != len(node_ids) or not node_ids:
            raise ValueError("Cora node IDs must be nonempty and unique")
        node_index = {node_id: index for index, node_id in enumerate(node_ids)}
        edges = []
        for line in cites.read_text().splitlines():
            source, target = line.split()
            if source not in node_index or target not in node_index:
                raise ValueError(f"Cora citation references unknown node: {source!r}, {target!r}")
            edges.append((node_index[source], node_index[target]))
        rng = random.Random(1729)
        indices = list(range(len(node_ids)))
        rng.shuffle(indices)
        train_end, val_end = int(len(indices) * 0.6), int(len(indices) * 0.8)
        masks = {"train": sorted(indices[:train_end]), "validation": sorted(indices[train_end:val_end]),
                 "test": sorted(indices[val_end:])}
        return {"node_ids": node_ids, "features": features, "labels": labels, "edges": edges,
                "masks": masks, "protocol": "full graph is visible transductively; deterministic seed 1729 node masks; not Planetoid split"}
