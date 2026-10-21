"""Deterministic local Zachary Karate Club graph (not a provider catalog dataset).

The undirected edge list and club assignments are the classic 34-member,
78-edge Zachary graph as exposed by NetworkX's karate_club_graph, derived from
Wayne W. Zachary (1977), ``An Information Flow Model for Conflict and Fission
in Small Groups``, Journal of Anthropological Research 33, 452-473.
No NetworkX import or runtime data download is required.
"""
from __future__ import annotations

import random
from typing import Any

_EDGES = (
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 10), (0, 11), (0, 12),
    (0, 13), (0, 17), (0, 19), (0, 21), (0, 31), (1, 2), (1, 3), (1, 7), (1, 13), (1, 17),
    (1, 19), (1, 21), (1, 30), (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28),
    (2, 32), (3, 7), (3, 12), (3, 13), (4, 6), (4, 10), (5, 6), (5, 10), (5, 16), (6, 16),
    (8, 30), (8, 32), (8, 33), (9, 33), (13, 33), (14, 32), (14, 33), (15, 32), (15, 33),
    (18, 32), (18, 33), (19, 33), (20, 32), (20, 33), (22, 32), (22, 33), (23, 25), (23, 27),
    (23, 29), (23, 32), (23, 33), (24, 25), (24, 27), (24, 31), (25, 31), (26, 29), (26, 33),
    (27, 33), (28, 31), (28, 33), (29, 32), (29, 33), (30, 32), (30, 33), (31, 32), (31, 33),
    (32, 33),
)
_MR_HI = frozenset((0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 19, 21))
REFERENCE = "Zachary (1977), Journal of Anthropological Research 33:452-473; NetworkX karate_club_graph reference encoding."


def karate_club(*, seed: int = 1729, train_fraction: float = 0.6,
                validation_fraction: float = 0.2) -> dict[str, Any]:
    """Return local feature/edge/label tensors and deterministic disjoint node masks.

    Features are identity one-hot vectors (34x34); edges are bidirectional
    directed arcs encoding 78 undirected edges. Club labels are 0=Mr. Hi,
    1=Officer. This is a reproducible local fixture, not a remotely hosted set.
    """
    import torch

    if train_fraction <= 0 or validation_fraction < 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("train/validation fractions must be nonnegative and leave a test partition")
    labels = [0 if node in _MR_HI else 1 for node in range(34)]
    buckets = [[], [], []]
    rng = random.Random(seed)
    for label in (0, 1):
        members = [node for node, value in enumerate(labels) if value == label]
        rng.shuffle(members)
        train_end = max(1, int(len(members) * train_fraction))
        validation_end = max(train_end, int(len(members) * (train_fraction + validation_fraction)))
        buckets[0].extend(members[:train_end])
        buckets[1].extend(members[train_end:validation_end])
        buckets[2].extend(members[validation_end:])
    directed = sorted((left, right) for edge in _EDGES for left, right in (edge, edge[::-1]))
    return {
        "features": torch.eye(34, dtype=torch.float32),
        "edge_index": torch.tensor(directed, dtype=torch.long).t().contiguous(),
        "labels": torch.tensor(labels, dtype=torch.long),
        "club_names": ["Mr. Hi", "Officer"],
        "train_mask": _mask(buckets[0]), "validation_mask": _mask(buckets[1]), "test_mask": _mask(buckets[2]),
        "undirected_edges": _EDGES, "reference": REFERENCE,
    }


def _mask(indices: list[int]):
    import torch
    mask = torch.zeros(34, dtype=torch.bool)
    if indices:
        mask[indices] = True
    return mask
