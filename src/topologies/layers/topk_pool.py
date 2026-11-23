import torch
from torch import nn


def _invalid_type(name, expected):
    return ValueError(f"{name} must be {expected}")


class TopKPool(nn.Module):
    component_id = "topk-pool"
    PRESETS = {"tiny": {"k": 1}, "small": {"k": 2}}

    def __init__(self, k, return_indices=False):
        super().__init__()
        if not isinstance(k, int) or isinstance(k, bool):
            raise _invalid_type("k", "an integer")
        if k <= 0:
            raise ValueError("k must be positive")
        if not isinstance(return_indices, bool):
            raise _invalid_type("return_indices", "a boolean")
        self.k = k
        self.return_indices = return_indices

    def forward(self, features, scores, return_indices=None):
        if return_indices is None:
            return_indices = self.return_indices
        elif not isinstance(return_indices, bool):
            raise _invalid_type("return_indices", "a boolean")
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError("features must be a rank-2 tensor")
        if not isinstance(scores, torch.Tensor):
            raise _invalid_type("scores", "a tensor")
        adapter_scores = scores.ndim == 2 and scores.shape[-1] > 1
        if adapter_scores:
            if scores.shape[0] != features.shape[0]:
                raise ValueError("features and score rows must have the same node count")
            scores = scores[:, 0]
        elif scores.ndim != 1:
            raise ValueError("scores must be a rank-1 tensor")
        if features.shape[0] != scores.shape[0]:
            raise ValueError("features and scores must have the same node count")
        if features.device != scores.device:
            raise ValueError("features and scores must be on the same device")
        if features.shape[0] == 0:
            raise ValueError("features must contain at least one node")
        if self.k > features.shape[0] and not adapter_scores:
            raise ValueError("k must not exceed the number of nodes")
        selected_k = min(self.k, features.shape[0]) if adapter_scores else self.k
        selected = torch.topk(scores, selected_k, sorted=False).indices
        indices = torch.sort(selected).values
        pooled = features.index_select(0, indices)
        gates = torch.sigmoid(scores.index_select(0, indices)).unsqueeze(-1)
        pooled = pooled * gates
        if return_indices:
            return pooled, indices
        return pooled
