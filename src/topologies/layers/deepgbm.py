import math

import torch
from torch import nn


class DeepGBM(nn.Module):
    component_id = "deepgbm"
    PRESETS = {
        "tiny": {"groups": 2, "trees_per_group": 2, "num_leaf_ids": 5, "embedding_dim": 4},
        "small": {"groups": 4, "trees_per_group": 4, "num_leaf_ids": 17, "embedding_dim": 8},
        "base": {"groups": 10, "trees_per_group": 10, "num_leaf_ids": 129, "embedding_dim": 20},
    }

    def __init__(
        self,
        groups,
        trees_per_group,
        num_leaf_ids,
        embedding_dim,
        dropout=0.5,
        task="regression",
    ):
        super().__init__()
        dimensions = {
            "groups": groups,
            "trees_per_group": trees_per_group,
            "num_leaf_ids": num_leaf_ids,
            "embedding_dim": embedding_dim,
        }
        for name, value in dimensions.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(dropout, (int, float))
            or isinstance(dropout, bool)
            or not math.isfinite(dropout)
            or not 0.0 <= dropout <= 1.0
        ):
            raise ValueError("dropout must be a finite number between 0 and 1")
        if task not in ("regression", "binary"):
            raise ValueError("task must be 'regression' or 'binary'")
        self.groups = groups
        self.trees_per_group = trees_per_group
        self.num_leaf_ids = num_leaf_ids
        self.embedding_dim = embedding_dim
        self.task = task
        self.embedding_weight = nn.Parameter(
            torch.empty(groups, trees_per_group * num_leaf_ids, embedding_dim)
        )
        self.group_head_weight = nn.Parameter(torch.empty(groups, embedding_dim))
        self.group_head_bias = nn.Parameter(torch.empty(groups))
        self.batch_norm = nn.BatchNorm1d(groups * embedding_dim)
        self.dropout = nn.Dropout(float(dropout))
        nn.init.normal_(self.embedding_weight, mean=0.0, std=math.sqrt(1.0 / trees_per_group))
        bound = math.sqrt(6.0 / (embedding_dim + 1))
        nn.init.uniform_(self.group_head_weight, -bound, bound)
        nn.init.zeros_(self.group_head_bias)

    @staticmethod
    def embedding_loss(group_outputs, group_targets):
        if not isinstance(group_outputs, torch.Tensor) or not isinstance(group_targets, torch.Tensor):
            raise TypeError("group_outputs and group_targets must be tensors")
        if group_outputs.shape != group_targets.shape:
            raise ValueError("group_outputs and group_targets must have the same shape")
        return torch.nn.functional.mse_loss(group_outputs, group_targets)

    def forward(self, leaf_indices, return_group_outputs=True):
        if not isinstance(leaf_indices, torch.Tensor):
            raise TypeError("leaf_indices must be a tensor")
        if leaf_indices.ndim != 2:
            raise ValueError("leaf_indices must have two dimensions with shape [batch, groups * trees_per_group]")
        expected_width = self.groups * self.trees_per_group
        if leaf_indices.shape[1] != expected_width:
            raise ValueError(f"leaf_indices width must equal groups * trees_per_group ({expected_width})")
        if leaf_indices.shape[0] == 0:
            raise ValueError("leaf_indices batch dimension must be positive")
        if leaf_indices.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("leaf_indices must have an integer dtype")
        if (leaf_indices < 0).any() or (leaf_indices >= self.num_leaf_ids).any():
            raise ValueError(f"leaf indices outside [0, {self.num_leaf_ids}) are invalid")
        batch_size = leaf_indices.shape[0]
        grouped_indices = leaf_indices.to(dtype=torch.long).reshape(
            batch_size, self.groups, self.trees_per_group
        )
        one_hot = torch.nn.functional.one_hot(grouped_indices, num_classes=self.num_leaf_ids)
        one_hot = one_hot.reshape(batch_size, self.groups, -1).to(dtype=self.embedding_weight.dtype)
        embeddings = torch.einsum("bgl,gld->bgd", one_hot, self.embedding_weight)
        embeddings = embeddings.reshape(batch_size, -1)
        if self.training and batch_size == 1:
            embeddings = torch.nn.functional.batch_norm(
                embeddings,
                self.batch_norm.running_mean,
                self.batch_norm.running_var,
                self.batch_norm.weight,
                self.batch_norm.bias,
                False,
                self.batch_norm.momentum,
                self.batch_norm.eps,
            )
        else:
            embeddings = self.batch_norm(embeddings)
        embeddings = self.dropout(embeddings).reshape(batch_size, self.groups, self.embedding_dim)
        group_outputs = (
            torch.einsum("bgd,gd->bg", embeddings, self.group_head_weight).unsqueeze(-1)
            + self.group_head_bias.reshape(1, self.groups, 1)
        )
        prediction = group_outputs.sum(dim=1)
        if self.task == "binary":
            prediction = prediction.sigmoid()
        if return_group_outputs:
            return prediction, group_outputs
        return prediction
