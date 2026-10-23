import math

import torch
from torch import nn


def _validation_error(message):
    return ValueError(message)


class TabR(nn.Module):
    component_id = "tabr"

    def __init__(
        self,
        input_dim,
        task,
        num_classes=None,
        context_size=96,
        attention_dropout=0.38920071545944357,
        ffn_dropout=0.38852797479169876,
        scale_scores=False,
        normalize_representations=False,
    ):
        super().__init__()
        if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if not isinstance(task, str) or task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'")
        if task == "classification":
            if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
                raise ValueError("num_classes must be an integer greater than one for classification")
        elif num_classes is not None:
            raise ValueError("num_classes must be None for regression")
        if isinstance(context_size, bool) or not isinstance(context_size, int) or context_size <= 0:
            raise ValueError("context_size must be a positive integer")
        for name, value in (("attention_dropout", attention_dropout), ("ffn_dropout", ffn_dropout)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise _validation_error(f"{name} must be a number in [0, 1)")
            if not math.isfinite(value) or value < 0.0 or value >= 1.0:
                raise ValueError(f"{name} must be a number in [0, 1)")
        if not isinstance(scale_scores, bool):
            raise _validation_error("scale_scores must be a boolean")
        if not isinstance(normalize_representations, bool):
            raise _validation_error("normalize_representations must be a boolean")

        self.input_dim = input_dim
        self.task = task
        self.num_classes = num_classes
        self.context_size = context_size
        self.scale_scores = scale_scores
        self.normalize_representations = normalize_representations
        self.shared_norm = nn.LayerNorm(input_dim) if normalize_representations else nn.Identity()
        self.key_projection = nn.Linear(input_dim, input_dim, bias=False)
        if task == "classification":
            self.label_embedding = nn.Embedding(num_classes, input_dim)
            self.label_projection = None
        else:
            self.label_embedding = None
            self.label_projection = nn.Linear(1, input_dim)
        self.value_network = nn.Sequential(
            nn.Linear(input_dim, 2 * input_dim),
            nn.ReLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(2 * input_dim, input_dim, bias=False),
        )
        self.attention_dropout = nn.Dropout(attention_dropout)

    def _validate_inputs(self, query_repr, context_repr, context_y, candidate_mask):
        if not isinstance(query_repr, torch.Tensor) or query_repr.ndim != 2 or query_repr.shape[0] == 0:
            raise ValueError("query_repr must be a non-empty rank-2 tensor with shape [B, input_dim]")
        if not isinstance(context_repr, torch.Tensor) or context_repr.ndim != 2 or context_repr.shape[0] == 0:
            raise ValueError("context_repr must be a non-empty rank-2 tensor with shape [N, input_dim]")
        if query_repr.shape[1] != self.input_dim or context_repr.shape[1] != self.input_dim:
            raise ValueError("query_repr and context_repr last dimensions must match input_dim")
        if not torch.is_floating_point(query_repr) or not torch.is_floating_point(context_repr):
            raise ValueError("query_repr and context_repr must use floating-point dtypes")
        if query_repr.device != context_repr.device or query_repr.dtype != context_repr.dtype:
            raise ValueError("query_repr and context_repr must have the same device and dtype")
        if not isinstance(context_y, torch.Tensor):
            raise _validation_error("context_y must be a tensor")
        if context_y.ndim == 0:
            raise ValueError("context_y must have one row per context candidate")
        if context_y.shape[0] != context_repr.shape[0]:
            raise ValueError("context_y must contain the same number of rows as context_repr")
        if context_y.device != query_repr.device:
            raise ValueError("context_y must be on the same device as query_repr")
        if self.task == "classification":
            if context_y.ndim != 1 and not (context_y.ndim == 2 and context_y.shape[1] == 1):
                raise ValueError("classification context_y must have shape [N] or [N, 1]")
            if context_y.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                raise ValueError("classification context_y must contain integer class IDs")
            if torch.any(context_y < 0) or torch.any(context_y >= self.num_classes):
                raise ValueError("classification context_y class IDs must be in [0, num_classes)")
        else:
            if context_y.ndim != 1 and not (context_y.ndim == 2 and context_y.shape[1] == 1):
                raise ValueError("regression context_y must have shape [N] or [N, 1]")
            if not torch.is_floating_point(context_y):
                raise ValueError("regression context_y must use a floating-point dtype")
        if candidate_mask is not None:
            expected_shape = (query_repr.shape[0], context_repr.shape[0])
            if not isinstance(candidate_mask, torch.Tensor) or candidate_mask.shape != expected_shape:
                raise ValueError("candidate_mask must have shape [B, N]")
            if candidate_mask.dtype != torch.bool:
                raise ValueError("candidate_mask must be a boolean tensor")
            if candidate_mask.device != query_repr.device:
                raise ValueError("candidate_mask must be on the same device as query_repr")

    def forward(self, query_repr, context_repr, context_y, candidate_mask=None):
        self._validate_inputs(query_repr, context_repr, context_y, candidate_mask)
        query = self.shared_norm(query_repr)
        context = self.shared_norm(context_repr)
        query_keys = self.key_projection(query)
        context_keys = self.key_projection(context)
        scores = -((query_keys[:, None, :] - context_keys[None, :, :]) ** 2).sum(dim=-1)
        if self.scale_scores:
            scores = scores * (self.input_dim ** -0.5)
        if candidate_mask is None:
            eligible = torch.ones_like(scores, dtype=torch.bool)
        else:
            eligible = candidate_mask
            scores = scores.masked_fill(~eligible, -torch.inf)
        if torch.any(eligible.sum(dim=1) < self.context_size):
            raise ValueError("each query must have at least context_size eligible candidates")
        selected_scores, selected_indices = scores.topk(self.context_size, dim=1)
        weights = self.attention_dropout(selected_scores.softmax(dim=1))
        selected_keys = context_keys[selected_indices]
        key_differences = query_keys[:, None, :] - selected_keys
        if self.task == "classification":
            if context_y.ndim == 2:
                context_y = context_y.squeeze(-1)
            selected_labels = self.label_embedding(context_y[selected_indices].long())
        else:
            selected_targets = context_y[selected_indices]
            if selected_targets.ndim == 2:
                selected_targets = selected_targets.squeeze(-1)
            selected_labels = self.label_projection(selected_targets.unsqueeze(-1))
        values = selected_labels + self.value_network(key_differences)
        return query + (weights.unsqueeze(-1) * values).sum(dim=1)
