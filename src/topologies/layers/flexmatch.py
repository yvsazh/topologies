import math
from numbers import Real

import torch
from torch import nn


class FlexMatch(nn.Module):
    component_id = "flexmatch"

    def __init__(self, base_threshold=0.95):
        super().__init__()
        if isinstance(base_threshold, bool) or not isinstance(base_threshold, Real):
            self._raise_value_error("base_threshold must be a finite number between 0 and 1")
        if not math.isfinite(float(base_threshold)) or not 0 < base_threshold < 1:
            raise ValueError("base_threshold must be a finite number between 0 and 1")
        self.base_threshold = float(base_threshold)

    def forward(self, class_counts, num_unlabeled):
        self._validate_counts(class_counts)
        pool_size_value = self._validate_pool_size(num_unlabeled)
        if class_counts.is_floating_point():
            dtype = class_counts.dtype
        else:
            dtype = torch.get_default_dtype()
        if dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32
        if pool_size_value > torch.finfo(dtype).max:
            dtype = torch.float64
        counts = class_counts.to(dtype=dtype)
        pool_size = torch.as_tensor(pool_size_value, device=counts.device, dtype=dtype)
        count_totals = counts.sum(dim=-1)
        if torch.any(count_totals > pool_size_value).item():
            raise ValueError("class_counts must sum to no more than num_unlabeled")
        unused_count = pool_size - count_totals
        maximum_count = counts.max(dim=-1).values
        denominator = torch.maximum(maximum_count, unused_count)
        if counts.ndim == 2:
            denominator = denominator.unsqueeze(-1)
        normalized_counts = counts / denominator
        return self.base_threshold * normalized_counts / (2 - normalized_counts)

    def select(self, confidences, pseudo_labels, class_counts, num_unlabeled):
        thresholds = self(class_counts, num_unlabeled)
        if thresholds.ndim != 1:
            raise ValueError("select requires a single class histogram")
        if not isinstance(confidences, torch.Tensor) or confidences.ndim != 1:
            raise ValueError("confidences must be a one-dimensional tensor")
        if not confidences.is_floating_point() or not torch.isfinite(confidences).all().item():
            raise ValueError("confidences must contain finite floating-point values")
        if torch.any((confidences < 0) | (confidences > 1)).item():
            raise ValueError("confidences must be between 0 and 1")
        if not isinstance(pseudo_labels, torch.Tensor) or pseudo_labels.ndim != 1:
            raise ValueError("pseudo_labels must be a one-dimensional tensor")
        if pseudo_labels.is_floating_point() or pseudo_labels.is_complex() or pseudo_labels.dtype == torch.bool:
            raise ValueError("pseudo_labels must contain integer class indices")
        if confidences.shape != pseudo_labels.shape:
            raise ValueError("confidences and pseudo_labels must have matching shapes")
        if confidences.device != thresholds.device or pseudo_labels.device != thresholds.device:
            raise ValueError("all inputs must be on the same device")
        if torch.any((pseudo_labels < 0) | (pseudo_labels >= thresholds.numel())).item():
            raise ValueError("pseudo_labels must be valid class indices")
        return confidences > thresholds[pseudo_labels.to(dtype=torch.long)]

    @staticmethod
    def _validate_counts(class_counts):
        if not isinstance(class_counts, torch.Tensor) or class_counts.ndim not in (1, 2):
            raise ValueError("class_counts must have shape [classes] or [batch, classes]")
        if class_counts.shape[-1] == 0 or class_counts.ndim == 2 and class_counts.shape[0] == 0:
            raise ValueError("class_counts must contain at least one histogram and class")
        if class_counts.is_complex() or class_counts.dtype == torch.bool:
            raise ValueError("class_counts must contain real numeric values")
        if not torch.isfinite(class_counts).all().item():
            raise ValueError("class_counts must contain finite values")
        if torch.any(class_counts < 0).item():
            raise ValueError("class_counts must be nonnegative")

    @staticmethod
    def _validate_pool_size(num_unlabeled):
        if isinstance(num_unlabeled, torch.Tensor):
            if num_unlabeled.ndim != 0 or num_unlabeled.is_complex() or num_unlabeled.dtype == torch.bool:
                raise ValueError("num_unlabeled must be a positive finite scalar")
            value = float(num_unlabeled.detach().item())
        elif isinstance(num_unlabeled, Real) and not isinstance(num_unlabeled, bool):
            value = float(num_unlabeled)
        else:
            FlexMatch._raise_value_error("num_unlabeled must be a positive finite scalar")
        if not math.isfinite(value) or value <= 0:
            raise ValueError("num_unlabeled must be a positive finite scalar")
        return value

    @staticmethod
    def _raise_value_error(message):
        raise ValueError(message)
