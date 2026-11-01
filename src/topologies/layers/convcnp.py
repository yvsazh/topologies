import math

import torch
from torch import nn


def _raise_value_error(message):
    raise ValueError(message)


class ConvCNP(nn.Module):
    component_id = "convcnp"

    def __init__(self, length_scale_init=0.03125, learn_length_scale=True, epsilon=1e-8):
        super().__init__()
        if isinstance(length_scale_init, bool) or not isinstance(length_scale_init, (int, float)):
            _raise_value_error("length_scale_init must be a finite positive number")
        if not math.isfinite(length_scale_init) or length_scale_init <= 0:
            raise ValueError("length_scale_init must be a finite positive number")
        if not isinstance(learn_length_scale, bool):
            _raise_value_error("learn_length_scale must be a bool")
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
            _raise_value_error("epsilon must be a finite positive number")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be a finite positive number")
        log_length_scale = torch.tensor(math.log(length_scale_init), dtype=torch.get_default_dtype())
        if learn_length_scale:
            self.log_length_scale = nn.Parameter(log_length_scale)
        else:
            self.register_buffer("log_length_scale", log_length_scale)
        self.epsilon = float(epsilon)

    @property
    def length_scale(self):
        return self.log_length_scale.exp()

    def forward(self, context_x, context_y, query_x):
        inputs = (("context_x", context_x), ("context_y", context_y), ("query_x", query_x))
        for name, value in inputs:
            if not isinstance(value, torch.Tensor):
                _raise_value_error(f"{name} must be a tensor")
            if not value.is_floating_point():
                raise ValueError(f"{name} must have a floating-point dtype")
        if context_x.ndim != 3 or context_x.shape[-1] != 1:
            raise ValueError("context_x must have shape [batch, context, 1]")
        if context_y.ndim != 3:
            raise ValueError("context_y must have shape [batch, context, channels]")
        if query_x.ndim != 3 or query_x.shape[-1] != 1:
            raise ValueError("query_x must have shape [batch, query, 1]")
        if context_x.shape[:2] != context_y.shape[:2]:
            raise ValueError("context_x and context_y must have matching batch and context dimensions")
        if context_x.shape[0] != query_x.shape[0]:
            raise ValueError("context_x and query_x must have matching batch dimensions")
        if context_y.shape[-1] < 1:
            raise ValueError("context_y must contain at least one value channel")
        if context_x.device != context_y.device or context_x.device != query_x.device:
            raise ValueError("all inputs must be on the same device")
        if context_x.dtype != context_y.dtype or context_x.dtype != query_x.dtype:
            raise ValueError("all inputs must have the same dtype")
        output_dtype = context_x.dtype
        compute_dtype = torch.float32 if output_dtype in (torch.float16, torch.bfloat16) else output_dtype
        context_x = context_x.to(dtype=compute_dtype)
        context_y = context_y.to(dtype=compute_dtype)
        query_x = query_x.to(dtype=compute_dtype)
        distances = query_x.unsqueeze(2) - context_x.unsqueeze(1)
        scale = self.log_length_scale.to(dtype=compute_dtype, device=distances.device).exp()
        weights = torch.exp(-0.5 * distances.square().squeeze(-1) / scale.square())
        density = weights.sum(dim=-1, keepdim=True)
        weighted_values = torch.matmul(weights, context_y)
        normalized_values = weighted_values / (density + self.epsilon)
        return torch.cat((density, normalized_values), dim=-1).to(dtype=output_dtype)
