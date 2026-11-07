import math

import torch
from torch import nn


class _CompensatedSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, embedding, scale, max_scale):
        ctx.save_for_backward(embedding)
        ctx.scale = scale
        ctx.max_scale = max_scale
        bounded = embedding.clamp(-6.0, 6.0)
        return torch.sigmoid((scale * bounded).clamp(-50.0, 50.0))

    @staticmethod
    def backward(ctx, gradient):
        (embedding,) = ctx.saved_tensors
        bounded = embedding.clamp(-6.0, 6.0)
        scaled = (ctx.scale * bounded).clamp(-50.0, 50.0)
        active = (embedding.abs() <= 6.0) & (scaled.abs() <= 50.0)
        standard = torch.sigmoid(bounded)
        compensated = ctx.max_scale * standard * (1.0 - standard)
        return gradient * compensated * active, None, None


class _ConditionedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, gradient_mask):
        ctx.save_for_backward(inputs, weight, gradient_mask)
        return torch.nn.functional.linear(inputs, weight)

    @staticmethod
    def backward(ctx, gradient):
        inputs, weight, gradient_mask = ctx.saved_tensors
        gradient_inputs = gradient.matmul(weight)
        flattened_gradient = gradient.reshape(-1, gradient.shape[-1])
        flattened_inputs = inputs.reshape(-1, inputs.shape[-1])
        gradient_weight = flattened_gradient.transpose(0, 1).matmul(flattened_inputs)
        return gradient_inputs, gradient_weight * gradient_mask, None


class HAT(nn.Module):
    component_id = "hat"
    PRESETS = {
        "tiny": {"in_features": 2, "out_features": 2, "num_tasks": 2},
        "small": {"in_features": 32, "out_features": 64, "num_tasks": 4},
    }

    def __init__(self, in_features, out_features, num_tasks, s_max=400, c=0.75):
        super().__init__()
        if type(in_features) is not int or in_features <= 0:
            raise ValueError("in_features must be a positive integer")
        if type(out_features) is not int or out_features <= 0:
            raise ValueError("out_features must be a positive integer")
        if type(num_tasks) is not int or num_tasks <= 0:
            raise ValueError("num_tasks must be a positive integer")
        if type(s_max) not in (int, float) or not math.isfinite(s_max) or s_max < 1:
            raise ValueError("s_max must be a finite number greater than or equal to 1")
        if type(c) not in (int, float) or not math.isfinite(c) or c < 0:
            raise ValueError("c must be a finite nonnegative number")
        self.in_features = in_features
        self.out_features = out_features
        self.num_tasks = num_tasks
        self.s_max = float(s_max)
        self.c = float(c)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.task_embeddings = nn.Parameter(torch.empty(num_tasks, out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.normal_(self.task_embeddings, mean=0.0, std=1.0)

    @staticmethod
    def _validate_attention(attention, name, expected_size=None):
        if not isinstance(attention, torch.Tensor) or attention.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional tensor")
        if expected_size is not None and attention.shape[0] != expected_size:
            raise ValueError(f"{name} must have length {expected_size}")
        if not attention.is_floating_point() or not torch.isfinite(attention).all():
            raise ValueError(f"{name} must contain finite floating-point values")
        if (attention < 0).any() or (attention > 1).any():
            raise ValueError(f"{name} values must be in [0, 1]")

    def _validate_scale(self, scale):
        if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("s must be a finite positive number")

    def _validate_task(self, task_id):
        if type(task_id) is not int or not 0 <= task_id < self.num_tasks:
            raise ValueError(f"task_id must be an integer in [0, {self.num_tasks})")

    def attention(self, task_id, s=1.0):
        self._validate_task(task_id)
        self._validate_scale(s)
        with torch.no_grad():
            self.task_embeddings.clamp_(-6.0, 6.0)
        return _CompensatedSigmoid.apply(self.task_embeddings[task_id], float(s), self.s_max)

    @staticmethod
    def cumulative_attention(current_attention, previous_attention=None):
        HAT._validate_attention(current_attention, "current_attention")
        if previous_attention is None:
            return current_attention
        HAT._validate_attention(previous_attention, "previous_attention", current_attention.shape[0])
        if previous_attention.device != current_attention.device or previous_attention.dtype != current_attention.dtype:
            raise ValueError("previous_attention must match current_attention device and dtype")
        return torch.maximum(current_attention, previous_attention)

    @staticmethod
    def condition_weight_gradient(gradient, previous_output_attention, previous_input_attention):
        if not isinstance(gradient, torch.Tensor) or gradient.ndim != 2:
            raise ValueError("gradient must be a two-dimensional tensor")
        if not gradient.is_floating_point() or not torch.isfinite(gradient).all():
            raise ValueError("gradient must contain finite floating-point values")
        HAT._validate_attention(previous_output_attention, "previous_output_attention", gradient.shape[0])
        HAT._validate_attention(previous_input_attention, "previous_input_attention", gradient.shape[1])
        if previous_output_attention.device != gradient.device or previous_input_attention.device != gradient.device:
            raise ValueError("attention masks and gradient must be on the same device")
        if previous_output_attention.dtype != gradient.dtype or previous_input_attention.dtype != gradient.dtype:
            raise ValueError("attention masks and gradient must have the same dtype")
        protection = torch.minimum(previous_output_attention[:, None], previous_input_attention[None, :])
        return (1.0 - protection) * gradient

    @staticmethod
    def _attention_groups(current_attention, previous_attention):
        current_group = current_attention if isinstance(current_attention, (tuple, list)) else (current_attention,)
        previous_group = previous_attention if isinstance(previous_attention, (tuple, list)) else (previous_attention,)
        if len(current_group) != len(previous_group) or not current_group:
            raise ValueError("current and previous attention groups must have the same nonzero length")
        return current_group, previous_group

    @staticmethod
    def capacity_regularizer(current_attention, previous_attention=None):
        if previous_attention is None:
            if isinstance(current_attention, (tuple, list)):
                previous_attention = tuple(torch.zeros_like(value) for value in current_attention)
            else:
                previous_attention = torch.zeros_like(current_attention)
        current_group, previous_group = HAT._attention_groups(current_attention, previous_attention)
        numerator = None
        denominator = None
        for current, previous in zip(current_group, previous_group):
            HAT._validate_attention(current, "current_attention")
            HAT._validate_attention(previous, "previous_attention", current.shape[-1])
            if current.shape != previous.shape:
                raise ValueError("current and previous attention shapes must match")
            if current.device != previous.device or current.dtype != previous.dtype:
                raise ValueError("current and previous attention must match device and dtype")
            term = current * (1.0 - previous)
            available = 1.0 - previous
            numerator = term.sum() if numerator is None else numerator + term.sum()
            denominator = available.sum() if denominator is None else denominator + available.sum()
        return numerator / denominator.clamp_min(torch.finfo(denominator.dtype).tiny)

    def capacity_penalty(self, current_attention, previous_attention=None):
        return self.c * self.capacity_regularizer(current_attention, previous_attention)

    def sigmoid_scale(self, batch_index, total_batches):
        if type(total_batches) is not int or total_batches < 2:
            raise ValueError("total_batches must be an integer greater than or equal to 2")
        if type(batch_index) is not int or not 1 <= batch_index <= total_batches:
            raise ValueError("batch_index must be an integer in [1, total_batches]")
        return 1.0 / self.s_max + (self.s_max - 1.0 / self.s_max) * (batch_index - 1) / (total_batches - 1)

    def compensate_embedding_gradient(self, gradient, embedding, s):
        self._validate_scale(s)
        if not isinstance(gradient, torch.Tensor) or not isinstance(embedding, torch.Tensor):
            raise TypeError("gradient and embedding must be tensors")
        if gradient.shape != embedding.shape or gradient.device != embedding.device or gradient.dtype != embedding.dtype:
            raise ValueError("gradient and embedding must have matching shape, device, and dtype")
        if not gradient.is_floating_point() or not embedding.is_floating_point():
            raise ValueError("gradient and embedding must be floating-point tensors")
        if not torch.isfinite(gradient).all() or not torch.isfinite(embedding).all():
            raise ValueError("gradient and embedding must contain finite values")
        bounded_embedding = embedding.clamp(-6.0, 6.0)
        scaled_embedding = (float(s) * bounded_embedding).clamp(-50.0, 50.0)
        factor = self.s_max * (torch.cosh(scaled_embedding) + 1.0)
        factor = factor / (float(s) * (torch.cosh(bounded_embedding) + 1.0))
        return gradient * factor

    def forward(self, x, task_id, s=1.0, previous_output_attention=None, previous_input_attention=None):
        self._validate_task(task_id)
        self._validate_scale(s)
        if not isinstance(x, torch.Tensor) or x.ndim < 2:
            raise ValueError("x must be a tensor with at least batch and feature dimensions")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"x final dimension must equal in_features ({self.in_features})")
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError("x must contain finite floating-point values")
        if x.device != self.weight.device or x.dtype != self.weight.dtype:
            raise ValueError("x must match the layer parameter device and dtype")
        if previous_output_attention is None:
            previous_output_attention = self.weight.new_zeros(self.out_features)
        if previous_input_attention is None:
            previous_input_attention = self.weight.new_zeros(self.in_features)
        HAT._validate_attention(previous_output_attention, "previous_output_attention", self.out_features)
        HAT._validate_attention(previous_input_attention, "previous_input_attention", self.in_features)
        if previous_output_attention.device != self.weight.device or previous_input_attention.device != self.weight.device:
            raise ValueError("attention masks must match the layer parameter device")
        if previous_output_attention.dtype != self.weight.dtype or previous_input_attention.dtype != self.weight.dtype:
            raise ValueError("attention masks must match the layer parameter dtype")
        gradient_mask = torch.ones_like(self.weight) - torch.minimum(
            previous_output_attention[:, None], previous_input_attention[None, :]
        )
        hidden = _ConditionedLinear.apply(x, self.weight, gradient_mask)
        return hidden * self.attention(task_id, s)
