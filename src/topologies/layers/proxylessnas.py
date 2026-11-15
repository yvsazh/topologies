import math

import torch
from torch import nn


def _value_error(message):
    return ValueError(message)


class ProxylessNAS(nn.Module):
    component_id = "proxylessnas"
    PRESETS = {"tiny": {"num_ops": 2}, "small": {"num_ops": 4}}

    def __init__(self, operations=None, num_ops=None, latencies=None, latency_weight=1.0):
        super().__init__()
        if operations is None:
            if num_ops is None:
                num_ops = 2
            if type(num_ops) is not int or num_ops < 1:
                raise ValueError("num_ops must be a positive integer")
            operations = [nn.Identity() for _ in range(num_ops)]
        else:
            if not isinstance(operations, (list, tuple)) or not operations:
                raise ValueError("operations must contain at least one candidate module")
            if not all(isinstance(operation, nn.Module) for operation in operations):
                raise ValueError("every operation must be an nn.Module")
            if num_ops is not None and (type(num_ops) is not int or num_ops != len(operations)):
                raise ValueError("num_ops must match the number of candidate operations")

        if isinstance(latency_weight, bool) or not isinstance(latency_weight, (int, float)):
            raise _value_error("latency_weight must be a finite positive number")
        try:
            latency_weight = float(latency_weight)
        except OverflowError:
            raise _value_error("latency_weight must be a finite positive number") from None
        if not math.isfinite(latency_weight) or latency_weight <= 0.0:
            raise ValueError("latency_weight must be a finite positive number")

        self.operations = nn.ModuleList(operations)
        self.alpha = nn.Parameter(torch.zeros(len(operations)), requires_grad=False)
        self.latency_weight = latency_weight
        self.last_operation_index = None
        self.last_gate = None
        if latencies is None:
            self.register_buffer("latencies", None)
        else:
            try:
                latency_values = torch.as_tensor(latencies, device=self.alpha.device)
            except (OverflowError, TypeError, ValueError, RuntimeError):
                raise ValueError("latencies must be a finite non-negative vector") from None
            if latency_values.ndim != 1 or latency_values.numel() != len(operations):
                raise ValueError("provide one latency value per candidate operation")
            if latency_values.dtype == torch.bool or latency_values.is_complex():
                raise ValueError("latencies must be a finite non-negative vector")
            latency_values = latency_values.to(dtype=self.alpha.dtype)
            if not bool(torch.isfinite(latency_values).all()) or bool((latency_values < 0).any()):
                raise ValueError("latencies must contain only finite and non-negative values")
            self.register_buffer("latencies", latency_values)

    @property
    def probabilities(self):
        self.alpha.requires_grad_(True)
        return torch.softmax(self.alpha, dim=0)

    @property
    def expected_latency(self):
        if self.latencies is None:
            raise ValueError("latencies are required to compute expected latency")
        return torch.sum(self.probabilities * self.latencies)

    @property
    def latency_penalty(self):
        return self.latency_weight * self.expected_latency

    def sample_operation_index(self):
        probabilities = torch.softmax(self.alpha.detach(), dim=0)
        return int(torch.multinomial(probabilities, num_samples=1).item())

    def architecture_loss(self, operation_indices, rewards):
        try:
            indices = torch.as_tensor(operation_indices, device=self.alpha.device)
        except (TypeError, ValueError, RuntimeError):
            raise ValueError("operation_indices must be valid candidate indices") from None
        if indices.ndim == 0:
            indices = indices.reshape(1)
        integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if indices.ndim != 1 or indices.dtype not in integer_dtypes:
            raise ValueError("operation_indices must be a one-dimensional integer sequence")
        if indices.numel() == 0 or bool((indices < 0).any()) or bool((indices >= len(self.operations)).any()):
            raise ValueError("operation_indices must refer to candidate operations")
        try:
            reward_values = torch.as_tensor(rewards, dtype=self.alpha.dtype, device=self.alpha.device)
        except (TypeError, ValueError, RuntimeError):
            raise ValueError("rewards must be finite values matching operation_indices") from None
        if reward_values.ndim == 0:
            reward_values = reward_values.reshape(1)
        if reward_values.ndim != 1 or reward_values.numel() != indices.numel():
            raise ValueError("rewards must have one value per operation index")
        if not bool(torch.isfinite(reward_values).all()):
            raise ValueError("rewards must contain only finite values")
        self.alpha.requires_grad_(True)
        log_probabilities = torch.log_softmax(self.alpha, dim=0)
        return -(reward_values * log_probabilities[indices]).mean()

    def forward(self, x, operation_index=None):
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a tensor")
        if operation_index is None:
            operation_index = (
                int(torch.argmax(self.alpha).item()) if not self.training else self.sample_operation_index()
            )
        if type(operation_index) is not int or not 0 <= operation_index < len(self.operations):
            raise ValueError("operation_index must be an integer referring to a candidate operation")
        self.last_operation_index = operation_index
        self.last_gate = torch.nn.functional.one_hot(
            torch.tensor(operation_index, device=self.alpha.device), num_classes=len(self.operations)
        ).to(dtype=self.alpha.dtype)
        return self.operations[operation_index](x)
