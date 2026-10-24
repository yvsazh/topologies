import math

import torch
from torch import nn
from torch.nn import functional as F


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class RigLLinear(nn.Module):
    component_id = "rigl"

    def __init__(
        self,
        in_features,
        out_features,
        sparsity,
        end_step,
        update_interval=100,
        drop_fraction=0.3,
        bias=True,
        seed=0,
    ):
        super().__init__()
        _require(
            not isinstance(in_features, bool) and isinstance(in_features, int) and in_features > 0,
            "in_features must be a positive integer",
        )
        _require(
            not isinstance(out_features, bool) and isinstance(out_features, int) and out_features > 0,
            "out_features must be a positive integer",
        )
        _require(
            not isinstance(sparsity, bool)
            and isinstance(sparsity, (int, float))
            and math.isfinite(sparsity)
            and 0 < sparsity < 1,
            "sparsity must be a finite number strictly between 0 and 1",
        )
        _require(
            not isinstance(end_step, bool) and isinstance(end_step, int) and end_step > 0,
            "end_step must be a positive integer",
        )
        _require(
            not isinstance(update_interval, bool)
            and isinstance(update_interval, int)
            and update_interval > 0,
            "update_interval must be a positive integer",
        )
        _require(
            not isinstance(drop_fraction, bool)
            and isinstance(drop_fraction, (int, float))
            and math.isfinite(drop_fraction)
            and 0 <= drop_fraction <= 1,
            "drop_fraction must be a finite number between 0 and 1",
        )
        _require(isinstance(bias, bool), "bias must be a boolean")
        _require(
            not isinstance(seed, bool) and isinstance(seed, int),
            "seed must be an integer",
        )
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = float(sparsity)
        self.end_step = end_step
        self.update_interval = update_interval
        self.drop_fraction = float(drop_fraction)
        self.seed = seed
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(in_features)
            nn.init.uniform_(self.bias, -bound, bound)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        size = out_features * in_features
        zero_count = math.floor(self.sparsity * size)
        initial_mask = torch.ones(size, dtype=self.weight.dtype)
        zero_indices = torch.randperm(size, generator=generator)[:zero_count]
        initial_mask[zero_indices] = 0
        self.register_buffer("mask", initial_mask.reshape(out_features, in_features))
        self.dense_gradient = None

    def _capture_dense_gradient(self, gradient):
        detached = gradient.detach()
        if self.dense_gradient is None:
            self.dense_gradient = detached.clone()
        else:
            self.dense_gradient.add_(detached)
        return gradient

    def forward(self, x):
        effective_weight = self.weight * self.mask
        if effective_weight.requires_grad:
            effective_weight.register_hook(self._capture_dense_gradient)
        return F.linear(x, effective_weight, self.bias)

    def drop_fraction_at(self, step):
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be a nonnegative integer")
        position = min(step, self.end_step) / self.end_step
        return self.drop_fraction * 0.5 * (1 + math.cos(math.pi * position))

    def update_mask(self, step, dense_gradient=None, optimizer=None):
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be a nonnegative integer")
        if dense_gradient is not None:
            if not isinstance(dense_gradient, torch.Tensor) or dense_gradient.shape != self.weight.shape:
                raise ValueError("dense_gradient must be a tensor with the same shape as weight")
            gradient = dense_gradient
        else:
            gradient = self.dense_gradient
        if step >= self.end_step or step % self.update_interval != 0:
            self.dense_gradient = None
            return False
        if gradient is None:
            raise ValueError("a dense effective-weight gradient is required for mask updates")
        if gradient.shape != self.weight.shape:
            raise ValueError("dense_gradient must have the same shape as weight")
        fraction = self.drop_fraction_at(step)
        flat_mask = self.mask.reshape(-1)
        active = torch.nonzero(flat_mask != 0, as_tuple=False).reshape(-1)
        drop_count = math.floor(active.numel() * fraction)
        if drop_count == 0:
            self.dense_gradient = None
            return False
        flat_weight = self.weight.detach().reshape(-1)
        flat_gradient = gradient.detach().to(device=self.weight.device).reshape(-1)
        dropped = active[torch.argsort(flat_weight[active].abs(), stable=True)[:drop_count]]
        updated_mask = flat_mask.clone()
        updated_mask[dropped] = 0
        inactive = torch.nonzero(updated_mask == 0, as_tuple=False).reshape(-1)
        grown = inactive[
            torch.argsort(flat_gradient[inactive].abs(), descending=True, stable=True)[:drop_count]
        ]
        updated_mask[grown] = 1
        with torch.no_grad():
            flat_mask.copy_(updated_mask)
            flat_weight[grown] = 0
            if optimizer is not None:
                for state_value in optimizer.state.get(self.weight, {}).values():
                    if isinstance(state_value, torch.Tensor) and state_value.shape == self.weight.shape:
                        state_value.reshape(-1)[grown] = 0
        self.dense_gradient = None
        return True
