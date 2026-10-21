import math

import torch
from torch import nn


def _is_boolean(value):
    return isinstance(value, bool)


def _is_finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


class VTrace(nn.Module):
    component_id = "v-trace"

    def __init__(self, clip_rho=1.0, clip_c=1.0, detach_targets=True):
        super().__init__()
        for name, value in (("clip_rho", clip_rho), ("clip_c", clip_c)):
            if not _is_finite_number(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if clip_rho < clip_c:
            raise ValueError("clip_rho must be greater than or equal to clip_c")
        if not _is_boolean(detach_targets):
            raise ValueError("detach_targets must be a boolean")
        self.clip_rho = float(clip_rho)
        self.clip_c = float(clip_c)
        self.detach_targets = detach_targets

    def forward(self, values, rewards, discounts, behavior_log_probs, target_log_probs, bootstrap_value):
        sequences = {
            "values": values,
            "rewards": rewards,
            "discounts": discounts,
            "behavior_log_probs": behavior_log_probs,
            "target_log_probs": target_log_probs,
        }
        for name, tensor in sequences.items():
            if not torch.is_tensor(tensor):
                raise ValueError(f"{name} must be a tensor")
            if tensor.ndim < 1 or tensor.shape[-1] < 1:
                raise ValueError(f"{name} must have a nonempty final time dimension")
            if not torch.is_floating_point(tensor):
                raise ValueError(f"{name} must have a floating-point dtype")
        if not torch.is_tensor(bootstrap_value) or not torch.is_floating_point(bootstrap_value):
            raise ValueError("bootstrap_value must be a floating-point tensor")
        for name, tensor in sequences.items():
            if tensor.shape != values.shape:
                raise ValueError(f"{name} must have the same shape as values")
            if tensor.device != values.device:
                raise ValueError(f"{name} must be on the same device as values")
        if bootstrap_value.shape != values.shape[:-1]:
            raise ValueError("bootstrap_value must match the leading dimensions of values")
        if bootstrap_value.device != values.device:
            raise ValueError("bootstrap_value must be on the same device as values")

        targets = []
        next_value = bootstrap_value
        next_target = bootstrap_value
        grad_enabled = torch.is_grad_enabled() and not self.detach_targets
        with torch.set_grad_enabled(grad_enabled):
            log_ratios = target_log_probs - behavior_log_probs
            rho_log_cap = math.log(self.clip_rho)
            c_log_cap = math.log(self.clip_c)
            for index in range(values.shape[-1] - 1, -1, -1):
                rho = torch.exp(log_ratios[..., index].clamp(max=rho_log_cap))
                c = torch.exp(log_ratios[..., index].clamp(max=c_log_cap))
                value = values[..., index]
                discount = discounts[..., index]
                target = value + rho * (rewards[..., index] + discount * next_value - value)
                target = target + discount * c * (next_target - next_value)
                targets.append(target)
                next_target = target
                next_value = value
        result = torch.stack(targets[::-1], dim=-1)
        return result.detach() if self.detach_targets else result
