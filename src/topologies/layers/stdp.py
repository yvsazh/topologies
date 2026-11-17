import math
import numbers

import torch
from torch import nn


class SpikeTimingUpdate(nn.Module):
    component_id = "stdp"

    def __init__(self, a_plus=0.005, a_minus=0.00525, tau_plus=20.0, tau_minus=20.0):
        super().__init__()
        for name, value in (
            ("a_plus", a_plus),
            ("a_minus", a_minus),
            ("tau_plus", tau_plus),
            ("tau_minus", tau_minus),
        ):
            try:
                if isinstance(value, bool) or not isinstance(value, numbers.Real):
                    raise TypeError
                value = float(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"{name} must be a positive finite real scalar") from error
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite real scalar")
            setattr(self, name, value)

    @staticmethod
    def _time_tensor(value, name, reference=None):
        if torch.is_tensor(value):
            if not torch.is_floating_point(value):
                raise ValueError(f"{name} must be a floating-point tensor or real scalar")
            result = value
        else:
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise ValueError(f"{name} must be a floating-point tensor or real scalar")
            try:
                scalar = float(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"{name} must be a finite real scalar") from error
            if not math.isfinite(scalar):
                raise ValueError(f"{name} must be a finite real scalar")
            options = {}
            if reference is not None:
                options = {"device": reference.device, "dtype": reference.dtype}
            try:
                result = torch.as_tensor(scalar, **options)
            except (TypeError, ValueError, RuntimeError, OverflowError) as error:
                raise ValueError(f"{name} must be representable as a floating-point value") from error
        if not torch.isfinite(result).all():
            raise ValueError(f"{name} must contain only finite values")
        return result

    def forward(self, pre_time, post_time):
        if torch.is_tensor(pre_time) and torch.is_tensor(post_time):
            if pre_time.device != post_time.device:
                raise ValueError("pre_time and post_time must be on the same device")
            pre = self._time_tensor(pre_time, "pre_time")
            post = self._time_tensor(post_time, "post_time")
        elif torch.is_tensor(pre_time):
            pre = self._time_tensor(pre_time, "pre_time")
            post = self._time_tensor(post_time, "post_time", pre)
        elif torch.is_tensor(post_time):
            post = self._time_tensor(post_time, "post_time")
            pre = self._time_tensor(pre_time, "pre_time", post)
        else:
            pre = self._time_tensor(pre_time, "pre_time")
            post = self._time_tensor(post_time, "post_time")

        try:
            pre, post = torch.broadcast_tensors(pre, post)
        except RuntimeError as error:
            raise ValueError("pre_time and post_time must have broadcastable shapes") from error
        delta = pre - post
        if not torch.isfinite(delta).all():
            raise ValueError("the difference between pre_time and post_time must be finite")

        potentiation_delta = torch.clamp(delta, max=0.0)
        depression_delta = torch.clamp(delta, min=0.0)
        potentiation = self.a_plus * torch.exp(potentiation_delta / self.tau_plus)
        depression = -self.a_minus * torch.exp(-depression_delta / self.tau_minus)
        result = torch.where(delta < 0, potentiation, depression)
        if not torch.isfinite(result).all():
            raise ValueError("the update is not finite in the input dtype")
        return result
