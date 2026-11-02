import torch
from torch import nn


class SharedExpert(nn.Module):
    component_id = "shared-expert"

    def __init__(self, *args, **kwargs):
        super().__init__()
        if args or kwargs:
            raise ValueError("SharedExpert does not accept constructor arguments")

    def forward(self, shared, routed, gates):
        if not isinstance(shared, torch.Tensor):
            raise TypeError("shared must be a tensor")
        if not isinstance(routed, torch.Tensor):
            raise TypeError("routed must be a tensor")
        if not isinstance(gates, torch.Tensor):
            raise TypeError("gates must be a tensor")
        if shared.ndim < 1:
            raise ValueError("shared must have at least one feature dimension")
        if routed.ndim < 2:
            raise ValueError("routed must have expert and feature dimensions")
        if gates.ndim < 1:
            raise ValueError("gates must have an expert dimension")
        if shared.shape[:-1] != routed.shape[:-2] or shared.shape[:-1] != gates.shape[:-1]:
            raise ValueError("shared, routed, and gates must have matching leading dimensions")
        if shared.shape[-1] != routed.shape[-1]:
            raise ValueError("shared and routed must have matching feature dimensions")
        if routed.shape[-2] != gates.shape[-1]:
            raise ValueError("routed and gates must have matching expert dimensions")
        if not shared.is_floating_point() or not routed.is_floating_point() or not gates.is_floating_point():
            raise ValueError("shared, routed, and gates must have floating-point dtypes")
        if shared.dtype != routed.dtype or shared.dtype != gates.dtype:
            raise ValueError("shared, routed, and gates must have matching dtypes")
        if shared.device != routed.device or shared.device != gates.device:
            raise ValueError("shared, routed, and gates must be on the same device")
        return shared + (routed * gates.unsqueeze(-1)).sum(dim=-2)
