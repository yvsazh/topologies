from numbers import Real

import torch
from torch import nn


class MetaSGD(nn.Module):
    component_id = "metasgd"
    PRESETS = {
        "tiny": {"alpha_init": torch.tensor([0.005])},
        "small": {"alpha_init": torch.full((4,), 0.005)},
    }

    def __init__(self, alpha_init=None):
        super().__init__()
        if alpha_init is None:
            raise ValueError("alpha_init is required")
        if not isinstance(alpha_init, torch.Tensor):
            if not isinstance(alpha_init, (list, tuple)) or not all(
                isinstance(value, Real) and not isinstance(value, bool) for value in alpha_init
            ):
                raise ValueError("alpha_init must be a finite floating-point vector")
            alpha_init = torch.tensor(alpha_init, dtype=torch.get_default_dtype())
        if (
            alpha_init.layout != torch.strided
            or not alpha_init.is_floating_point()
            or alpha_init.ndim != 1
            or alpha_init.numel() == 0
        ):
            raise ValueError("alpha_init must be a non-empty one-dimensional floating-point tensor")
        if not bool(torch.isfinite(alpha_init).all().item()):
            raise ValueError("alpha_init must contain only finite values")
        self.alpha = nn.Parameter(alpha_init.detach().clone())

    def forward(self, params, support_grads):
        if not isinstance(params, torch.Tensor) or not isinstance(support_grads, torch.Tensor):
            raise TypeError("params and support_grads must be tensors")
        if params.shape != support_grads.shape:
            raise ValueError("params and support_grads shapes must match")
        if params.ndim not in (1, 2):
            raise ValueError("params and support_grads must have shape [P] or [B, P]")
        if params.shape[-1] != self.alpha.numel():
            raise ValueError("final parameter dimension must match alpha_init")
        if params.dtype != support_grads.dtype or params.device != support_grads.device:
            raise ValueError("params and support_grads dtype and device must match")
        if params.dtype != self.alpha.dtype or params.device != self.alpha.device:
            raise ValueError("params and support_grads dtype and device must match alpha")
        return params - self.alpha * support_grads
