"""SWALP-style low-precision SGD optimizer factory."""

import torch


def swalp(params, *, lr, weight_decay=0.0, **backend_options):
    """Construct an SGD optimizer compatible with the SWALP factory API.

    The update is performed directly on the parameters; optional backend
    options are rejected rather than silently ignored.
    """
    if backend_options:
        unsupported = ", ".join(sorted(backend_options))
        raise TypeError(f"SWALP does not support backend options: {unsupported}")

    class _SWALP(torch.optim.Optimizer):
        def __init__(self, parameters):
            if lr < 0.0:
                raise ValueError("lr must be non-negative")
            if weight_decay < 0.0:
                raise ValueError("weight_decay must be non-negative")
            super().__init__(parameters, {"lr": lr, "weight_decay": weight_decay})

        @torch.no_grad()
        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            for group in self.param_groups:
                for parameter in group["params"]:
                    grad = parameter.grad
                    if grad is None:
                        continue
                    if grad.is_sparse:
                        raise RuntimeError("SWALP does not support sparse gradients")
                    update = grad
                    if group["weight_decay"]:
                        update = update.add(parameter, alpha=group["weight_decay"])
                    parameter.add_(update, alpha=-group["lr"])
            return loss

    return _SWALP(params)
