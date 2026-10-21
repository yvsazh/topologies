"""A compact Schedule-Free SGD optimizer factory."""

import torch


def schedule_free(params, *, lr, weight_decay=0.0, **backend_options):
    """Build a minimal Schedule-Free SGD optimizer."""
    if backend_options:
        unsupported = ", ".join(sorted(backend_options))
        raise TypeError(f"Schedule-Free SGD does not support backend options: {unsupported}")

    class _ScheduleFree(torch.optim.Optimizer):
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
                        raise RuntimeError("Schedule-Free SGD does not support sparse gradients")
                    if group["weight_decay"]:
                        parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                    parameter.add_(grad, alpha=-group["lr"])
            return loss

    return _ScheduleFree(params)
