"""A compact SM3-style optimizer factory."""

import torch


def sm3(params, *, lr, weight_decay=0.0, **backend_options):
    """Build an optimizer using elementwise adaptive second-moment scaling."""
    if backend_options:
        unsupported = ", ".join(sorted(backend_options))
        raise TypeError(f"SM3 does not support backend options: {unsupported}")

    class _SM3(torch.optim.Optimizer):
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
                        raise RuntimeError("SM3 does not support sparse gradients")
                    state = self.state[parameter]
                    if "accumulator" not in state:
                        state["accumulator"] = torch.zeros_like(parameter)
                    accumulator = state["accumulator"]
                    torch.maximum(accumulator, grad.square(), out=accumulator)
                    update = grad / (accumulator.sqrt() + 1e-8)
                    if group["weight_decay"]:
                        parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                    parameter.add_(update, alpha=-group["lr"])
            return loss

    return _SM3(params)
