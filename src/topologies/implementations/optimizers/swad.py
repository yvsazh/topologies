"""Simple Weight-Averaged Descent (SWAD) optimizer factory."""

import torch


def swad(params, *, lr, weight_decay=0.0, **backend_options):
    """Build a small SGD optimizer suitable for SWAD-style training."""
    momentum = backend_options.pop("momentum", 0.0)
    dampening = backend_options.pop("dampening", 0.0)
    nesterov = backend_options.pop("nesterov", False)
    maximize = backend_options.pop("maximize", False)
    if backend_options:
        raise TypeError(f"unsupported SWAD options: {', '.join(backend_options)}")
    if lr < 0 or weight_decay < 0 or momentum < 0 or dampening < 0:
        raise ValueError("lr, weight_decay, momentum, and dampening must be non-negative")
    if nesterov and (momentum <= 0 or dampening != 0):
        raise ValueError("Nesterov momentum requires positive momentum and zero dampening")

    class _SWAD(torch.optim.Optimizer):
        def __init__(self, parameters):
            super().__init__(parameters, {"lr": lr, "weight_decay": weight_decay,
                                              "momentum": momentum, "dampening": dampening,
                                              "nesterov": nesterov, "maximize": maximize})

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
                        raise RuntimeError("SWAD does not support sparse gradients")
                    grad = -grad if group["maximize"] else grad
                    if group["weight_decay"]:
                        grad = grad.add(parameter, alpha=group["weight_decay"])
                    if group["momentum"]:
                        state = self.state[parameter]
                        if "momentum_buffer" not in state:
                            buffer = state["momentum_buffer"] = grad.clone().detach()
                        else:
                            buffer = state["momentum_buffer"]
                            buffer.mul_(group["momentum"]).add_(grad, alpha=1 - group["dampening"])
                        grad = grad.add(buffer, alpha=group["momentum"]) if group["nesterov"] else buffer
                    parameter.add_(grad, alpha=-group["lr"])
            return loss

    return _SWAD(params)
