"""A lightweight GrokFast optimizer factory using EMA gradient filtering."""

import torch


def grokfast(params, *, lr, weight_decay=0.0, **backend_options):
    """Build an SGD optimizer with GrokFast-style EMA gradient amplification."""
    if backend_options:
        unsupported = ", ".join(sorted(backend_options))
        raise TypeError(f"GrokFast does not support backend options: {unsupported}")

    class _GrokFast(torch.optim.Optimizer):
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
                        raise RuntimeError("GrokFast does not support sparse gradients")
                    state = self.state[parameter]
                    if "grad_ema" not in state:
                        state["grad_ema"] = torch.zeros_like(
                            parameter, memory_format=torch.preserve_format
                        )
                    grad_ema = state["grad_ema"]
                    update = grad.add(grad_ema, alpha=2.0)
                    if group["weight_decay"]:
                        update = update.add(parameter, alpha=group["weight_decay"])
                    parameter.add_(update, alpha=-group["lr"])
                    grad_ema.mul_(0.98).add_(grad, alpha=0.02)
            return loss

    return _GrokFast(params)
