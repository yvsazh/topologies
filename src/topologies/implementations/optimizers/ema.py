"""Exponential moving average weight optimizer factory."""

import torch


def ema(params, *, lr, weight_decay=0.0, **backend_options):
    """Build SGD that maintains an exponential moving average of its weights.

    The parameters contain the averaged weights after every step, so they can
    be used directly for evaluation.
    """
    if backend_options:
        unsupported = ", ".join(sorted(backend_options))
        raise TypeError(f"EMA does not support backend options: {unsupported}")

    class _EMA(torch.optim.Optimizer):
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
                decay = group["lr"] * group["weight_decay"]
                for parameter in group["params"]:
                    grad = parameter.grad
                    if grad is None:
                        continue
                    if grad.is_sparse:
                        raise RuntimeError("EMA does not support sparse gradients")
                    if decay:
                        parameter.mul_(1.0 - decay)
                    parameter.add_(grad, alpha=-group["lr"])
                    state = self.state[parameter]
                    if "ema" not in state:
                        state["ema"] = parameter.detach().clone()
                    else:
                        state["ema"].lerp_(parameter, group["lr"])
                    parameter.copy_(state["ema"])
            return loss

    return _EMA(params)
