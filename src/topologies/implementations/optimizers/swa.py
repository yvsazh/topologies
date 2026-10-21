"""Stochastic Weight Averaging optimizer factory."""

import torch


def swa(params, *, lr, weight_decay=0.0, **backend_options):
    """Build SGD with an online average of its parameter trajectory.

    Parameters contain the running SWA average after each step; callers can
    use them directly for evaluation once training has completed.
    """
    if backend_options:
        unsupported = ", ".join(sorted(backend_options))
        raise TypeError(f"SWA does not support backend options: {unsupported}")

    class _SWA(torch.optim.Optimizer):
        def __init__(self, parameters):
            if lr < 0.0:
                raise ValueError("lr must be non-negative")
            if weight_decay < 0.0:
                raise ValueError("weight_decay must be non-negative")
            super().__init__(parameters, {"lr": lr, "weight_decay": weight_decay})
            self._steps = 0

        @torch.no_grad()
        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            self._steps += 1
            for group in self.param_groups:
                for parameter in group["params"]:
                    grad = parameter.grad
                    if grad is None:
                        continue
                    if grad.is_sparse:
                        raise RuntimeError("SWA does not support sparse gradients")
                    if group["weight_decay"]:
                        parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                    parameter.add_(grad, alpha=-group["lr"])
                    state = self.state[parameter]
                    if "swa" not in state:
                        state["swa"] = parameter.detach().clone()
                    else:
                        state["swa"].lerp_(parameter, 1.0 / self._steps)
                    parameter.copy_(state["swa"])
            return loss

    return _SWA(params)
