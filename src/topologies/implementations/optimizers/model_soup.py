"""Online model-soup averaging optimizer factory."""

import torch


def model_soup(params, *, lr, weight_decay=0.0, **backend_options):
    """Apply SGD updates while maintaining the running parameter average.

    The parameters are replaced by their online average after every update,
    making the optimizer useful as a lightweight trajectory-soup baseline.
    """
    if backend_options:
        unsupported = ", ".join(sorted(backend_options))
        raise TypeError(f"Model soup does not support backend options: {unsupported}")

    class _ModelSoup(torch.optim.Optimizer):
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
                        raise RuntimeError("Model soup does not support sparse gradients")
                    if group["weight_decay"]:
                        parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                    parameter.add_(grad, alpha=-group["lr"])
                    state = self.state[parameter]
                    if "soup" not in state:
                        state["soup"] = parameter.detach().clone()
                    else:
                        state["soup"].lerp_(parameter, 1.0 / self._steps)
                    parameter.copy_(state["soup"])
            return loss

    return _ModelSoup(params)
