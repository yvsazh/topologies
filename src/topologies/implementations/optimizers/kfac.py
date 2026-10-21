"""Experimental diagonal Kronecker-factor optimizer."""

import torch


def kfac(params, *, lr, weight_decay=0.0, **backend_options):
    """Build a small diagonal K-FAC-style optimizer.

    The diagonal gradient second moment approximates the Kronecker factors
    without requiring module hooks or matrix inversions.
    """
    class _KFAC(torch.optim.Optimizer):
        def __init__(self, parameters):
            defaults = {"lr": lr, "weight_decay": weight_decay, **backend_options}
            super().__init__(parameters, defaults)

        @torch.no_grad()
        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            for group in self.param_groups:
                beta = group.get("beta", 0.95)
                eps = group.get("eps", 1e-8)
                for parameter in group["params"]:
                    grad = parameter.grad
                    if grad is None:
                        continue
                    if grad.is_sparse:
                        raise RuntimeError("K-FAC does not support sparse gradients")
                    state = self.state[parameter]
                    if "factor" not in state:
                        state["factor"] = torch.zeros_like(parameter)
                    factor = state["factor"]
                    factor.mul_(beta).addcmul_(grad, grad, value=1.0 - beta)
                    update = grad / (factor.sqrt() + eps)
                    if group["weight_decay"]:
                        update = update.add(parameter, alpha=group["weight_decay"])
                    parameter.add_(update, alpha=-group["lr"])
            return loss

    return _KFAC(params)
