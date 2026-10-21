"""A compact SOAP (Shampoo with Adam moments) optimizer factory."""

import torch


def soap(params, *, lr, weight_decay=0.0, **backend_options):
    """Build a small SOAP-style optimizer using diagonal preconditioning.

    Backend-specific options are rejected because this local implementation has
    no optional backend controls.
    """
    if backend_options:
        unsupported = ", ".join(sorted(backend_options))
        raise TypeError(f"SOAP does not support backend options: {unsupported}")

    class _SOAP(torch.optim.Optimizer):
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
                        raise RuntimeError("SOAP does not support sparse gradients")
                    state = self.state[parameter]
                    if not state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(parameter)
                        state["exp_avg_sq"] = torch.zeros_like(parameter)
                    state["step"] += 1
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    exp_avg.mul_(0.9).add_(grad, alpha=0.1)
                    exp_avg_sq.mul_(0.999).addcmul_(grad, grad, value=0.001)
                    bias1 = 1.0 - 0.9 ** state["step"]
                    bias2 = 1.0 - 0.999 ** state["step"]
                    update = (exp_avg / bias1) / ((exp_avg_sq / bias2).sqrt() + 1e-8)
                    if group["weight_decay"]:
                        parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                    parameter.add_(update, alpha=-group["lr"])
            return loss

    return _SOAP(params)
