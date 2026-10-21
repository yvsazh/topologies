"""A compact Sophia-G optimizer implementation."""

import torch


def sophia(params, *, lr, weight_decay=0.0, **backend_options):
    """Build a Sophia optimizer.

    ``backend_options`` may specify ``betas``, ``rho`` (update clipping), and
    ``eps``. Hessian-diagonal estimates use the available squared-gradient
    proxy every ``hessian_interval`` steps; callers can replace this by setting
    ``parameter.grad_hessian`` before the corresponding step.
    """
    optimizer_class = getattr(torch.optim, "Sophia", None)
    if optimizer_class is not None:
        return optimizer_class(params, lr=lr, weight_decay=weight_decay, **backend_options)
    return _Sophia(params, lr=lr, weight_decay=weight_decay, **backend_options)


class _Sophia(torch.optim.Optimizer):
    def __init__(self, params, *, lr, weight_decay=0.0, betas=(0.965, 0.99),
                 rho=0.04, eps=1e-12, hessian_interval=10):
        if lr < 0 or weight_decay < 0 or rho <= 0 or eps < 0 or hessian_interval < 1:
            raise ValueError("invalid Sophia hyperparameter")
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "rho": rho,
                        "eps": eps, "hessian_interval": hessian_interval}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("Sophia does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(parameter)
                    state["hessian"] = torch.zeros_like(parameter)
                state["step"] += 1
                momentum, hessian = state["momentum"], state["hessian"]
                momentum.mul_(beta1).add_(grad, alpha=1 - beta1)
                if state["step"] % group["hessian_interval"] == 0:
                    estimate = getattr(parameter, "grad_hessian", None)
                    hessian.mul_(beta2).add_(grad.square() if estimate is None else estimate,
                                             alpha=1 - beta2)
                update = (momentum / (hessian + group["eps"])).clamp(
                    min=-group["rho"], max=group["rho"]
                )
                if group["weight_decay"]:
                    parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-group["lr"])
        return loss
