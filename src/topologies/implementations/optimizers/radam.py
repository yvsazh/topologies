"""Rectified Adam optimizer factory."""

import math

import torch


def radam(params, *, lr, weight_decay=0.0, **backend_options):
    """Build RAdam, preferring the implementation supplied by PyTorch."""
    radam_class = getattr(torch.optim, "RAdam", None)
    if radam_class is not None:
        return radam_class(params, lr=lr, weight_decay=weight_decay, **backend_options)

    betas = backend_options.pop("betas", (0.9, 0.999))
    eps = backend_options.pop("eps", 1e-8)
    if backend_options:
        raise TypeError(f"unsupported RAdam options: {', '.join(backend_options)}")
    if lr < 0:
        raise ValueError("lr must be non-negative")
    if eps < 0:
        raise ValueError("eps must be non-negative")
    if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
        raise ValueError("betas must be in [0, 1)")

    class _RAdam(torch.optim.Optimizer):
        def __init__(self, parameters):
            super().__init__(parameters, {"lr": lr, "weight_decay": weight_decay,
                                               "betas": betas, "eps": eps})

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
                        raise RuntimeError("RAdam does not support sparse gradients")
                    state = self.state[parameter]
                    if not state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(parameter)
                        state["exp_avg_sq"] = torch.zeros_like(parameter)
                    state["step"] += 1
                    step = state["step"]
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    if group["weight_decay"]:
                        grad = grad.add(parameter, alpha=group["weight_decay"])
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    beta2_power = beta2 ** step
                    rho_inf = 2 / (1 - beta2) - 1
                    rho = rho_inf - 2 * step * beta2_power / (1 - beta2_power)
                    step_size = group["lr"] / (1 - beta1 ** step)
                    if rho > 5:
                        rectification = math.sqrt(
                            ((rho - 4) * (rho - 2) * rho_inf)
                            / ((rho_inf - 4) * (rho_inf - 2) * rho)
                        )
                        denominator = exp_avg_sq.sqrt().div_(math.sqrt(1 - beta2_power)).add_(group["eps"])
                        parameter.addcdiv_(exp_avg, denominator, value=-step_size * rectification)
                    else:
                        parameter.add_(exp_avg, alpha=-step_size)
            return loss

    return _RAdam(params)
