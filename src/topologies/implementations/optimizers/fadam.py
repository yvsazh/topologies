"""A compact Adam-style FAdam optimizer factory."""

import torch


def fadam(params, *, lr, weight_decay=0.0, **backend_options):
    """Create an Adam-style optimizer, forwarding common options."""
    betas = backend_options.pop("betas", (0.9, 0.999))
    eps = backend_options.pop("eps", 1e-8)
    if backend_options:
        raise TypeError(f"unsupported FAdam options: {', '.join(backend_options)}")
    if lr < 0:
        raise ValueError("lr must be non-negative")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if eps < 0:
        raise ValueError("eps must be non-negative")
    if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
        raise ValueError("betas must be in [0, 1)")

    class _FAdam(torch.optim.Optimizer):
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
                        raise RuntimeError("FAdam does not support sparse gradients")
                    state = self.state[parameter]
                    if not state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(parameter)
                        state["exp_avg_sq"] = torch.zeros_like(parameter)
                    state["step"] += 1
                    step = state["step"]
                    if group["weight_decay"]:
                        grad = grad.add(parameter, alpha=group["weight_decay"])
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    correction1 = 1 - beta1 ** step
                    correction2 = 1 - beta2 ** step
                    denominator = exp_avg_sq.sqrt().div_(correction2 ** 0.5).add_(group["eps"])
                    parameter.addcdiv_(exp_avg, denominator, value=-group["lr"] / correction1)
            return loss

    return _FAdam(params)
