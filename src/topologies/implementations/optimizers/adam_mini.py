"""Adam-mini optimizer factory."""

import torch


def adam_mini(params, *, lr, weight_decay=0.0, **backend_options):
    """Build Adam-mini, using PyTorch's implementation when available.

    The local fallback shares one second-moment accumulator across all
    parameters in each parameter group, retaining Adam-mini's reduced-state
    approach while remaining a useful lightweight optimizer for small models.
    """
    adam_mini_class = getattr(torch.optim, "AdamMini", None)
    if adam_mini_class is not None:
        return adam_mini_class(params, lr=lr, weight_decay=weight_decay, **backend_options)

    betas = backend_options.pop("betas", (0.9, 0.999))
    eps = backend_options.pop("eps", 1e-8)
    if backend_options:
        raise TypeError(f"unsupported Adam-mini options: {', '.join(backend_options)}")
    if lr < 0:
        raise ValueError("lr must be non-negative")
    if eps < 0:
        raise ValueError("eps must be non-negative")
    if len(betas) != 2 or not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
        raise ValueError("betas must be a pair of values in [0, 1)")

    class _AdamMini(torch.optim.Optimizer):
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
                params_with_grad = [p for p in group["params"] if p.grad is not None]
                if not params_with_grad:
                    continue
                if any(p.grad.is_sparse for p in params_with_grad):
                    raise RuntimeError("Adam-mini does not support sparse gradients")
                state = self.state[params_with_grad[0]]
                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg_sq"] = torch.zeros((), device=params_with_grad[0].device,
                                                       dtype=params_with_grad[0].dtype)
                state["step"] += 1
                step = state["step"]
                squared_sum = torch.zeros_like(state["exp_avg_sq"])
                count = 0
                for parameter in params_with_grad:
                    grad = parameter.grad
                    if group["weight_decay"]:
                        grad = grad.add(parameter, alpha=group["weight_decay"])
                    squared_sum.add_(grad.detach().to(squared_sum.dtype).square().sum())
                    count += grad.numel()
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg_sq.mul_(beta2).add_(squared_sum / count, alpha=1 - beta2)
                correction1 = 1 - beta1 ** step
                correction2 = 1 - beta2 ** step
                denominator = (exp_avg_sq / correction2).sqrt().add_(group["eps"])
                for parameter in params_with_grad:
                    parameter_state = self.state[parameter]
                    if "exp_avg" not in parameter_state:
                        parameter_state["exp_avg"] = torch.zeros_like(parameter)
                    grad = parameter.grad
                    if group["weight_decay"]:
                        grad = grad.add(parameter, alpha=group["weight_decay"])
                    exp_avg = parameter_state["exp_avg"]
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    parameter.addcdiv_(exp_avg, denominator, value=-group["lr"] / correction1)
            return loss

    return _AdamMini(params)
