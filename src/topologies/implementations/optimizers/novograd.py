"""A small, local-state implementation of the NovoGrad optimizer."""

import torch


def novograd(params, *, lr, weight_decay=0.0, **backend_options):
    """Create a NovoGrad optimizer.

    NovoGrad normalizes each parameter's gradient by a running second moment
    of its squared L2 norm, then applies momentum to the normalized gradient.
    """
    torch_optimizer = getattr(torch.optim, "NovoGrad", None)
    if torch_optimizer is not None:
        return torch_optimizer(params, lr=lr, weight_decay=weight_decay, **backend_options)

    class _NovoGrad(torch.optim.Optimizer):
        def __init__(self, params, lr, weight_decay, options):
            beta1 = options.pop("beta1", 0.95)
            beta2 = options.pop("beta2", 0.98)
            eps = options.pop("eps", 1e-8)
            if options:
                unknown = ", ".join(sorted(options))
                raise TypeError(f"unexpected backend option(s): {unknown}")
            if lr < 0 or weight_decay < 0 or eps <= 0:
                raise ValueError("lr and weight_decay must be non-negative and eps positive")
            if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
                raise ValueError("beta1 and beta2 must be in [0, 1)")
            super().__init__(params, {"lr": lr, "weight_decay": weight_decay,
                                          "beta1": beta1, "beta2": beta2, "eps": eps})

        @torch.no_grad()
        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            for group in self.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue
                    grad = parameter.grad
                    if grad.is_sparse:
                        raise RuntimeError("NovoGrad does not support sparse gradients")
                    state = self.state[parameter]
                    if not state:
                        state["step"] = 0
                        state["exp_avg_sq"] = torch.zeros((), device=parameter.device,
                                                          dtype=parameter.dtype)
                        state["exp_avg"] = torch.zeros_like(parameter)
                    state["step"] += 1
                    grad_norm_sq = torch.sum(grad * grad)
                    second = state["exp_avg_sq"]
                    second.mul_(group["beta2"]).add_(grad_norm_sq, alpha=1 - group["beta2"])
                    normalized = grad / (second.sqrt() + group["eps"])
                    if group["weight_decay"]:
                        normalized = normalized.add(parameter, alpha=group["weight_decay"])
                    first = state["exp_avg"]
                    first.mul_(group["beta1"]).add_(normalized)
                    parameter.add_(first, alpha=-group["lr"])
            return loss

    return _NovoGrad(params, lr, weight_decay, dict(backend_options))
