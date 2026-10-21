"""Shampoo optimizer factory."""

import torch


def shampoo(params, *, lr, weight_decay=0.0, **backend_options):
    """Build Shampoo, preferring the implementation supplied by PyTorch."""
    shampoo_class = getattr(torch.optim, "Shampoo", None)
    if shampoo_class is not None:
        return shampoo_class(params, lr=lr, weight_decay=weight_decay, **backend_options)

    beta = backend_options.pop("beta", 0.9)
    epsilon = backend_options.pop("epsilon", 1e-4)
    if backend_options:
        raise TypeError(f"unsupported Shampoo options: {', '.join(backend_options)}")
    if lr < 0:
        raise ValueError("lr must be non-negative")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if not 0 <= beta < 1:
        raise ValueError("beta must be in [0, 1)")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    class _Shampoo(torch.optim.Optimizer):
        def __init__(self, parameters):
            super().__init__(parameters, {"lr": lr, "weight_decay": weight_decay,
                                               "beta": beta, "epsilon": epsilon})

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
                        raise RuntimeError("Shampoo does not support sparse gradients")
                    state = self.state[parameter]
                    if "preconditioner" not in state:
                        state["preconditioner"] = torch.zeros_like(parameter)
                        state["step"] = 0
                    state["step"] += 1
                    if group["weight_decay"]:
                        grad = grad.add(parameter, alpha=group["weight_decay"])
                    preconditioner = state["preconditioner"]
                    preconditioner.mul_(group["beta"]).addcmul_(
                        grad, grad, value=1 - group["beta"]
                    )
                    # Diagonal Shampoo preconditioning; epsilon stabilizes near-zero
                    # second moments while retaining a per-coordinate update.
                    update = grad / (preconditioner.sqrt() + group["epsilon"])
                    parameter.add_(update, alpha=-group["lr"])
            return loss

    return _Shampoo(params)
