"""Polynomially decaying learning-rate optimizer."""

import torch


def polynomial_decay(params, *, lr, weight_decay=0.0, **backend_options):
    """Create SGD with a per-step polynomial learning-rate decay.

    ``total_steps`` sets the decay horizon (default 1000), and ``power`` sets
    its exponent (default 1). Other backend options are not supported.
    """
    total_steps = backend_options.pop("total_steps", 1000)
    power = backend_options.pop("power", 1.0)
    if backend_options:
        unsupported = ", ".join(sorted(backend_options))
        raise TypeError(f"polynomial_decay does not support backend options: {unsupported}")
    if lr < 0.0:
        raise ValueError("lr must be non-negative")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if power < 0.0:
        raise ValueError("power must be non-negative")

    class _PolynomialDecay(torch.optim.Optimizer):
        def __init__(self, parameters):
            super().__init__(parameters, {"lr": lr, "weight_decay": weight_decay})
            self._step_count = 0

        @torch.no_grad()
        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            self._step_count += 1
            factor = max(1.0 - self._step_count / total_steps, 0.0) ** power
            for group in self.param_groups:
                for parameter in group["params"]:
                    grad = parameter.grad
                    if grad is None:
                        continue
                    if grad.is_sparse:
                        raise RuntimeError("polynomial_decay does not support sparse gradients")
                    update = grad
                    if group["weight_decay"]:
                        update = update.add(parameter, alpha=group["weight_decay"])
                    parameter.add_(update, alpha=-group["lr"] * factor)
            return loss

    return _PolynomialDecay(params)
