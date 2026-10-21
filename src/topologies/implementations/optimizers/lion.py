"""A compact Lion optimizer factory."""

import torch


def lion(params, *, lr, weight_decay=0.0, **backend_options):
    """Build Lion, preferring PyTorch's implementation when available."""
    native = getattr(torch.optim, "Lion", None)
    if native is not None:
        return native(params, lr=lr, weight_decay=weight_decay, **backend_options)
    if backend_options:
        unsupported = ", ".join(sorted(backend_options))
        raise TypeError(f"fallback Lion does not support backend options: {unsupported}")

    class _Lion(torch.optim.Optimizer):
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
                        raise RuntimeError("Lion does not support sparse gradients")
                    state = self.state[parameter]
                    if not state:
                        state["exp_avg"] = torch.zeros_like(
                            parameter, memory_format=torch.preserve_format
                        )
                    exp_avg = state["exp_avg"]
                    update = exp_avg.mul(0.9).add(grad, alpha=0.1).sign()
                    if group["weight_decay"]:
                        parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                    parameter.add_(update, alpha=-group["lr"])
                    exp_avg.mul_(0.99).add_(grad, alpha=0.01)
            return loss

    return _Lion(params)
