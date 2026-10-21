"""AdEMAMix optimizer factory with a compact PyTorch fallback."""

import torch


def ademamix(params, *, lr, weight_decay=0.0, **backend_options):
    """Build AdEMAMix, preferring a native PyTorch implementation."""
    native = getattr(torch.optim, "AdEMAMix", None)
    if native is not None:
        return native(params, lr=lr, weight_decay=weight_decay, **backend_options)
    if backend_options:
        unsupported = ", ".join(sorted(backend_options))
        raise TypeError(f"fallback AdEMAMix does not support backend options: {unsupported}")

    class _AdEMAMix(torch.optim.Optimizer):
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
                        raise RuntimeError("AdEMAMix does not support sparse gradients")
                    state = self.state[parameter]
                    if not state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(parameter)
                        state["exp_avg_slow"] = torch.zeros_like(parameter)
                        state["exp_avg_sq"] = torch.zeros_like(parameter)
                    state["step"] += 1
                    step = state["step"]
                    beta1, beta2, beta3 = 0.9, 0.999, 0.9999
                    momentum, slow, variance = state["exp_avg"], state["exp_avg_slow"], state["exp_avg_sq"]
                    momentum.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    slow.mul_(beta3).add_(grad, alpha=1.0 - beta3)
                    variance.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    # Bias-correct the two fast statistics; ramp up the slow EMA contribution.
                    corrected_m = momentum / (1.0 - beta1**step)
                    corrected_v = variance / (1.0 - beta2**step)
                    alpha = 2.0 * min(1.0, step / 1000.0)
                    update = (corrected_m + alpha * slow) / (corrected_v.sqrt() + 1e-8)
                    if group["weight_decay"]:
                        parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                    parameter.add_(update, alpha=-group["lr"])
            return loss

    return _AdEMAMix(params)
