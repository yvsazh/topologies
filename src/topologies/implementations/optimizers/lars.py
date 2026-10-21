"""Layer-wise Adaptive Rate Scaling (LARS) optimizer."""

import torch


def lars(params, *, lr, weight_decay=0.0, **backend_options):
    """Build a LARS optimizer with a layer-wise trust ratio."""
    implementation = getattr(torch.optim, "LARS", None)
    if implementation is not None:
        return implementation(params, lr=lr, weight_decay=weight_decay, **backend_options)

    class _LARS(torch.optim.Optimizer):
        def __init__(self, parameters):
            defaults = {"lr": lr, "weight_decay": weight_decay, **backend_options}
            super().__init__(parameters, defaults)

        @torch.no_grad()
        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            for group in self.param_groups:
                group_lr = group["lr"]
                decay = group["weight_decay"]
                trust_coefficient = group.get("trust_coefficient", 0.001)
                eps = group.get("eps", 1e-8)
                for parameter in group["params"]:
                    grad = parameter.grad
                    if grad is None:
                        continue
                    if grad.is_sparse:
                        raise RuntimeError("LARS does not support sparse gradients")
                    update = grad
                    if decay:
                        update = update.add(parameter, alpha=decay)
                    parameter_norm = torch.norm(parameter)
                    update_norm = torch.norm(update)
                    if parameter_norm > 0 and update_norm > 0:
                        trust_ratio = trust_coefficient * parameter_norm / (update_norm + eps)
                    else:
                        trust_ratio = 1.0
                    parameter.add_(update, alpha=-group_lr * trust_ratio)
            return loss

    return _LARS(params)
