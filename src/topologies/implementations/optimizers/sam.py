"""Sharpness-Aware Minimization (SAM) optimizer factory."""

import torch


def sam(params, *, lr, weight_decay=0.0, **backend_options):
    """Create a two-pass SAM optimizer.

    Each ``step`` evaluates the gradient at an adversarially perturbed point,
    then applies that gradient at the original parameters. Callers should
    provide a closure that clears gradients, computes the loss, and backprops.
    """
    rho = backend_options.pop("rho", 0.05)
    adaptive = backend_options.pop("adaptive", False)
    if backend_options:
        unsupported = ", ".join(sorted(backend_options))
        raise TypeError(f"SAM does not support backend options: {unsupported}")

    class _SAM(torch.optim.Optimizer):
        def __init__(self, parameters):
            if lr < 0.0:
                raise ValueError("lr must be non-negative")
            if weight_decay < 0.0:
                raise ValueError("weight_decay must be non-negative")
            if rho < 0.0:
                raise ValueError("rho must be non-negative")
            super().__init__(parameters, {"lr": lr, "weight_decay": weight_decay})

        @torch.no_grad()
        def step(self, closure=None):
            if closure is None:
                raise RuntimeError("SAM requires a closure for its second gradient evaluation")
            with torch.enable_grad():
                loss = closure()

            norm_parts = []
            for group in self.param_groups:
                for parameter in group["params"]:
                    grad = parameter.grad
                    if grad is None:
                        continue
                    if grad.is_sparse:
                        raise RuntimeError("SAM does not support sparse gradients")
                    scale = parameter.abs() if adaptive else 1.0
                    norm_parts.append((scale * grad).norm(p=2))
            if not norm_parts:
                return loss
            grad_norm = torch.norm(torch.stack(norm_parts), p=2)
            scale = rho / (grad_norm + 1e-12)

            for group in self.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue
                    state = self.state[parameter]
                    perturbation = (parameter.pow(2) if adaptive else 1.0) * parameter.grad * scale
                    state["sam_perturbation"] = perturbation
                    parameter.add_(perturbation)

            self.zero_grad(set_to_none=True)
            with torch.enable_grad():
                closure()

            for group in self.param_groups:
                for parameter in group["params"]:
                    perturbation = self.state[parameter].pop("sam_perturbation", None)
                    if perturbation is None:
                        continue
                    parameter.sub_(perturbation)
                    grad = parameter.grad
                    if grad is None:
                        continue
                    if group["weight_decay"]:
                        grad = grad.add(parameter, alpha=group["weight_decay"])
                    parameter.add_(grad, alpha=-group["lr"])
            return loss

    return _SAM(params)
