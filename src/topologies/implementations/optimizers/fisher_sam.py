"""Fisher-preconditioned Sharpness-Aware Minimization optimizer."""

import torch


def fisher_sam(params, *, lr, weight_decay=0.0, **backend_options):
    """Build Fisher-SAM; ``step`` requires a loss/backward closure.

    The first-pass squared gradients provide a diagonal empirical-Fisher
    metric. The normalized adversarial perturbation is formed in that metric,
    and the second-pass gradient updates the unperturbed parameters.
    """
    rho = backend_options.pop("rho", 0.05)
    eps = backend_options.pop("eps", 1e-12)
    if backend_options:
        unsupported = ", ".join(sorted(backend_options))
        raise TypeError(f"Fisher-SAM does not support backend options: {unsupported}")

    class _FisherSAM(torch.optim.Optimizer):
        def __init__(self, parameters):
            if lr < 0.0 or weight_decay < 0.0 or rho < 0.0 or eps <= 0.0:
                raise ValueError("lr, weight_decay and rho must be non-negative; eps must be positive")
            super().__init__(parameters, {"lr": lr, "weight_decay": weight_decay})

        @torch.no_grad()
        def step(self, closure=None):
            if closure is None:
                raise RuntimeError("Fisher-SAM requires a closure for its second gradient evaluation")
            with torch.enable_grad():
                loss = closure()

            parts = []
            for group in self.param_groups:
                for parameter in group["params"]:
                    grad = parameter.grad
                    if grad is None:
                        continue
                    if grad.is_sparse:
                        raise RuntimeError("Fisher-SAM does not support sparse gradients")
                    state = self.state[parameter]
                    fisher = state.setdefault("fisher", grad.detach().square().clone())
                    fisher.copy_(grad.detach().square())
                    direction = grad / (fisher.sqrt() + eps)
                    state["fisher_sam_direction"] = direction
                    parts.append((direction * fisher.sqrt()).norm())
            if not parts:
                return loss
            scale = rho / (torch.stack(parts).norm() + eps)
            for group in self.param_groups:
                for parameter in group["params"]:
                    direction = self.state[parameter].get("fisher_sam_direction")
                    if direction is not None:
                        perturbation = direction * scale
                        self.state[parameter]["fisher_sam_perturbation"] = perturbation
                        parameter.add_(perturbation)

            self.zero_grad(set_to_none=True)
            with torch.enable_grad():
                closure()

            for group in self.param_groups:
                for parameter in group["params"]:
                    state = self.state[parameter]
                    perturbation = state.pop("fisher_sam_perturbation", None)
                    state.pop("fisher_sam_direction", None)
                    if perturbation is None:
                        continue
                    parameter.sub_(perturbation)
                    grad = parameter.grad
                    if grad is not None:
                        if group["weight_decay"]:
                            grad = grad.add(parameter, alpha=group["weight_decay"])
                        parameter.add_(grad, alpha=-group["lr"])
            return loss

    return _FisherSAM(params)
