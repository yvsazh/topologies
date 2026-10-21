"""Adaptive Sharpness-Aware Minimization optimizer."""

import torch


class _ASAM(torch.optim.Optimizer):
    def __init__(self, params, *, lr, weight_decay=0.0, rho=0.5, eta=0.01):
        if lr < 0.0:
            raise ValueError(f"invalid learning rate: {lr}")
        if rho < 0.0:
            raise ValueError(f"invalid rho: {rho}")
        if weight_decay < 0.0:
            raise ValueError(f"invalid weight_decay: {weight_decay}")
        super().__init__(params, {"lr": lr, "weight_decay": weight_decay, "rho": rho, "eta": eta})

    @torch.no_grad()
    def _adaptive_grad_norm(self):
        norms = []
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    scale = parameter.abs().add(group["eta"])
                    norms.append(torch.norm(scale * parameter.grad, p=2))
        if not norms:
            return None
        return torch.norm(torch.stack(norms), p=2)

    @torch.no_grad()
    def step(self, closure=None):
        """Apply an ASAM step; closure enables the canonical two-pass update.

        Without a closure, perform the available first-order descent step. This
        keeps the optimizer usable with the conventional backward()/step() API.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        norm = self._adaptive_grad_norm()
        if norm is None:
            return loss

        perturbations = []
        if closure is not None:
            for group in self.param_groups:
                scale = group["rho"] / (norm + 1e-12)
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue
                    perturbation = parameter.abs().add(group["eta"]) ** 2 * parameter.grad * scale
                    parameter.add_(perturbation)
                    perturbations.append((parameter, perturbation))
            self.zero_grad(set_to_none=True)
            with torch.enable_grad():
                loss = closure()
            for parameter, perturbation in perturbations:
                parameter.sub_(perturbation)

        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    if group["weight_decay"]:
                        parameter.grad.add_(parameter, alpha=group["weight_decay"])
                    parameter.add_(parameter.grad, alpha=-group["lr"])
        return loss


def asam(params, *, lr, weight_decay=0.0, **backend_options):
    """Build an ASAM optimizer, forwarding supported options to its backend."""
    return _ASAM(params, lr=lr, weight_decay=weight_decay, **backend_options)
