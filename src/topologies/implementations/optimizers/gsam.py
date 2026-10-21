"""Gradient sharpness-aware minimization optimizer factory."""

import torch


def gsam(params, *, lr, weight_decay=0.0, **backend_options):
    """Construct a lightweight GSAM-style optimizer.

    A closure performs the second forward/backward pass at the perturbed
    parameters. Without a closure this behaves as the base-gradient update,
    which permits standard Optimizer usage where no second pass is available.
    """
    rho = backend_options.pop("rho", 0.05)
    if backend_options:
        raise TypeError(f"unsupported GSAM options: {', '.join(backend_options)}")
    if lr < 0:
        raise ValueError("lr must be non-negative")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if rho < 0:
        raise ValueError("rho must be non-negative")

    class _GSAM(torch.optim.Optimizer):
        def __init__(self, parameters):
            super().__init__(parameters, {"lr": lr, "weight_decay": weight_decay, "rho": rho})

        @torch.no_grad()
        def step(self, closure=None):
            if closure is None:
                return self._base_step()
            with torch.enable_grad():
                loss = closure()
            parameters = [p for group in self.param_groups for p in group["params"]
                          if p.grad is not None]
            norm = torch.norm(torch.stack([p.grad.norm(2) for p in parameters]), 2) if parameters else None
            if norm is None or norm.item() == 0:
                self._base_step()
                return loss
            perturbations = []
            for group in self.param_groups:
                scale = group["rho"] / (norm + 1e-12)
                for p in group["params"]:
                    if p.grad is not None:
                        perturbation = p.grad * scale
                        p.add_(perturbation)
                        perturbations.append((p, perturbation))
            try:
                self.zero_grad()
                with torch.enable_grad():
                    loss = closure()
            finally:
                for p, perturbation in perturbations:
                    p.sub_(perturbation)
            self._base_step()
            return loss

        def _base_step(self):
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if group["weight_decay"]:
                        grad = grad.add(p, alpha=group["weight_decay"])
                    p.add_(grad, alpha=-group["lr"])

    return _GSAM(params)
