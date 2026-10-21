"""A lightweight Flora gradient-descent optimizer."""

import torch


class Flora(torch.optim.Optimizer):
    """Apply gradient descent with optional coupled weight decay."""

    def __init__(self, params, *, lr, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        super().__init__(params, {"lr": lr, "weight_decay": weight_decay})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if weight_decay:
                    gradient = gradient.add(parameter, alpha=weight_decay)
                parameter.add_(gradient, alpha=-lr)
        return loss


def flora(params, *, lr, weight_decay=0.0, **backend_options):
    """Build the local Flora optimizer; backend options are unsupported."""
    if backend_options:
        unknown = ", ".join(sorted(backend_options))
        raise TypeError(f"Unsupported Flora backend option(s): {unknown}")
    return Flora(params, lr=lr, weight_decay=weight_decay)
