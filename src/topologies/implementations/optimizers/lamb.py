"""Layer-wise adaptive moments optimizer (LAMB)."""

import torch


class _LAMB(torch.optim.Optimizer):
    """Small eager LAMB implementation using local parameter tensors."""

    def __init__(self, params, *, lr, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-6):
        if lr < 0:
            raise ValueError("lr must be non-negative")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if eps < 0:
            raise ValueError("eps must be non-negative")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("betas must be in [0, 1)")
        super().__init__(params, {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("LAMB does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias1 = 1 - beta1 ** state["step"]
                bias2 = 1 - beta2 ** state["step"]
                update = (exp_avg / bias1) / ((exp_avg_sq / bias2).sqrt() + group["eps"])
                if group["weight_decay"]:
                    update = update.add(parameter, alpha=group["weight_decay"])
                weight_norm = torch.linalg.vector_norm(parameter)
                update_norm = torch.linalg.vector_norm(update)
                trust_ratio = torch.where(
                    (weight_norm > 0) & (update_norm > 0), weight_norm / update_norm,
                    torch.ones_like(weight_norm),
                )
                parameter.add_(update * trust_ratio, alpha=-group["lr"])
        return loss


def lamb(params, *, lr, weight_decay=0.0, **backend_options):
    """Create a LAMB optimizer, forwarding supported options to its implementation."""
    implementation = getattr(torch.optim, "LAMB", None)
    if implementation is not None:
        return implementation(params, lr=lr, weight_decay=weight_decay, **backend_options)
    return _LAMB(params, lr=lr, weight_decay=weight_decay, **backend_options)
