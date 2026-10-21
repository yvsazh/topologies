"""AdaBelief optimizer, with the standard PyTorch optimizer interface."""

import torch


class _AdaBelief(torch.optim.Optimizer):
    """Small single-tensor implementation of AdaBelief."""

    def __init__(self, params, *, lr, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
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
                    raise RuntimeError("AdaBelief does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_var"] = torch.zeros_like(parameter)
                state["step"] += 1
                if group["weight_decay"]:
                    grad = grad.add(parameter, alpha=group["weight_decay"])
                mean, variance = state["exp_avg"], state["exp_avg_var"]
                residual = grad - mean
                mean.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                variance.mul_(beta2).addcmul_(residual, residual, value=1.0 - beta2)
                step = state["step"]
                mean_hat = mean / (1.0 - beta1**step)
                variance_hat = variance / (1.0 - beta2**step)
                parameter.addcdiv_(mean_hat, variance_hat.sqrt().add_(group["eps"]), value=-group["lr"])
        return loss


def adabelief(params, *, lr, weight_decay=0.0, **backend_options):
    """Construct AdaBelief; backend options are forwarded to the optimizer."""
    optimizer_type = getattr(torch.optim, "AdaBelief", _AdaBelief)
    return optimizer_type(params, lr=lr, weight_decay=weight_decay, **backend_options)
