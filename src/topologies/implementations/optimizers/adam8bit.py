"""Adam optimizer with moment buffers quantized to 8-bit levels."""

import torch


def _quantize_8bit(tensor):
    """Symmetrically quantize a tensor to 256 representable levels."""
    if tensor.numel() == 0:
        return tensor
    scale = tensor.abs().max() / 127
    if scale == 0:
        return tensor
    return (tensor / scale).round().clamp(-128, 127) * scale


def adam8bit(params, *, lr, weight_decay=0.0, **backend_options):
    """Construct Adam while quantizing first and second moments to 8-bit levels."""
    betas = backend_options.pop("betas", (0.9, 0.999))
    eps = backend_options.pop("eps", 1e-8)
    amsgrad = backend_options.pop("amsgrad", False)
    if backend_options:
        raise TypeError(f"unsupported Adam8bit options: {', '.join(backend_options)}")
    if lr < 0:
        raise ValueError("lr must be non-negative")
    if eps < 0:
        raise ValueError("eps must be non-negative")
    if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
        raise ValueError("betas must be in [0, 1)")

    class _Adam8bit(torch.optim.Optimizer):
        def __init__(self, parameters):
            super().__init__(parameters, {"lr": lr, "weight_decay": weight_decay,
                                              "betas": betas, "eps": eps, "amsgrad": amsgrad})

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
                        raise RuntimeError("Adam8bit does not support sparse gradients")
                    state = self.state[parameter]
                    if not state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(parameter)
                        state["exp_avg_sq"] = torch.zeros_like(parameter)
                        if group["amsgrad"]:
                            state["max_exp_avg_sq"] = torch.zeros_like(parameter)
                    state["step"] += 1
                    step = state["step"]
                    if group["weight_decay"]:
                        grad = grad.add(parameter, alpha=group["weight_decay"])
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    exp_avg.copy_(_quantize_8bit(exp_avg.mul(beta1).add(grad, alpha=1 - beta1)))
                    exp_avg_sq.copy_(_quantize_8bit(
                        exp_avg_sq.mul(beta2).addcmul(grad, grad, value=1 - beta2)
                    ))
                    variance = exp_avg_sq
                    if group["amsgrad"]:
                        variance = state["max_exp_avg_sq"]
                        torch.maximum(variance, exp_avg_sq, out=variance)
                    bias1 = 1 - beta1 ** step
                    bias2 = 1 - beta2 ** step
                    denominator = variance.sqrt().div_(bias2 ** 0.5).add_(group["eps"])
                    parameter.addcdiv_(exp_avg, denominator, value=-group["lr"] / bias1)
            return loss

    return _Adam8bit(params)
