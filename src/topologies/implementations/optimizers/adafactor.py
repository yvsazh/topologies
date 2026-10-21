"""A compact Adafactor optimizer factory."""

import torch


def adafactor(params, *, lr, weight_decay=0.0, **backend_options):
    """Build Adafactor, preferring PyTorch's implementation when available."""
    native = getattr(torch.optim, "Adafactor", None)
    if native is not None:
        return native(params, lr=lr, weight_decay=weight_decay, **backend_options)
    if backend_options:
        unsupported = ", ".join(sorted(backend_options))
        raise TypeError(f"fallback Adafactor does not support backend options: {unsupported}")

    class _Adafactor(torch.optim.Optimizer):
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
                        raise RuntimeError("Adafactor does not support sparse gradients")
                    state = self.state[parameter]
                    if not state:
                        state["step"] = 0
                        if parameter.ndim >= 2:
                            state["row"] = torch.zeros_like(grad.mean(dim=-1), memory_format=torch.preserve_format)
                            state["col"] = torch.zeros_like(grad.mean(dim=-2), memory_format=torch.preserve_format)
                        else:
                            state["variance"] = torch.zeros_like(parameter)
                    state["step"] += 1
                    decay = 1.0 - state["step"] ** -0.8
                    squared = grad.to(torch.float32).square().add_(1e-30)
                    if parameter.ndim >= 2:
                        row, col = state["row"], state["col"]
                        row.mul_(decay).add_(squared.mean(dim=-1), alpha=1.0 - decay)
                        col.mul_(decay).add_(squared.mean(dim=-2), alpha=1.0 - decay)
                        row_factor = (row / row.mean(dim=-1, keepdim=True).clamp_min(1e-30)).rsqrt()
                        col_factor = col.rsqrt()
                        update = squared.new_ones(())
                        update = grad * (row_factor.unsqueeze(-1) * col_factor.unsqueeze(-2)).to(grad.dtype)
                    else:
                        variance = state["variance"]
                        variance.mul_(decay).add_(squared, alpha=1.0 - decay)
                        update = grad * variance.rsqrt().to(grad.dtype)
                    rms = update.float().square().mean().sqrt().clamp_min(1e-30)
                    update = update / torch.maximum(rms / 1.0, rms.new_ones(())).to(update.dtype)
                    if group["weight_decay"]:
                        parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                    parameter.add_(update, alpha=-group["lr"])
            return loss

    return _Adafactor(params)
