"""AdaHessian optimizer factory with Hutchinson Hessian-diagonal estimates."""

import torch


def adahessian(params, *, lr, weight_decay=0.0, **backend_options):
    """Create an AdaHessian optimizer.

    Hessian diagonal magnitude is estimated using Hutchinson probes from the
    gradient graph. Call ``loss.backward(create_graph=True)`` before ``step``.
    """
    betas = backend_options.pop("betas", (0.9, 0.999))
    eps = backend_options.pop("eps", 1e-8)
    hessian_power = backend_options.pop("hessian_power", 1.0)
    n_samples = backend_options.pop("n_samples", 1)
    if backend_options:
        raise TypeError(f"unsupported AdaHessian options: {', '.join(backend_options)}")
    if lr < 0 or weight_decay < 0 or eps < 0:
        raise ValueError("lr, weight_decay, and eps must be non-negative")
    if len(betas) != 2 or not all(0 <= beta < 1 for beta in betas):
        raise ValueError("betas must contain values in [0, 1)")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")

    class _AdaHessian(torch.optim.Optimizer):
        def __init__(self, parameters):
            super().__init__(parameters, {"lr": lr, "weight_decay": weight_decay,
                                               "betas": betas, "eps": eps,
                                               "hessian_power": hessian_power,
                                               "n_samples": n_samples})

        @torch.no_grad()
        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            for group in self.param_groups:
                parameters = [p for p in group["params"] if p.grad is not None]
                grads = [p.grad for p in parameters]
                if any(not grad.requires_grad for grad in grads):
                    raise RuntimeError("AdaHessian requires loss.backward(create_graph=True)")
                hessians = [torch.zeros_like(p) for p in parameters]
                for _ in range(group["n_samples"]):
                    vectors = [torch.empty_like(p).bernoulli_(0.5).mul_(2).sub_(1)
                               for p in parameters]
                    products = torch.autograd.grad(
                        grads, parameters, grad_outputs=vectors,
                        retain_graph=True, allow_unused=True)
                    for index, (product, vector) in enumerate(zip(products, vectors)):
                        if product is not None:
                            hessians[index].add_(product * vector)
                for hessian in hessians:
                    hessian.div_(group["n_samples"]).abs_()
                beta1, beta2 = group["betas"]
                for parameter, grad, hessian in zip(parameters, grads, hessians):
                    state = self.state[parameter]
                    if not state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(parameter)
                        state["exp_hessian"] = torch.zeros_like(parameter)
                    state["step"] += 1
                    step = state["step"]
                    grad = grad.detach()
                    if group["weight_decay"]:
                        grad = grad.add(parameter, alpha=group["weight_decay"])
                    avg, avg_hessian = state["exp_avg"], state["exp_hessian"]
                    avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    avg_hessian.mul_(beta2).add_(hessian, alpha=1 - beta2)
                    corrected_avg = avg / (1 - beta1 ** step)
                    corrected_hessian = avg_hessian / (1 - beta2 ** step)
                    denominator = corrected_hessian.pow(group["hessian_power"]).add_(group["eps"])
                    parameter.addcdiv_(corrected_avg, denominator, value=-group["lr"])
            return loss

    return _AdaHessian(params)
