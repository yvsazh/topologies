import torch


def sgd(params, *, lr, weight_decay=0.0, momentum=0.9, dampening=0.0, nesterov=False, **backend_options):
    return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum,
                           dampening=dampening, nesterov=nesterov, **backend_options)


def adam(params, *, lr, weight_decay=0.0, eps=1e-8, betas=(0.9, 0.999), amsgrad=False, **backend_options):
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, eps=eps, betas=betas,
                            amsgrad=amsgrad, **backend_options)


def adamw(params, *, lr, weight_decay=0.01, eps=1e-8, betas=(0.9, 0.999), amsgrad=False, **backend_options):
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, eps=eps, betas=betas,
                             amsgrad=amsgrad, **backend_options)


def rmsprop(params, *, lr, alpha=0.99, eps=1e-8, weight_decay=0.0, momentum=0.0,
            centered=False, **backend_options):
    return torch.optim.RMSprop(params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay,
                               momentum=momentum, centered=centered, **backend_options)


def adagrad(params, *, lr=0.01, lr_decay=0.0, weight_decay=0.0,
            initial_accumulator_value=0.0, eps=1e-10, **backend_options):
    return torch.optim.Adagrad(params, lr=lr, lr_decay=lr_decay, weight_decay=weight_decay,
                               initial_accumulator_value=initial_accumulator_value, eps=eps,
                               **backend_options)
