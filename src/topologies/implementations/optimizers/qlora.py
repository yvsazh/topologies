"""Adam optimizer suitable for optimizing QLoRA adapter parameters."""

import torch


def qlora(params, *, lr, weight_decay=0.0, **backend_options):
    """Build an Adam optimizer for trainable low-rank adapter parameters.

    Quantized base-model weights remain frozen by the model; this optimizer
    handles the trainable parameters passed to it using standard Adam updates.
    """
    betas = backend_options.pop("betas", (0.9, 0.999))
    eps = backend_options.pop("eps", 1e-8)
    amsgrad = backend_options.pop("amsgrad", False)
    if backend_options:
        raise TypeError(f"unsupported QLoRA options: {', '.join(backend_options)}")
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay,
                            betas=betas, eps=eps, amsgrad=amsgrad)
