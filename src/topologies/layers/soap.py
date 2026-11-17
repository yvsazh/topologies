import math

import torch
from torch import nn


class SoapStep(nn.Module):
    component_id = "soap"

    def __init__(
        self,
        beta1=0.95,
        beta2=0.95,
        eps=1e-8,
        lr=3e-3,
        preconditioning_frequency=10,
    ):
        super().__init__()
        if isinstance(beta1, bool) or not isinstance(beta1, (int, float)) or not 0 <= beta1 < 1:
            raise ValueError("beta1 must be a number in [0, 1)")
        if isinstance(beta2, bool) or not isinstance(beta2, (int, float)) or not 0 <= beta2 < 1:
            raise ValueError("beta2 must be a number in [0, 1)")
        if (
            isinstance(eps, bool)
            or not isinstance(eps, (int, float))
            or not math.isfinite(eps)
            or eps <= 0
        ):
            raise ValueError("eps must be a finite positive number")
        if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr < 0:
            raise ValueError("lr must be a finite nonnegative number")
        if (
            isinstance(preconditioning_frequency, bool)
            or not isinstance(preconditioning_frequency, int)
            or preconditioning_frequency <= 0
        ):
            raise ValueError("preconditioning_frequency must be a positive integer")
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.lr = float(lr)
        self.preconditioning_frequency = preconditioning_frequency
        self.register_buffer("M", None)
        self.register_buffer("V", None)
        self.register_buffer("L", None)
        self.register_buffer("R", None)
        self.register_buffer("Q_L", None)
        self.register_buffer("Q_R", None)
        self.register_buffer("_initialized", torch.tensor(False))
        self.register_buffer("_active_updates", torch.tensor(0, dtype=torch.long))

    def forward(self, weight, gradient):
        if not isinstance(weight, torch.Tensor) or not isinstance(gradient, torch.Tensor):
            raise TypeError("weight and gradient must be tensors")
        if weight.ndim != 2 or gradient.ndim != 2:
            raise ValueError("weight and gradient must be 2-D tensors")
        if weight.shape != gradient.shape:
            raise ValueError("weight and gradient must have the same shape")
        if not weight.is_floating_point() or not gradient.is_floating_point():
            raise ValueError("weight and gradient must have floating point dtypes")
        if weight.device != gradient.device or weight.dtype != gradient.dtype:
            raise ValueError("weight and gradient must have the same device and dtype")
        if not self.training:
            return weight
        initialized = bool(self._initialized.item())
        if initialized and self.M.shape != weight.shape:
            self._initialized = torch.tensor(False, device=weight.device)
            self._active_updates = torch.zeros((), dtype=torch.long, device=weight.device)
            initialized = False
        if initialized and (self.M.device != weight.device or self.M.dtype != weight.dtype):
            raise ValueError("weight device and dtype must remain constant after SOAP state initialization")
        if not initialized:
            self.M = torch.zeros_like(weight)
            self.V = torch.zeros_like(weight)
            self.L = ((1 - self.beta2) * (gradient @ gradient.T)).detach()
            self.R = ((1 - self.beta2) * (gradient.T @ gradient)).detach()
            self.Q_L = torch.linalg.eigh(self.L)[1].detach()
            self.Q_R = torch.linalg.eigh(self.R)[1].detach()
            self._initialized = torch.tensor(True, device=weight.device)
            return weight

        projected_gradient = self.Q_L.T @ gradient @ self.Q_R
        first_moment = self.beta1 * self.M + (1 - self.beta1) * gradient
        moment_projection = self.Q_L.T @ first_moment @ self.Q_R
        self.V = self.beta2 * self.V + (1 - self.beta2) * projected_gradient.square()
        update_number = int(self._active_updates.item()) + 1
        corrected_first_moment = moment_projection / (1 - self.beta1**update_number)
        corrected_second_moment = self.V / (1 - self.beta2**update_number)
        second_moment_root = torch.sqrt(
            corrected_second_moment.clamp_min(torch.finfo(corrected_second_moment.dtype).tiny)
        )
        projected_update = corrected_first_moment / (second_moment_root + self.eps)
        update = self.Q_L @ projected_update @ self.Q_R.T
        output = weight - self.lr * update

        self.L = self.beta2 * self.L + (1 - self.beta2) * (gradient @ gradient.T)
        self.R = self.beta2 * self.R + (1 - self.beta2) * (gradient.T @ gradient)
        self.M = first_moment.detach()
        self.V = self.V.detach()
        self.L = self.L.detach()
        self.R = self.R.detach()
        self._active_updates = self._active_updates + 1
        if int(self._active_updates.item()) % self.preconditioning_frequency == 0:
            self.Q_L = torch.linalg.eigh(self.L)[1].detach()
            self.Q_R = torch.linalg.eigh(self.R)[1].detach()
        return output
