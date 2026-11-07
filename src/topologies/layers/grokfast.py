from math import isfinite
from numbers import Real

import torch
from torch import nn


class GrokfastStep(nn.Module):
    component_id = "grokfast"

    def __init__(self, alpha=0.98, lamb=2.0):
        super().__init__()
        if isinstance(alpha, bool) or not isinstance(alpha, Real):
            alpha = float("nan")
        else:
            try:
                alpha = float(alpha)
            except (OverflowError, ValueError):
                alpha = float("nan")
        if not isfinite(alpha) or alpha < 0 or alpha >= 1:
            raise ValueError("alpha must be a finite number in [0, 1)")
        if isinstance(lamb, bool) or not isinstance(lamb, Real):
            lamb = float("nan")
        else:
            try:
                lamb = float(lamb)
            except (OverflowError, ValueError):
                lamb = float("nan")
        if not isfinite(lamb) or lamb < 0:
            raise ValueError("lamb must be a finite nonnegative number")
        self.alpha = alpha
        self.lamb = lamb
        self._emas = {}
        self._hook_handles = []

    def _validate_gradient(self, gradient):
        if not isinstance(gradient, torch.Tensor):
            raise TypeError("gradient must be a torch.Tensor")
        if gradient.layout != torch.strided:
            raise ValueError("GrokfastStep requires dense gradients")

    def _transform(self, gradient, state_key):
        self._validate_gradient(gradient)
        previous = self._emas.get(state_key)
        if previous is None:
            ema = gradient.detach().clone()
        else:
            previous = previous.to(device=gradient.device, dtype=gradient.dtype)
            ema = previous * self.alpha + gradient.detach() * (1.0 - self.alpha)
        self._emas[state_key] = ema.detach()
        return gradient + self.lamb * ema

    def forward(self, gradient):
        self._validate_gradient(gradient)
        return gradient + self.lamb * gradient

    def _make_hook(self, state_key):
        def hook(gradient):
            return self._transform(gradient, state_key)

        return hook

    def bind(self, model):
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        self.unbind()
        for parameter in model.parameters():
            if parameter.requires_grad:
                self._hook_handles.append(
                    parameter.register_hook(self._make_hook(id(parameter)))
                )
        return self

    def unbind(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._emas.clear()
        return self
