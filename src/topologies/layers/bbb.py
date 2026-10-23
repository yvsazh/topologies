import math
import numbers

import torch
from torch import nn
from torch.nn import functional as F


class BayesLinear(nn.Module):
    component_id = "bbb"

    def _require(self, condition, message):
        if not condition:
            raise ValueError(message)

    def __init__(self, in_features, out_features, prior_std, bias=True):
        super().__init__()
        self._require(
            isinstance(in_features, int) and not isinstance(in_features, bool) and in_features > 0,
            "in_features must be a positive integer",
        )
        self._require(
            isinstance(out_features, int) and not isinstance(out_features, bool) and out_features > 0,
            "out_features must be a positive integer",
        )
        self._require(
            isinstance(prior_std, numbers.Real) and not isinstance(prior_std, bool),
            "prior_std must be a finite positive number",
        )
        self._require(
            math.isfinite(float(prior_std)) and prior_std > 0,
            "prior_std must be a finite positive number",
        )
        self._require(isinstance(bias, bool), "bias must be a boolean")
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = float(prior_std)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_rho = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_rho", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        nn.init.constant_(self.weight_rho, -5.0)
        if self.bias_mu is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias_mu, -bound, bound)
            nn.init.constant_(self.bias_rho, -5.0)

    def forward(self, x, weight_epsilon=None, bias_epsilon=None):
        if x.dim() == 0 or x.shape[-1] != self.in_features:
            raise ValueError(f"x must have a last dimension of {self.in_features}")
        if weight_epsilon is None:
            weight_epsilon = torch.randn_like(self.weight_mu)
        elif weight_epsilon.shape != self.weight_mu.shape:
            raise ValueError("weight_epsilon must match the weight parameter shape")
        weight = self.weight_mu + F.softplus(self.weight_rho) * weight_epsilon
        if self.bias_mu is None:
            if bias_epsilon is not None:
                raise ValueError("bias_epsilon cannot be supplied when bias is disabled")
            return F.linear(x, weight)
        if bias_epsilon is None:
            bias_epsilon = torch.randn_like(self.bias_mu)
        elif bias_epsilon.shape != self.bias_mu.shape:
            raise ValueError("bias_epsilon must match the bias parameter shape")
        bias = self.bias_mu + F.softplus(self.bias_rho) * bias_epsilon
        return F.linear(x, weight, bias)

    def _kl_term(self, mean, rho):
        scale = F.softplus(rho)
        prior_std = mean.new_tensor(self.prior_std)
        return (
            torch.log(prior_std / scale)
            + (scale.square() + mean.square()) / (2 * prior_std.square())
            - 0.5
        ).sum()

    def kl_divergence(self):
        kl = self._kl_term(self.weight_mu, self.weight_rho)
        if self.bias_mu is not None:
            kl = kl + self._kl_term(self.bias_mu, self.bias_rho)
        return kl
