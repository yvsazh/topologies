import math
import numbers

import torch
from torch import nn


def _is_finite_real(value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


class StructuredNeuralClf(nn.Module):
    architecture_id = "structured-neural-clf"
    PRESETS = {
        "tiny": {"mu": 0.5, "nu": 2.0, "K": 1, "n": 2, "mk": 2, "L": 1},
        "small": {"mu": 0.5, "nu": 2.0, "K": 2, "n": 2, "mk": 6, "L": 2},
    }

    def __init__(self, mu=0.5, nu=2.0, K=2, n=2, mk=6, L=2):
        super().__init__()
        for name, value in (("n", n), ("K", K), ("mk", mk), ("L", L)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not _is_finite_real(mu) or not _is_finite_real(nu):
            raise ValueError("mu and nu must be finite real numbers")
        if mu <= 0 or nu < mu:
            raise ValueError("bounds must satisfy 0 < mu <= nu")

        self.mu = float(mu)
        self.nu = float(nu)
        self.K = K
        self.n = n
        self.mk = mk
        self.L = L
        self.width = mk * L
        self.Fp = nn.ParameterList()
        self.Fq = nn.ParameterList()
        self.d = nn.ParameterList()
        self.Fa = nn.ParameterList()
        self.Fb = nn.ModuleList()
        self.b = nn.ParameterList()
        self.by = nn.ParameterList()
        for _ in range(K):
            self.Fp.append(nn.Parameter(torch.empty(n, n).uniform_(-1.0, 1.0)))
            self.Fq.append(nn.Parameter(torch.empty(self.width, n).uniform_(-1.0, 1.0)))
            self.d.append(nn.Parameter(torch.empty(L, mk).uniform_(-1.0, 1.0)))
            self.Fa.append(nn.Parameter(torch.empty(L, mk, mk).uniform_(-1.0, 1.0)))
            block_fb = nn.ParameterList(
                [nn.Parameter(torch.empty(0, mk).uniform_(-1.0, 1.0))]
                + [nn.Parameter(torch.empty(mk, mk).uniform_(-1.0, 1.0)) for _ in range(L - 1)]
            )
            self.Fb.append(block_fb)
            self.b.append(nn.Parameter(torch.empty(L, mk).uniform_(-1.0, 1.0)))
            self.by.append(nn.Parameter(torch.empty(n).uniform_(-1.0, 1.0)))
        self.orthogonal_A = nn.ParameterList(
            [nn.Parameter(torch.zeros(n, n)) for _ in range(K + 1)]
        )
        self.orthogonal_q = nn.ParameterList(
            [nn.Parameter(torch.zeros(n)) for _ in range(K + 1)]
        )

    def _cayley(self, matrix, extension):
        identity = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
        z = matrix.t() - matrix + extension.t() @ extension
        inverse = torch.linalg.inv(identity + z)
        top = inverse @ (identity - z)
        bottom = -2.0 * extension @ inverse
        return torch.cat((top, bottom), dim=0)

    def _feed_through(self, x, block_index):
        global_frame = self._cayley(self.Fp[block_index], self.Fq[block_index])
        q_blocks = torch.split(global_frame[self.n :], self.mk, dim=0)
        mu_k = self.mu ** (1.0 / self.K)
        nu_k = self.nu ** (1.0 / self.K)
        gamma = nu_k - mu_k
        inputs = []
        recurrences = []
        outputs = []
        previous_a = None
        previous_psi = None

        for layer_index in range(self.L):
            frame = self._cayley(self.Fa[block_index][layer_index], self.Fb[block_index][layer_index])
            a = frame[: self.mk].t()
            b = frame[self.mk :].t()
            psi = torch.diag(torch.exp(self.d[block_index][layer_index]))
            psi_inverse = torch.diag(torch.exp(-self.d[block_index][layer_index]))
            s = a @ q_blocks[layer_index]
            if layer_index > 0:
                s = s - b @ q_blocks[layer_index - 1]
                v = 2.0 * b @ previous_a.t()
                recurrences.append(psi_inverse @ v @ previous_psi)
            else:
                recurrences.append(x.new_zeros((self.mk, self.mk)))
            scale = math.sqrt(2.0 * gamma)
            inputs.append(scale * psi_inverse @ s)
            outputs.append(math.sqrt(gamma / 2.0) * s.t() @ psi)
            previous_a = a
            previous_psi = psi

        hidden = []
        activation = x.new_zeros((x.shape[0], self.mk))
        for layer_index in range(self.L):
            activation = torch.relu(
                activation @ recurrences[layer_index].t()
                + x @ inputs[layer_index].t()
                + self.b[block_index][layer_index]
            )
            hidden.append(activation)

        result = mu_k * x + self.by[block_index]
        for layer_index in range(self.L):
            result = result + hidden[layer_index] @ outputs[layer_index].t()
        return result

    def _orthogonal(self, x, layer_index):
        parameter = self.orthogonal_A[layer_index]
        skew = parameter - parameter.t()
        identity = torch.eye(self.n, device=parameter.device, dtype=parameter.dtype)
        matrix = (identity - skew) @ torch.linalg.inv(identity + skew)
        return x @ matrix.t() + self.orthogonal_q[layer_index]

    def _bi_lip(self, x):
        x = self._orthogonal(x, 0)
        for block_index in range(self.K):
            x = self._feed_through(x, block_index)
            x = self._orthogonal(x, block_index + 1)
        return x

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[-1] != self.n:
            raise ValueError(f"x must have shape (batch, {self.n})")
        if not x.is_floating_point():
            raise ValueError("x must have a floating-point dtype")
        origin = torch.zeros((1, self.n), device=x.device, dtype=x.dtype)
        difference = self._bi_lip(x) - self._bi_lip(origin)
        return 0.5 * difference.square().sum(dim=-1)
